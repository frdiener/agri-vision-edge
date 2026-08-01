"""
Tests for the prediction-integrity guard.

Regression cover for the i.MX8MP sweep: the Teflon delegate handed an fp32 SSD
graph returned NaN boxes and a constant out-of-range score tensor, and the
pipeline reported **AP 85.8** for it -- higher than any healthy INT8 run --
because ``pycocotools`` treats a NaN IoU as a match at every IoU threshold.

The first test pins that mechanism (it is the reason this guard has to exist and
why a warning is not enough); the rest pin the guard itself.
"""

from __future__ import annotations

import contextlib
import io
import json
import math

import pytest

from agri_vision_edge.evaluation.integrity import (
    CorruptPredictionsError,
    check_predictions,
    prediction_integrity,
)


def _pred(bbox, score=0.9, image_id=1, category_id=1):
    return {
        "image_id": image_id,
        "category_id": category_id,
        "bbox": list(bbox),
        "score": score,
    }


def _gt_dataset():
    return {
        "images": [{"id": 1, "file_name": "a.png", "width": 100, "height": 100}],
        "categories": [{"id": 1, "name": "crop"}],
        "annotations": [
            {
                "id": i,
                "image_id": 1,
                "category_id": 1,
                "bbox": [10 * i, 10 * i, 8, 8],
                "area": 64,
                "iscrowd": 0,
            }
            for i in range(1, 6)
        ],
    }


def test_nan_boxes_make_pycocotools_report_a_high_threshold_independent_ap():
    """
    Why this guard exists: NaN boxes score *better* than real ones.

    ``COCOeval`` skips a candidate with ``if ious[dind, gind] < iou: continue``.
    ``NaN < t`` is ``False``, so the detection is accepted -- at every IoU
    threshold, since the comparison never depends on it. Hence a near-perfect AP
    and the tell-tale ``AP == AP50``.
    """

    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    coco = COCO()
    coco.dataset = _gt_dataset()
    with contextlib.redirect_stdout(io.StringIO()):
        coco.createIndex()

    nan = float("nan")
    detections = [
        _pred([nan, nan, nan, nan], score=1.0 - 0.01 * i) for i in range(20)
    ]

    with contextlib.redirect_stdout(io.StringIO()):
        evaluator = COCOeval(coco, coco.loadRes(detections), "bbox")
        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()

    ap, ap50 = evaluator.stats[0], evaluator.stats[1]

    assert ap > 0.9, "NaN boxes should (pathologically) score near-perfectly"
    assert ap == pytest.approx(ap50, abs=1e-12), "AP == AP50 is the fingerprint"


def test_clean_predictions_pass():
    predictions = [_pred([1, 2, 3, 4], score=0.5), _pred([5, 6, 7, 8], score=0.0)]

    integrity = check_predictions(predictions)

    assert not integrity.corrupt
    assert integrity.total == 2
    assert integrity.non_finite_boxes == 0
    assert (integrity.score_min, integrity.score_max) == (0.0, 0.5)


@pytest.mark.parametrize(
    "bbox",
    [
        [float("nan")] * 4,
        [float("inf"), 0.0, 1.0, 1.0],
        [0.0, 0.0, 1.0],  # wrong arity
    ],
)
def test_non_finite_or_malformed_boxes_are_rejected(bbox):
    with pytest.raises(CorruptPredictionsError, match="non-finite"):
        check_predictions([_pred(bbox)])


@pytest.mark.parametrize("score", [1.9, 6.0, -0.5])
def test_out_of_range_scores_are_rejected(score):
    # A "score" of 6.0 means the tensor being read is not a score tensor.
    with pytest.raises(CorruptPredictionsError, match=r"outside \[0, 1\]"):
        check_predictions([_pred([1, 2, 3, 4], score=score)])


def test_non_finite_scores_are_rejected():
    with pytest.raises(CorruptPredictionsError):
        check_predictions([_pred([1, 2, 3, 4], score=float("nan"))])


def test_degenerate_boxes_are_reported_but_not_fatal():
    # A detector may legitimately emit a zero-area box.
    integrity = check_predictions([_pred([1, 2, 0, 5]), _pred([1, 2, 3, 4])])

    assert integrity.degenerate_boxes == 1
    assert not integrity.corrupt


def test_strict_false_downgrades_to_a_warning(capsys):
    integrity = check_predictions([_pred([float("nan")] * 4)], strict=False)

    assert integrity.corrupt
    assert "Corrupt predictions" in capsys.readouterr().out


def test_message_names_the_source():
    with pytest.raises(CorruptPredictionsError, match="run-42/predictions.json"):
        check_predictions([_pred([float("nan")] * 4)], source="run-42/predictions.json")


def test_prediction_integrity_counts_every_symptom():
    integrity = prediction_integrity(
        [
            _pred([float("nan")] * 4, score=6.0),
            _pred([1, 2, 0, 4], score=0.5),
            _pred([1, 2, 3, 4], score=0.5),
        ]
    )

    assert integrity.total == 3
    assert integrity.non_finite_boxes == 1
    assert integrity.out_of_range_scores == 1
    assert integrity.degenerate_boxes == 1
    assert integrity.to_dict()["corrupt"] is True


#
# Integration with the evaluation entry points
#


def _write(tmp_path, predictions):
    ann = tmp_path / "ann.json"
    ann.write_text(json.dumps(_gt_dataset()))
    pred = tmp_path / "predictions.json"
    pred.write_text(json.dumps(predictions))
    return ann, pred


def test_evaluate_predictions_refuses_corrupt_input(tmp_path):
    from agri_vision_edge.evaluation.coco import evaluate_predictions

    ann, pred = _write(tmp_path, [_pred([float("nan")] * 4)])

    with pytest.raises(CorruptPredictionsError):
        evaluate_predictions(ann, pred)


def test_evaluate_predictions_allow_corrupt_opt_out(tmp_path):
    from agri_vision_edge.evaluation.coco import evaluate_predictions

    ann, pred = _write(tmp_path, [_pred([float("nan")] * 4)])

    with contextlib.redirect_stdout(io.StringIO()):
        metrics = evaluate_predictions(ann, pred, allow_corrupt=True)

    assert math.isfinite(metrics["AP"])


def test_evaluate_model_dir_skips_and_leaves_no_metrics(tmp_path):
    from agri_vision_edge.evaluation.coco import evaluate_model_dir

    run = tmp_path / "tiled_broken_model"
    run.mkdir()
    ann, _ = _write(tmp_path, [])
    (run / "predictions.json").write_text(json.dumps([_pred([float("nan")] * 4)]))

    # A stale result from an earlier, healthy run must not survive.
    (run / "metrics.json").write_text(json.dumps({"AP": 0.85}))

    with contextlib.redirect_stdout(io.StringIO()) as out:
        ok = evaluate_model_dir(run, ann)

    assert ok is False
    assert not (run / "metrics.json").exists()
    assert (run / "metrics_invalid.json").exists()
    assert "[skip]" in out.getvalue()


def test_benchmark_artifacts_record_integrity(tmp_path):
    from agri_vision_edge.evaluation.artifacts import save_benchmark_artifacts
    from agri_vision_edge.evaluation.benchmark import BenchmarkResult

    class _Runtime:
        active_delegate = None
        input_details: list = []
        output_details: list = []

    result = BenchmarkResult(
        predictions=[_pred([float("nan")] * 4, score=6.0), _pred([1, 2, 3, 4])],
        latencies_ms=[1.0, 2.0],
    )

    with contextlib.redirect_stdout(io.StringIO()) as out:
        save_benchmark_artifacts(
            output_dir=tmp_path / "run",
            benchmark_result=result,
            runtime=_Runtime(),
            model_name="m.tflite",
            delegate=None,
        )

    runtime_json = json.loads((tmp_path / "run" / "runtime.json").read_text())
    integrity = runtime_json["predictions_integrity"]

    assert integrity["corrupt"] is True
    assert integrity["non_finite_boxes"] == 1
    assert integrity["out_of_range_scores"] == 1
    assert "unusable predictions" in out.getvalue()
