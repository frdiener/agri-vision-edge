"""
Artifact wiring for the confidence sweep: ``ave evaluate`` -> report figures.

Covers what :mod:`tests.test_score_sweep` does not: that the sweep is written
beside ``metrics.json``, removed when a run is rejected, and loadable by the
reporting layer.

Requires ``pycocotools`` (the ``prep`` dependency group).
"""

from __future__ import annotations

import contextlib
import io
import json

import pytest

pytest.importorskip("pycocotools")
pytest.importorskip("matplotlib")

import matplotlib  # noqa: E402

matplotlib.use("Agg")

from agri_vision_edge.evaluation.coco import evaluate_model_dir  # noqa: E402
from agri_vision_edge.evaluation.score_sweep import (  # noqa: E402
    SCORE_SWEEP_FILENAME,
)

#: Parsed by ``benchmark_report.parse_run_name``.
RUN_NAME = "untiled_ssd-mn2_mc_phenobench_fp32_ptq_320_fastnms"


def _gt():
    return {
        "images": [{"id": 1, "file_name": "x.png", "width": 100, "height": 100}],
        "categories": [{"id": 1, "name": "crop"}, {"id": 2, "name": "weed"}],
        "annotations": [
            {
                "id": 1,
                "image_id": 1,
                "category_id": 1,
                "bbox": [10, 10, 20, 20],
                "area": 400,
                "iscrowd": 0,
            },
            {
                "id": 2,
                "image_id": 1,
                "category_id": 2,
                "bbox": [60, 60, 20, 20],
                "area": 400,
                "iscrowd": 0,
            },
        ],
    }


def _preds():
    return [
        {"image_id": 1, "category_id": 1, "bbox": [10, 10, 20, 20], "score": 0.9},
        {"image_id": 1, "category_id": 2, "bbox": [62, 60, 20, 20], "score": 0.6},
        {"image_id": 1, "category_id": 2, "bbox": [0, 0, 8, 8], "score": 0.2},
    ]


def _tree(tmp_path, predictions=None, platform="x86_cpu"):
    """Build a minimal ``benchmark_results``-shaped tree with one run."""

    root = tmp_path / "benchmark_results"
    run_dir = root / platform / RUN_NAME
    run_dir.mkdir(parents=True)

    (run_dir / "predictions.json").write_text(
        json.dumps(_preds() if predictions is None else predictions)
    )
    (run_dir / "latency.json").write_text(
        json.dumps(
            {
                "mean_latency_ms": 10.0,
                "median_latency_ms": 10.0,
                "min_latency_ms": 9.0,
                "max_latency_ms": 11.0,
                "latencies_ms": [10.0] * 5,
            }
        )
    )
    (run_dir / "runtime.json").write_text(
        json.dumps(
            {
                "model": RUN_NAME,
                "delegate": None,
                "backend": "cpu",
                "format": "tflite",
                "input_details": [],
                "output_details": [],
            }
        )
    )

    gt_path = tmp_path / "gt.json"
    gt_path.write_text(json.dumps(_gt()))

    return root, run_dir, gt_path


def _evaluate(run_dir, gt_path, **kwargs):
    with contextlib.redirect_stdout(io.StringIO()) as out:
        ok = evaluate_model_dir(run_dir, gt_path, **kwargs)

    return ok, out.getvalue()


# =========================================================
# Artifact writing
# =========================================================


def test_evaluate_writes_the_sweep_beside_metrics(tmp_path):
    _, run_dir, gt_path = _tree(tmp_path)

    ok, _ = _evaluate(run_dir, gt_path)

    assert ok
    assert (run_dir / "metrics.json").is_file()
    assert (run_dir / SCORE_SWEEP_FILENAME).is_file()


def test_evaluating_twice_leaves_metrics_json_byte_identical(tmp_path):
    """
    Backfilling the sweep must be purely additive: re-running evaluation is how
    an existing tree picks up score_sweep.json, and it must not rewrite the
    recorded metrics to do so.
    """

    _, run_dir, gt_path = _tree(tmp_path)

    _evaluate(run_dir, gt_path, score_sweep=False)

    metrics_path = run_dir / "metrics.json"

    before = metrics_path.read_bytes()
    before_mtime = metrics_path.stat().st_mtime_ns
    assert not (run_dir / SCORE_SWEEP_FILENAME).exists()

    _evaluate(run_dir, gt_path)

    assert metrics_path.read_bytes() == before
    # Unchanged content must not even be rewritten.
    assert metrics_path.stat().st_mtime_ns == before_mtime
    assert (run_dir / SCORE_SWEEP_FILENAME).is_file()


def test_operating_points_are_derived_not_stored(tmp_path):
    """Operating points come from the sweep; metrics.json stays threshold-free."""

    from agri_vision_edge.evaluation.score_sweep import (
        load_score_sweep,
        operating_points,
    )

    _, run_dir, gt_path = _tree(tmp_path)

    _evaluate(run_dir, gt_path)

    metrics = json.loads((run_dir / "metrics.json").read_text())
    assert "operating_points" not in metrics

    sweep = load_score_sweep(run_dir / SCORE_SWEEP_FILENAME)
    points = operating_points(sweep)

    assert set(points) == {"f1_best@0.50", "f1_best@0.75"}

    point = points["f1_best@0.50"]

    for key in ("score", "f1", "precision", "recall"):
        assert isinstance(point[key], float)

    assert set(point["per_class"]) == {"crop", "weed"}


def test_sweep_can_be_disabled(tmp_path):
    _, run_dir, gt_path = _tree(tmp_path)

    _evaluate(run_dir, gt_path, score_sweep=False)

    metrics = json.loads((run_dir / "metrics.json").read_text())

    assert not (run_dir / SCORE_SWEEP_FILENAME).exists()
    assert "operating_points" not in metrics


def test_corrupt_predictions_remove_a_stale_sweep(tmp_path):
    """A rejected run must not leave last evaluation's curve behind."""

    _, run_dir, gt_path = _tree(tmp_path)

    ok, _ = _evaluate(run_dir, gt_path)
    assert ok and (run_dir / SCORE_SWEEP_FILENAME).is_file()

    # Re-run with NaN boxes, which the integrity check refuses.
    (run_dir / "predictions.json").write_text(
        json.dumps(
            [
                {
                    "image_id": 1,
                    "category_id": 1,
                    "bbox": [float("nan")] * 4,
                    "score": 0.9,
                }
            ]
        )
    )

    ok, _ = _evaluate(run_dir, gt_path)

    assert not ok
    assert not (run_dir / "metrics.json").exists()
    assert not (run_dir / SCORE_SWEEP_FILENAME).exists()
    assert (run_dir / "metrics_invalid.json").is_file()


def test_run_with_no_predictions_writes_no_sweep(tmp_path):
    _, run_dir, gt_path = _tree(tmp_path, predictions=[])

    ok, _ = _evaluate(run_dir, gt_path)

    assert ok
    assert not (run_dir / SCORE_SWEEP_FILENAME).exists()


def test_console_reports_the_operating_point(tmp_path):
    _, run_dir, gt_path = _tree(tmp_path)

    _, output = _evaluate(run_dir, gt_path)

    assert "best F1" in output
    assert "IoU 0.50" in output


# =========================================================
# Reporting layer
# =========================================================


def test_report_loads_and_plots_the_sweep(tmp_path):
    import agri_vision_edge.evaluation.benchmark_report as br

    root, run_dir, gt_path = _tree(tmp_path)

    _evaluate(run_dir, gt_path)

    sweeps = br.load_score_sweeps(root)

    assert list(sweeps) == [("x86_cpu", RUN_NAME)]

    sweep = sweeps["x86_cpu", RUN_NAME]

    # Default: one panel per IoU threshold the sweep carries.
    figure = br.plot_pr_f1_vs_confidence(sweep)

    assert figure is not None
    assert len(figure.axes) == 2

    for axes, iou in zip(figure.axes, (0.5, 0.75), strict=True):
        # Precision, recall, F1 -- the best-F1 marker follows them.
        assert [line.get_label() for line in axes.lines][:3] == [
            "Precision",
            "Recall",
            "F1",
        ]
        assert axes.get_title() == f"IoU {iou:.2f}"
        assert axes.get_xlabel() == "Confidence threshold"

    # A scalar pins the figure to a single IoU.
    single = br.plot_pr_f1_vs_confidence(sweep, 0.75)

    assert len(single.axes) == 1
    assert single.axes[0].get_title() == "IoU 0.75"

    runs, _ = br.load_benchmark_results(root)

    comparison = br.plot_f1_vs_confidence(sweeps, runs)

    assert comparison is not None
    assert len(comparison.axes) == 2


def test_report_frame_is_long_format(tmp_path):
    import agri_vision_edge.evaluation.benchmark_report as br

    root, run_dir, gt_path = _tree(tmp_path)

    _evaluate(run_dir, gt_path)

    frame = br.score_sweep_frame(br.load_score_sweeps(root))

    assert not frame.empty
    assert {"platform", "run", "class", "iou", "score", "f1"} <= set(frame.columns)
    # Two named classes plus the pooled aggregate, at two IoU thresholds.
    assert set(frame["class"]) == {"crop", "weed", "all"}
    assert set(frame["iou"]) == {0.5, 0.75}


def test_operating_point_table_has_a_row_per_run(tmp_path):
    import agri_vision_edge.evaluation.benchmark_report as br

    root, run_dir, gt_path = _tree(tmp_path)

    _evaluate(run_dir, gt_path)

    table = br.operating_point_table(br.load_score_sweeps(root))

    assert len(table) == 1
    assert {"conf@0.50", "F1@0.50", "conf@0.75", "F1@0.75"} <= set(table.columns)


def test_missing_sweeps_are_simply_absent(tmp_path):
    """Runs evaluated before the sweep existed must not break the loader."""

    import agri_vision_edge.evaluation.benchmark_report as br

    root, run_dir, gt_path = _tree(tmp_path)

    _evaluate(run_dir, gt_path, score_sweep=False)

    assert br.load_score_sweeps(root) == {}
    assert br.plot_f1_vs_confidence({}, None) is None
    assert br.plot_pr_f1_vs_confidence(None) is None
