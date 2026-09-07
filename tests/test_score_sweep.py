"""
Unit tests for the confidence sweep.

The fixture is small enough to work precision/recall/F1 out by hand at both IoU
thresholds, pinning the ``dtIgnore`` bookkeeping and the threshold-invariance
:mod:`agri_vision_edge.evaluation.score_sweep` relies on.

Requires ``pycocotools`` (the ``prep`` dependency group).
"""

from __future__ import annotations

import contextlib
import io
import json

import numpy as np
import pytest

pytest.importorskip("pycocotools")

from agri_vision_edge.evaluation.coco import prepare_evaluation  # noqa: E402
from agri_vision_edge.evaluation.score_sweep import (  # noqa: E402
    MICRO_CLASS,
    ScoreSweep,
    compute_score_sweep,
    operating_points,
)


def _write(tmp_path, name, obj):
    path = tmp_path / name
    path.write_text(json.dumps(obj))
    return path


def _gt(extra_annotations=()):
    """Three well-separated 20x20 plants in one image."""

    annotations = [
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
            "category_id": 1,
            "bbox": [60, 60, 20, 20],
            "area": 400,
            "iscrowd": 0,
        },
        {
            "id": 3,
            "image_id": 1,
            "category_id": 1,
            "bbox": [10, 60, 20, 20],
            "area": 400,
            "iscrowd": 0,
        },
    ]

    return {
        "images": [{"id": 1, "file_name": "x.png", "width": 100, "height": 100}],
        "categories": [{"id": 1, "name": "weed"}],
        "annotations": [*annotations, *extra_annotations],
    }


def _preds():
    return [
        # Exact hit on plant 1 -> TP at both IoU thresholds.
        {"image_id": 1, "category_id": 1, "bbox": [10, 10, 20, 20], "score": 0.9},
        # Shifted hit on plant 2: IoU = 320/480 = 0.667 -> TP@.50 but FP@.75.
        {"image_id": 1, "category_id": 1, "bbox": [64, 60, 20, 20], "score": 0.8},
        # Nowhere near anything -> FP at both.
        {"image_id": 1, "category_id": 1, "bbox": [0, 0, 5, 5], "score": 0.7},
    ]


def _sweep(tmp_path, gt=None, preds=None, **kwargs):
    gt_path = _write(tmp_path, "gt.json", gt or _gt())
    pred_path = _write(tmp_path, "pred.json", preds or _preds())

    with contextlib.redirect_stdout(io.StringIO()):
        prepared = prepare_evaluation(gt_path, pred_path)
        return compute_score_sweep(prepared, **kwargs)


def _at(sweep, iou, threshold):
    """Read the curve at the grid point closest to ``threshold``."""

    curve = sweep.curve(iou, MICRO_CLASS)
    index = int(np.argmin(np.abs(curve["score"] - threshold)))

    return {k: v[index] for k, v in curve.items()}


# =========================================================
# The fixture's IoUs are what the test claims they are
# =========================================================


def test_shifted_box_iou_straddles_the_two_thresholds(tmp_path):
    sweep = _sweep(tmp_path)

    # TP counts at threshold 0 differ only because of that middle detection.
    assert _at(sweep, 0.5, 0.0)["tp"] == 2
    assert _at(sweep, 0.75, 0.0)["tp"] == 1


# =========================================================
# Precision / recall / F1 at IoU 0.50
# =========================================================


@pytest.mark.parametrize(
    ("threshold", "tp", "fp", "precision", "recall"),
    [
        (0.0, 2, 1, 2 / 3, 2 / 3),
        (0.75, 2, 0, 1.0, 2 / 3),
        (0.85, 1, 0, 1.0, 1 / 3),
    ],
)
def test_curve_at_iou_50(tmp_path, threshold, tp, fp, precision, recall):
    point = _at(_sweep(tmp_path), 0.5, threshold)

    assert point["tp"] == tp
    assert point["fp"] == fp
    assert point["precision"] == pytest.approx(precision)
    assert point["recall"] == pytest.approx(recall)

    expected_f1 = 2 * precision * recall / (precision + recall)
    assert point["f1"] == pytest.approx(expected_f1)


@pytest.mark.parametrize(
    ("threshold", "tp", "fp", "precision"),
    [
        (0.0, 1, 2, 1 / 3),
        (0.75, 1, 1, 0.5),
        (0.85, 1, 0, 1.0),
    ],
)
def test_curve_at_iou_75(tmp_path, threshold, tp, fp, precision):
    point = _at(_sweep(tmp_path), 0.75, threshold)

    assert point["tp"] == tp
    assert point["fp"] == fp
    assert point["precision"] == pytest.approx(precision)
    # Only one of three plants is ever found at this IoU.
    assert point["recall"] == pytest.approx(1 / 3)


def test_precision_is_nan_above_the_highest_score(tmp_path):
    """No predictions survive, so precision has no value -- and must not read 0."""

    point = _at(_sweep(tmp_path), 0.5, 0.95)

    assert point["tp"] == 0
    assert np.isnan(point["precision"])
    assert point["recall"] == pytest.approx(0.0)


# =========================================================
# Recall denominator
# =========================================================


def test_recall_denominator_counts_undetected_ground_truth(tmp_path):
    sweep = _sweep(tmp_path)

    assert sweep.per_class["weed"].n_gt == 3


def test_iscrowd_ground_truth_is_excluded_from_n_gt(tmp_path):
    """pycocotools treats iscrowd as ignore; it must not inflate the denominator."""

    crowd = {
        "id": 4,
        "image_id": 1,
        "category_id": 1,
        "bbox": [40, 0, 10, 10],
        "area": 100,
        "iscrowd": 1,
    }

    sweep = _sweep(tmp_path, gt=_gt(extra_annotations=[crowd]))

    assert sweep.per_class["weed"].n_gt == 3


def test_detection_on_ignored_gt_is_neither_tp_nor_fp(tmp_path):
    """A hit on a do-not-care box must vanish from both counters, not become a FP."""

    crowd = {
        "id": 4,
        "image_id": 1,
        "category_id": 1,
        "bbox": [40, 0, 10, 10],
        "area": 100,
        "iscrowd": 1,
    }

    preds = [
        *_preds(),
        {"image_id": 1, "category_id": 1, "bbox": [40, 0, 10, 10], "score": 0.95},
    ]

    baseline = _at(_sweep(tmp_path), 0.5, 0.0)

    sweep = _sweep(
        tmp_path,
        gt=_gt(extra_annotations=[crowd]),
        preds=preds,
    )
    point = _at(sweep, 0.5, 0.0)

    assert point["tp"] == baseline["tp"]
    assert point["fp"] == baseline["fp"]


# =========================================================
# Operating points
# =========================================================


def test_best_f1_finds_the_expected_operating_point(tmp_path):
    sweep = _sweep(tmp_path)

    best = sweep.best_f1(0.5)

    # Dropping only the stray FP leaves P=1, R=2/3 -> F1=0.8.
    assert best["f1"] == pytest.approx(0.8)
    assert best["precision"] == pytest.approx(1.0)
    assert best["recall"] == pytest.approx(2 / 3)
    # Any threshold in (0.7, 0.8] is equivalent on this data.
    assert 0.7 < best["score"] <= 0.8


def test_best_f1_never_recommends_a_threshold_below_the_export_floor(tmp_path):
    """
    Nothing is emitted below the export floor, so every threshold under it
    describes the same operating point. A plain argmax would report the lowest,
    i.e. 0.0; the floor is reported instead.
    """

    # All detections sit above a 0.05 floor, and F1 is maximised by keeping
    # every one of them, so the optimum lies on the flat sub-floor plateau.
    preds = [
        {"image_id": 1, "category_id": 1, "bbox": [10, 10, 20, 20], "score": 0.30},
        {"image_id": 1, "category_id": 1, "bbox": [60, 60, 20, 20], "score": 0.10},
        {"image_id": 1, "category_id": 1, "bbox": [10, 60, 20, 20], "score": 0.051},
    ]

    sweep = _sweep(tmp_path, preds=preds)

    assert sweep.min_score == pytest.approx(0.051)

    best = sweep.best_f1(0.5)

    # Perfect detection of all three plants: the optimum is "keep everything",
    # which must remain reachable -- the floor bin itself is not masked.
    assert best["f1"] == pytest.approx(1.0)
    # ...and it is reported at the floor rather than at 0.0.
    assert 0.049 <= best["score"] <= 0.052


def test_operating_points_cover_every_iou_and_class(tmp_path):
    points = operating_points(_sweep(tmp_path))

    assert set(points) == {"f1_best@0.50", "f1_best@0.75"}
    assert points["f1_best@0.50"]["per_class"]["weed"]["f1"] == pytest.approx(0.8)


# =========================================================
# Structural properties
# =========================================================


def test_counts_are_monotonically_non_increasing_in_the_threshold(tmp_path):
    """A cumulative "at or above" count can only shrink as the threshold rises."""

    sweep = _sweep(tmp_path)

    for iou in sweep.iou_thresholds:
        curve = sweep.curve(iou)

        assert np.all(np.diff(curve["tp"]) <= 0)
        assert np.all(np.diff(curve["fp"]) <= 0)


def test_all_detections_are_accounted_for_at_threshold_zero(tmp_path):
    sweep = _sweep(tmp_path)

    for iou in sweep.iou_thresholds:
        point = _at(sweep, iou, 0.0)
        assert point["tp"] + point["fp"] == sweep.n_detections


def test_max_dets_defaults_to_no_truncation(tmp_path):
    sweep = _sweep(tmp_path)

    assert sweep.max_dets_observed == 3
    assert sweep.max_dets_used == 3


def test_explicit_max_dets_truncates(tmp_path):
    """A binding maxDets cap must actually drop the lowest-scoring detections."""

    sweep = _sweep(tmp_path, max_dets=2)

    point = _at(sweep, 0.5, 0.0)

    assert sweep.max_dets_observed == 3
    assert point["tp"] + point["fp"] == 2


# =========================================================
# Round-trip
# =========================================================


def test_sweep_records_the_partial_convention(tmp_path):
    """Curves made with different partial settings are not comparable."""

    gt_path = _write(tmp_path, "gt.json", _gt())
    pred_path = _write(tmp_path, "pred.json", _preds())

    with contextlib.redirect_stdout(io.StringIO()):
        default = compute_score_sweep(prepare_evaluation(gt_path, pred_path))
        ignored = compute_score_sweep(
            prepare_evaluation(gt_path, pred_path, ignore_partials=True)
        )

    assert default.ignore_partials is False
    assert ignored.ignore_partials is True

    # Survives the artifact round trip.
    restored = ScoreSweep.from_dict(json.loads(json.dumps(ignored.to_dict())))
    assert restored.ignore_partials is True
    assert restored.partial_threshold == ignored.partial_threshold


def test_serialisation_round_trip_preserves_curves(tmp_path):
    sweep = _sweep(tmp_path)

    restored = ScoreSweep.from_dict(json.loads(json.dumps(sweep.to_dict())))

    assert restored.per_class["weed"].n_gt == sweep.per_class["weed"].n_gt
    assert restored.iou_thresholds == sweep.iou_thresholds

    for iou in sweep.iou_thresholds:
        original = sweep.curve(iou)
        loaded = restored.curve(iou)

        for key in ("precision", "recall", "f1"):
            np.testing.assert_allclose(
                original[key], loaded[key], equal_nan=True
            )


def test_micro_average_pools_classes(tmp_path):
    """Two categories must pool into summed counts, not be averaged."""

    gt = _gt()
    gt["categories"].append({"id": 2, "name": "crop"})
    gt["annotations"].append(
        {
            "id": 4,
            "image_id": 1,
            "category_id": 2,
            "bbox": [10, 10, 20, 20],
            "area": 400,
            "iscrowd": 0,
        }
    )

    preds = [
        *_preds(),
        {"image_id": 1, "category_id": 2, "bbox": [10, 10, 20, 20], "score": 0.6},
    ]

    sweep = _sweep(tmp_path, gt=gt, preds=preds)

    _, _, n_gt = sweep.counts(MICRO_CLASS)

    assert n_gt == 4
    assert _at(sweep, 0.5, 0.0)["tp"] == 3
