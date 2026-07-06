"""
End-to-end test for the ``ignore_partials`` knob on the pycocotools eval path.

Builds a tiny synthetic COCO ground-truth with one normally-scored plant and one
partial ("do-not-care") plant, plus a prediction that (correctly) hits the
scored plant and a higher-scoring prediction that lands on the partial plant.
With the knob off the partial hit is a false positive that drags AP down; with
the knob on it is suppressed and AP recovers.

Requires ``pycocotools`` (the ``prep`` dependency group).
"""

from __future__ import annotations

import contextlib
import io
import json

import pytest

pytest.importorskip("pycocotools")

from agri_vision_edge.evaluation.coco import evaluate_predictions  # noqa: E402


def _write(tmp_path, name, obj):
    path = tmp_path / name
    path.write_text(json.dumps(obj))
    return path


def _gt():
    return {
        "images": [{"id": 1, "file_name": "x.png", "width": 100, "height": 100}],
        "categories": [{"id": 1, "name": "crop"}],
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
                "category_id": 1,
                "bbox": [60, 60, 20, 20],
                "area": 400,
                "iscrowd": 0,
                "ignore": 1,
                "partial": 1,
                "visibility": 0.3,
            },
        ],
    }


def _preds():
    return [
        # Higher-scoring detection on the partial plant -> FP unless ignored.
        {"image_id": 1, "category_id": 1, "bbox": [60, 60, 20, 20], "score": 0.95},
        # Correct detection on the scored plant.
        {"image_id": 1, "category_id": 1, "bbox": [10, 10, 20, 20], "score": 0.90},
    ]


def _evaluate(tmp_path, ignore_partials):
    gt_path = _write(tmp_path, "gt.json", _gt())
    pred_path = _write(tmp_path, "pred.json", _preds())
    with contextlib.redirect_stdout(io.StringIO()):
        return evaluate_predictions(
            gt_path, pred_path, ignore_partials=ignore_partials
        )


def test_knob_off_penalizes_partial_hit(tmp_path):
    metrics = _evaluate(tmp_path, ignore_partials=False)
    # The partial hit is an unmatched false positive scored above the TP.
    assert metrics["AP50"] < 1.0


def test_knob_on_suppresses_partial_hit(tmp_path):
    metrics = _evaluate(tmp_path, ignore_partials=True)
    assert metrics["AP50"] == pytest.approx(1.0)


def test_knob_on_beats_knob_off(tmp_path):
    off = _evaluate(tmp_path, ignore_partials=False)
    on = _evaluate(tmp_path, ignore_partials=True)
    assert on["AP50"] > off["AP50"]
