"""
Tests for the faithful upstream eval path that don't require the torch stack.

Only the COCO->YOLO conversion and the missing-deps import guard are exercised
here; the actual upstream evaluation needs the optional ``faithful-eval`` extra.
"""

from __future__ import annotations

import pytest

from agri_vision_edge.evaluation.faithful import (
    coco_predictions_to_yolo_lines,
    evaluate_faithful,
    upstream_label_map,
)


def test_yolo_conversion_normalizes_and_centers():
    info = {"file_name": "img.png", "width": 1024.0, "height": 1024.0}
    preds = [{"category_id": 2, "bbox": [462.0, 487.0, 100.0, 50.0], "score": 0.8}]
    (line,) = coco_predictions_to_yolo_lines(preds, info)
    label, cx, cy, w, h, score = line.split()
    assert label == "2"
    assert float(cx) == pytest.approx((462.0 + 50.0) / 1024.0)
    assert float(cy) == pytest.approx((487.0 + 25.0) / 1024.0)
    assert float(w) == pytest.approx(100.0 / 1024.0)
    assert float(h) == pytest.approx(50.0 / 1024.0)
    assert float(score) == pytest.approx(0.8)


def test_yolo_conversion_empty():
    info = {"file_name": "img.png", "width": 512.0, "height": 512.0}
    assert coco_predictions_to_yolo_lines([], info) == []


def test_multiclass_label_map_is_identity():
    coco = {"categories": [{"id": 1, "name": "crop"}, {"id": 2, "name": "weed"}]}
    assert upstream_label_map(coco) == {1: 1, 2: 2}


def test_single_class_weed_is_remapped_to_the_upstream_weed_label():
    # The sc bundle numbers its sole weed category 1, which is upstream's *crop*
    # label; writing it through unchanged scores weeds against crop ground truth
    # and produces near-zero mAP.
    coco = {"categories": [{"id": 1, "name": "weed"}]}
    label_map = upstream_label_map(coco)
    assert label_map == {1: 2}

    info = {"file_name": "img.png", "width": 1024.0, "height": 1024.0}
    preds = [{"category_id": 1, "bbox": [0.0, 0.0, 10.0, 10.0], "score": 0.5}]
    (line,) = coco_predictions_to_yolo_lines(preds, info, label_map)
    assert line.split()[0] == "2"


def test_unknown_category_is_rejected():
    with pytest.raises(ValueError, match="upstream"):
        upstream_label_map({"categories": [{"id": 1, "name": "plant"}]})


def test_faithful_requires_optional_deps_when_absent():
    # torch/torchvision/torchmetrics are not part of the default env; the guard
    # must raise a clear, actionable ImportError rather than a bare ModuleError.
    pytest.importorskip  # noqa: B018 - keep import side effect explicit
    try:
        import torch  # noqa: F401
    except ImportError:
        with pytest.raises(ImportError, match="faithful-eval"):
            evaluate_faithful("a.json", "b.json", "raw", "val")
    else:  # pragma: no cover - only when the extra is installed
        pytest.skip("faithful-eval extra is installed")
