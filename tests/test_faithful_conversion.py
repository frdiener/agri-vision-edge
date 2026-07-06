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
