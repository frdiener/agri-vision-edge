"""
Tests for partial-aware plant-box generation from synthetic PhenoBench masks.
"""

from __future__ import annotations

import numpy as np

from agri_vision_edge.data.plant_boxes import plant_boxes_from_masks
from agri_vision_edge.data.tiling import generate_plant_bboxes


def _blank(size=20):
    return np.zeros((size, size), dtype=np.int32)


def test_full_visibility_not_partial():
    semantics = _blank()
    instances = _blank()
    visibility = _blank()
    # A crop instance (label 1), fully visible.
    semantics[2:6, 2:6] = 1
    instances[2:6, 2:6] = 1
    visibility[2:6, 2:6] = 255

    (box,) = plant_boxes_from_masks(semantics, instances, visibility)
    assert box["label"] == 1
    assert box["is_partial"] is False
    assert box["visibility"] == 1.0


def test_low_visibility_is_partial():
    semantics = _blank()
    instances = _blank()
    visibility = _blank()
    semantics[2:6, 2:6] = 2  # weed
    instances[2:6, 2:6] = 1
    visibility[2:6, 2:6] = 100  # 100/255 ~= 0.39 <= 0.5

    (box,) = plant_boxes_from_masks(semantics, instances, visibility)
    assert box["label"] == 2
    assert box["is_partial"] is True


def test_border_partial_class_remapped_and_flagged():
    semantics = _blank()
    instances = _blank()
    visibility = _blank()
    # partial-crop (semantic 3) -> label 1, flagged partial even at high vis.
    semantics[2:6, 2:6] = 3
    instances[2:6, 2:6] = 1
    visibility[2:6, 2:6] = 255

    (box,) = plant_boxes_from_masks(semantics, instances, visibility)
    assert box["label"] == 1
    assert box["is_partial"] is True


def test_instance_id_reused_across_classes():
    semantics = _blank()
    instances = _blank()
    visibility = _blank()
    # Same instance id 1 for a crop and a weed region -> two distinct boxes.
    semantics[2:5, 2:5] = 1
    instances[2:5, 2:5] = 1
    visibility[2:5, 2:5] = 255
    semantics[10:14, 10:14] = 2
    instances[10:14, 10:14] = 1
    visibility[10:14, 10:14] = 255

    boxes = plant_boxes_from_masks(semantics, instances, visibility)
    labels = sorted(b["label"] for b in boxes)
    assert labels == [1, 2]


def test_no_visibility_mask_only_border_flagged():
    semantics = _blank()
    instances = _blank()
    semantics[2:6, 2:6] = 1  # normal crop, no visibility info
    instances[2:6, 2:6] = 1

    (box,) = plant_boxes_from_masks(semantics, instances, plant_visibility=None)
    assert "visibility" not in box
    assert box["is_partial"] is False


def test_tiled_generate_uses_upstream_visibility():
    # Without instance_areas there is no tile-cut fraction, so the effective
    # visibility (and the is_partial decision) comes from the upstream
    # plant_visibility mask alone.
    semantics = _blank()
    instances = _blank()
    visibility = _blank()
    # A well-visible weed (label 2), visibility 255 -> not partial.
    semantics[2:8, 2:8] = 2
    instances[2:8, 2:8] = 1
    visibility[2:8, 2:8] = 255
    # A low-visibility crop (label 1), visibility 100/255 ~= 0.39 -> partial.
    semantics[10:16, 10:16] = 1
    instances[10:16, 10:16] = 2
    visibility[10:16, 10:16] = 100

    boxes = generate_plant_bboxes(
        semantics,
        instances,
        partial_threshold=0.5,
        plant_visibility=visibility,
    )
    by_label = {b["label"]: b for b in boxes}
    assert by_label[2]["is_partial"] is False
    assert by_label[1]["is_partial"] is True
    assert by_label[1]["visibility"] < 0.5
