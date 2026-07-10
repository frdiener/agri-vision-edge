"""
Tests for the combined do-not-care criterion in tiled bbox generation.

A plant's ``is_partial`` flag must follow the upstream PhenoBench
``visibility <= 0.5`` rule applied to its *effective visibility* -- the fraction
of the whole plant visible within a tile. That fraction combines two
independent reductions:

* upstream frame visibility (occlusion / frame border), and
* the tile-slice cut (pixels surviving inside the tile).

so a plant that was fully visible in the frame but sliced away by a tile border
is flagged partial just like an originally-partial plant.
"""

from __future__ import annotations

import numpy as np

from agri_vision_edge.data.tiling import (
    compute_instance_areas,
    generate_plant_bboxes,
)


def _full_frame():
    """A single crop instance (id=1, label=1) filling a 10x10 frame."""
    semantics = np.zeros((10, 10), dtype=np.int32)
    instances = np.zeros((10, 10), dtype=np.int32)
    visibility = np.zeros((10, 10), dtype=np.int32)
    semantics[:, :] = 1
    instances[:, :] = 1
    visibility[:, :] = 255  # fully visible in the original frame
    return semantics, instances, visibility


def test_tile_cut_flags_partial_even_when_frame_fully_visible():
    semantics, instances, visibility = _full_frame()
    instance_areas = compute_instance_areas(semantics, instances)
    assert instance_areas[(1, 1)] == 100

    # Tile keeps only the left 4 columns -> 40 / 100 = 0.4 of the plant.
    cols = slice(0, 4)
    (box,) = generate_plant_bboxes(
        semantics[:, cols],
        instances[:, cols],
        instance_areas=instance_areas,
        partial_threshold=0.5,
        plant_visibility=visibility[:, cols],
    )

    # Upstream visibility is 1.0, but the tile cut drops effective visibility
    # to 0.4 <= 0.5, so the box must be flagged do-not-care.
    assert box["is_partial"] is True
    assert box["visibility"] == 0.4


def test_tile_keeping_majority_is_not_partial():
    semantics, instances, visibility = _full_frame()
    instance_areas = compute_instance_areas(semantics, instances)

    # Tile keeps 6 of 10 columns -> 0.6 > 0.5, still scored.
    cols = slice(0, 6)
    (box,) = generate_plant_bboxes(
        semantics[:, cols],
        instances[:, cols],
        instance_areas=instance_areas,
        partial_threshold=0.5,
        plant_visibility=visibility[:, cols],
    )

    assert box["is_partial"] is False
    assert box["visibility"] == 0.6


def test_upstream_and_tile_cut_combine_by_product():
    semantics, instances, visibility = _full_frame()
    visibility[:, :] = 204  # 204/255 = 0.8 frame visibility
    instance_areas = compute_instance_areas(semantics, instances)

    # Tile keeps 8 of 10 columns -> tile fraction 0.8; combined 0.8*0.8 = 0.64.
    cols = slice(0, 8)
    (box,) = generate_plant_bboxes(
        semantics[:, cols],
        instances[:, cols],
        instance_areas=instance_areas,
        partial_threshold=0.5,
        plant_visibility=visibility[:, cols],
    )

    assert box["visibility"] == 0.8 * 0.8
    assert box["is_partial"] is False

    # Tighter tile: keep 5 of 10 columns -> combined 0.8*0.5 = 0.4 <= 0.5.
    cols = slice(0, 5)
    (box,) = generate_plant_bboxes(
        semantics[:, cols],
        instances[:, cols],
        instance_areas=instance_areas,
        partial_threshold=0.5,
        plant_visibility=visibility[:, cols],
    )

    assert box["visibility"] == 0.8 * 0.5
    assert box["is_partial"] is True


def test_upstream_only_without_tile_areas():
    # No instance_areas -> no tile-cut fraction; falls back to upstream alone.
    semantics, instances, visibility = _full_frame()
    visibility[:, :] = 100  # 100/255 ~= 0.39 <= 0.5

    (box,) = generate_plant_bboxes(
        semantics,
        instances,
        instance_areas=None,
        partial_threshold=0.5,
        plant_visibility=visibility,
    )

    assert box["is_partial"] is True
    assert abs(box["visibility"] - 100 / 255) < 1e-9
