"""
Tests for the tiled ground-truth / annotations grid check.

``ave evaluate --faithful`` joins a COCO bundle to a materialized raw tiled tree
by file name. A 2x2 and a 3x3-with-half-overlap cut of the same 1024 frame both
produce 512 px tiles named ``_tile0..``, so pointing the evaluator at the wrong
tree resolves every mask and reports plausible numbers computed against the
wrong crops. Only the tile *count* distinguishes them, which is what
:func:`check_tiling_consistency` compares.
"""

from __future__ import annotations

import json

import pytest

from agri_vision_edge.evaluation.faithful import (
    annotation_tile_indices,
    check_tiling_consistency,
)


def _index(file_names):
    return {
        i: {"file_name": name, "width": 512.0, "height": 512.0}
        for i, name in enumerate(file_names, start=1)
    }


def _tiled_index(tiles_per_frame, frames=("a", "b")):
    return _index(
        [f"{frame}_tile{i}.png" for frame in frames for i in range(tiles_per_frame)]
    )


def _tree(tmp_path, rows, cols, overlap=0.5):
    root = tmp_path / f"raw_tiled_{rows}x{cols}"
    root.mkdir()
    (root / "tiling_config.json").write_text(
        json.dumps({"rows": rows, "cols": cols, "overlap": overlap})
    )
    return root


def test_annotation_tile_indices_reads_the_marker():
    assert annotation_tile_indices(_tiled_index(4)) == {0, 1, 2, 3}


def test_annotation_tile_indices_none_for_full_frames():
    assert annotation_tile_indices(_index(["05-15_00028_P0030852.png"])) is None


def test_matching_grid_passes(tmp_path):
    check_tiling_consistency(_tiled_index(9), _tree(tmp_path, 3, 3))


def test_2x2_annotations_against_a_3x3_tree_raises(tmp_path):
    # The regression this guard exists for: every mask would resolve.
    with pytest.raises(ValueError, match="Tiling mismatch"):
        check_tiling_consistency(_tiled_index(4), _tree(tmp_path, 3, 3))


def test_3x3_annotations_against_a_2x2_tree_raises(tmp_path):
    with pytest.raises(ValueError, match="Tiling mismatch"):
        check_tiling_consistency(_tiled_index(9), _tree(tmp_path, 2, 2, overlap=0.0))


def test_full_frame_annotations_against_a_tiled_tree_raises(tmp_path):
    with pytest.raises(ValueError, match="full-frame"):
        check_tiling_consistency(
            _index(["05-15_00028_P0030852.png"]),
            _tree(tmp_path, 3, 3),
        )


def test_tree_without_recorded_geometry_is_not_checked(tmp_path):
    # Legacy trees carry no tiling_config.json; there is nothing to compare
    # against, and _stage's missing-mask error still catches wrong names.
    root = tmp_path / "legacy"
    root.mkdir()

    check_tiling_consistency(_tiled_index(4), root)
    check_tiling_consistency(_index(["frame.png"]), root)


def test_untiled_tree_and_untiled_annotations_pass(tmp_path):
    root = tmp_path / "raw_full"
    root.mkdir()

    check_tiling_consistency(_index(["05-15_00028_P0030852.png"]), root)
