"""
Tests for uniform tile geometry.

Overlapped tiling must produce same-size square tiles (fixed window + stride),
never clamped/non-square edge tiles -- those would be aspect-distorted when the
exporter resizes each tile to a square.
"""

from __future__ import annotations

from agri_vision_edge.data.tiling import compute_tiles


def test_3x3_overlap_half_is_uniform_512():
    tiles = compute_tiles(1024, 1024, rows=3, cols=3, overlap=0.5)
    assert len(tiles) == 9
    assert all((t.width, t.height) == (512, 512) for t in tiles)
    # Stride 256: columns start at 0, 256, 512.
    xs = sorted({t.x0 for t in tiles})
    assert xs == [0, 256, 512]


def test_2x2_no_overlap_backward_compatible():
    tiles = compute_tiles(1024, 1024, rows=2, cols=2, overlap=0.0)
    assert len(tiles) == 4
    assert all((t.width, t.height) == (512, 512) for t in tiles)


def test_single_tile_is_full_frame():
    (tile,) = compute_tiles(1024, 1024, rows=1, cols=1, overlap=0.0)
    assert (tile.width, tile.height) == (1024, 1024)


def test_tiles_cover_and_stay_in_bounds():
    w = h = 1000
    tiles = compute_tiles(w, h, rows=3, cols=3, overlap=0.5)
    for t in tiles:
        assert 0 <= t.x0 < t.x1 <= w
        assert 0 <= t.y0 < t.y1 <= h
    # Last tile reaches the frame edge.
    assert max(t.x1 for t in tiles) == w
    assert max(t.y1 for t in tiles) == h
