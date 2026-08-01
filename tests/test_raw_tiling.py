"""
Tests for materializing a raw PhenoBench tree as tiles on disk.

The materialized tree (``phenobench_raw_tiled``) is joined to the exported COCO
bundles purely **by file name**: ``ave benchmark`` reads its tiles as images and
``ave evaluate --faithful`` stages its masks as ground truth. A tree cut with a
different grid still resolves ``_tile0.._tile3`` and evaluates against the wrong
crops instead of failing, so naming, geometry and the recorded config are all
pinned here.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
from PIL import Image

from agri_vision_edge.data.raw_tiling import (
    RAW_SUBDIRS,
    TILING_CONFIG_NAME,
    materialize_tiled_dataset,
    read_tiling_config,
    tile_name,
)
from agri_vision_edge.data.tiling import compute_tiles

FRAME = 128


def _rgb(seed: int) -> Image.Image:
    rng = np.random.default_rng(seed)
    return Image.fromarray(rng.integers(0, 256, (FRAME, FRAME, 3), dtype=np.uint8))


def _mask16(seed: int) -> Image.Image:
    rng = np.random.default_rng(seed)
    array = rng.integers(0, 5000, (FRAME, FRAME), dtype=np.uint16)
    image = Image.fromarray(array)
    assert image.mode == "I;16"
    return image


def _visibility(seed: int) -> Image.Image:
    rng = np.random.default_rng(seed)
    return Image.fromarray(rng.integers(0, 256, (FRAME, FRAME), dtype=np.uint8))


@pytest.fixture
def raw_root(tmp_path):
    """A miniature raw PhenoBench tree: annotated train/val + images-only test."""

    root = tmp_path / "raw_full"

    factories = {
        "images": _rgb,
        "semantics": _mask16,
        "plant_instances": _mask16,
        "leaf_instances": _mask16,
        "plant_visibility": _visibility,
        "leaf_visibility": _visibility,
    }

    seed = 0

    for split, subdirs in (
        ("train", RAW_SUBDIRS),
        ("val", RAW_SUBDIRS),
        ("test", ("images",)),
    ):
        for index in range(2):
            name = f"{split}_{index:03d}.png"
            for subdir in subdirs:
                directory = root / split / subdir
                directory.mkdir(parents=True, exist_ok=True)
                factories[subdir](seed).save(directory / name)
                seed += 1

    return root


def test_tile_name_starts_at_zero():
    # TiledPhenoBench names the first tile `_tile0`; the exported annotations
    # follow it, so the materialized files must too.
    assert tile_name("05-15_00028_P0030852", 0) == "05-15_00028_P0030852_tile0.png"
    assert tile_name("frame", 8) == "frame_tile8.png"


def test_tile_names_match_the_dataset_wrapper(raw_root, tmp_path):
    """
    The materialized names must equal what ``TiledPhenoBench`` emits.

    The COCO bundles are exported from that wrapper and carry its
    ``image_name``s; the benchmark and the faithful evaluator then look those up
    as files in the materialized tree. Compare against the wrapper itself rather
    than a hand-written expectation, so the two cannot drift apart.
    """

    from agri_vision_edge.data.tiling import TiledPhenoBench

    class _StubDataset:
        """Minimal stand-in for ``PhenoBench`` (which needs the real corpus)."""

        def __init__(self, split_dir, names):
            self.split_dir = split_dir
            self.names = names

        def __len__(self):
            return len(self.names)

        def __getitem__(self, index):
            name = self.names[index]
            return {
                "image": Image.open(self.split_dir / "images" / name),
                "semantics": np.array(Image.open(self.split_dir / "semantics" / name)),
                "plant_instances": np.array(
                    Image.open(self.split_dir / "plant_instances" / name)
                ),
                "image_name": name,
            }

    materialize_tiled_dataset(raw_root, tmp_path / "tiled", rows=3, cols=3, overlap=0.5)

    wrapper = TiledPhenoBench(
        _StubDataset(raw_root / "val", ["val_000.png", "val_001.png"]),
        rows=3,
        cols=3,
        overlap=0.5,
    )

    expected = sorted(wrapper[i]["image_name"] for i in range(len(wrapper)))
    produced = sorted(
        p.name for p in (tmp_path / "tiled" / "val" / "images").glob("*.png")
    )

    assert produced == expected


def test_3x3_overlap_half_produces_nine_tiles_per_frame(raw_root, tmp_path):
    dest = tmp_path / "tiled"

    stats = materialize_tiled_dataset(
        raw_root,
        dest,
        rows=3,
        cols=3,
        overlap=0.5,
    )

    assert stats["splits"]["train"]["frames"] == 2
    assert stats["splits"]["train"]["expected_tiles"] == 18

    for split in ("train", "val"):
        for subdir in RAW_SUBDIRS:
            assert len(list((dest / split / subdir).glob("*.png"))) == 18

    # The test split ships images only; nothing else may be invented for it.
    assert len(list((dest / "test" / "images").glob("*.png"))) == 18
    assert sorted(p.name for p in (dest / "test").iterdir()) == ["images"]


def test_tiles_are_the_compute_tiles_crops(raw_root, tmp_path):
    dest = tmp_path / "tiled"

    materialize_tiled_dataset(raw_root, dest, rows=3, cols=3, overlap=0.5)

    source = np.array(Image.open(raw_root / "val" / "images" / "val_000.png"))
    tiles = compute_tiles(FRAME, FRAME, rows=3, cols=3, overlap=0.5)

    for index, tile in enumerate(tiles):
        produced = np.array(
            Image.open(dest / "val" / "images" / f"val_000_tile{index}.png")
        )
        expected = source[tile.y0 : tile.y1, tile.x0 : tile.x1]
        assert np.array_equal(produced, expected), f"tile {index} mismatch"


def test_16bit_masks_survive_the_cut(raw_root, tmp_path):
    dest = tmp_path / "tiled"

    materialize_tiled_dataset(raw_root, dest, rows=2, cols=2, overlap=0.0)

    source_path = raw_root / "train" / "plant_instances" / "train_000.png"
    tile_path = dest / "train" / "plant_instances" / "train_000_tile3.png"

    with Image.open(source_path) as source, Image.open(tile_path) as produced:
        assert produced.mode == source.mode == "I;16"
        source_array = np.array(source)
        produced_array = np.array(produced)

    assert produced_array.dtype == np.uint16
    half = FRAME // 2
    assert np.array_equal(produced_array, source_array[half:, half:])


def test_tiling_config_records_the_geometry(raw_root, tmp_path):
    dest = tmp_path / "tiled"

    materialize_tiled_dataset(raw_root, dest, rows=3, cols=3, overlap=0.5)

    recorded = json.loads((dest / TILING_CONFIG_NAME).read_text())

    assert recorded["rows"] == 3
    assert recorded["cols"] == 3
    assert recorded["overlap"] == 0.5
    assert recorded["tile_width"] == recorded["tile_height"] == FRAME // 2
    assert recorded["source_dataset"] == str(raw_root)

    config = read_tiling_config(dest)
    assert config is not None
    assert (config.rows, config.cols, config.overlap) == (3, 3, 0.5)
    assert config.tiles_per_image == 9


def test_read_tiling_config_missing_is_none(tmp_path):
    assert read_tiling_config(tmp_path) is None


def test_existing_destination_is_refused(raw_root, tmp_path):
    dest = tmp_path / "tiled"
    dest.mkdir()

    # Writing a second geometry into an existing tree would leave _tile0.._tileN
    # names spanning two grids -- exactly the silent mismatch this guards.
    with pytest.raises(FileExistsError):
        materialize_tiled_dataset(raw_root, dest)

    materialize_tiled_dataset(raw_root, dest, exist_ok=True)
    assert (dest / TILING_CONFIG_NAME).exists()


def test_workers_produce_identical_output(raw_root, tmp_path):
    inline = tmp_path / "inline"
    parallel = tmp_path / "parallel"

    materialize_tiled_dataset(raw_root, inline, rows=3, cols=3, overlap=0.5)
    materialize_tiled_dataset(
        raw_root, parallel, rows=3, cols=3, overlap=0.5, workers=2
    )

    for path in sorted(inline.rglob("*.png")):
        other = parallel / path.relative_to(inline)
        assert other.exists(), other
        assert np.array_equal(np.array(Image.open(path)), np.array(Image.open(other)))


def test_missing_source_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        materialize_tiled_dataset(tmp_path / "nope", tmp_path / "out")
