"""Materialize an in-memory PhenoBench tiling as a raw on-disk tree.

Files use zero-based row-major ``{stem}_tile{index}.png`` names matching
``TiledPhenoBench``. Geometry and names must match the exported annotations: two
grids can share names for different crops and silently score against the wrong
ground truth. ``tiling_config.json`` records the geometry for validation.

PIL cropping preserves source modes, including 16-bit annotation masks.
"""

from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

from .tiling import compute_tiles

#: PhenoBench per-split sub-directories, in the order they are processed.
#: ``images`` is the only one the ``test`` split ships; the rest are annotations
#: and are skipped when absent.
RAW_SUBDIRS = (
    "images",
    "semantics",
    "plant_instances",
    "leaf_instances",
    "plant_visibility",
    "leaf_visibility",
)

#: Sub-directories the official (faithful) evaluator reads. Kept here so callers
#: can materialize a lean tree without pulling in the unused leaf masks.
GT_SUBDIRS = (
    "images",
    "semantics",
    "plant_instances",
    "plant_visibility",
)

DEFAULT_SPLITS = ("train", "val", "test")

TILING_CONFIG_NAME = "tiling_config.json"


def tile_name(stem: str, tile_index: int, suffix: str = ".png") -> str:
    """
    The materialized file name for tile ``tile_index`` of frame ``stem``.

    Must stay identical to ``TiledPhenoBench.__getitem__``'s ``image_name``:
    the COCO annotations exported from that wrapper are matched against these
    files by name.
    """

    return f"{stem}_tile{tile_index}{suffix}"


@dataclass(frozen=True)
class TilingConfig:
    """Geometry a materialized tree was cut with."""

    rows: int
    cols: int
    overlap: float
    source_dataset: str | None = None
    tile_width: int | None = None
    tile_height: int | None = None

    @property
    def tiles_per_image(self) -> int:
        return self.rows * self.cols

    def to_dict(self) -> dict:
        return {
            "rows": self.rows,
            "cols": self.cols,
            "overlap": self.overlap,
            "source_dataset": self.source_dataset,
            "tile_width": self.tile_width,
            "tile_height": self.tile_height,
        }


def read_tiling_config(root: str | Path) -> TilingConfig | None:
    """
    The geometry recorded in ``root/tiling_config.json`` (``None`` when absent).
    """

    path = Path(root) / TILING_CONFIG_NAME

    if not path.exists():
        return None

    raw = json.loads(path.read_text())

    return TilingConfig(
        rows=int(raw["rows"]),
        cols=int(raw["cols"]),
        overlap=float(raw.get("overlap", 0.0)),
        source_dataset=raw.get("source_dataset"),
        tile_width=raw.get("tile_width"),
        tile_height=raw.get("tile_height"),
    )


def tile_file(
    source: str | Path,
    dest_dir: str | Path,
    *,
    rows: int,
    cols: int,
    overlap: float,
) -> list[Path]:
    """
    Cut one image into ``rows x cols`` tiles written to ``dest_dir``.

    Returns the written paths, in tile order. ``Image.crop`` keeps the source
    mode, so 16-bit masks stay 16-bit.
    """

    source = Path(source)
    dest_dir = Path(dest_dir)

    written: list[Path] = []

    with Image.open(source) as image:
        image.load()

        tiles = compute_tiles(
            width=image.width,
            height=image.height,
            rows=rows,
            cols=cols,
            overlap=overlap,
        )

        for tile_index, tile in enumerate(tiles):
            out_path = dest_dir / tile_name(
                source.stem,
                tile_index,
                source.suffix,
            )

            image.crop(
                (
                    tile.x0,
                    tile.y0,
                    tile.x1,
                    tile.y1,
                )
            ).save(out_path)

            written.append(out_path)

    return written


def _tile_frame(job) -> int:
    """
    Worker: tile one frame across every sub-directory that has it.

    Takes/returns plain data so it can cross a process boundary.
    """

    (
        file_name,
        source_split,
        dest_split,
        subdirs,
        rows,
        cols,
        overlap,
    ) = job

    written = 0

    for subdir in subdirs:
        source = Path(source_split) / subdir / file_name

        if not source.exists():
            continue

        written += len(
            tile_file(
                source,
                Path(dest_split) / subdir,
                rows=rows,
                cols=cols,
                overlap=overlap,
            )
        )

    return written


def materialize_tiled_dataset(
    source_root: str | Path,
    dest_root: str | Path,
    *,
    rows: int = 3,
    cols: int = 3,
    overlap: float = 0.5,
    splits=DEFAULT_SPLITS,
    subdirs=RAW_SUBDIRS,
    workers: int | None = None,
    progress=None,
    exist_ok: bool = False,
) -> dict:
    """Write a tiled PhenoBench tree and return its config and split counts.

    ``overlap`` is a fraction in ``[0, 1)``, not pixels. Missing splits and
    subdirectories are skipped. Existing destinations are refused unless
    ``exist_ok`` because mixing geometries under colliding names is unsafe.
    ``workers`` values above one use a process pool; ``progress`` accepts an
    iterable and ``desc=``.
    """

    source_root = Path(source_root)
    dest_root = Path(dest_root)

    if not source_root.is_dir():
        raise FileNotFoundError(f"Source dataset not found: {source_root}")

    if dest_root.exists() and not exist_ok:
        raise FileExistsError(
            f"Destination {dest_root} already exists. Remove or rename it "
            "first -- overwriting in place would mix two tile geometries "
            "under names that silently collide."
        )

    dest_root.mkdir(parents=True, exist_ok=True)

    jobs: list[tuple] = []
    split_stats: dict[str, dict] = {}

    for split in splits:
        source_split = source_root / split

        if not source_split.is_dir():
            continue

        image_dir = source_split / "images"

        if not image_dir.is_dir():
            raise FileNotFoundError(
                f"Split {split!r} has no images/ directory: {image_dir}"
            )

        file_names = sorted(p.name for p in image_dir.glob("*.png"))

        present = [subdir for subdir in subdirs if (source_split / subdir).is_dir()]

        for subdir in present:
            (dest_root / split / subdir).mkdir(parents=True, exist_ok=True)

        split_stats[split] = {
            "frames": len(file_names),
            "subdirs": present,
            "tiles_per_frame": rows * cols,
            "expected_tiles": len(file_names) * rows * cols,
        }

        for file_name in file_names:
            jobs.append(
                (
                    file_name,
                    str(source_split),
                    str(dest_root / split),
                    tuple(present),
                    rows,
                    cols,
                    overlap,
                )
            )

    if not split_stats:
        raise FileNotFoundError(f"No splits {tuple(splits)} found under {source_root}")

    def _run(iterable):
        if progress is None:
            return iterable
        return progress(iterable, desc=f"tiling {rows}x{cols}@{overlap}")

    total_written = 0

    if workers and workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for count in _run(pool.map(_tile_frame, jobs, chunksize=8)):
                total_written += count
    else:
        for job in _run(iter(jobs)):
            total_written += _tile_frame(job)

    # Record geometry because filenames do not identify the source grid.
    tile_width = tile_height = None

    probe_split = next(iter(split_stats))
    probe = sorted((source_root / probe_split / "images").glob("*.png"))

    if probe:
        with Image.open(probe[0]) as image:
            first_tile = compute_tiles(
                width=image.width,
                height=image.height,
                rows=rows,
                cols=cols,
                overlap=overlap,
            )[0]

        tile_width = first_tile.width
        tile_height = first_tile.height

    config = TilingConfig(
        rows=rows,
        cols=cols,
        overlap=overlap,
        source_dataset=str(source_root),
        tile_width=tile_width,
        tile_height=tile_height,
    )

    (dest_root / TILING_CONFIG_NAME).write_text(json.dumps(config.to_dict(), indent=2))

    return {
        **config.to_dict(),
        "splits": split_stats,
        "files_written": total_written,
    }


__all__ = [
    "DEFAULT_SPLITS",
    "GT_SUBDIRS",
    "RAW_SUBDIRS",
    "TILING_CONFIG_NAME",
    "TilingConfig",
    "materialize_tiled_dataset",
    "read_tiling_config",
    "tile_file",
    "tile_name",
]
