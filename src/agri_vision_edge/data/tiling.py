"""
PhenoBench tiling utilities.

Tiling is applied before bbox generation.

Bounding boxes are regenerated from
cropped semantics + plant_instances.

The primary filtering criterion is
instance visibility (pixel count),
which naturally removes tiny tile-border
fragments without relying on bbox shape.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

# --------------------------------------------------
# Geometry
# --------------------------------------------------


@dataclass(frozen=True)
class Tile:
    x0: int
    y0: int
    x1: int
    y1: int

    @property
    def width(self) -> int:
        return self.x1 - self.x0

    @property
    def height(self) -> int:
        return self.y1 - self.y0


def compute_tiles(
    width: int,
    height: int,
    rows: int = 2,
    cols: int = 2,
    overlap: float = 0.0,
) -> list[Tile]:

    if rows < 1:
        raise ValueError("rows must be >= 1")

    if cols < 1:
        raise ValueError("cols must be >= 1")

    if not (0.0 <= overlap < 1.0):
        raise ValueError(
            "overlap must be in [0, 1)"
        )

    tile_w = width / cols
    tile_h = height / rows

    overlap_w = tile_w * overlap
    overlap_h = tile_h * overlap

    tiles = []

    for r in range(rows):
        for c in range(cols):

            x0 = int(
                round(
                    c * tile_w
                    - overlap_w / 2
                )
            )

            y0 = int(
                round(
                    r * tile_h
                    - overlap_h / 2
                )
            )

            x1 = int(
                round(
                    (c + 1) * tile_w
                    + overlap_w / 2
                )
            )

            y1 = int(
                round(
                    (r + 1) * tile_h
                    + overlap_h / 2
                )
            )

            x0 = max(0, x0)
            y0 = max(0, y0)

            x1 = min(width, x1)
            y1 = min(height, y1)

            tiles.append(
                Tile(
                    x0=x0,
                    y0=y0,
                    x1=x1,
                    y1=y1,
                )
            )

    return tiles


def crop_array(
    array: np.ndarray,
    tile: Tile,
):
    return array[
        tile.y0:tile.y1,
        tile.x0:tile.x1,
    ]


def compute_instance_areas(
    semantics: np.ndarray,
    plant_instances: np.ndarray,
):
    areas = {}

    for label in (1, 2):
        ids = np.unique(
            plant_instances[
                (semantics == label)
                & (plant_instances > 0)
            ]
        )

        for instance_id in ids:
            areas[(label, int(instance_id))] = int(
                (
                    (plant_instances == instance_id)
                    & (semantics == label)
                ).sum()
            )

    return areas


# --------------------------------------------------
# Filtering
# --------------------------------------------------


@dataclass(frozen=True)
class FilterConfig:
    min_instance_pixels: int = 0

    min_bbox_width: int = 0
    min_bbox_height: int = 0

    min_bbox_area: int = 0

    min_visible_fraction: float = 0.0


# --------------------------------------------------
# BBox regeneration
# --------------------------------------------------


def generate_plant_bboxes(
    semantics: np.ndarray,
    plant_instances: np.ndarray,
    filter_config: FilterConfig,
    instance_areas: dict[
        tuple[int, int],
        int,
    ] | None = None,
):
    """
    Regenerate plant bboxes from cropped masks.

    If instance_areas is provided, visibility
    fractions are computed against the original
    uncropped instance area and can be filtered
    via filter_config.min_visible_fraction.
    """

    boxes = []

    for label in (1, 2):

        instance_ids = np.unique(
            plant_instances[
                (semantics == label)
                & (plant_instances > 0)
            ]
        )

        for instance_id in instance_ids:

            mask = (
                (plant_instances == instance_id)
                & (semantics == label)
            )

            visible_pixels = int(
                mask.sum()
            )

            if (
                visible_pixels
                < filter_config.min_instance_pixels
            ):
                continue

            visible_fraction = None

            if instance_areas is not None:

                original_pixels = instance_areas.get(
                    (
                        int(label),
                        int(instance_id),
                    )
                )

                if (
                    original_pixels is not None
                    and original_pixels > 0
                ):
                    visible_fraction = (
                        visible_pixels
                        / original_pixels
                    )

                    if (
                        visible_fraction
                        < filter_config.min_visible_fraction
                    ):
                        continue

            ys, xs = np.where(mask)

            if len(xs) == 0:
                continue

            xmin = int(xs.min())
            xmax = int(xs.max())

            ymin = int(ys.min())
            ymax = int(ys.max())

            width = xmax - xmin
            height = ymax - ymin

            area = width * height

            if (
                width
                < filter_config.min_bbox_width
            ):
                continue

            if (
                height
                < filter_config.min_bbox_height
            ):
                continue

            if (
                area
                < filter_config.min_bbox_area
            ):
                continue

            bbox = {
                "label": int(label),
                "corner": (
                    xmin,
                    ymin,
                ),
                "center": (
                    xmin + width // 2,
                    ymin + height // 2,
                ),
                "width": int(width),
                "height": int(height),
                "visible_pixels": visible_pixels,
            }

            if visible_fraction is not None:
                bbox[
                    "visible_fraction"
                ] = float(
                    visible_fraction
                )

            boxes.append(bbox)

    return boxes


# --------------------------------------------------
# Dataset indexing
# --------------------------------------------------


def decode_tile_index(
    index: int,
    tiles_per_image: int,
):
    image_index = (
        index // tiles_per_image
    )

    tile_index = (
        index % tiles_per_image
    )

    return (
        image_index,
        tile_index,
    )


# --------------------------------------------------
# Sample tiling
# --------------------------------------------------


def tile_sample(
    sample,
    tile: Tile,
    filter_config: FilterConfig,
):

    image = np.asarray(
        sample["image"]
    )

    semantics = sample["semantics"]

    plant_instances = sample[
        "plant_instances"
    ]

    image_tile = crop_array(
        image,
        tile,
    )

    semantics_tile = crop_array(
        semantics,
        tile,
    )

    instances_tile = crop_array(
        plant_instances,
        tile,
    )

    instance_areas = compute_instance_areas(
        semantics,
        plant_instances,
    )

    plant_bboxes = generate_plant_bboxes(
        semantics_tile,
        instances_tile,
        filter_config,
        instance_areas=instance_areas,
    )

    result = dict(sample)

    result["image"] = Image.fromarray(
        image_tile
    )

    result["semantics"] = semantics_tile

    result["plant_instances"] = (
        instances_tile
    )

    result["plant_bboxes"] = (
        plant_bboxes
    )

    result["tile"] = tile

    return result


# --------------------------------------------------
# Dataset wrapper
# --------------------------------------------------


class TiledPhenoBench:
    """
    Dataset wrapper that expands
    each image into rows × cols tiles.

    Requires:

        semantics
        plant_instances

    target_types.

    plant_bboxes are regenerated
    after tiling.
    """

    def __init__(
        self,
        dataset,
        rows: int = 2,
        cols: int = 2,
        overlap: float = 0.0,
        filter_config: FilterConfig | None = None,
    ):
        self.dataset = dataset

        self.rows = rows
        self.cols = cols

        self.overlap = overlap

        self.filter_config = (
            filter_config
            if filter_config is not None
            else FilterConfig()
        )

        self.tiles_per_image = (
            rows * cols
        )

        first = dataset[0]

        image = np.asarray(
            first["image"]
        )

        h, w = image.shape[:2]

        self.tiles = compute_tiles(
            width=w,
            height=h,
            rows=rows,
            cols=cols,
            overlap=overlap,
        )

    def __len__(self):

        return (
            len(self.dataset)
            * self.tiles_per_image
        )

    def __getitem__(
        self,
        index,
    ):
        image_index, tile_index = (
            decode_tile_index(
                index,
                self.tiles_per_image,
            )
        )

        sample = self.dataset[
            image_index
        ]

        tile = self.tiles[
            tile_index
        ]

        tiled = tile_sample(
            sample,
            tile,
            self.filter_config,
        )

        image_path = Path(sample["image_name"])

        tiled["image_name"] = (
            f"{image_path.stem}_tile{tile_index}{image_path.suffix}"
        )

        return tiled

    def tile_info(
        self,
        index,
    ):
        image_index, tile_index = (
            decode_tile_index(
                index,
                self.tiles_per_image,
            )
        )

        return {
            "image_index": image_index,
            "tile_index": tile_index,
            "tile": self.tiles[tile_index],
        }

    @property
    def source_dataset(self):
        return self.dataset


class ConcatDataset:
    """
    Concatenate multiple datasets.

    Example
    -------

    full_dataset = PhenoBench(...)

    tiled_dataset = TiledPhenoBench(
        full_dataset,
        rows=2,
        cols=2,
        overlap=0.25,
    )

    train_dataset = ConcatDataset(
        full_dataset,
        tiled_dataset,
    )

    len(train_dataset)

    ==

    len(full_dataset)
    +
    len(tiled_dataset)
    """

    def __init__(
        self,
        *datasets,
    ):
        if not datasets:
            raise ValueError(
                "At least one dataset "
                "must be provided."
            )

        self.datasets = list(
            datasets
        )

        self.cumulative_sizes = []

        total = 0

        for dataset in self.datasets:

            total += len(dataset)

            self.cumulative_sizes.append(
                total
            )

    def __len__(self):

        return self.cumulative_sizes[-1]

    def _locate(
        self,
        index,
    ):
        if index < 0:
            index += len(self)

        if (
            index < 0
            or index >= len(self)
        ):
            raise IndexError(index)

        dataset_idx = 0

        while (
            index
            >= self.cumulative_sizes[
                dataset_idx
            ]
        ):
            dataset_idx += 1

        previous = (
            0
            if dataset_idx == 0
            else self.cumulative_sizes[
                dataset_idx - 1
            ]
        )

        sample_idx = (
            index - previous
        )

        return (
            dataset_idx,
            sample_idx,
        )

    def __getitem__(
        self,
        index,
    ):
        dataset_idx, sample_idx = (
            self._locate(index)
        )

        return self.datasets[
            dataset_idx
        ][
            sample_idx
        ]

    def dataset_info(
        self,
        index,
    ):
        """
        Debug helper.
        """

        dataset_idx, sample_idx = (
            self._locate(index)
        )

        return {
            "dataset_index":
                dataset_idx,
            "sample_index":
                sample_idx,
            "dataset_type":
                type(
                    self.datasets[
                        dataset_idx
                    ]
                ).__name__,
        }

__all__ = [
    "Tile",
    "FilterConfig",
    "compute_tiles",
    "crop_array",
    "generate_plant_bboxes",
    "decode_tile_index",
    "tile_sample",
    "TiledPhenoBench",
    "ConcatDataset"
]
