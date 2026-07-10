"""
PhenoBench tiling utilities.

Tiling is applied before bbox generation.

Bounding boxes are regenerated from
cropped semantics + plant_instances.

Boxes are never dropped: every plant
instance in a tile yields a box. Instead,
tile-border fragments and otherwise
partially-visible plants are *tagged*
``is_partial`` (do-not-care) following the
upstream PhenoBench ``visibility <= 0.5``
rule applied to each box's effective
visibility -- the fraction of the whole
plant visible within the tile, combining
original-frame occlusion with the tile cut
(see ``generate_plant_bboxes``).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

# Visibility masks store a 0..255 visible-fraction; normalize to [0, 1].
_VISIBILITY_SCALE = 255.0

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

    # Uniform tile size + stride so all tiles are the SAME size and together
    # cover the frame exactly, overlapping by `overlap`. Solving
    # ``(cols - 1) * stride + tile = width`` with ``stride = tile * (1 - overlap)``
    # gives ``tile = width / ((cols - 1) * (1 - overlap) + 1)``. This avoids the
    # non-square, clamped edge tiles the naive ``width / cols`` extend-and-clamp
    # produced for overlap > 0 (those get stretched to a square on export). For
    # overlap == 0 it reduces to the old ``width / cols`` grid.
    tile_w = width / ((cols - 1) * (1.0 - overlap) + 1)
    tile_h = height / ((rows - 1) * (1.0 - overlap) + 1)

    stride_w = tile_w * (1.0 - overlap)
    stride_h = tile_h * (1.0 - overlap)

    tiles = []

    for r in range(rows):
        for c in range(cols):

            x0 = int(round(c * stride_w))
            y0 = int(round(r * stride_h))

            # Uniform extent; the construction lands the last tile on the frame
            # edge (modulo rounding), so clamp defensively.
            x1 = min(width, int(round(c * stride_w + tile_w)))
            y1 = min(height, int(round(r * stride_h + tile_h)))

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
# BBox regeneration
# --------------------------------------------------


def _combine_visibility(
    upstream_visibility: float | None,
    visible_fraction: float | None,
) -> float | None:
    """
    Fraction of the WHOLE plant visible within a tile.

    ``upstream_visibility`` (fraction visible in the original frame) and
    ``visible_fraction`` (fraction of the frame-visible plant surviving the tile
    cut) are independent reductions, so the overall visible fraction is their
    product. Either may be ``None`` when its source is unavailable, in which case
    the other is returned alone; when both are ``None`` the caller cannot decide
    partiality and gets ``None``.
    """

    if upstream_visibility is None:
        return visible_fraction

    if visible_fraction is None:
        return upstream_visibility

    return upstream_visibility * visible_fraction


def generate_plant_bboxes(
    semantics: np.ndarray,
    plant_instances: np.ndarray,
    instance_areas: dict[
        tuple[int, int],
        int,
    ] | None = None,
    partial_threshold: float | None = None,
    plant_visibility: np.ndarray | None = None,
):
    """
    Regenerate plant bboxes from cropped masks.

    Every plant instance present in the (cropped) masks yields a box; there is
    no size- or visibility-based *dropping*. Instead, partially-visible plants
    -- including tile-border fragments -- are *tagged* ``is_partial``
    (do-not-care) so downstream evaluation can ignore them without removing them.

    If ``instance_areas`` is provided, the per-instance ``visible_fraction``
    (fraction of the original uncropped instance surviving the tile cut) is
    computed and stored on the box; it also feeds the ``is_partial`` decision.

    Partiality applies the **upstream PhenoBench criterion** (``<=
    partial_threshold``) to the box's *effective visibility* -- the fraction of
    the **whole plant** that is visible within this tile. Two independent
    reductions combine (by product) into that fraction so tile-slice borders are
    handled exactly like the upstream visibility rule:

    * ``upstream_visibility`` -- the fraction of the plant that was visible in
      the **original frame** (occlusion / frame border), read per instance from
      the cropped ``plant_visibility`` mask as ``max(plant_visibility[mask]) /
      255``. Available when ``plant_visibility`` is given.
    * ``visible_fraction`` -- the fraction of the frame-visible plant that
      survives the **tile cut** (``visible_pixels / original_pixels``).
      Available when ``instance_areas`` is given.

    The effective visibility is ``upstream_visibility * visible_fraction`` when
    both are known, or whichever single source is available otherwise; a box is
    flagged ``is_partial`` when it is ``<= partial_threshold``. This means a
    plant that was fully visible in the frame but is sliced away by a tile
    border (leaving ``<= partial_threshold`` of its pixels) is flagged partial,
    consistently with an originally-partial plant. The effective visibility is
    also stored on the box as ``visibility``.
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

            ys, xs = np.where(mask)

            if len(xs) == 0:
                continue

            xmin = int(xs.min())
            xmax = int(xs.max())

            ymin = int(ys.min())
            ymax = int(ys.max())

            width = xmax - xmin
            height = ymax - ymin

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

            # Upstream visibility: the per-instance visibility value read from
            # the cropped plant_visibility mask, normalized to [0, 1]. This is
            # the fraction of the plant that was visible in the ORIGINAL frame
            # (occlusion / frame border), before any tile cut.
            upstream_visibility = None

            if plant_visibility is not None:
                upstream_visibility = (
                    float(plant_visibility[mask].max())
                    / _VISIBILITY_SCALE
                )

            # Effective visibility (do-not-care criterion): the fraction of the
            # WHOLE plant visible within this tile. Frame occlusion
            # (upstream_visibility) and the tile cut (visible_fraction) are
            # independent reductions, so they combine by product. Applying the
            # upstream ``<= partial_threshold`` rule to it flags both
            # originally-partial plants and plants sliced away by a tile border,
            # keeping tile borders consistent with the upstream 0.5 criterion.
            effective_visibility = _combine_visibility(
                upstream_visibility,
                visible_fraction,
            )

            if effective_visibility is not None:
                bbox["visibility"] = float(effective_visibility)

            if (
                partial_threshold is not None
                and effective_visibility is not None
            ):
                bbox["is_partial"] = bool(
                    effective_visibility <= partial_threshold
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
    partial_threshold: float | None = None,
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

    # Upstream visibility mask (optional): cropped per tile so per-box
    # visibility is read against the same tile coordinate frame.
    visibility = sample.get("plant_visibility")
    visibility_tile = (
        crop_array(np.asarray(visibility), tile)
        if visibility is not None
        else None
    )

    instance_areas = compute_instance_areas(
        semantics,
        plant_instances,
    )

    plant_bboxes = generate_plant_bboxes(
        semantics_tile,
        instances_tile,
        instance_areas=instance_areas,
        partial_threshold=partial_threshold,
        plant_visibility=visibility_tile,
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
        partial_threshold: float | None = None,
    ):
        self.dataset = dataset

        self.rows = rows
        self.cols = cols

        self.overlap = overlap

        # When set, boxes whose effective visibility (upstream frame visibility
        # combined with the tile-cut fraction) is <= this are tagged is_partial
        # (do-not-care). See generate_plant_bboxes.
        self.partial_threshold = partial_threshold

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
            partial_threshold=self.partial_threshold,
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
