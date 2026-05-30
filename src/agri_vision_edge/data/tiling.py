"""
Dataset tiling utilities.

Provides:

- tile geometry generation
- image/mask cropping
- tiled dataset wrapper

Tiling is performed before bbox extraction.

This allows upstream PhenoBench bbox generation
to remain the single source of truth.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import numpy as np

@dataclass(frozen=True)
class Tile:
    """
    Image tile.

    Coordinates follow numpy slicing:

        image[y0:y1, x0:x1]
    """
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
):
    """
    Partition image into rows × cols tiles.

    Returns
    -------
    list[Tile]
    """

    xs = np.linspace(
        0,
        width,
        cols + 1,
        dtype=int,
    )

    ys = np.linspace(
        0,
        height,
        rows + 1,
        dtype=int,
    )

    tiles = []

    for r in range(rows):

        for c in range(cols):

            tiles.append(

                Tile(
                    x0=int(xs[c]),
                    y0=int(ys[r]),
                    x1=int(xs[c + 1]),
                    y1=int(ys[r + 1]),
                )
            )

    return tiles


def crop_array(
    array: np.ndarray,
    tile: Tile,
):
    """
    Crop image or mask.
    """

    return array[
        tile.y0:tile.y1,
        tile.x0:tile.x1,
    ]


def decode_tile_index(
    index: int,
    tiles_per_image: int,
):
    """
    Global index ->

        image_index
        tile_index
    """

    image_index = (
        index // tiles_per_image
    )

    tile_index = (
        index % tiles_per_image
    )

    return image_index, tile_index


def tile_sample(
    sample,
    tile: Tile,
):
    """
    Create tiled sample.

    Expected keys:

        image
        semantics
    """

    image = np.asarray(
        sample["image"]
    )

    semantics = np.asarray(
        sample["semantics"]
    )

    image_tile = crop_array(
        image,
        tile,
    )

    semantics_tile = crop_array(
        semantics,
        tile,
    )

    result = dict(sample)

    result["image"] = image_tile
    result["semantics"] = semantics_tile

    return result



class TiledDataset:
    """
    Generic dataset tiling wrapper.

    Produces rows × cols samples
    per upstream sample.
    """

    def __init__(
        self,
        dataset,
        rows: int = 2,
        cols: int = 2,
    ):
        self.dataset = dataset

        self.rows = rows
        self.cols = cols

        self.tiles_per_image = (
            rows * cols
        )

        #
        # Determine geometry once.
        #

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

        return tile_sample(
            sample,
            tile,
        )
