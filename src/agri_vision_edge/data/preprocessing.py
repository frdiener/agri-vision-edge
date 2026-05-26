"""
Detection preprocessing utilities.

This module intentionally does NOT perform
bounding-box extraction anymore.

Bounding boxes should originate from the
upstream PhenoBench bbox API.

Responsibilities:

- image resizing
- bbox resizing
- bbox normalization
- dataset splitting
"""

from __future__ import annotations

import random

from typing import Sequence

import cv2
import numpy as np


DEFAULT_TARGET_SIZE = 320


def resize_image_and_boxes(
    image: np.ndarray,
    boxes: Sequence[Sequence[float]],
    size: int = DEFAULT_TARGET_SIZE,
):
    """
    Resize image and scale bounding boxes.

    Args:
        image:
            RGB image.

        boxes:
            Bounding boxes in:

                [xmin, ymin, xmax, ymax]

        size:
            Target square image size.

    Returns:
        Tuple:

            image_resized,
            boxes_resized
    """

    h, w = image.shape[:2]

    scale_x = size / float(w)
    scale_y = size / float(h)

    image_resized = cv2.resize(
        image,
        (size, size),
        interpolation=cv2.INTER_LINEAR,
    )

    boxes_resized = []

    for xmin, ymin, xmax, ymax in boxes:

        boxes_resized.append([
            xmin * scale_x,
            ymin * scale_y,
            xmax * scale_x,
            ymax * scale_y,
        ])

    return image_resized, boxes_resized


def normalize_boxes(
    boxes,
    image_size,
):
    """
    Normalize bounding boxes to [0, 1].

    Args:
        boxes:
            Bounding boxes in pixel coordinates:

                [xmin, ymin, xmax, ymax]

        image_size:
            Square image size.

    Returns:
        Normalized bounding boxes.
    """

    normalized = []

    for xmin, ymin, xmax, ymax in boxes:

        normalized.append([
            xmin / image_size,
            ymin / image_size,
            xmax / image_size,
            ymax / image_size,
        ])

    return normalized


def split_indices(
    n: int,
    val_ratio: float = 0.5,
    seed: int = 42,
):
    """
    Split indices deterministically.

    Args:
        n:
            Number of samples.

        val_ratio:
            Validation fraction.

        seed:
            Random seed.

    Returns:
        val_indices,
        test_indices
    """

    indices = list(range(n))

    rng = random.Random(seed)

    rng.shuffle(indices)

    split = int(n * val_ratio)

    return (
        indices[:split],
        indices[split:],
    )
