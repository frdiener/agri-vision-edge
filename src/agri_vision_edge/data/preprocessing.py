"""
Preprocessing utilities for converting PhenoBench segmentation data
into TensorFlow Object Detection training data.

This module provides functions to:

1. Extract bounding boxes from instance + semantic masks
2. Resize images and scale bounding boxes to model input size
3. Normalize bounding boxes for TensorFlow Object Detection API
4. Split datasets into validation and test subsets

Typical usage:

    from agri_vision_edge.data.preprocessing import (
        process_sample,
        split_indices,
    )

    image = load_rgb_image(...)
    instances = load_instance_mask(...)
    semantics = load_semantic_mask(...)

    image_resized, boxes_normalized, labels = process_sample(
        image=image,
        instances=instances,
        semantics=semantics,
        size=320,
    )

Notes:
- Bounding boxes are extracted BEFORE resizing.
- All downstream steps (training, inference, quantization)
  must use identical preprocessing.
- Input images are expected to be uint8 in the range [0, 255].
"""

import random
from typing import List, Sequence, Tuple

import cv2
import numpy as np


DEFAULT_TARGET_SIZE = 320
DEFAULT_ALLOWED_CLASSES = (1, 2)
DEFAULT_MIN_AREA = 20


def extract_boxes(
    instances: np.ndarray,
    semantics: np.ndarray,
    allowed_classes: Tuple[int, ...] = DEFAULT_ALLOWED_CLASSES,
    min_area: int = DEFAULT_MIN_AREA,
) -> Tuple[List[List[int]], List[int]]:
    """
    Extract bounding boxes and class labels from instance and semantic masks.

    Each unique instance ID (>0) is treated as one object. The class label
    is determined via majority voting over the semantic mask.

    Args:
        instances (np.ndarray):
            Instance mask of shape (H, W).
            Values >0 correspond to individual plants.
        semantics (np.ndarray):
            Semantic mask of shape (H, W).
            Contains class IDs per pixel.
        allowed_classes (Tuple[int, ...]):
            Class IDs to keep.
        min_area (int):
            Minimum bounding box area in pixels.

    Returns:
        Tuple[List[List[int]], List[int]]:
            boxes:
                Bounding boxes in pixel coordinates:
                [xmin, ymin, xmax, ymax]
            labels:
                Corresponding class IDs.

    Notes:
        - Instances with classes not in `allowed_classes`
          are skipped.
        - Very small objects can be filtered via `min_area`.
    """
    boxes: List[List[int]] = []
    labels: List[int] = []

    instance_ids = np.unique(instances)
    instance_ids = instance_ids[instance_ids > 0]

    for inst_id in instance_ids:
        mask = instances == inst_id

        if not np.any(mask):
            continue

        ys, xs = np.where(mask)

        xmin = int(xs.min())
        xmax = int(xs.max())

        ymin = int(ys.min())
        ymax = int(ys.max())

        area = (xmax - xmin + 1) * (ymax - ymin + 1)

        if area < min_area:
            continue

        class_pixels = semantics[mask].astype(np.int32)
        cls = int(np.bincount(class_pixels).argmax())

        if cls not in allowed_classes:
            continue

        boxes.append([xmin, ymin, xmax, ymax])
        labels.append(cls)

    return boxes, labels


def resize_image_and_boxes(
    image: np.ndarray,
    boxes: Sequence[Sequence[int]],
    size: int = DEFAULT_TARGET_SIZE,
) -> Tuple[np.ndarray, List[List[float]]]:
    """
    Resize image to a fixed square size and scale bounding boxes.

    Args:
        image (np.ndarray):
            RGB image of shape (H, W, C).
        boxes (Sequence[Sequence[int]]):
            Bounding boxes in pixel coordinates:
            [xmin, ymin, xmax, ymax].
        size (int):
            Target square image size.

    Returns:
        Tuple[np.ndarray, List[List[float]]]:
            image_resized:
                Resized RGB image of shape (size, size, C).
            boxes_scaled:
                Bounding boxes scaled to resized image coordinates.

    Notes:
        - Aspect ratio is NOT preserved.
        - PhenoBench images are square, so distortion is negligible.
        - Output boxes are NOT normalized yet.
    """
    h, w = image.shape[:2]

    scale_x = size / float(w)
    scale_y = size / float(h)

    image_resized = cv2.resize(
        image,
        (size, size),
        interpolation=cv2.INTER_LINEAR,
    )

    boxes_scaled: List[List[float]] = []

    for xmin, ymin, xmax, ymax in boxes:
        boxes_scaled.append([
            xmin * scale_x,
            ymin * scale_y,
            xmax * scale_x,
            ymax * scale_y,
        ])

    return image_resized, boxes_scaled


def normalize_boxes(
    boxes: Sequence[Sequence[float]],
    image_size: int,
) -> List[List[float]]:
    """
    Normalize bounding boxes to the [0, 1] range.

    Args:
        boxes (Sequence[Sequence[float]]):
            Bounding boxes in pixel coordinates:
            [xmin, ymin, xmax, ymax].
        image_size (int):
            Square image size.

    Returns:
        List[List[float]]:
            Normalized bounding boxes.
    """
    norm_boxes: List[List[float]] = []

    for xmin, ymin, xmax, ymax in boxes:
        norm_boxes.append([
            xmin / image_size,
            ymin / image_size,
            xmax / image_size,
            ymax / image_size,
        ])

    return norm_boxes


def process_sample(
    image: np.ndarray,
    instances: np.ndarray,
    semantics: np.ndarray,
    size: int = DEFAULT_TARGET_SIZE,
    allowed_classes: Tuple[int, ...] = DEFAULT_ALLOWED_CLASSES,
    min_area: int = DEFAULT_MIN_AREA,
):
    """
    Full preprocessing pipeline for one sample.

    Combines:
        - instance → bounding boxes
        - resizing
        - normalization

    Args:
        image (np.ndarray):
            RGB image.
        instances (np.ndarray):
            Instance mask.
        semantics (np.ndarray):
            Semantic mask.
        size (int):
            Target image size.
        allowed_classes (Tuple[int, ...]):
            Allowed class IDs.
        min_area (int):
            Minimum bounding box area.

    Returns:
        Tuple:
            image_resized (np.ndarray):
                Resized RGB image.
            boxes_normalized (List[List[float]]):
                Normalized bounding boxes.
            labels (List[int]):
                Class labels.
    """
    boxes, labels = extract_boxes(
        instances,
        semantics,
        allowed_classes=allowed_classes,
        min_area=min_area,
    )

    image_resized, boxes_resized = resize_image_and_boxes(
        image,
        boxes,
        size=size,
    )

    boxes_normalized = normalize_boxes(
        boxes_resized,
        image_size=size,
    )

    return image_resized, boxes_normalized, labels


def split_indices(
    n: int,
    val_ratio: float = 0.5,
    seed: int = 42,
):
    """
    Split indices into validation and test subsets.

    Args:
        n (int):
            Total number of samples.
        val_ratio (float):
            Fraction assigned to the validation split.
        seed (int):
            Random seed for reproducibility.

    Returns:
        Tuple[List[int], List[int]]:
            val_idx:
                Validation indices.
            test_idx:
                Test indices.
    """
    indices = list(range(n))

    rng = random.Random(seed)
    rng.shuffle(indices)

    split = int(n * val_ratio)

    val_idx = indices[:split]
    test_idx = indices[split:]

    return val_idx, test_idx
