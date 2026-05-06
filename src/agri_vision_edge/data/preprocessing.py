"""
Preprocessing utilities for converting PhenoBench segmentation data
into object detection training data.

This module provides functions to:

1. Extract bounding boxes from instance + semantic masks
2. Resize images and scale bounding boxes to model input size
3. Normalize bounding boxes for TensorFlow Object Detection API

Typical usage (dataset preparation):

    from agri_vision_edge.data.preprocessing import (
        extract_boxes,
        resize_image_and_boxes,
        normalize_boxes,
    )

    image = load_rgb_image(...)
    instances = load_instance_mask(...)
    semantics = load_semantic_mask(...)

    boxes, labels = extract_boxes(instances, semantics)

    image_resized, boxes_resized = resize_image_and_boxes(
        image, boxes, size=320
    )

    boxes_normalized = normalize_boxes(boxes_resized, image_size=320)

Notes:
- Bounding boxes are extracted BEFORE resizing.
- All downstream steps (training, inference, quantization) must use the same preprocessing.
- Input images are expected to be uint8 in range [0, 255].
"""

from typing import List, Tuple
import numpy as np
import cv2


def extract_boxes(
    instances: np.ndarray,
    semantics: np.ndarray,
    allowed_classes: Tuple[int, ...] = (1, 2),
    min_area: int = 0,
) -> Tuple[List[List[int]], List[int]]:
    """
    Extract bounding boxes and class labels from instance and semantic masks.

    Each unique instance ID (>0) is treated as one object. The class label is
    determined via majority voting over the semantic mask.

    Args:
        instances (np.ndarray):
            Instance mask (H, W), values >0 correspond to individual plants.
        semantics (np.ndarray):
            Semantic mask (H, W), containing class IDs per pixel.
        allowed_classes (Tuple[int, ...], optional):
            Class IDs to keep (default: (1, 2) → crop, weed).
        min_area (int, optional):
            Minimum bounding box area in pixels to keep an instance.

    Returns:
        Tuple[List[List[int]], List[int]]:
            - boxes: list of [xmin, ymin, xmax, ymax] (pixel coordinates)
            - labels: list of class IDs

    Notes:
        - Instances with classes not in allowed_classes are skipped.
        - Very small objects can be filtered via min_area.
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

        xmin, xmax = xs.min(), xs.max()
        ymin, ymax = ys.min(), ys.max()

        area = (xmax - xmin + 1) * (ymax - ymin + 1)
        if area < min_area:
            continue

        class_pixels = semantics[mask].astype(np.int32)
        cls = np.bincount(class_pixels).argmax()

        if cls not in allowed_classes:
            continue

        boxes.append([xmin, ymin, xmax, ymax])
        labels.append(int(cls))

    return boxes, labels


def resize_image_and_boxes(
    image: np.ndarray,
    boxes: List[List[int]],
    size: int = 320,
) -> Tuple[np.ndarray, List[List[float]]]:
    """
    Resize image to fixed square size and scale bounding boxes accordingly.

    Args:
        image (np.ndarray):
            Input RGB image (H, W, C).
        boxes (List[List[int]]):
            Bounding boxes [xmin, ymin, xmax, ymax] in pixel coordinates.
        size (int, optional):
            Target size (default: 320).

    Returns:
        Tuple[np.ndarray, List[List[float]]]:
            - resized image (size, size, C)
            - scaled bounding boxes (pixel coordinates)

    Notes:
        - No aspect ratio preservation (PhenoBench is square → safe).
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
    boxes: List[List[float]],
    image_size: int,
) -> List[List[float]]:
    """
    Normalize bounding boxes to [0, 1] range.

    Args:
        boxes (List[List[float]]):
            Bounding boxes in pixel coordinates.
        image_size (int):
            Size of the (square) image.

    Returns:
        List[List[float]]:
            Normalized boxes [xmin, ymin, xmax, ymax].

    Notes:
        - Required for TensorFlow Object Detection API.
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
    size: int = 320,
    allowed_classes: Tuple[int, ...] = (1, 2),
    min_area: int = 0,
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
            Minimum box area.

    Returns:
        Tuple:
            image_resized (np.ndarray),
            boxes_normalized (List[List[float]]),
            labels (List[int])
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

    boxes_normalized = normalize_boxes(boxes_resized, size)

    return image_resized, boxes_normalized, labels

import random


def split_indices(n, val_ratio=0.5, seed=42):
    """
    Split indices into val/test.

    val_ratio = fraction that goes into validation set
    """
    indices = list(range(n))

    rng = random.Random(seed)
    rng.shuffle(indices)

    split = int(n * val_ratio)

    val_idx = indices[:split]
    test_idx = indices[split:]

    return val_idx, test_idx
