"""
Representative dataset utilities for TensorFlow Lite quantization.

This module provides:

1. Representative dataset generators for post-training quantization (PTQ)
2. Deterministic representative dataset index selection

The representative dataset must use the SAME preprocessing pipeline
as training to ensure correct activation calibration during quantization.

Typical usage (TFLite conversion):

    rep_indices = build_rep_indices(
        train_dataset,
        num_samples=200,
        seed=42,
    )

    converter.representative_dataset = lambda: representative_dataset(
        dataset=train_dataset,
        indices=rep_indices,
    )

Notes:
- Representative samples are yielded as float32 tensors.
- TensorFlow Lite internally computes quantization parameters.
- Images are resized identically to training preprocessing.
- Empty samples are skipped automatically.
"""

import random
from typing import Iterable, List, Optional

import numpy as np

from third_party.phenobench import PhenoBench
from .preprocessing import (
    DEFAULT_ALLOWED_CLASSES,
    DEFAULT_MIN_AREA,
    DEFAULT_TARGET_SIZE,
    process_sample,
)


DEFAULT_REPRESENTATIVE_SAMPLES = 200
DEFAULT_REPRESENTATIVE_SEED = 42


def representative_dataset(
    dataset: PhenoBench,
    indices: Optional[Iterable[int]] = None,
    num_samples: int = 100,
    size: int = DEFAULT_TARGET_SIZE,
):
    """
    Create a representative dataset generator for TFLite quantization.

    Args:
        dataset (PhenoBench):
            Dataset instance, typically the training split.
        indices (Optional[Iterable[int]]):
            Subset of dataset indices to use.
            If None, the full dataset is used.
        num_samples (int):
            Maximum number of representative samples to yield.
        size (int):
            Input image size.
            Must match the training preprocessing pipeline.

    Yields:
        List[np.ndarray]:
            Batched float32 input tensor:
            [1, H, W, 3]

    Notes:
        - Uses identical preprocessing as training.
        - Only images are yielded (no labels).
        - Samples are yielded as float32 tensors.
        - TFLite internally computes int8 quantization parameters.
        - Empty samples are skipped automatically.
    """
    if indices is None:
        indices = range(len(dataset))

    count = 0

    for i in indices:
        if count >= num_samples:
            break

        sample = dataset[i]

        image = np.array(sample["image"], dtype=np.uint8)
        instances = sample["plant_instances"]
        semantics = sample["semantics"]

        image_resized, boxes, _ = process_sample(
            image=image,
            instances=instances,
            semantics=semantics,
            size=size,
            allowed_classes=DEFAULT_ALLOWED_CLASSES,
            min_area=DEFAULT_MIN_AREA,
        )

        # Skip empty samples
        if len(boxes) == 0:
            continue

        # TFLite representative datasets expect float32 inputs
        image_resized = image_resized.astype(np.float32)

        # Add batch dimension → (1, H, W, 3)
        yield [np.expand_dims(image_resized, axis=0)]

        count += 1


def build_rep_indices(
    dataset: PhenoBench,
    num_samples: int = DEFAULT_REPRESENTATIVE_SAMPLES,
    seed: int = DEFAULT_REPRESENTATIVE_SEED,
) -> List[int]:
    """
    Build deterministic representative dataset indices.

    Args:
        dataset (PhenoBench):
            Dataset instance used for representative sampling.
        num_samples (int):
            Number of representative samples to select.
        seed (int):
            Random seed for reproducibility.

    Returns:
        List[int]:
            Randomized representative dataset indices.

    Notes:
        - Sampling is deterministic given the same seed.
        - Indices can be serialized and reused across notebooks.
    """
    indices = list(range(len(dataset)))

    rng = random.Random(seed)
    rng.shuffle(indices)

    return indices[:num_samples]
