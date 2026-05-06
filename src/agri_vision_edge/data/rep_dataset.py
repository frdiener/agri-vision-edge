"""
Representative dataset generator for TFLite quantization.

Provides a generator that yields preprocessed input samples
matching the training pipeline.

Usage (TFLite):

    converter.representative_dataset = lambda: representative_dataset(...)
"""

from typing import Iterable, Optional
import numpy as np
import random

from .phenobench_loader import PhenoBench
from .preprocessing import process_sample


def representative_dataset(
    dataset: PhenoBench,
    indices: Optional[Iterable[int]] = None,
    num_samples: int = 100,
    size: int = 320,
):
    """
    Create a representative dataset generator for TFLite quantization.

    Args:
        dataset (PhenoBench):
            Dataset instance (typically train split).
        indices (Optional[Iterable[int]]):
            Subset of dataset indices. If None, uses full dataset.
        num_samples (int):
            Maximum number of samples to yield.
        size (int):
            Input size (must match training).

    Yields:
        List[np.ndarray]:
            Single input tensor [1, H, W, 3] as float32.

    Notes:
        - Uses the SAME preprocessing as training.
        - Only images are yielded (no labels).
        - Output is batched (batch=1) as required by TFLite.
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

        image_resized, boxes, labels = process_sample(
            image=image,
            instances=instances,
            semantics=semantics,
            size=size,
            allowed_classes=(1, 2),
            min_area=20,
        )

        # skip empty samples (important for stability)
        if not boxes:
            continue

        # convert to float32 if your model expects it
        image_resized = image_resized.astype(np.float32)

        # add batch dimension → (1, H, W, 3)
        yield [np.expand_dims(image_resized, axis=0)]

        count += 1


def build_rep_indices(dataset, num_samples=200, seed=42):
    indices = list(range(len(dataset)))
    rng = random.Random(seed)
    rng.shuffle(indices)

    return indices[:num_samples]
