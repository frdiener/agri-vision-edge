"""
Representative dataset utilities for TFLite PTQ.
"""

from __future__ import annotations

import random

import numpy as np

from .coco import (
    phenobench_bbox_to_xyxy,
)

from .preprocessing import (
    resize_image_and_boxes,
)


DEFAULT_REPRESENTATIVE_SAMPLES = 200


def representative_dataset(
    dataset,
    indices=None,
    num_samples=100,
    size=320,
):
    """
    TFLite representative dataset generator.
    """

    if indices is None:
        indices = range(len(dataset))

    count = 0

    for i in indices:

        if count >= num_samples:
            break

        sample = dataset[i]

        image = np.array(
            sample["image"],
            dtype=np.uint8,
        )

        boxes = [

            phenobench_bbox_to_xyxy(
                bbox
            )

            for bbox in sample[
                "plant_bboxes"
            ]
        ]

        #
        # Skip empty samples
        #

        if not boxes:
            continue

        image_resized, _ = (
            resize_image_and_boxes(
                image,
                boxes,
                size=size,
            )
        )

        image_resized = (
            image_resized.astype(
                np.float32
            )
        )

        yield [
            np.expand_dims(
                image_resized,
                axis=0,
            )
        ]

        count += 1

def normalized_representative_dataset():
    """
    Normalized Representative Dataset for use with raw backbone quantization without full SSD wrapper.
    """
    for sample in representative_dataset(
        dataset=train_dataset,
        indices=rep_ds_indices,
        num_samples=200,
        size=IMAGE_SIZE,
    ):
        x = sample[0]

        x = (
            2.0 / 255.0
        ) * x - 1.0

        yield [x]


def build_rep_indices(
    dataset,
    num_samples=DEFAULT_REPRESENTATIVE_SAMPLES,
    seed=42,
):

    indices = list(range(len(dataset)))

    rng = random.Random(seed)

    rng.shuffle(indices)

    return indices[:num_samples]
