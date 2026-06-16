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

def normalized_representative_dataset(
    dataset,
    indices=None,
    num_samples=100,
    size=320,
):
    """
    Representative dataset yielding inputs normalized to [-1, 1].

    Use this whenever the converted graph expects already-preprocessed input,
    e.g. ``SSDModule.inference_fn`` (which calls ``model.predict`` WITHOUT the
    SSD preprocessing step) or a raw backbone without the full SSD wrapper.
    Feeding the raw [0, 255] ``representative_dataset`` to such a graph
    saturates calibration and collapses detection scores to sigmoid(0) = 0.5.
    """

    for sample in representative_dataset(
        dataset=dataset,
        indices=indices,
        num_samples=num_samples,
        size=size,
    ):
        yield [(2.0 / 255.0) * sample[0] - 1.0]


def build_rep_indices(
    dataset,
    num_samples=DEFAULT_REPRESENTATIVE_SAMPLES,
    seed=42,
):

    indices = list(range(len(dataset)))

    rng = random.Random(seed)

    rng.shuffle(indices)

    return indices[:num_samples]
