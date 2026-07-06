"""
TFRecord export for TensorFlow Object Detection API.

Consumes upstream PhenoBench bounding boxes
instead of manually extracting boxes from masks.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import tensorflow as tf

from .categories import (
    build_class_names,
)
from .coco import (
    phenobench_bbox_to_xyxy,
)
from .datasets import (
    DatasetDefinition,
)
from .preprocessing import (
    normalize_boxes,
    resize_image_and_boxes,
)

DEFAULT_TARGET_SIZE = 320


def pil_to_numpy(img):

    return np.array(
        img,
        dtype=np.uint8,
    )


def create_tf_example(
    image,
    boxes,
    labels,
    categories,
    is_partial=None,
):
    """
    Create TensorFlow Example.

    Args:
        image:
            RGB image.

        boxes:
            Normalized bounding boxes.

        labels:
            Exported class labels.

        categories:
            Exported category definitions.

        is_partial:
            Optional per-box partial ("do-not-care") flags (0/1), aligned with
            ``boxes``. Written as the canonical ``image/object/is_partial`` and
            mirrored into ``image/object/is_crowd`` so the TFOD eval input
            pipeline surfaces it as ``groundtruth_is_crowd`` (PhenoBench has no
            genuine crowds), letting the trainer's ignore_partials knob identify
            partial ground-truth without a custom decoder. Defaults to all-zero.
    """

    class_names = build_class_names(
        categories
    )

    height, width = image.shape[:2]

    encoded = tf.io.encode_jpeg(
        image
    ).numpy()

    xmins = [b[0] for b in boxes]
    ymins = [b[1] for b in boxes]

    xmaxs = [b[2] for b in boxes]
    ymaxs = [b[3] for b in boxes]

    classes = [
        int(label)
        for label in labels
    ]

    classes_text = [
        class_names[label]
        for label in labels
    ]

    if is_partial is None:
        is_partial = [0] * len(boxes)

    partial_flags = [
        int(bool(flag))
        for flag in is_partial
    ]

    feature = {

        "image/height":
            tf.train.Feature(
                int64_list=tf.train.Int64List(
                    value=[height]
                )
            ),

        "image/width":
            tf.train.Feature(
                int64_list=tf.train.Int64List(
                    value=[width]
                )
            ),

        "image/encoded":
            tf.train.Feature(
                bytes_list=tf.train.BytesList(
                    value=[encoded]
                )
            ),

        "image/format":
            tf.train.Feature(
                bytes_list=tf.train.BytesList(
                    value=[b"jpeg"]
                )
            ),

        "image/object/bbox/xmin":
            tf.train.Feature(
                float_list=tf.train.FloatList(
                    value=xmins
                )
            ),

        "image/object/bbox/xmax":
            tf.train.Feature(
                float_list=tf.train.FloatList(
                    value=xmaxs
                )
            ),

        "image/object/bbox/ymin":
            tf.train.Feature(
                float_list=tf.train.FloatList(
                    value=ymins
                )
            ),

        "image/object/bbox/ymax":
            tf.train.Feature(
                float_list=tf.train.FloatList(
                    value=ymaxs
                )
            ),

        "image/object/class/label":
            tf.train.Feature(
                int64_list=tf.train.Int64List(
                    value=classes
                )
            ),

        "image/object/class/text":
            tf.train.Feature(
                bytes_list=tf.train.BytesList(
                    value=classes_text
                )
            ),

        "image/object/is_partial":
            tf.train.Feature(
                int64_list=tf.train.Int64List(
                    value=partial_flags
                )
            ),

        # Mirror of is_partial: TFOD's standard decoder surfaces this as
        # groundtruth_is_crowd, giving the trainer eval a decoder-native handle
        # on partials (PhenoBench has no real crowds).
        "image/object/is_crowd":
            tf.train.Feature(
                int64_list=tf.train.Int64List(
                    value=partial_flags
                )
            ),
    }

    return tf.train.Example(
        features=tf.train.Features(
            feature=feature
        )
    )


def build_record(
    target,
    dataset,
    dataset_definition: DatasetDefinition,
    indices: Iterable[int] | None = None,
    target_size: int = DEFAULT_TARGET_SIZE,
    skip_negatives=True,
    include_partials=False,
):
    """
    Build TFRecord dataset.

    Args:
        target:
            Output TFRecord path.

        dataset:
            PhenoBench dataset. To carry partials, wrap it in
            ``agri_vision_edge.data.plant_boxes.PartialAwarePhenoBench`` so each
            ``plant_bboxes`` entry gains an ``is_partial`` flag.

        dataset_definition:
            Canonical dataset definition.

        indices:
            Optional subset indices.

        target_size:
            Target image size.

        skip_negatives:
            Do not include images without GT instances.

        include_partials:
            When ``True``, partial ("do-not-care") plants are written to the
            record flagged (``image/object/is_partial`` / ``is_crowd``) so the
            trainer's ignore_partials eval knob can suppress detections on them.
            When ``False`` (the default) partial boxes are dropped and the
            output is unchanged from the pre-partials behaviour.
    """

    writer = tf.io.TFRecordWriter(
        str(target)
    )

    if indices is None:
        indices = range(len(dataset))

    written = 0
    negatives = 0

    for i in indices:

        sample = dataset[i]

        image = pil_to_numpy(
            sample["image"]
        )

        raw_boxes = []

        labels = []

        partial_flags = []

        for bbox in sample["plant_bboxes"]:

            source_label = int(
                bbox["label"]
            )

            #
            # Skip labels not exported
            # by this dataset definition
            #

            if (
                source_label
                not in dataset_definition.label_mapping
            ):
                continue

            is_partial = bool(bbox.get("is_partial", False))

            #
            # Partial ("do-not-care") plants: drop unless partials are
            # requested, in which case keep them flagged.
            #

            if is_partial and not include_partials:
                continue

            target_label = (
                dataset_definition.label_mapping[
                    source_label
                ]
            )

            labels.append(
                target_label
            )

            partial_flags.append(
                1 if is_partial else 0
            )

            raw_boxes.append(
                phenobench_bbox_to_xyxy(
                    bbox
                )
            )

        if not raw_boxes:
            negatives += 1
            if skip_negatives:
                continue

        image_resized, boxes_resized = (
            resize_image_and_boxes(
                image,
                raw_boxes,
                size=target_size,
            )
        )

        boxes_normalized = normalize_boxes(
            boxes_resized,
            image_size=target_size,
        )

        example = create_tf_example(

            image=image_resized,

            boxes=boxes_normalized,

            labels=labels,

            categories=dataset_definition.categories,

            is_partial=partial_flags,
        )

        writer.write(
            example.SerializeToString()
        )

        written += 1

    writer.close()

    print(
        f"{target} → written: {written}, {negatives} hard negatives "
        f"{'skipped' if skip_negatives else 'included'}."
    )

    return {

        "written":
            written,

        "target_size":
            target_size,

        "hard_negatives":
            negatives if not skip_negatives else 0,
    }
