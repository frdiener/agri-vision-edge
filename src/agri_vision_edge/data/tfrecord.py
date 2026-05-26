"""
TFRecord export for TensorFlow Object Detection API.

Consumes upstream PhenoBench bounding boxes
instead of manually extracting boxes from masks.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
import tensorflow as tf

from .categories import (
    build_class_names,
)

from .coco import (
    phenobench_bbox_to_xyxy,
)

from .preprocessing import (
    resize_image_and_boxes,
    normalize_boxes,
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
):
    """
    Create TensorFlow Example.
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
    }

    return tf.train.Example(
        features=tf.train.Features(
            feature=feature
        )
    )


def build_record(
    target,
    dataset,
    categories,
    indices: Iterable[int] | None = None,
    target_size: int = DEFAULT_TARGET_SIZE,
):
    """
    Build TFRecord dataset.
    """

    writer = tf.io.TFRecordWriter(
        str(target)
    )

    if indices is None:
        indices = range(len(dataset))

    written = 0

    for i in indices:

        sample = dataset[i]

        image = pil_to_numpy(
            sample["image"]
        )

        raw_boxes = []

        labels = []

        for bbox in sample["plant_bboxes"]:

            raw_boxes.append(
                phenobench_bbox_to_xyxy(
                    bbox
                )
            )

            labels.append(
                bbox["label"]
            )

        if not raw_boxes:
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
            categories=categories,
        )

        writer.write(
            example.SerializeToString()
        )

        written += 1

    writer.close()

    print(
        f"{target} → written: {written}"
    )
