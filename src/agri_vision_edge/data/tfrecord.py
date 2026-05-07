"""
TFRecord builder for PhenoBench → TensorFlow Object Detection API.

This module converts PhenoBench samples (RGB image + instance mask +
semantic mask) into TFRecord files compatible with the TensorFlow
Object Detection API.

Pipeline:
    PhenoBench sample
        → process_sample (extract boxes, resize, normalize)
        → create_tf_example (serialize to TF Example)
        → TFRecord

Key assumptions:
- Images are resized to a fixed square size (e.g. 320x320)
- Bounding boxes are normalized to [0, 1]
- Class IDs follow the PhenoBench convention:
    1 = crop
    2 = weed

Typical usage:

    dataset = PhenoBench(
        root="/data/phenobench",
        split="train",
        target_types=["semantics", "plant_instances"],
    )

    build_record("train.record", dataset, with_tqdm=True)
"""

from typing import Iterable, Optional, Sequence

import numpy as np
import tensorflow as tf

from ..third_party.phenobench import PhenoBench
from .preprocessing import process_sample


DEFAULT_TARGET_SIZE = 320
DEFAULT_ALLOWED_CLASSES = (1, 2)
DEFAULT_MIN_AREA = 20

CLASS_NAMES = {
    1: b"crop",
    2: b"weed",
}


def pil_to_numpy(img) -> np.ndarray:
    """
    Convert a PIL image to a NumPy array.

    Args:
        img:
            PIL RGB image.

    Returns:
        np.ndarray:
            RGB image array of shape (H, W, 3), dtype uint8.
    """
    return np.array(img, dtype=np.uint8)


def create_tf_example(
    image: np.ndarray,
    boxes: Sequence[Sequence[float]],
    labels: Sequence[int],
) -> tf.train.Example:
    """
    Create a TensorFlow Example from one preprocessed sample.

    Args:
        image (np.ndarray):
            RGB image (H, W, 3), uint8.
        boxes (Sequence[Sequence[float]]):
            Normalized bounding boxes in the format:
            [xmin, ymin, xmax, ymax].
        labels (Sequence[int]):
            Class IDs.

    Returns:
        tf.train.Example:
            Serialized TF Example for TFRecord writing.
    """
    height, width = image.shape[:2]

    # Encode image as JPEG bytes
    encoded = tf.io.encode_jpeg(image).numpy()

    xmins = [float(b[0]) for b in boxes]
    ymins = [float(b[1]) for b in boxes]
    xmaxs = [float(b[2]) for b in boxes]
    ymaxs = [float(b[3]) for b in boxes]

    classes = [int(label) for label in labels]
    classes_text = [CLASS_NAMES[label] for label in labels]

    feature = {
        "image/height": tf.train.Feature(
            int64_list=tf.train.Int64List(value=[height])
        ),
        "image/width": tf.train.Feature(
            int64_list=tf.train.Int64List(value=[width])
        ),
        "image/encoded": tf.train.Feature(
            bytes_list=tf.train.BytesList(value=[encoded])
        ),

        "image/object/bbox/xmin": tf.train.Feature(
            float_list=tf.train.FloatList(value=xmins)
        ),
        "image/object/bbox/xmax": tf.train.Feature(
            float_list=tf.train.FloatList(value=xmaxs)
        ),
        "image/object/bbox/ymin": tf.train.Feature(
            float_list=tf.train.FloatList(value=ymins)
        ),
        "image/object/bbox/ymax": tf.train.Feature(
            float_list=tf.train.FloatList(value=ymaxs)
        ),

        "image/object/class/label": tf.train.Feature(
            int64_list=tf.train.Int64List(value=classes)
        ),
        "image/object/class/text": tf.train.Feature(
            bytes_list=tf.train.BytesList(value=classes_text)
        ),
    }

    return tf.train.Example(
        features=tf.train.Features(feature=feature)
    )


def build_record(
    target: str,
    dataset: PhenoBench,
    indices: Optional[Iterable[int]] = None,
    target_size: int = DEFAULT_TARGET_SIZE,
    with_tqdm: bool = False,
):
    """
    Build a TFRecord file from a PhenoBench dataset.

    Each sample is processed using `process_sample`, converted into
    a TensorFlow Example, and written to disk.

    Args:
        target (str):
            Output path for the TFRecord file.
        dataset (PhenoBench):
            Dataset instance.
        indices (Optional[Iterable[int]]):
            Subset of dataset indices to process.
            If None, all samples are used.
        target_size (int):
            Target square image size used during preprocessing.
        with_tqdm (bool):
            If True, display a progress bar.

    Returns:
        dict:
            Statistics about the generated TFRecord:
            {
                "written": int,
                "skipped": int,
            }

    Notes:
        - Samples without valid bounding boxes are skipped.
        - Images are resized and boxes normalized before serialization.
        - Ensure the dataset includes:
            target_types = ["semantics", "plant_instances"]
    """
    writer = tf.io.TFRecordWriter(target)

    written = 0
    skipped = 0

    if indices is None:
        indices = range(len(dataset))

    if with_tqdm:
        from tqdm import tqdm
        iterator = tqdm(indices)
    else:
        iterator = indices

    for i in iterator:
        sample = dataset[i]

        image = pil_to_numpy(sample["image"])
        instances = sample["plant_instances"]
        semantics = sample["semantics"]

        image_resized, boxes, labels = process_sample(
            image=image,
            instances=instances,
            semantics=semantics,
            size=target_size,
            allowed_classes=DEFAULT_ALLOWED_CLASSES,
            min_area=DEFAULT_MIN_AREA,
        )

        # Skip empty samples
        if len(boxes) == 0:
            skipped += 1
            continue

        example = create_tf_example(
            image_resized,
            boxes,
            labels,
        )

        writer.write(example.SerializeToString())

        written += 1

        if with_tqdm:
            iterator.set_postfix(
                written=written,
                skipped=skipped,
            )

    writer.close()

    print(f"{target} → written: {written}, skipped: {skipped}")

    return {
        "written": written,
        "skipped": skipped,
    }
