"""
YOLO (Darknet/Ultralytics) export utilities.

Materializes a PhenoBench dataset into the on-disk layout expected by the
official YOLOv7 trainer (WongKinYiu/yolov7), whose dataloader
(`LoadImagesAndLabels`) reads images and one ``.txt`` label file per image
rather than a TFRecord/COCO bundle:

    <root>/
    ├── images/<split>/<stem>.png
    ├── labels/<split>/<stem>.txt
    └── data.yaml

Each label line is::

    <class> <cx> <cy> <w> <h>

with a 0-based class index and box coordinates normalized to ``[0, 1]`` by the
image dimensions (YOLO convention).

The exporter consumes the same ``plant_bboxes`` samples used by the COCO export
(:mod:`agri_vision_edge.data.coco`), so it works transparently over a plain
``PhenoBench(target_types=["plant_bboxes"])`` dataset (full images) or a
``TiledPhenoBench`` wrapper (per-tile regenerated boxes). Class identity and
semantic remapping come from a canonical :class:`DatasetDefinition`.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from .coco import phenobench_bbox_to_xyxy
from .datasets import DatasetDefinition


def yolo_class_names(
    dataset_definition: DatasetDefinition,
) -> list[str]:
    """
    Ordered class names for ``data.yaml``.

    YOLO class indices are 0-based and contiguous; the canonical
    ``DatasetDefinition`` uses 1-based COCO category IDs, so name ``i`` here
    corresponds to exported category ID ``i + 1``.
    """

    categories = sorted(
        dataset_definition.categories,
        key=lambda c: c["id"],
    )

    return [str(c["name"]) for c in categories]


def phenobench_bbox_to_yolo(
    bbox,
    image_width: int,
    image_height: int,
):
    """
    Convert an upstream PhenoBench bbox to a normalized YOLO box.

    Returns:
        (cx, cy, w, h) normalized to [0, 1] and clipped to the image bounds.
    """

    xmin, ymin, xmax, ymax = phenobench_bbox_to_xyxy(bbox)

    # Clip to the image frame -- tiled boxes are already in-tile, but border
    # fragments can land a single pixel outside after rounding.
    xmin = min(max(xmin, 0.0), image_width)
    xmax = min(max(xmax, 0.0), image_width)
    ymin = min(max(ymin, 0.0), image_height)
    ymax = min(max(ymax, 0.0), image_height)

    cx = (xmin + xmax) / 2.0 / image_width
    cy = (ymin + ymax) / 2.0 / image_height

    w = (xmax - xmin) / image_width
    h = (ymax - ymin) / image_height

    return cx, cy, w, h


def export_yolo_split(
    dataset,
    dataset_definition: DatasetDefinition,
    output_dir: str | Path,
    split: str,
    indices=None,
    min_box_size: float = 0.0,
) -> dict:
    """
    Export one dataset split as images + YOLO label files.

    Args:
        dataset:
            A PhenoBench-style dataset yielding samples with ``image`` (PIL) and
            ``plant_bboxes`` (e.g. ``PhenoBench(target_types=["plant_bboxes"])``
            or a ``TiledPhenoBench`` wrapper).

        dataset_definition:
            Canonical definition supplying the label remapping. Boxes whose
            upstream label is absent from ``label_mapping`` are skipped.

        output_dir:
            Dataset root; ``images/<split>`` and ``labels/<split>`` are created
            beneath it.

        split:
            Split name (``train`` / ``val``) used for the sub-directories.

        indices:
            Optional subset of dataset indices (defaults to the whole dataset).

        min_box_size:
            Drop boxes whose normalized width or height is below this value
            (degenerate fragments). ``0.0`` keeps everything.

    Returns:
        Summary dict with image and box counts.
    """

    output_dir = Path(output_dir)

    images_dir = output_dir / "images" / split
    labels_dir = output_dir / "labels" / split

    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    label_mapping = dataset_definition.label_mapping

    if indices is None:
        indices = range(len(dataset))

    num_images = 0
    num_boxes = 0
    num_skipped_boxes = 0

    for dataset_index in indices:
        sample = dataset[dataset_index]

        image = sample["image"]

        if not isinstance(image, Image.Image):
            image = Image.fromarray(np.asarray(image))

        width, height = image.size

        stem = Path(sample["image_name"]).stem

        image.save(images_dir / f"{stem}.png")

        lines = []

        for bbox in sample["plant_bboxes"]:
            source_label = int(bbox["label"])

            if source_label not in label_mapping:
                continue

            # COCO category ID (1-based) -> YOLO class index (0-based).
            yolo_class = label_mapping[source_label] - 1

            cx, cy, w, h = phenobench_bbox_to_yolo(
                bbox,
                width,
                height,
            )

            if w <= min_box_size or h <= min_box_size:
                num_skipped_boxes += 1
                continue

            lines.append(f"{yolo_class} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

        # An image with no boxes still gets an (empty) label file so YOLOv7
        # treats it as a valid background image rather than missing-label.
        (labels_dir / f"{stem}.txt").write_text("\n".join(lines))

        num_images += 1
        num_boxes += len(lines)

    return {
        "split": split,
        "images": num_images,
        "boxes": num_boxes,
        "skipped_boxes": num_skipped_boxes,
        "images_dir": str(images_dir),
        "labels_dir": str(labels_dir),
    }


def write_data_yaml(
    output_dir: str | Path,
    dataset_definition: DatasetDefinition,
    train_split: str = "train",
    val_split: str = "val",
    test_split: str | None = None,
) -> Path:
    """
    Write the YOLOv7 ``data.yaml`` describing splits and classes.

    Paths are written relative to ``output_dir`` (the dataset root), matching
    the ``images/<split>`` layout produced by :func:`export_yolo_split`.
    """

    output_dir = Path(output_dir)

    names = yolo_class_names(dataset_definition)

    lines = [
        f"path: {output_dir}",
        f"train: images/{train_split}",
        f"val: images/{val_split}",
    ]

    if test_split is not None:
        lines.append(f"test: images/{test_split}")

    lines.append(f"nc: {len(names)}")

    names_repr = ", ".join(f"'{n}'" for n in names)
    lines.append(f"names: [{names_repr}]")

    data_yaml = output_dir / "data.yaml"
    data_yaml.write_text("\n".join(lines) + "\n")

    return data_yaml
