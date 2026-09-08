"""Export PhenoBench samples to the YOLOv7 image/label layout.

Labels use ``<class> <cx> <cy> <w> <h>`` with contiguous zero-based classes and
coordinates normalized to ``[0, 1]``. Full and tiled datasets share the same
``plant_bboxes`` interface and :class:`DatasetDefinition` remapping.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
from PIL import Image

from .coco import phenobench_bbox_to_xyxy
from .datasets import DatasetDefinition


def yolo_class_names(
    dataset_definition: DatasetDefinition,
) -> list[str]:
    """Return class names in zero-based YOLO order.

    Name ``i`` corresponds to the definition's one-based category ``i + 1``.
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

    # Clip rounding-induced one-pixel overflow at tile borders.
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
    source_images_dir: str | Path | None = None,
    include_partials: bool = False,
) -> dict:
    """Export one split under ``images/<split>`` and ``labels/<split>``.

    Labels absent from ``dataset_definition.label_mapping`` are skipped.
    ``min_box_size`` is a normalized width/height floor. Partials are excluded
    by default because YOLO has no do-not-care flag. When ``source_images_dir``
    contains a sample, its image is copied byte-for-byte instead of re-encoded.
    Returns image and box counts.
    """

    output_dir = Path(output_dir)
    source_images_dir = (
        Path(source_images_dir) if source_images_dir is not None else None
    )

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
        dest_image = images_dir / f"{stem}.png"

        # Byte-exact copy for full images; re-encode only when there is no
        # matching source file (tiles synthesised in-memory).
        source_image = (
            source_images_dir / sample["image_name"]
            if source_images_dir is not None
            else None
        )
        if source_image is not None and source_image.exists():
            shutil.copyfile(source_image, dest_image)
        else:
            image.save(dest_image)

        lines = []

        for bbox in sample["plant_bboxes"]:
            source_label = int(bbox["label"])

            if source_label not in label_mapping:
                continue

            # Drop partial ("do-not-care") plants unless explicitly kept.
            if bbox.get("is_partial", False) and not include_partials:
                num_skipped_boxes += 1
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

        # Empty label files mark valid background images for YOLOv7.
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
    dest: str | Path | None = None,
) -> Path:
    """Write YOLOv7 split and class metadata.

    Split paths are absolute because WongKinYiu/yolov7 ignores ``path:`` and
    resolves relative paths from its own checkout. ``dest`` may be outside a
    read-only ``output_dir``.
    """

    output_dir = Path(output_dir)

    dest = Path(dest) if dest is not None else output_dir / "data.yaml"

    names = yolo_class_names(dataset_definition)

    images_root = output_dir / "images"

    lines = [
        f"train: {images_root / train_split}",
        f"val: {images_root / val_split}",
    ]

    if test_split is not None:
        lines.append(f"test: {images_root / test_split}")

    lines.append(f"nc: {len(names)}")

    names_repr = ", ".join(f"'{n}'" for n in names)
    lines.append(f"names: [{names_repr}]")

    dest.write_text("\n".join(lines) + "\n")

    return dest
