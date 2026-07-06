"""
COCO export utilities.

Provides:

- COCO annotation export
- bbox conversion helpers
- image registry creation

Exports are generated from canonical
DatasetDefinition objects to support:

- multiclass detection
- binary detection
- semantic label remapping
- framework-independent evaluation
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def phenobench_bbox_to_xyxy(
    bbox,
):
    """
    Convert upstream PhenoBench bbox structure
    into canonical xyxy format.

    Args:
        bbox:
            PhenoBench bbox dictionary.

    Returns:
        [xmin, ymin, xmax, ymax]
    """

    xmin, ymin = bbox["corner"]

    width = bbox["width"]
    height = bbox["height"]

    xmax = xmin + width
    ymax = ymin + height

    return [
        float(xmin),
        float(ymin),
        float(xmax),
        float(ymax),
    ]


def xyxy_to_coco(
    box,
):
    """
    Convert xyxy box to COCO xywh.

    Args:
        box:
            [xmin, ymin, xmax, ymax]

    Returns:
        [x, y, width, height]
    """

    xmin, ymin, xmax, ymax = box

    return [
        float(xmin),
        float(ymin),
        float(xmax - xmin),
        float(ymax - ymin),
    ]


def export_coco_annotations(
    target,
    dataset,
    dataset_definition,
    indices=None,
    include_partials=False,
):
    """
    Export dataset split as COCO annotations.

    Args:
        target:
            Output annotations.json path.

        dataset:
            PhenoBench dataset configured with:

                target_types=["plant_bboxes"]

            To carry partials, wrap it in
            ``agri_vision_edge.data.plant_boxes.PartialAwarePhenoBench`` so each
            ``plant_bboxes`` entry gains ``is_partial`` / ``visibility``.

        dataset_definition:
            Canonical dataset definition.

        indices:
            Optional subset indices.

        include_partials:
            When ``True``, partial ("do-not-care") plants are emitted as
            annotations flagged ``ignore=1`` plus a custom ``partial=1`` and, if
            available, ``visibility``. They are kept out of scoring by default
            and consumed by the ``ignore_partials`` evaluation knob (see
            :mod:`agri_vision_edge.evaluation.partials`). When ``False`` (the
            default) partial boxes are skipped and the output is unchanged from
            the pre-partials behaviour.

    Returns:
        COCO dictionary.
    """

    target = Path(target)

    if indices is None:
        indices = range(len(dataset))

    images = []

    annotations = []

    annotation_id = 1

    for image_id, dataset_index in enumerate(
        indices,
        start=1,
    ):

        sample = dataset[dataset_index]

        image = np.array(sample["image"])

        h, w = image.shape[:2]

        image_name = sample["image_name"]

        images.append({

            "id":
                int(image_id),

            "file_name":
                str(image_name),

            "width":
                int(w),

            "height":
                int(h),
        })

        for bbox in sample["plant_bboxes"]:

            source_label = int(
                bbox["label"]
            )

            #
            # Skip labels not exported by this
            # dataset definition
            #

            if (
                source_label
                not in dataset_definition.label_mapping
            ):
                continue

            is_partial = bool(bbox.get("is_partial", False))

            #
            # Partial ("do-not-care") plants: skip entirely unless partials are
            # requested, in which case emit them flagged so the evaluation
            # ignore_partials knob can suppress detections landing on them.
            #

            if is_partial and not include_partials:
                continue

            target_label = (
                dataset_definition.label_mapping[
                    source_label
                ]
            )

            xyxy = phenobench_bbox_to_xyxy(
                bbox
            )

            coco_bbox = xyxy_to_coco(
                xyxy
            )

            area = (
                coco_bbox[2]
                * coco_bbox[3]
            )

            annotation = {

                "id":
                    int(annotation_id),

                "image_id":
                    int(image_id),

                "category_id":
                    int(target_label),

                "bbox": [
                    float(v)
                    for v in coco_bbox
                ],

                "area":
                    float(area),

                "iscrowd":
                    0,
            }

            if is_partial:
                # COCO-standard ignore + explicit partial marker; keep visibility
                # when the source carried it (upstream do-not-care criterion).
                annotation["ignore"] = 1
                annotation["partial"] = 1

                visibility = bbox.get("visibility")
                if visibility is not None:
                    annotation["visibility"] = float(visibility)

            annotations.append(annotation)

            annotation_id += 1

    coco = {

        "images":
            images,

        "annotations":
            annotations,

        "categories":
            dataset_definition.categories,
    }

    with open(target, "w") as f:

        json.dump(
            coco,
            f,
            indent=2,
        )

    print(
        f"Wrote COCO annotations: {target}"
    )

    return {

        "images":
            len(images),

        "annotations":
            len(annotations),

        "categories":
            len(dataset_definition.categories),
    }
