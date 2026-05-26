"""
COCO export utilities.

Provides:

- COCO annotation export
- bbox conversion helpers
- image registry creation
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
        xmin,
        ymin,
        xmax,
        ymax,
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
        xmin,
        ymin,
        xmax - xmin,
        ymax - ymin,
    ]


def export_coco_annotations(
    target,
    dataset,
    categories,
    indices=None,
):
    """
    Export dataset split as COCO annotations.

    Args:
        target:
            Output annotations.json path.

        dataset:
            PhenoBench dataset configured with:

                target_types=["plant_bboxes"]

        categories:
            COCO category definitions.

        indices:
            Optional subset indices.

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

            annotations.append({

                "id":
                    int(annotation_id),

                "image_id":
                    int(image_id),

                "category_id":
                    int(bbox["label"]),

                "bbox": [
                    float(v)
                    for v in coco_bbox
                ],

                "area":
                    float(area),

                "iscrowd":
                    0,
            })
            
            annotation_id += 1

    coco = {

        "images":
            images,

        "annotations":
            annotations,

        "categories":
            categories,
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

    return coco
