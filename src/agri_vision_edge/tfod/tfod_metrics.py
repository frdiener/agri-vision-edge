from __future__ import annotations

import json
from pathlib import Path

from ..evaluation.metrics import CocoMetrics

PathLike = str | Path


TFOD_COCO_MAPPING = {

    #
    # AP
    #

    "DetectionBoxes_Precision/mAP":
        "map",

    "DetectionBoxes_Precision/mAP@.50IOU":
        "map50",

    "DetectionBoxes_Precision/mAP@.75IOU":
        "map75",

    "DetectionBoxes_Precision/mAP (small)":
        "aps",

    "DetectionBoxes_Precision/mAP (medium)":
        "apm",

    "DetectionBoxes_Precision/mAP (large)":
        "apl",

    #
    # AR
    #

    "DetectionBoxes_Recall/AR@1":
        "ar1",

    "DetectionBoxes_Recall/AR@10":
        "ar10",

    "DetectionBoxes_Recall/AR@100":
        "ar100",

    "DetectionBoxes_Recall/AR@100 (small)":
        "ars",

    "DetectionBoxes_Recall/AR@100 (medium)":
        "arm",

    "DetectionBoxes_Recall/AR@100 (large)":
        "arl",
}


def load_tfod_best_metrics(
    path: PathLike,
) -> CocoMetrics:

    path = Path(path)

    with open(path) as f:
        data = json.load(f)

    raw_metrics = data["all_metrics"]

    coco_kwargs = {}

    for tfod_name, canonical_name in (
        TFOD_COCO_MAPPING.items()
    ):

        coco_kwargs[canonical_name] = (
            raw_metrics[tfod_name]
        )

    return CocoMetrics(

        **coco_kwargs,

        step=data["step"],

        evaluator="tfod",

        selection_metric=data[
            "metric_name"
        ],

        selection_value=data[
            "metric_value"
        ],
    )
