
from __future__ import annotations

import json

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


METRIC_NAMES = [
    "AP",
    "AP50",
    "AP75",
    "APS",
    "APM",
    "APL",
    "AR1",
    "AR10",
    "AR100",
    "ARS",
    "ARM",
    "ARL",
]


def evaluate_predictions(
    annotations_path,
    predictions_path,
):

    with open(predictions_path) as f:

        predictions = json.load(f)

    if not predictions:

        return {
            name: 0.0
            for name in METRIC_NAMES
        }

    coco_gt = COCO(
        str(annotations_path)
    )

    coco_dt = coco_gt.loadRes(
        str(predictions_path)
    )

    evaluator = COCOeval(
        coco_gt,
        coco_dt,
        "bbox",
    )

    evaluator.evaluate()
    evaluator.accumulate()
    evaluator.summarize()

    return {
        name: float(value)
        for name, value in zip(
            METRIC_NAMES,
            evaluator.stats,
        )
    }
