from __future__ import annotations

import json
from pathlib import Path

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
    annotations_path: str | Path,
    predictions_path: str | Path,
) -> dict:
    """
    Evaluate a COCO predictions file.
    """

    with open(predictions_path) as f:

        predictions = json.load(f)

    #
    # No detections
    #

    if not predictions:

        print(
            "[warning] no predictions:"
            f" {predictions_path}"
        )

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


def save_metrics(
    metrics: dict,
    output_path: str | Path,
):

    with open(output_path, "w") as f:

        json.dump(
            metrics,
            f,
            indent=2,
        )


def evaluate_model_dir(
    model_dir: Path,
    annotations_path: Path,
):
    """
    Evaluate one benchmark directory.
    """

    predictions_path = (
        model_dir /
        "predictions.json"
    )

    error_path = (
        model_dir /
        "error.json"
    )

    metrics_path = (
        model_dir /
        "metrics.json"
    )

    #
    # Failed benchmark
    #

    if error_path.exists():

        print(
            f"[skip] "
            f"{model_dir.name}"
            " (failed benchmark)"
        )

        return False

    #
    # No predictions
    #

    if not predictions_path.exists():

        print(
            f"[skip] "
            f"{model_dir.name}"
            " (missing predictions)"
        )

        return False

    print(
        f"\n=== Evaluating: "
        f"{model_dir.name} ==="
    )

    metrics = (
        evaluate_predictions(
            annotations_path,
            predictions_path,
        )
    )

    save_metrics(
        metrics,
        metrics_path,
    )

    print()

    print(
        f"AP:   "
        f"{metrics['AP']:.4f}"
    )

    print(
        f"AP50: "
        f"{metrics['AP50']:.4f}"
    )

    print(
        f"AP75: "
        f"{metrics['AP75']:.4f}"
    )

    return True
