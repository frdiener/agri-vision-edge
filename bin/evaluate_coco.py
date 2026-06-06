#!/usr/bin/env python3

"""
Evaluate COCO predictions.

Supports:

- single predictions.json evaluation
- benchmark directory evaluation
- metrics.json generation
- failed benchmark skipping
"""

from __future__ import annotations

import argparse
import json

from pathlib import Path

from agri_vision_edge.evaluation.coco import (
    evaluate_predictions,
    evaluate_model_dir,
)


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "annotations",

        help=(
            "COCO annotations JSON"
        ),
    )

    parser.add_argument(
        "input",

        help=(
            "predictions.json or "
            "benchmark_results directory"
        ),
    )

    args = parser.parse_args()

    annotations_path = Path(
        args.annotations
    )

    input_path = Path(
        args.input
    )

    #
    # Single predictions file
    #

    if input_path.is_file():

        metrics = (
            evaluate_predictions(
                annotations_path,
                input_path,
            )
        )

        print()

        print(
            json.dumps(
                metrics,
                indent=2,
            )
        )

        return

    #
    # Benchmark directory
    #

    model_dirs = sorted(

        p
        for p in input_path.iterdir()

        if p.is_dir()
    )

    success = 0
    skipped = 0

    for model_dir in model_dirs:

        ok = evaluate_model_dir(
            model_dir,
            annotations_path,
        )

        if ok:
            success += 1
        else:
            skipped += 1

    print()

    print(
        f"completed: "
        f"{success} evaluated, "
        f"{skipped} skipped"
    )


if __name__ == "__main__":

    main()
