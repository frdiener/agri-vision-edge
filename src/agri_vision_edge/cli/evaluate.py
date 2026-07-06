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
    evaluate_model_dir,
    evaluate_predictions,
    print_per_class,
    save_metrics,
)
from agri_vision_edge.evaluation.partials import DEFAULT_PARTIAL_THRESHOLD


def _run_faithful(args):
    """
    Run the official PhenoBench evaluator on a single predictions file.
    """

    if not args.phenobench_dir:
        raise SystemExit("--faithful requires --phenobench-dir")

    input_path = Path(args.input)

    if not input_path.is_file():
        raise SystemExit(
            "--faithful expects a predictions.json file, not a directory"
        )

    # Imported lazily so the lightweight path never pulls the torch stack.
    from agri_vision_edge.evaluation.faithful import evaluate_faithful

    metrics = evaluate_faithful(
        annotations_path=args.annotations,
        predictions_path=input_path,
        phenobench_dir=args.phenobench_dir,
        split=args.split,
    )

    metrics_path = input_path.with_name("metrics_faithful.json")
    save_metrics(metrics, metrics_path)

    print()
    print(json.dumps(metrics, indent=2))
    print(f"\nwrote {metrics_path}")

    return


def main(argv=None):

    parser = argparse.ArgumentParser(prog="ave evaluate")

    parser.add_argument(
        "annotations",
        help=("COCO annotations JSON"),
    )

    parser.add_argument(
        "input",
        help=("predictions.json or benchmark_results directory"),
    )

    parser.add_argument(
        "--ignore-partials",
        action="store_true",
        help=(
            "Treat PhenoBench partial (border / low-visibility) plants as "
            "do-not-care: drop detections that land on them instead of counting "
            "them as false positives (upstream containment rule). Requires the "
            "annotations to carry partial/ignore/visibility flags."
        ),
    )

    parser.add_argument(
        "--partial-threshold",
        type=float,
        default=DEFAULT_PARTIAL_THRESHOLD,
        help=(
            "Visibility/containment threshold for the partial rule "
            f"(default: {DEFAULT_PARTIAL_THRESHOLD})."
        ),
    )

    parser.add_argument(
        "--faithful",
        action="store_true",
        help=(
            "Use the official PhenoBench evaluator (torchmetrics mAP + upstream "
            "partial filtering) for leaderboard-comparable numbers instead of "
            "the lightweight pycocotools path. Requires full-image predictions "
            "and the 'faithful-eval' extra; --phenobench-dir is required."
        ),
    )

    parser.add_argument(
        "--phenobench-dir",
        help=(
            "Root of the raw PhenoBench dataset (with train/val/test splits). "
            "Required with --faithful."
        ),
    )

    parser.add_argument(
        "--split",
        default="val",
        help="Dataset split the predictions correspond to (default: val).",
    )

    args = parser.parse_args(argv)

    #
    # Faithful upstream evaluation (single predictions file only)
    #

    if args.faithful:
        return _run_faithful(args)

    annotations_path = Path(args.annotations)

    input_path = Path(args.input)

    #
    # Single predictions file
    #

    if input_path.is_file():
        metrics = evaluate_predictions(
            annotations_path,
            input_path,
            ignore_partials=args.ignore_partials,
            partial_threshold=args.partial_threshold,
        )

        # Persist beside the predictions file (mirrors directory mode), so the
        # aggregate + per-class metrics land in written data, not just stdout.
        metrics_path = input_path.with_name("metrics.json")

        save_metrics(
            metrics,
            metrics_path,
        )

        print()

        print(
            json.dumps(
                metrics,
                indent=2,
            )
        )

        print_per_class(metrics)

        print(f"\nwrote {metrics_path}")

        return

    #
    # Benchmark directory
    #

    model_dirs = sorted(p for p in input_path.iterdir() if p.is_dir())

    success = 0
    skipped = 0

    for model_dir in model_dirs:
        ok = evaluate_model_dir(
            model_dir,
            annotations_path,
            ignore_partials=args.ignore_partials,
            partial_threshold=args.partial_threshold,
        )

        if ok:
            success += 1
        else:
            skipped += 1

    print()

    print(f"completed: {success} evaluated, {skipped} skipped")


if __name__ == "__main__":
    raise SystemExit(main())
