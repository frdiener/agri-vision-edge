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
from agri_vision_edge.evaluation.integrity import CorruptPredictionsError
from agri_vision_edge.evaluation.partials import DEFAULT_PARTIAL_THRESHOLD


def _infer_annotations_path(
    annotations_root: Path,
    run_name: str,
) -> Path | None:
    """
    Infer the matching test-bundle annotations for a benchmark run directory.

    Directory-mode benchmark sweeps may contain both ``tiled_`` and
    ``untiled_`` runs. In that case, pass the test-bundle directory as the
    annotations argument and infer the concrete annotations JSON from each run
    directory's prefix plus its ``sc`` / ``mc`` class token.
    """

    if "_mc_" in run_name:
        cls = "mc"
    elif "_sc_" in run_name:
        cls = "sc"
    else:
        return None

    if run_name.startswith("tiled_"):
        return annotations_root / f"annotations_{cls}_tiled.json"

    if run_name.startswith("untiled_"):
        return annotations_root / f"annotations_{cls}.json"

    return None


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

    try:
        metrics = evaluate_faithful(
            annotations_path=args.annotations,
            predictions_path=input_path,
            phenobench_dir=args.phenobench_dir,
            split=args.split,
            allow_corrupt=args.allow_corrupt_predictions,
        )
    except CorruptPredictionsError as exc:
        # Exit with the actual reason instead of a traceback: sweep drivers
        # report a generic "faithful eval failed" on non-zero exit, so the
        # cause has to be legible on the last line.
        raise SystemExit(f"[error] {exc}") from exc

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
        help=(
            "COCO annotations JSON, or a test-bundle directory when evaluating "
            "a benchmark_results directory containing tiled_/untiled_ runs"
        ),
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
        "--allow-corrupt-predictions",
        action="store_true",
        help=(
            "Score predictions that contain non-finite boxes or out-of-range "
            "scores instead of refusing them. The resulting numbers are "
            "meaningless -- pycocotools matches a NaN box at every IoU "
            "threshold, which inflates AP and makes AP == AP50. Only useful to "
            "inspect a known-broken run."
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
        try:
            metrics = evaluate_predictions(
                annotations_path,
                input_path,
                ignore_partials=args.ignore_partials,
                partial_threshold=args.partial_threshold,
                allow_corrupt=args.allow_corrupt_predictions,
            )
        except CorruptPredictionsError as exc:
            raise SystemExit(f"[error] {exc}") from exc

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
        run_annotations_path = annotations_path

        if annotations_path.is_dir():
            inferred_annotations_path = _infer_annotations_path(
                annotations_path,
                model_dir.name,
            )

            if inferred_annotations_path is None:
                print(
                    f"[skip] {model_dir.name} "
                    "(cannot infer tiled_/untiled_ annotations)"
                )
                skipped += 1
                continue

            if not inferred_annotations_path.exists():
                print(
                    f"[skip] {model_dir.name} "
                    f"(missing annotations: {inferred_annotations_path})"
                )
                skipped += 1
                continue

            run_annotations_path = inferred_annotations_path

        ok = evaluate_model_dir(
            model_dir,
            run_annotations_path,
            ignore_partials=args.ignore_partials,
            partial_threshold=args.partial_threshold,
            allow_corrupt=args.allow_corrupt_predictions,
        )

        if ok:
            success += 1
        else:
            skipped += 1

    print()

    print(f"completed: {success} evaluated, {skipped} skipped")


if __name__ == "__main__":
    raise SystemExit(main())
