"""
Benchmark TensorFlow Lite models.

Supports:

- single model benchmarking
- directory model sweeps
- delegate acceleration
- latency collection
- COCO prediction export
- COCO annotation-driven image selection
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from agri_vision_edge.evaluation.artifacts import (
    save_benchmark_artifacts,
    save_failure_artifact,
)
from agri_vision_edge.evaluation.benchmark import (
    benchmark_runtime,
)
from agri_vision_edge.evaluation.dataset import (
    load_coco_images,
)
from agri_vision_edge.runtime.inference.factory import (
    build_runtime,
)


def collect_models(
    path: str | Path,
) -> list[Path]:
    """
    Collect TensorFlow Lite models.

    Args:
        path:
            Single .tflite model or
            directory containing models.

    Returns:
        List of model paths.
    """

    path = Path(path)

    if path.is_file():
        return [path]

    return sorted(path.glob("*.tflite"))


def benchmark_model(
    *,
    model_path: Path,
    image_records,
    output_root: Path,
    delegate: str | None,
    iou: float,
):
    """
    Benchmark a single model.
    """

    print(f"\n=== Benchmarking: {model_path.name} ===")

    # The factory selects SSD (post-NMS) vs YOLOv7-tiny (raw grids) by output
    # shape; score_threshold stays 0.0 so COCO eval sees every detection (the
    # YOLO runtime floors candidates internally to keep NMS tractable).
    runtime = build_runtime(
        model_path=model_path,
        delegate_path=delegate,
        iou_threshold=iou,
    )

    result = benchmark_runtime(
        runtime,
        image_records,
    )

    output_dir = output_root / model_path.stem

    save_benchmark_artifacts(
        output_dir=output_dir,
        benchmark_result=result,
        runtime=runtime,
        model_name=model_path.name,
        delegate=delegate,
    )

    mean_latency = sum(result.latencies_ms) / len(result.latencies_ms)

    print(f"mean latency: {mean_latency:.2f} ms")

    print(f"exported {len(result.predictions)} prediction(s)")


def env_var(value: str) -> tuple[str, str]:
    try:
        key, val = value.split("=", 1)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"expected KEY=VALUE, got {value!r}"
        ) from exc

    if not key:
        raise argparse.ArgumentTypeError("environment-variable name cannot be empty")

    return key, val


def main(argv=None):
    """
    CLI entrypoint.
    """

    parser = argparse.ArgumentParser(prog="ave benchmark")

    parser.add_argument(
        "models",
        help=("Single .tflite model or directory of models"),
    )

    parser.add_argument(
        "images",
        help=("Directory containing images"),
    )

    parser.add_argument(
        "--annotations",
        required=True,
        help=("COCO annotations JSON"),
    )

    parser.add_argument(
        "--output-dir",
        default="benchmark_results",
    )

    parser.add_argument(
        "--iou",
        type=float,
        default=0.65,
        help=(
            "IoU threshold for YOLO NMS (ignored for SSD, whose graph already "
            "runs NMS) (default: %(default)s)"
        ),
    )

    parser.add_argument(
        "--delegate",
        default="/usr/lib/libteflon.so",
        help=(
            "Path to the TFLite delegate, or 'none' to run on CPU "
            "(use for fp32 models — the NPU delegate is for INT8)"
        ),
    )

    parser.add_argument(
        "-e",
        "--env",
        metavar="KEY=VALUE",
        action="append",
        type=env_var,
        default=[],
        help="Set an environment variable; may be passed multiple times.",
    )

    args = parser.parse_args(argv)

    env = dict(args.env)
    os.environ.update(env)

    # The Teflon/NPU delegate targets INT8; routing an fp32 graph through it
    # silently degrades results. 'none' (or empty) keeps the model on CPU.
    delegate = args.delegate
    if delegate is not None and delegate.strip().lower() in ("", "none"):
        delegate = None

    model_paths = collect_models(args.models)

    image_records = load_coco_images(
        args.images,
        args.annotations,
    )

    output_root = Path(args.output_dir)

    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(f"Found {len(model_paths)} model(s)")

    print(f"Found {len(image_records)} annotated image(s)")

    print(f"Environment: {env}")

    success = 0
    failed = 0

    for model_path in model_paths:
        try:
            benchmark_model(
                model_path=model_path,
                image_records=image_records,
                output_root=output_root,
                delegate=delegate,
                iou=args.iou,
            )

            success += 1

        except Exception as e:
            failed += 1

            print(f"\n[error] {model_path.name}")

            print(f"        {type(e).__name__}: {e}")

            save_failure_artifact(
                output_dir=output_root / model_path.stem,
                exception=e,
            )

    print()

    print(f"completed: {success} succeeded, {failed} failed")

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
