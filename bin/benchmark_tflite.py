#!/usr/bin/env python3

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

from agri_vision_edge.runtime.inference.tflite import (
    TFLiteRuntime,
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

    return sorted(
        path.glob("*.tflite")
    )


def benchmark_model(
    *,
    model_path: Path,
    image_records,
    output_root: Path,
    delegate: str | None,
):
    """
    Benchmark a single model.
    """

    print(
        f"\n=== Benchmarking: "
        f"{model_path.name} ==="
    )

    runtime = TFLiteRuntime(
        model_path=model_path,
        delegate_path=delegate,
    )

    result = benchmark_runtime(
        runtime,
        image_records,
    )

    output_dir = (
        output_root /
        model_path.stem
    )

    save_benchmark_artifacts(
        output_dir=output_dir,
        benchmark_result=result,
        runtime=runtime,
        model_name=model_path.name,
        delegate=delegate,
    )

    mean_latency = (
        sum(result.latencies_ms)
        / len(result.latencies_ms)
    )

    print(
        f"mean latency: "
        f"{mean_latency:.2f} ms"
    )

    print(
        f"exported "
        f"{len(result.predictions)} "
        f"prediction(s)"
    )


def main():
    """
    CLI entrypoint.
    """

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "models",

        help=(
            "Single .tflite model "
            "or directory of models"
        ),
    )

    parser.add_argument(
        "images",

        help=(
            "Directory containing images"
        ),
    )

    parser.add_argument(
        "--annotations",

        required=True,

        help=(
            "COCO annotations JSON"
        ),
    )

    parser.add_argument(
        "--output-dir",

        default="benchmark_results",
    )

    parser.add_argument(
        "--delegate",

        default="/usr/lib/libteflon.so",
    )

    args = parser.parse_args()

    model_paths = collect_models(
        args.models
    )

    image_records = load_coco_images(
        args.images,
        args.annotations,
    )

    output_root = Path(
        args.output_dir
    )

    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        f"Found "
        f"{len(model_paths)} "
        f"model(s)"
    )

    print(
        f"Found "
        f"{len(image_records)} "
        f"annotated image(s)"
    )

    success = 0
    failed = 0

    for model_path in model_paths:

        try:

            benchmark_model(
                model_path=model_path,
                image_records=image_records,
                output_root=output_root,
                delegate=args.delegate,
            )

            success += 1

        except Exception as e:

            failed += 1

            print(
                f"\n[error] "
                f"{model_path.name}"
            )

            print(
                f"        "
                f"{type(e).__name__}: {e}"
            )

            save_failure_artifact(
                output_dir=
                    output_root /
                    model_path.stem,

                exception=e,
            )

    print()

    print(
        f"completed: "
        f"{success} succeeded, "
        f"{failed} failed"
    )


if __name__ == "__main__":

    main()
