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

This benchmark uses image IDs from a COCO
annotations JSON rather than inferring IDs
from filenames.

Only images referenced in the annotations
file are benchmarked.

Missing images are skipped gracefully.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time

from pathlib import Path

import cv2

from agri_vision_edge.runtime.inference.tflite import (
    TFLiteRuntime,
)


IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
}


def collect_models(path):
    """
    Collect TensorFlow Lite models.

    Args:
        path:
            Single .tflite file or
            directory containing models.

    Returns:
        List of model paths.
    """

    path = Path(path)

    #
    # Single model
    #

    if path.is_file():

        return [path]

    #
    # Directory sweep
    #

    return sorted(
        path.glob("*.tflite")
    )


def collect_images_from_annotations(
    images_dir,
    annotations_path,
):
    """
    Collect benchmark images from a COCO
    annotations JSON.

    Only images referenced in the
    annotations file are returned.

    Missing images are skipped.

    Args:
        images_dir:
            Directory containing images.

        annotations_path:
            COCO annotations JSON.

    Returns:
        List of image records:

            {
                "id": int,
                "file_name": str,
                "path": Path,
            }
    """

    images_dir = Path(images_dir)

    with open(annotations_path) as f:

        coco = json.load(f)

    image_entries = coco["images"]

    image_records = []

    missing = []

    for entry in image_entries:

        file_name = entry["file_name"]

        image_path = (
            images_dir / file_name
        )

        #
        # Skip missing files
        #

        if not image_path.exists():

            missing.append(file_name)

            continue

        #
        # Validate extension
        #

        if (
            image_path.suffix.lower()
            not in IMAGE_EXTENSIONS
        ):

            continue

        image_records.append({

            "id":
                entry["id"],

            "file_name":
                file_name,

            "path":
                image_path,
        })

    if missing:

        print(
            f"[warning] missing "
            f"{len(missing)} image(s)"
        )

        preview = missing[:10]

        for name in preview:

            print(
                f"  - {name}"
            )

        if len(missing) > 10:

            print(
                f"  ... and "
                f"{len(missing) - 10} more"
            )

    if not image_records:

        raise RuntimeError(
            "No annotation images found "
            "in image directory"
        )

    return image_records


def benchmark_model(
    model_path,
    image_records,
    output_root,
    delegate,
):
    """
    Benchmark a TensorFlow Lite model.

    Args:
        model_path:
            Path to .tflite model.

        image_records:
            Image records from COCO
            annotations.

        output_root:
            Root output directory.

        delegate:
            Optional delegate path.
    """

    print(
        f"\n=== Benchmarking: "
        f"{model_path.name} ==="
    )

    runtime = TFLiteRuntime(
        model_path=model_path,

        delegate_path=delegate,
    )

    output_dir = (
        output_root /
        model_path.stem
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    #
    # Warmup
    #

    warmup = cv2.imread(
        str(image_records[0]["path"])
    )

    warmup = cv2.cvtColor(
        warmup,
        cv2.COLOR_BGR2RGB,
    )

    for _ in range(20):

        runtime.predict(warmup)

    #
    # Benchmark
    #

    latencies_ms = []

    predictions = []

    for record in image_records:

        image_id = record["id"]

        image_path = record["path"]

        image = cv2.imread(
            str(image_path)
        )

        #
        # Skip unreadable images
        #

        if image is None:

            print(
                f"[warning] failed to read:"
                f" {image_path}"
            )

            continue

        image = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB,
        )

        h, w = image.shape[:2]

        start = time.perf_counter()

        detections = runtime.predict(
            image
        )

        end = time.perf_counter()

        latency_ms = (
            end - start
        ) * 1000

        latencies_ms.append(
            latency_ms
        )

        #
        # COCO predictions
        #

        for det in detections:

            ymin, xmin, ymax, xmax = (
                det.bbox
            )

            bbox = [

                xmin * w,
                ymin * h,

                (xmax - xmin) * w,
                (ymax - ymin) * h,
            ]

            predictions.append({

                "image_id":
                    image_id,

                "category_id":
                    det.category_id,

                "bbox":
                    bbox,

                "score":
                    det.score,
            })

    #
    # Latency metrics
    #

    latency_results = {

        "mean_latency_ms":
            statistics.mean(
                latencies_ms
            ),

        "median_latency_ms":
            statistics.median(
                latencies_ms
            ),

        "min_latency_ms":
            min(latencies_ms),

        "max_latency_ms":
            max(latencies_ms),

        "latencies_ms":
            latencies_ms,
    }

    #
    # Runtime metadata
    #

    runtime_results = {

        "model":
            model_path.name,

        "delegate":
            delegate,

        "input_details":
            runtime.input_details,

        "output_details":
            runtime.output_details,
    }

    #
    # Save artifacts
    #

    with open(
        output_dir / "latency.json",
        "w",
    ) as f:

        json.dump(
            latency_results,
            f,
            indent=2,
            default=str,
        )

    with open(
        output_dir / "predictions.json",
        "w",
    ) as f:

        json.dump(
            predictions,
            f,
            indent=2,
        )

    with open(
        output_dir / "runtime.json",
        "w",
    ) as f:

        json.dump(
            runtime_results,
            f,
            indent=2,
            default=str,
        )

    print(
        f"mean latency: "
        f"{latency_results['mean_latency_ms']:.2f} ms"
    )

    print(
        f"exported "
        f"{len(predictions)} "
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

    image_records = (
        collect_images_from_annotations(
            args.images,
            args.annotations,
        )
    )

    output_root = Path(
        args.output_dir
    )

    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        f"Found {len(model_paths)} "
        f"model(s)"
    )

    print(
        f"Found "
        f"{len(image_records)} "
        f"annotated image(s)"
    )

    for model_path in model_paths:

        benchmark_model(
            model_path=model_path,

            image_records=image_records,

            output_root=output_root,

            delegate=args.delegate,
        )


if __name__ == "__main__":

    main()
