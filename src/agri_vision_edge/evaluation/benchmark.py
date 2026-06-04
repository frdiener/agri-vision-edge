"""
Runtime benchmarking.
"""

from __future__ import annotations

import statistics
import time

from dataclasses import dataclass

import cv2

from .export import (
    detections_to_coco,
)


@dataclass
class BenchmarkResult:

    predictions: list[dict]

    latencies_ms: list[float]


def benchmark_runtime(
    runtime,
    image_records,
    *,
    warmup_iterations: int = 20,
):

    warmup_image = cv2.imread(
        str(image_records[0].path)
    )

    warmup_image = cv2.cvtColor(
        warmup_image,
        cv2.COLOR_BGR2RGB,
    )

    for _ in range(
        warmup_iterations
    ):
        runtime.predict(
            warmup_image
        )

    predictions = []

    latencies_ms = []

    for record in image_records:

        image = cv2.imread(
            str(record.path)
        )

        if image is None:

            print(
                "[warning] failed "
                f"to read {record.path}"
            )

            continue

        image = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB,
        )

        height, width = (
            image.shape[:2]
        )

        start = (
            time.perf_counter()
        )

        detections = (
            runtime.predict(
                image
            )
        )

        end = (
            time.perf_counter()
        )

        latencies_ms.append(
            (end - start)
            * 1000
        )

        predictions.extend(
            detections_to_coco(
                image_id=
                    record.image_id,

                image_width=
                    width,

                image_height=
                    height,

                detections=
                    detections,
            )
        )

    return BenchmarkResult(
        predictions=predictions,
        latencies_ms=latencies_ms,
    )


def latency_summary(
    latencies_ms,
):

    return {

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
