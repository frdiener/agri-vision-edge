#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import time

from pathlib import Path

import cv2

from agri_vision_edge.runtime.inference.tflite import (
    TFLiteRuntime,
)


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "model",
    )

    parser.add_argument(
        "images",
    )

    parser.add_argument(
        "--delegate",
    )

    args = parser.parse_args()

    runtime = TFLiteRuntime(
        model_path=args.model,

        delegate_path=args.delegate,
    )

    image_paths = sorted(
        Path(args.images).glob("*.jpg")
    )

    #
    # Warmup
    #

    warmup_image = cv2.imread(
        str(image_paths[0])
    )

    warmup_image = cv2.cvtColor(
        warmup_image,
        cv2.COLOR_BGR2RGB,
    )

    for _ in range(20):

        runtime.predict(
            warmup_image
        )

    #
    # Benchmark
    #

    latencies_ms = []

    predictions = []

    for image_path in image_paths:

        image = cv2.imread(
            str(image_path)
        )

        image = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB,
        )

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
        # COCO prediction export
        #

        image_id = int(
            image_path.stem
        )

        for det in detections:

            ymin, xmin, ymax, xmax = (
                det.bbox
            )

            h, w = image.shape[:2]

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
                    det.class_id,

                "bbox":
                    bbox,

                "score":
                    det.score,
            })

    results = {

        "mean_latency_ms":
            sum(latencies_ms)
            / len(latencies_ms),

        "latencies_ms":
            latencies_ms,
    }

    with open(
        "latency.json",
        "w",
    ) as f:

        json.dump(
            results,
            f,
            indent=2,
        )

    with open(
        "predictions.json",
        "w",
    ) as f:

        json.dump(
            predictions,
            f,
            indent=2,
        )


if __name__ == "__main__":

    main()
