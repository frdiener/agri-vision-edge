"""
Export helpers.
"""

from __future__ import annotations

import json

from pathlib import Path

from ..runtime.inference.base import (
    Detection,
)


def detections_to_coco(
    *,
    image_id: int,
    image_width: int,
    image_height: int,
    detections: list[Detection],
) -> list[dict]:

    results = []

    for det in detections:

        ymin, xmin, ymax, xmax = (
            det.bbox
        )

        results.append({

            "image_id":
                image_id,

            "category_id":
                det.category_id,

            "bbox": [

                xmin * image_width,
                ymin * image_height,

                (xmax - xmin)
                * image_width,

                (ymax - ymin)
                * image_height,
            ],

            "score":
                det.score,
        })

    return results


def save_json(
    obj,
    path: str | Path,
):

    with open(path, "w") as f:

        json.dump(
            obj,
            f,
            indent=2,
            default=str,
        )
