"""
Dataset loading helpers.
"""

from __future__ import annotations

import json

from dataclasses import dataclass
from pathlib import Path


IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
}


@dataclass(slots=True)
class ImageRecord:

    image_id: int

    file_name: str

    path: Path


def load_coco_images(
    images_dir: str | Path,
    annotations_path: str | Path,
) -> list[ImageRecord]:

    images_dir = Path(images_dir)

    with open(annotations_path) as f:
        coco = json.load(f)

    records = []
    missing = []

    for image in coco["images"]:

        path = (
            images_dir /
            image["file_name"]
        )

        if not path.exists():

            missing.append(
                image["file_name"]
            )

            continue

        if (
            path.suffix.lower()
            not in IMAGE_EXTENSIONS
        ):
            continue

        records.append(
            ImageRecord(
                image_id=image["id"],
                file_name=image["file_name"],
                path=path,
            )
        )

    if not records:

        raise RuntimeError(
            "No benchmark images found."
        )

    if missing:

        print(
            f"[warning] missing "
            f"{len(missing)} image(s)"
        )

    return records
