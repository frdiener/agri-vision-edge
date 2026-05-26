"""
TensorFlow Object Detection label map utilities.
"""

from __future__ import annotations

from pathlib import Path

from .categories import (
    build_category_map,
)


def write_label_map(
    target,
    categories,
):
    """
    Write TFOD label map.

    Args:
        target:
            Output path.

        categories:
            Category definitions.
    """

    category_map = build_category_map(
        categories
    )

    target = Path(target)

    lines = []

    for class_id in sorted(category_map):

        class_name = (
            category_map[class_id]
        )

        lines.extend([
            "item {",
            f"  id: {class_id}",
            f'  name: "{class_name}"',
            "}",
            "",
        ])

    target.write_text(
        "\n".join(lines)
    )

    print(
        f"Wrote label map: {target}"
    )
