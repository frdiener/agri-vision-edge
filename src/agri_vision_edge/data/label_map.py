"""
Utilities for TensorFlow Object Detection label maps.
"""

from pathlib import Path


DEFAULT_LABELS = {
    1: "crop",
    2: "weed",
}


def write_label_map(
    target,
    labels=DEFAULT_LABELS,
):
    """
    Write a TensorFlow Object Detection API label map.

    Args:
        target:
            Output path for label_map.pbtxt.
        labels:
            Mapping from class ID to class name.
    """
    target = Path(target)

    lines = []

    for class_id, class_name in labels.items():
        lines.extend([
            "item {",
            f"  id: {class_id}",
            f'  name: "{class_name}"',
            "}",
            "",
        ])

    target.write_text("\n".join(lines))

    print(f"Wrote label map to: {target}")
