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
    labels=None,
):
    """
    Write a TensorFlow Object Detection API label map.

    Args:
        target:
            Output path for label_map.pbtxt.
        labels:
            Mapping from class ID to class name.
    """
    if labels is None:
        labels = DEFAULT_LABELS

    target = Path(target)

    lines = []

    for class_id in sorted(labels):
        class_name = labels[class_id]

        lines.extend([
            "item {",
            f"  id: {class_id}",
            f'  name: "{class_name}"',
            "}",
            "",
        ])

    target.write_text("\n".join(lines))

    print(f"Wrote label map to: {target}")
