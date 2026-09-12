"""Generate partial-aware plant boxes from PhenoBench masks.

Unlike the upstream loader, this module retains border classes 3 and 4 as
``is_partial`` boxes. Visibility is normalized to ``[0, 1]`` and instance IDs
remain scoped by semantic label because IDs may be reused across classes.
"""

from __future__ import annotations

import numpy as np

from ..evaluation.partials import DEFAULT_PARTIAL_THRESHOLD

# Partial (border) semantic classes -> their fully-visible counterparts.
PARTIAL_LABEL_REMAP = {3: 1, 4: 2}

# Visibility masks store a 0..255 visible-fraction; normalize to [0, 1].
_VISIBILITY_SCALE = 255.0


def plant_boxes_from_masks(
    semantics: np.ndarray,
    plant_instances: np.ndarray,
    plant_visibility: np.ndarray | None = None,
    partial_threshold: float = DEFAULT_PARTIAL_THRESHOLD,
) -> list[dict]:
    """Generate boxes for raw PhenoBench semantic classes 1–4.

    Classes 3–4 are remapped and always partial; a 0–255 visibility mask adds
    normalized ``visibility`` and marks values at or below the threshold partial.
    """

    boxes: list[dict] = []

    for raw_label in (1, 2, 3, 4):
        instance_ids = np.unique(plant_instances[semantics == raw_label])

        for instance_id in instance_ids:
            if instance_id == 0:
                continue

            mask = (plant_instances == instance_id) & (semantics == raw_label)

            ys, xs = np.where(mask)

            if len(xs) == 0:
                continue

            xmin = int(xs.min())
            xmax = int(xs.max())
            ymin = int(ys.min())
            ymax = int(ys.max())

            width = xmax - xmin
            height = ymax - ymin

            target_label = PARTIAL_LABEL_REMAP.get(raw_label, raw_label)

            is_partial = raw_label in PARTIAL_LABEL_REMAP

            box: dict[str, object] = {
                "label": int(target_label),
                "corner": (xmin, ymin),
                "center": (xmin + width // 2, ymin + height // 2),
                "width": int(width),
                "height": int(height),
            }

            if plant_visibility is not None:
                # A plant instance carries a single visibility value; take the
                # max over its mask (they are equal) and normalize to [0, 1].
                visibility = float(plant_visibility[mask].max()) / _VISIBILITY_SCALE
                box["visibility"] = visibility
                is_partial = is_partial or (visibility <= partial_threshold)

            box["is_partial"] = bool(is_partial)

            boxes.append(box)

    return boxes


class PartialAwarePhenoBench:
    """Replace a PhenoBench sample's boxes with partial-aware boxes from its masks.

    The wrapped dataset must expose semantics and instances and must retain raw
    partial classes with ``ignore_partial=False``; visibility is optional.
    """

    def __init__(
        self,
        dataset,
        partial_threshold: float = DEFAULT_PARTIAL_THRESHOLD,
    ):
        self.dataset = dataset
        self.partial_threshold = partial_threshold

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        sample = dict(self.dataset[index])

        semantics = np.asarray(sample["semantics"])
        plant_instances = np.asarray(sample["plant_instances"])

        visibility = sample.get("plant_visibility")
        if visibility is not None:
            visibility = np.asarray(visibility)

        sample["plant_bboxes"] = plant_boxes_from_masks(
            semantics,
            plant_instances,
            plant_visibility=visibility,
            partial_threshold=self.partial_threshold,
        )

        return sample

    @property
    def source_dataset(self):
        return self.dataset


__all__ = [
    "PARTIAL_LABEL_REMAP",
    "plant_boxes_from_masks",
    "PartialAwarePhenoBench",
]
