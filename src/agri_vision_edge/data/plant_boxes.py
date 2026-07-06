"""
Partial-aware plant bounding-box generation from PhenoBench masks.

The upstream :class:`phenobench.PhenoBench` loader builds ``plant_bboxes`` only
for the fully-visible semantic classes (``1`` crop, ``2`` weed) and silently
drops the partial (border) plants labelled ``3`` (partial-crop) / ``4``
(partial-weed); it also never surfaces the per-instance visibility. That is
lossy for evaluation: the official PhenoBench protocol treats partials as
*do-not-care* (see :mod:`agri_vision_edge.evaluation.partials`), which requires
knowing *where* the partials are so a detection landing on one is not penalized.

This module regenerates boxes for **all** instances directly from the
``semantics`` / ``plant_instances`` / ``plant_visibility`` masks, carrying a
per-box ``visibility`` (in ``[0, 1]``) and an ``is_partial`` flag. It mirrors the
loader's ``(label, instance)``-scoped box construction (robust to instance ids
reused across classes) and the upstream partiality criterion
(``visibility <= threshold``), additionally flagging the border classes ``3``/``4``.

The boxes are emitted in the same dict shape the loader uses
(``label`` / ``corner`` / ``center`` / ``width`` / ``height``) plus ``visibility``
and ``is_partial``, so the COCO / TFRecord exporters consume them unchanged.
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
    """
    Regenerate all plant boxes (including partials) with visibility.

    Parameters
    ----------
    semantics:
        ``H x W`` semantic mask with the raw PhenoBench classes
        (``1`` crop, ``2`` weed, ``3`` partial-crop, ``4`` partial-weed).
    plant_instances:
        ``H x W`` instance-id mask.
    plant_visibility:
        Optional ``H x W`` visibility mask (0..255). When given, each box gets a
        ``visibility`` in ``[0, 1]`` and ``is_partial`` follows the upstream
        criterion ``visibility <= partial_threshold``. Border classes ``3``/``4``
        are always flagged partial regardless.
    partial_threshold:
        Visibility at or below which a box is partial (default ``0.5``).

    Returns
    -------
    list[dict]
        Boxes with keys ``label`` (remapped to ``1``/``2``), ``corner``,
        ``center``, ``width``, ``height``, ``is_partial`` and -- when a
        visibility mask is supplied -- ``visibility``.
    """

    boxes: list[dict] = []

    for raw_label in (1, 2, 3, 4):

        instance_ids = np.unique(
            plant_instances[semantics == raw_label]
        )

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
    """
    Wrap a :class:`phenobench.PhenoBench` so ``plant_bboxes`` includes partials.

    The wrapped dataset must expose the ``semantics``, ``plant_instances`` and
    (optionally, but recommended) ``plant_visibility`` target types. Each sample
    is returned with ``plant_bboxes`` replaced by the partial-aware boxes from
    :func:`plant_boxes_from_masks`, so the COCO / TFRecord exporters can carry
    the do-not-care boxes through with an ``ignore`` flag.

    Use with ``ignore_partial=False`` on the underlying loader (so the raw
    semantics ``3``/``4`` reach this wrapper); passing ``ignore_partial=True``
    would have already masked them away.
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
