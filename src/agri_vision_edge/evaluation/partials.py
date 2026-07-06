"""
Partial-plant ("do-not-care") filtering for detection evaluation.

PhenoBench's plants that are only partially visible -- because they are cut off
by the image border (the semantic ``partial-crop`` / ``partial-weed`` classes) or
otherwise mostly occluded -- are treated as *do-not-care* by the official
benchmark: a missed partial is not a false negative, and a detection that fires
on a partial is not a false positive. See
``phenobench.evaluation.evaluate_plant_bounding_boxes`` /
``phenobench.evaluation.auxiliary.filter.filter_partials_boxes``.

This module ports that exact rule with **numpy only** (no torch), so the
lightweight pycocotools evaluation path -- and on-device evaluation, where only
the core ``numpy`` dependency is available -- can reproduce it. The heavyweight,
byte-for-byte upstream reproduction (torchmetrics mAP) lives behind the optional
``eval-faithful`` path in :mod:`agri_vision_edge.evaluation.faithful`.

Upstream criterion (``filter_partials_boxes``), replicated here:

* A ground-truth box is *partial* when its visibility is ``<= threshold``
  (default ``0.5``).
* Every partial ground-truth box is dropped from scoring.
* A prediction is dropped when the fraction of *its own area* that lies inside
  any single partial ground-truth box exceeds ``threshold``.

Upstream measures that fraction by rasterizing both boxes onto the full image
canvas and dividing the intersection pixel count by the prediction pixel count.
For axis-aligned boxes this equals the analytic ``intersection_area /
prediction_area`` computed here (canvas-size independent, and free of the
inclusive ``+1`` pixel rounding, whose effect is sub-pixel).
"""

from __future__ import annotations

import numpy as np

DEFAULT_PARTIAL_THRESHOLD = 0.5


def xywh_to_xyxy(boxes: np.ndarray) -> np.ndarray:
    """
    Convert ``[x, y, w, h]`` boxes (COCO convention) to ``[x0, y0, x1, y1]``.

    Accepts an ``(N, 4)`` array; returns an ``(N, 4)`` array. An empty input
    yields an empty ``(0, 4)`` array.
    """

    boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)

    x0 = boxes[:, 0]
    y0 = boxes[:, 1]
    x1 = x0 + boxes[:, 2]
    y1 = y0 + boxes[:, 3]

    return np.stack([x0, y0, x1, y1], axis=1)


def containment_fractions(
    pred_xyxy: np.ndarray,
    gt_xyxy: np.ndarray,
) -> np.ndarray:
    """
    Fraction of each prediction's area contained in each ground-truth box.

    Returns an ``(n_pred, n_gt)`` matrix where entry ``[i, j]`` is
    ``area(pred_i ∩ gt_j) / area(pred_i)``. Predictions with zero area yield
    ``0`` (they cannot be meaningfully contained). Mirrors upstream's
    ``sum(gt & pred) / sum(pred)``.
    """

    pred_xyxy = np.asarray(pred_xyxy, dtype=np.float64).reshape(-1, 4)
    gt_xyxy = np.asarray(gt_xyxy, dtype=np.float64).reshape(-1, 4)

    n_pred = pred_xyxy.shape[0]
    n_gt = gt_xyxy.shape[0]

    if n_pred == 0 or n_gt == 0:
        return np.zeros((n_pred, n_gt), dtype=np.float64)

    # Pairwise intersection extents: [n_pred, n_gt].
    inter_x0 = np.maximum(pred_xyxy[:, None, 0], gt_xyxy[None, :, 0])
    inter_y0 = np.maximum(pred_xyxy[:, None, 1], gt_xyxy[None, :, 1])
    inter_x1 = np.minimum(pred_xyxy[:, None, 2], gt_xyxy[None, :, 2])
    inter_y1 = np.minimum(pred_xyxy[:, None, 3], gt_xyxy[None, :, 3])

    inter_w = np.clip(inter_x1 - inter_x0, a_min=0.0, a_max=None)
    inter_h = np.clip(inter_y1 - inter_y0, a_min=0.0, a_max=None)
    inter_area = inter_w * inter_h

    pred_w = np.clip(pred_xyxy[:, 2] - pred_xyxy[:, 0], a_min=0.0, a_max=None)
    pred_h = np.clip(pred_xyxy[:, 3] - pred_xyxy[:, 1], a_min=0.0, a_max=None)
    pred_area = pred_w * pred_h  # [n_pred]

    # Avoid division by zero; zero-area predictions get a fraction of 0.
    safe_area = np.where(pred_area > 0.0, pred_area, 1.0)
    fractions = inter_area / safe_area[:, None]
    fractions[pred_area <= 0.0, :] = 0.0

    return fractions


def partial_prediction_mask(
    pred_boxes_xywh: np.ndarray,
    partial_gt_boxes_xywh: np.ndarray,
    threshold: float = DEFAULT_PARTIAL_THRESHOLD,
) -> np.ndarray:
    """
    Boolean mask over predictions: ``True`` where a prediction must be dropped.

    A prediction is dropped when the fraction of its own area contained in *any*
    single partial ground-truth box is strictly greater than ``threshold``
    (upstream uses ``score > 0.5``).

    Parameters
    ----------
    pred_boxes_xywh:
        ``(n_pred, 4)`` predictions in ``[x, y, w, h]``.
    partial_gt_boxes_xywh:
        ``(n_gt, 4)`` partial ground-truth boxes in ``[x, y, w, h]``.
    threshold:
        Containment threshold (default ``0.5``).

    Returns
    -------
    ``(n_pred,)`` boolean array. All ``False`` when there are no partial GT
    boxes.
    """

    pred_boxes_xywh = np.asarray(pred_boxes_xywh, dtype=np.float64).reshape(-1, 4)

    n_pred = pred_boxes_xywh.shape[0]

    partial_gt_boxes_xywh = np.asarray(
        partial_gt_boxes_xywh, dtype=np.float64
    ).reshape(-1, 4)

    if n_pred == 0 or partial_gt_boxes_xywh.shape[0] == 0:
        return np.zeros((n_pred,), dtype=bool)

    fractions = containment_fractions(
        xywh_to_xyxy(pred_boxes_xywh),
        xywh_to_xyxy(partial_gt_boxes_xywh),
    )

    return np.any(fractions > threshold, axis=1)


def is_partial_annotation(
    annotation: dict,
    threshold: float = DEFAULT_PARTIAL_THRESHOLD,
) -> bool:
    """
    Decide whether a COCO annotation denotes a partial ("do-not-care") plant.

    Recognizes, in order of precedence:

    * an explicit ``partial`` flag (``1``/``True``),
    * a ``visibility`` value ``<= threshold`` (the upstream criterion), or
    * the COCO-standard ``ignore`` flag (``1``/``True``).

    Falls back to ``False`` (a normally-scored box) when none is present, so
    annotations produced before partials were carried keep their old meaning.
    """

    if annotation.get("partial"):
        return True

    visibility = annotation.get("visibility")
    if visibility is not None:
        return float(visibility) <= threshold

    return bool(annotation.get("ignore"))


def split_annotations_by_partial(
    annotations: list[dict],
    threshold: float = DEFAULT_PARTIAL_THRESHOLD,
) -> tuple[list[dict], list[dict]]:
    """
    Split COCO annotations into ``(scored, partial)`` lists.

    ``scored`` are the normally-evaluated ground-truth boxes; ``partial`` are the
    do-not-care boxes (per :func:`is_partial_annotation`).
    """

    scored: list[dict] = []
    partial: list[dict] = []

    for annotation in annotations:
        if is_partial_annotation(annotation, threshold):
            partial.append(annotation)
        else:
            scored.append(annotation)

    return scored, partial


def filter_predictions_against_partials(
    predictions: list[dict],
    partial_annotations: list[dict],
    threshold: float = DEFAULT_PARTIAL_THRESHOLD,
) -> list[dict]:
    """
    Drop predictions that land on partial ("do-not-care") ground-truth boxes.

    Groups both by ``image_id`` and applies the upstream containment rule
    (:func:`partial_prediction_mask`) per image. Predictions on images with no
    partial ground-truth pass through untouched. The returned list preserves
    input order.
    """

    if not partial_annotations:
        return list(predictions)

    partials_by_image: dict[object, list[list[float]]] = {}
    for annotation in partial_annotations:
        partials_by_image.setdefault(
            annotation["image_id"], []
        ).append(annotation["bbox"])

    kept: list[dict] = []

    # Group predictions per image so the mask is computed once per image.
    preds_by_image: dict[object, list[dict]] = {}
    for prediction in predictions:
        preds_by_image.setdefault(
            prediction["image_id"], []
        ).append(prediction)

    for image_id, image_preds in preds_by_image.items():
        image_partials = partials_by_image.get(image_id)

        if not image_partials:
            kept.extend(image_preds)
            continue

        drop_mask = partial_prediction_mask(
            np.array([p["bbox"] for p in image_preds], dtype=np.float64),
            np.array(image_partials, dtype=np.float64),
            threshold=threshold,
        )

        for prediction, drop in zip(image_preds, drop_mask, strict=True):
            if not drop:
                kept.append(prediction)

    return kept


__all__ = [
    "DEFAULT_PARTIAL_THRESHOLD",
    "xywh_to_xyxy",
    "containment_fractions",
    "partial_prediction_mask",
    "is_partial_annotation",
    "split_annotations_by_partial",
    "filter_predictions_against_partials",
]
