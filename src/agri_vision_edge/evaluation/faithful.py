"""
Faithful upstream PhenoBench detection evaluation.

The lightweight path (:mod:`agri_vision_edge.evaluation.coco`) scores with
pycocotools and merely *ports* the partial-plant rule, so its numbers stay
comparable across the whole pipeline. For leaderboard comparability we also want
the *official* number, which the PhenoBench authors compute with a different
stack (torchmetrics' ``MeanAveragePrecision`` + their own partial filtering) in
``phenobench.evaluation.evaluate_plant_bounding_boxes``.

This module drives that upstream evaluator unchanged: it converts our COCO
``predictions.json`` into the per-image YOLO ``.txt`` files it expects, stages a
ground-truth tree containing exactly the evaluated images (symlinked from the raw
PhenoBench dataset), and calls ``evaluate_plant_detection``. The heavy
``torch`` / ``torchvision`` / ``torchmetrics`` dependencies live behind the
optional ``faithful-eval`` extra and are imported lazily here, so the default
lightweight path never needs them.

Notes
-----
* Predictions must be in their annotation's pixel space and each annotation
  ``file_name`` must match the corresponding PhenoBench mask filename.
* The stock upstream evaluator hard-codes a ``1024 x 1024`` canvas and assumes
  every frame has at least one plant. To support the tiled datasets (e.g. 512
  tiles, some of which are empty background) we detect the (uniform) image size
  from the annotations and, for the duration of the call, patch the upstream
  canvas constants to it and make its ground-truth conversion empty-safe --
  otherwise tiles are silently mis-scaled and empty tiles crash it. The
  algorithm itself (torchmetrics mAP + the official partial filtering) is
  untouched.
* **Tiled eval is tile-wise**: it applies the official evaluator per tile, which
  is internally consistent but is NOT the official full-frame leaderboard
  number (that requires stitching tile predictions back to 1024 frames first).
  A warning is emitted when the image size is not 1024.
* Category ids are written straight through (``1`` crop, ``2`` weed), matching
  the semantic labels the upstream ground-truth uses.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

# Mask sub-directories the upstream evaluator reads per split.
_GT_SUBDIRS = ("plant_instances", "semantics", "plant_visibility")


def _require_upstream():
    """
    Import the upstream evaluator, raising a clear error if deps are missing.
    """

    try:
        from phenobench.evaluation.evaluate_plant_bounding_boxes import (
            evaluate_plant_detection,
        )
    except ImportError as exc:  # pragma: no cover - exercised only without deps
        raise ImportError(
            "Faithful evaluation needs the optional 'faithful-eval' extra "
            "(torch, torchvision, torchmetrics). Install it with "
            "`uv sync --extra faithful-eval` or "
            "`pip install agri-vision-edge[faithful-eval]`."
        ) from exc

    return evaluate_plant_detection


def _load_image_index(annotations_path: Path) -> dict[int, dict]:
    """
    Map ``image_id -> {file_name, width, height}`` from a COCO annotations file.
    """

    with open(annotations_path) as f:
        coco = json.load(f)

    return {
        int(image["id"]): {
            "file_name": image["file_name"],
            "width": float(image["width"]),
            "height": float(image["height"]),
        }
        for image in coco["images"]
    }


def coco_predictions_to_yolo_lines(
    predictions: list[dict],
    image_info: dict,
) -> list[str]:
    """
    Convert one image's COCO predictions to upstream YOLO lines.

    Each line is ``label cx cy w h score`` with the box normalized to ``[0, 1]``
    by the image size (upstream rescales by the fixed ``1024`` canvas). ``label``
    is the COCO ``category_id`` written through unchanged.
    """

    width = image_info["width"]
    height = image_info["height"]

    lines: list[str] = []

    for pred in predictions:
        x, y, w, h = pred["bbox"]

        cx = (x + w / 2.0) / width
        cy = (y + h / 2.0) / height
        nw = w / width
        nh = h / height

        label = int(pred["category_id"])
        score = float(pred.get("score", 1.0))

        lines.append(f"{label} {cx} {cy} {nw} {nh} {score}")

    return lines


def _detect_image_size(image_index: dict[int, dict]) -> tuple[int, int]:
    """
    Return the single ``(width, height)`` shared by every annotated image.

    The upstream evaluator uses one global canvas size, so mixed sizes cannot be
    evaluated faithfully in a single pass.
    """

    sizes = {
        (int(info["width"]), int(info["height"]))
        for info in image_index.values()
    }

    if len(sizes) != 1:
        raise ValueError(
            "Faithful evaluation requires a single, uniform image size (the "
            f"upstream evaluator uses one global canvas); got {sorted(sizes)}."
        )

    return sizes.pop()


def _patch_upstream_for_size(width: int, height: int):
    """
    Adapt the upstream evaluator to ``width x height`` images, empty-safely.

    The stock upstream hard-codes a ``1024 x 1024`` canvas (``convert.IMG_WIDTH``
    / ``IMG_HEIGHT`` -- used to scale the normalized YOLO predictions and to
    rasterize boxes in the partial filter) and its ``cvt_gt_to_bbox_map`` raises
    on a frame with no instances. We patch both for the duration of the call so
    tiled / non-1024 and empty-tile inputs evaluate correctly; the scoring
    algorithm is otherwise unchanged. Returns a ``restore()`` callable.
    """

    import torch
    from phenobench.evaluation import evaluate_plant_bounding_boxes as _epb
    from phenobench.evaluation.auxiliary import convert as _convert

    saved = {
        "convert_w": _convert.IMG_WIDTH,
        "convert_h": _convert.IMG_HEIGHT,
        "epb_w": getattr(_epb, "IMG_WIDTH", None),
        "epb_h": getattr(_epb, "IMG_HEIGHT", None),
        "cvt_gt": _epb.cvt_gt_to_bbox_map,
    }

    _convert.IMG_WIDTH = width
    _convert.IMG_HEIGHT = height
    if hasattr(_epb, "IMG_WIDTH"):
        _epb.IMG_WIDTH = width
    if hasattr(_epb, "IMG_HEIGHT"):
        _epb.IMG_HEIGHT = height

    _orig_cvt = saved["cvt_gt"]

    def _empty_safe_cvt_gt(instance_map, semantics, visibility):
        # A tile with no plant instances yields an empty ground-truth (all its
        # predictions become false positives) -- upstream's torch.stack chokes on
        # that, so return the empty structure torchmetrics expects instead.
        ids = torch.unique(instance_map)
        ids = ids[ids != 0]
        if ids.numel() == 0:
            return [
                {
                    "labels": torch.zeros((0,), dtype=torch.uint8),
                    "boxes": torch.zeros((0, 4), dtype=torch.float32),
                    "visibility": torch.zeros((0,), dtype=torch.float32),
                }
            ]
        return _orig_cvt(instance_map, semantics, visibility)

    _epb.cvt_gt_to_bbox_map = _empty_safe_cvt_gt

    def restore():
        _convert.IMG_WIDTH = saved["convert_w"]
        _convert.IMG_HEIGHT = saved["convert_h"]
        if saved["epb_w"] is not None:
            _epb.IMG_WIDTH = saved["epb_w"]
        if saved["epb_h"] is not None:
            _epb.IMG_HEIGHT = saved["epb_h"]
        _epb.cvt_gt_to_bbox_map = saved["cvt_gt"]

    return restore


def _stage(
    image_index: dict[int, dict],
    predictions_path: Path,
    phenobench_dir: Path,
    split: str,
    workdir: Path,
) -> tuple[Path, Path, Path]:
    """
    Build the temporary GT tree + YOLO prediction tree the evaluator expects.

    Returns ``(staged_phenobench_dir, prediction_dir, export_dir)``.
    """

    with open(predictions_path) as f:
        predictions = json.load(f)

    preds_by_image: dict[int, list[dict]] = {}
    for pred in predictions:
        preds_by_image.setdefault(int(pred["image_id"]), []).append(pred)

    staged_root = workdir / "gt"
    split_dir = staged_root / split
    for sub in _GT_SUBDIRS:
        (split_dir / sub).mkdir(parents=True, exist_ok=True)

    pred_dir = workdir / "pred"
    (pred_dir / "plant_bboxes").mkdir(parents=True, exist_ok=True)

    export_dir = workdir / "export"
    export_dir.mkdir(parents=True, exist_ok=True)

    src_split = Path(phenobench_dir) / split

    for image_id, info in image_index.items():
        stem = Path(info["file_name"]).stem

        # Symlink the three GT masks for exactly this image so the upstream
        # evaluator iterates only over our evaluated set.
        for sub in _GT_SUBDIRS:
            src = src_split / sub / f"{stem}.png"
            if not src.exists():
                raise FileNotFoundError(
                    f"Missing PhenoBench ground-truth mask: {src}. Each "
                    "annotation file_name must match a mask under "
                    f"{phenobench_dir}/{split}/ (for tiled eval, point "
                    "--phenobench-dir at the tiled raw dataset)."
                )
            os.symlink(src.resolve(), split_dir / sub / f"{stem}.png")

        # One YOLO txt per image (empty when there are no predictions).
        lines = coco_predictions_to_yolo_lines(
            preds_by_image.get(image_id, []),
            info,
        )
        (pred_dir / "plant_bboxes" / f"{stem}.txt").write_text(
            "\n".join(lines)
        )

    return staged_root, pred_dir, export_dir


def evaluate_faithful(
    annotations_path: str | Path,
    predictions_path: str | Path,
    phenobench_dir: str | Path,
    split: str = "val",
) -> dict:
    """
    Run the official PhenoBench plant-detection evaluation on our predictions.

    Parameters
    ----------
    annotations_path:
        COCO annotations JSON. Resolves ``image_id -> file_name`` for staging,
        drives the prediction normalization, and determines the evaluation
        canvas size (all images must share one size). Predictions come from the
        benchmark already in this annotation-pixel space, so a model that ran
        inference at a smaller resolution (e.g. 320) needs no special handling.
    predictions_path:
        COCO ``predictions.json`` in annotation-pixel space.
    phenobench_dir:
        Root of the PhenoBench dataset (the directory containing
        ``train`` / ``val`` / ``test`` splits with their mask sub-folders). For
        tiled evaluation this is the tiled raw dataset (512 tiles).
    split:
        Which split the predictions correspond to (``val`` by default).

    Returns
    -------
    dict
        The upstream ``eval_results`` (``mAP`` / ``mAP_50`` / ``mAP_75`` and
        per-class ``mAP_cls``), as torchmetrics percentages.
    """

    evaluate_plant_detection = _require_upstream()

    annotations_path = Path(annotations_path)
    predictions_path = Path(predictions_path)
    phenobench_dir = Path(phenobench_dir)

    image_index = _load_image_index(annotations_path)
    width, height = _detect_image_size(image_index)

    if (width, height) != (1024, 1024):
        print(
            f"[faithful] image size is {width}x{height}, not 1024x1024: running "
            "the official evaluator per image (e.g. tile-wise). This is "
            "internally consistent but NOT the official full-frame leaderboard "
            "number, which requires stitching predictions back to 1024 frames.",
            file=sys.stderr,
        )

    restore = _patch_upstream_for_size(width, height)
    try:
        with tempfile.TemporaryDirectory(prefix="ave-faithful-") as tmp:
            workdir = Path(tmp)

            staged_root, pred_dir, export_dir = _stage(
                image_index,
                predictions_path,
                phenobench_dir,
                split,
                workdir,
            )

            return evaluate_plant_detection(
                {
                    "phenobench_dir": staged_root,
                    "prediction_dir": pred_dir,
                    "export": export_dir,
                    "split": split,
                }
            )
    finally:
        restore()


__all__ = [
    "coco_predictions_to_yolo_lines",
    "evaluate_faithful",
]
