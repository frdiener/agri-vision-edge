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
* The upstream evaluator operates on full ``1024 x 1024`` frames. Predictions
  must therefore be in the original-image pixel space and their annotation
  ``file_name`` must match the raw PhenoBench mask filenames (i.e. full-image,
  not tiled, exports).
* Category ids are written straight through (``1`` crop, ``2`` weed), matching
  the semantic labels the upstream ground-truth uses.
"""

from __future__ import annotations

import json
import os
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


def _stage(
    annotations_path: Path,
    predictions_path: Path,
    phenobench_dir: Path,
    split: str,
    workdir: Path,
) -> tuple[Path, Path, Path]:
    """
    Build the temporary GT tree + YOLO prediction tree the evaluator expects.

    Returns ``(staged_phenobench_dir, prediction_dir, export_dir)``.
    """

    image_index = _load_image_index(annotations_path)

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
                    f"Missing upstream ground-truth mask: {src}. Faithful "
                    "evaluation requires full-image predictions whose "
                    "file_name matches the raw PhenoBench masks."
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
        COCO annotations JSON (used only to resolve ``image_id -> file_name`` and
        image sizes for the YOLO conversion).
    predictions_path:
        COCO ``predictions.json`` in original-image pixel space.
    phenobench_dir:
        Root of the raw PhenoBench dataset (the directory containing
        ``train`` / ``val`` / ``test`` splits with their mask sub-folders).
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

    with tempfile.TemporaryDirectory(prefix="ave-faithful-") as tmp:
        workdir = Path(tmp)

        staged_root, pred_dir, export_dir = _stage(
            annotations_path,
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


__all__ = [
    "coco_predictions_to_yolo_lines",
    "evaluate_faithful",
]
