"""Adapt COCO predictions to the official PhenoBench detector evaluator.

Inputs must share annotation pixel geometry; tiled results remain tile-wise.
Use ``mAP_plants`` when upstream metrics include unpredicted classes.
"""

from __future__ import annotations

import json
import os
import re
import sys
import tempfile
from pathlib import Path

from .integrity import check_predictions

# Mask sub-directories the upstream evaluator reads per split.
_GT_SUBDIRS = ("plant_instances", "semantics", "plant_visibility")

#: Trailing ``_tile<N>`` marker that ``TiledPhenoBench`` / the materialized
#: tiled tree append to every tile's file name.
_TILE_SUFFIX = re.compile(r"_tile(\d+)$")

#: Semantic label ids used by the upstream PhenoBench ground truth, in the class
#: order its ``mAP_cls`` list follows.
UPSTREAM_LABELS = {"crop": 1, "weed": 2}
UPSTREAM_CLASS_NAMES = ["crop", "weed"]

#: PhenoBench semantic ids for *partial* plants. The upstream evaluator's
#: ``cvt_gt_to_bbox_map`` reads instance labels straight out of ``semantics``
#: and never calls its own ``convert_partial_semantics`` helper, so these leak
#: into the ground truth as two extra "classes" that no model can predict.
UPSTREAM_PARTIAL_LABELS = {3: "partial-crop", 4: "partial-weed"}


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


def _load_coco(annotations_path: Path) -> dict:
    with open(annotations_path) as f:
        return json.load(f)


def _image_index(coco: dict) -> dict[int, dict]:
    """
    Map ``image_id -> {file_name, width, height}`` from a COCO annotations file.
    """

    return {
        int(image["id"]): {
            "file_name": image["file_name"],
            "width": float(image["width"]),
            "height": float(image["height"]),
        }
        for image in coco["images"]
    }


def upstream_label_map(coco: dict) -> dict[int, int]:
    """
    Map our COCO ``category_id`` to the upstream semantic label, by name.

    Multi-class bundles use ``1 crop / 2 weed``; single-class bundles use
    ``1 weed``. Name-based matching handles both regimes. Unknown names raise an
    error.
    """

    label_map: dict[int, int] = {}

    for category in coco.get("categories", []):
        name = str(category["name"]).strip().lower()

        if name not in UPSTREAM_LABELS:
            raise ValueError(
                f"Category {name!r} has no upstream PhenoBench counterpart "
                f"(known: {sorted(UPSTREAM_LABELS)}). Faithful evaluation "
                "compares against the official crop/weed ground truth."
            )

        label_map[int(category["id"])] = UPSTREAM_LABELS[name]

    return label_map


def coco_predictions_to_yolo_lines(
    predictions: list[dict],
    image_info: dict,
    label_map: dict[int, int] | None = None,
) -> list[str]:
    """
    Convert one image's COCO predictions to upstream YOLO lines.

    Each line is ``label cx cy w h score`` with the box normalized to ``[0, 1]``
    by the image size (upstream rescales by the fixed ``1024`` canvas).
    ``label`` is the COCO ``category_id`` translated through ``label_map`` into
    the upstream semantic id; without a map it is written through unchanged
    (correct only when our ids already match upstream's).
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

        category_id = int(pred["category_id"])
        label = label_map.get(category_id, category_id) if label_map else category_id
        score = float(pred.get("score", 1.0))

        lines.append(f"{label} {cx} {cy} {nw} {nh} {score}")

    return lines


def annotation_tile_indices(image_index: dict[int, dict]) -> set[int] | None:
    """
    The ``_tile<N>`` indices used by the annotations, or ``None`` if untiled.

    Returns ``None`` when any filename lacks the marker.
    """

    indices: set[int] = set()

    for info in image_index.values():
        match = _TILE_SUFFIX.search(Path(info["file_name"]).stem)

        if match is None:
            return None

        indices.add(int(match.group(1)))

    return indices


def check_tiling_consistency(
    image_index: dict[int, dict],
    phenobench_dir: Path,
) -> None:
    """
    Refuse a ground-truth tree cut with a different grid than the annotations.

    A 2x2 grid and a half-overlap 3x3 grid both produce 512 px tiles with shared
    ``_tile<N>`` names. Compare tile counts with ``tiling_config.json`` to detect
    this mismatch. Legacy trees without recorded geometry are not checked.
    """

    # Imported here, not at module scope: `agri_vision_edge.data` pulls in
    # TensorFlow, and this module is meant to stay importable without it.
    from ..data.raw_tiling import read_tiling_config

    config = read_tiling_config(phenobench_dir)

    if config is None:
        return

    indices = annotation_tile_indices(image_index)

    if indices is None:
        raise ValueError(
            f"{phenobench_dir} is a tiled ground-truth tree "
            f"({config.rows}x{config.cols}, overlap {config.overlap}) but the "
            "annotations are full-frame (no '_tile<N>' names). Point "
            "--phenobench-dir at the untiled raw dataset instead."
        )

    expected = config.tiles_per_image
    found = max(indices) + 1

    if found != expected:
        raise ValueError(
            f"Tiling mismatch: the annotations use {found} tiles per frame "
            f"(indices up to _tile{max(indices)}) but {phenobench_dir} was cut "
            f"{config.rows}x{config.cols} with overlap {config.overlap} "
            f"= {expected} tiles per frame. Both grids produce the same tile "
            "size and the same '_tile<N>' names, so evaluating across them "
            "would silently score against the wrong crops. Re-materialize the "
            "tiled dataset with the geometry recorded in the exported bundle's "
            "dataset_metadata.json (see scripts/materialize_raw_tiled.py)."
        )


def annotate_class_metrics(
    results: dict,
    predicted_classes: list[str],
    images_without_predictions: int,
) -> dict:
    """Add named per-class and plant-only metrics to upstream results.

    ``mAP_cls`` is label-ID ordered: crop and weed first, then leaked partial
    classes that dilute the upstream aggregate.
    """

    per_class = list(results.get("mAP_cls") or [])

    plant_scores = per_class[: len(UPSTREAM_CLASS_NAMES)]

    ap_per_class = dict(
        zip(UPSTREAM_CLASS_NAMES, plant_scores, strict=False)
    )

    partial_scores = per_class[len(UPSTREAM_CLASS_NAMES):]

    comparable = [
        ap_per_class[name]
        for name in predicted_classes
        if name in ap_per_class
    ]

    results["ap_per_class"] = ap_per_class
    results["ap_partial_classes"] = partial_scores
    results["mAP_plants"] = (
        round(sum(comparable) / len(comparable), 2) if comparable else None
    )
    results["upstream_class_count"] = len(per_class)
    results["images_without_predictions"] = images_without_predictions

    # Include leaked partial classes so `class_names` describes `mAP_cls`.
    results["class_names"] = UPSTREAM_CLASS_NAMES[: len(per_class)] + [
        UPSTREAM_PARTIAL_LABELS.get(3 + i, f"extra-{i}")
        for i in range(max(0, len(per_class) - len(UPSTREAM_CLASS_NAMES)))
    ]
    results["predicted_classes"] = predicted_classes

    if partial_scores:
        print(
            "[faithful] upstream scored "
            f"{len(per_class)} classes, not {len(UPSTREAM_CLASS_NAMES)}: "
            "PhenoBench's partial semantic ids (3/4) leaked into the ground "
            f"truth for the {images_without_predictions} evaluated image(s) "
            "that got no predictions (upstream's partial filter only runs when "
            "an image has at least one prediction). They can never be "
            "predicted, so each drags the reported 'mAP' toward 0 -- use "
            "'mAP_plants' for a comparable number.",
            file=sys.stderr,
        )

    return results


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

    Upstream hard-codes a ``1024 x 1024`` canvas in ``convert.IMG_WIDTH`` and
    ``IMG_HEIGHT`` for YOLO scaling and partial-filter rasterization.
    ``cvt_gt_to_bbox_map`` raises
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
        # Return the empty structure expected by torchmetrics when a tile has no
        # plant instances; upstream's torch.stack rejects that case.
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
    label_map: dict[int, int] | None = None,
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
            label_map,
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
    allow_corrupt: bool = False,
) -> dict:
    """Evaluate annotation-space COCO predictions with official PhenoBench metrics.

    ``phenobench_dir`` must contain masks matching the selected split and image
    geometry. Returns percentage metrics augmented with named class AP and
    ``mAP_plants``; corrupt predictions raise unless ``allow_corrupt`` is true.
    """

    evaluate_plant_detection = _require_upstream()

    annotations_path = Path(annotations_path)
    predictions_path = Path(predictions_path)
    phenobench_dir = Path(phenobench_dir)

    coco = _load_coco(annotations_path)
    image_index = _image_index(coco)
    label_map = upstream_label_map(coco)
    predicted_classes = [
        name
        for name in UPSTREAM_CLASS_NAMES
        if UPSTREAM_LABELS[name] in set(label_map.values())
    ]

    check_tiling_consistency(image_index, phenobench_dir)

    with open(predictions_path) as f:
        predictions = json.load(f)

    check_predictions(
        predictions,
        source=predictions_path,
        strict=not allow_corrupt,
    )

    # Upstream's partial filter only removes partial ground truth on images that
    # have at least one prediction, so the count of prediction-less images
    # explains any phantom classes in `mAP_cls` (see annotate_class_metrics).
    images_without_predictions = len(image_index) - len(
        {int(p["image_id"]) for p in predictions}
    )

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
                label_map,
            )

            results = dict(
                evaluate_plant_detection(
                    {
                        "phenobench_dir": staged_root,
                        "prediction_dir": pred_dir,
                        "export": export_dir,
                        "split": split,
                    }
                )
            )
    finally:
        restore()

    # `mAP_cls` is a bare list whose length varies with the run; without the
    # class order it cannot be read back unambiguously, and the bare `mAP` is
    # not comparable to anything.
    return annotate_class_metrics(
        results,
        predicted_classes,
        images_without_predictions,
    )


__all__ = [
    "annotate_class_metrics",
    "annotation_tile_indices",
    "check_tiling_consistency",
    "coco_predictions_to_yolo_lines",
    "evaluate_faithful",
    "upstream_label_map",
    "UPSTREAM_CLASS_NAMES",
    "UPSTREAM_LABELS",
    "UPSTREAM_PARTIAL_LABELS",
]
