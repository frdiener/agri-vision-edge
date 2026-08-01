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
* Tiled ground truth must have been cut with the **same grid** as the exported
  annotations. Different grids share the tile size and the ``_tile<N>`` names,
  so a mismatch resolves every mask and scores against the wrong crops instead
  of failing; :func:`check_tiling_consistency` compares the annotations' tile
  count against the grid the tree recorded in ``tiling_config.json``.
* Prediction labels are remapped from our COCO ``category_id`` to the upstream
  semantic ids (``1`` crop, ``2`` weed) *by category name*. The multi-class
  bundle happens to agree numerically, but the single-class (weed-only) bundle
  numbers its sole ``weed`` category ``1`` -- writing that through unchanged
  labels every weed as a crop and yields garbage.
* The upstream ground truth always carries both classes, so ``mAP`` is averaged
  over crop *and* weed even for a weed-only model, whose crop AP is
  structurally 0. For single-class models the comparable number is the weed
  entry of ``mAP_cls``, not ``mAP``; the emitted metrics therefore name the
  classes (``class_names``) and record which of them the model can predict
  (``predicted_classes``).
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

    Our bundles number their categories per class regime -- multi-class is
    ``1 crop / 2 weed`` (which matches upstream), but single-class is ``1 weed``
    (which does not). Matching on the name keeps both regimes correct; anything
    outside the upstream vocabulary is a hard error, since silently writing an
    unknown id through is exactly the failure mode this map exists to prevent.
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

    Returns ``None`` as soon as any file name lacks the marker -- a mixed set is
    not a tiled bundle.
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

    This is the one mismatch staging cannot catch. Grids share the tile *size*
    (2x2 and 3x3-with-half-overlap both yield 512 px tiles on a 1024 frame) and
    the name space (``_tile0..``), so 2x2 annotations pointed at a 3x3 tree
    resolve every mask and score against the **wrong crops** -- silently, with
    plausible-looking numbers. Only the tile *count* separates them, so compare
    that against the grid the tree recorded in ``tiling_config.json``.

    No-ops for trees without a recorded geometry (legacy) -- there is nothing to
    compare against, and the missing-mask error in :func:`_stage` still catches
    the case where the names do not exist at all.
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
    """
    Make upstream's ``mAP`` / ``mAP_cls`` readable, and flag when it is diluted.

    Upstream builds its metric with ``MeanAveragePrecision(class_metrics=True)``
    and reports ``mAP`` as the **unweighted mean over whichever classes appear**
    in the union of ground truth and predictions. Two upstream quirks make that
    set vary between runs of the *same* model family:

    1. ``cvt_gt_to_bbox_map`` labels each instance with its raw ``semantics``
       value and never applies ``convert_partial_semantics``, so PhenoBench's
       partial ids ``3`` / ``4`` survive as extra classes.
    2. ``filter_partials_boxes`` nests its ground-truth removal loop *inside*
       the per-prediction loop, so an image with **zero** predictions keeps all
       of its partial ground truth -- and with it those extra classes.

    Together they mean a model that misses whole images is penalised twice: once
    for the misses, and again because each extra class contributes ``0`` to the
    average. Measured on the i.MX8MP sweep, the number of extra classes tracked
    the count of prediction-less images exactly (0 -> 2 classes, a handful -> 3,
    hundreds -> 4).

    ``mAP_cls`` is ordered by ascending label id and crop (``1``) / weed (``2``)
    are always present in the PhenoBench ground truth, so entries ``0`` and
    ``1`` are always crop and weed and anything beyond them is a partial class.
    This adds:

    * ``ap_per_class`` -- ``{"crop": ..., "weed": ...}``.
    * ``ap_partial_classes`` -- the phantom entries, if any.
    * ``mAP_plants`` -- the comparable aggregate: the mean over the classes this
      model can actually emit (crop + weed for multi-class, weed alone for a
      weed-only model). This is what lines up with the pycocotools ``AP``.
    * ``upstream_class_count`` / ``images_without_predictions`` -- the evidence
      for how much ``mAP`` was diluted.
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

    # `class_names` used to claim ["crop", "weed"] unconditionally, which is
    # wrong whenever the partial classes leak in -- keep it describing what
    # `mAP_cls` actually holds.
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
    allow_corrupt:
        Score predictions containing non-finite boxes / out-of-range scores
        instead of refusing them. The result is meaningless; see
        :mod:`agri_vision_edge.evaluation.integrity`.

    Returns
    -------
    dict
        The upstream ``eval_results`` (``mAP`` / ``mAP_50`` / ``mAP_75`` and
        per-class ``mAP_cls``) as torchmetrics percentages, annotated by
        :func:`annotate_class_metrics`.

        **Read ``mAP_plants``, not ``mAP``.** Upstream's ``mAP`` is an
        unweighted mean over every class it happened to score, which includes
        classes this model cannot emit (``crop`` for a weed-only model) and
        PhenoBench's partial semantic ids when its partial filter did not run
        (see :func:`annotate_class_metrics`). ``mAP_plants`` averages only the
        classes in ``predicted_classes`` and is what lines up with the
        pycocotools ``AP`` in ``metrics.json``.
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
