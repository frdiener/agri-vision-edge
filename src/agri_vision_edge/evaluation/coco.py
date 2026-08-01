from __future__ import annotations

import json
from pathlib import Path

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from .integrity import (
    CorruptPredictionsError,
    check_predictions,
)
from .partials import (
    DEFAULT_PARTIAL_THRESHOLD,
    filter_predictions_against_partials,
    split_annotations_by_partial,
)

METRIC_NAMES = [
    "AP",
    "AP50",
    "AP75",
    "APS",
    "APM",
    "APL",
    "AR1",
    "AR10",
    "AR100",
    "ARS",
    "ARM",
    "ARL",
]


def _per_class_metrics(
    evaluator: COCOeval,
    coco_gt: COCO,
) -> dict:
    """
    Extract the 12 COCO metrics for each category from an *accumulated*
    ``COCOeval``.

    ``COCOeval`` already evaluates per category — ``eval['precision']`` has shape
    ``[T, R, K, A, M]`` (IoU thresholds, recall thresholds, categories, area
    ranges, max-detections) and ``eval['recall']`` has shape ``[T, K, A, M]``.
    Slicing out a single category ``k`` and averaging exactly as
    ``COCOeval.summarize`` does reproduces the per-class equivalent of the
    aggregate stats. Returns ``{category_name: {metric: value}}``.
    """

    eval_result = getattr(evaluator, "eval", None)

    if not eval_result:
        return {}

    precision = eval_result["precision"]  # [T, R, K, A, M]
    recall = eval_result["recall"]  # [T, K, A, M]

    cat_ids = list(evaluator.params.catIds)
    id_to_name = {c["id"]: c["name"] for c in coco_gt.loadCats(cat_ids)}

    def _mean(values) -> float:
        # COCO convention: -1 marks "not applicable"; average only valid entries.
        valid = values[values > -1]
        return float(valid.mean()) if valid.size else -1.0

    # Area-range axis: 0=all, 1=small, 2=medium, 3=large.
    # Max-detection axis: 0=1, 1=10, 2=100. IoU axis: 0=0.50, 5=0.75.
    per_class = {}

    for k, cat_id in enumerate(cat_ids):
        values = [
            _mean(precision[:, :, k, 0, 2]),  # AP   @[.50:.95]
            _mean(precision[0, :, k, 0, 2]),  # AP50
            _mean(precision[5, :, k, 0, 2]),  # AP75
            _mean(precision[:, :, k, 1, 2]),  # APS
            _mean(precision[:, :, k, 2, 2]),  # APM
            _mean(precision[:, :, k, 3, 2]),  # APL
            _mean(recall[:, k, 0, 0]),  # AR1
            _mean(recall[:, k, 0, 1]),  # AR10
            _mean(recall[:, k, 0, 2]),  # AR100
            _mean(recall[:, k, 1, 2]),  # ARS
            _mean(recall[:, k, 2, 2]),  # ARM
            _mean(recall[:, k, 3, 2]),  # ARL
        ]

        name = id_to_name.get(cat_id, str(cat_id))
        per_class[name] = dict(zip(METRIC_NAMES, values, strict=False))

    return per_class


def _coco_from_dict(dataset: dict) -> COCO:
    """
    Build an in-memory ``COCO`` from an already-parsed annotation dict.

    ``COCO(path)`` only loads from a file; constructing empty and assigning
    ``dataset`` lets us score against a *filtered* ground-truth (partials
    removed) without writing a temporary file.
    """

    coco = COCO()
    coco.dataset = dataset
    coco.createIndex()

    return coco


def evaluate_predictions(
    annotations_path: str | Path,
    predictions_path: str | Path,
    ignore_partials: bool = False,
    partial_threshold: float = DEFAULT_PARTIAL_THRESHOLD,
    allow_corrupt: bool = False,
) -> dict:
    """
    Evaluate a COCO predictions file.

    Returns the 12 aggregate (class-averaged) metrics in ``METRIC_NAMES`` plus a
    ``per_class`` entry mapping each category name to its own 12 metrics.

    Partial ("do-not-care") ground-truth annotations (flagged ``partial`` /
    ``ignore`` / low ``visibility`` -- see
    :mod:`agri_vision_edge.evaluation.partials`) are always excluded from the
    scored ground-truth, matching the pre-partials behaviour. When
    ``ignore_partials`` is set, the PhenoBench rule is additionally applied to
    the predictions: any detection whose area is more than ``partial_threshold``
    contained inside a partial ground-truth box is dropped, so a hit on a
    partial plant is not counted as a false positive. Scoring stays
    pycocotools-based, so numbers remain comparable across the pipeline.
    """

    with open(predictions_path) as f:
        predictions = json.load(f)

    #
    # No detections
    #

    if not predictions:
        print(f"[warning] no predictions: {predictions_path}")

        metrics = dict.fromkeys(METRIC_NAMES, 0.0)
        metrics["per_class"] = {}

        return metrics

    # Non-finite boxes make pycocotools match every detection at every IoU
    # threshold, which yields a high AP (with AP == AP50) instead of an error.
    # Refuse before that number can be written to metrics.json.
    check_predictions(
        predictions,
        source=predictions_path,
        strict=not allow_corrupt,
    )

    with open(annotations_path) as f:
        gt_dataset = json.load(f)

    scored_annotations, partial_annotations = split_annotations_by_partial(
        gt_dataset.get("annotations", []),
        threshold=partial_threshold,
    )

    #
    # Drop detections that land on partial plants (do-not-care), per the
    # upstream PhenoBench containment rule.
    #

    if ignore_partials and partial_annotations:
        predictions = filter_predictions_against_partials(
            predictions,
            partial_annotations,
            threshold=partial_threshold,
        )

    #
    # Score against the non-partial ground-truth only.
    #

    scored_gt_dataset = dict(gt_dataset)
    scored_gt_dataset["annotations"] = scored_annotations

    coco_gt = _coco_from_dict(scored_gt_dataset)

    coco_dt = coco_gt.loadRes(predictions)

    evaluator = COCOeval(
        coco_gt,
        coco_dt,
        "bbox",
    )

    evaluator.evaluate()
    evaluator.accumulate()
    evaluator.summarize()

    metrics = {
        name: float(value)
        for name, value in zip(
            METRIC_NAMES,
            evaluator.stats,
            strict=False,
        )
    }

    metrics["per_class"] = _per_class_metrics(evaluator, coco_gt)

    return metrics


def save_metrics(
    metrics: dict,
    output_path: str | Path,
):

    with open(output_path, "w") as f:
        json.dump(
            metrics,
            f,
            indent=2,
        )


def evaluate_model_dir(
    model_dir: Path,
    annotations_path: Path,
    ignore_partials: bool = False,
    partial_threshold: float = DEFAULT_PARTIAL_THRESHOLD,
    allow_corrupt: bool = False,
):
    """
    Evaluate one benchmark directory.
    """

    predictions_path = model_dir / "predictions.json"

    error_path = model_dir / "error.json"

    metrics_path = model_dir / "metrics.json"

    #
    # Failed benchmark
    #

    if error_path.exists():
        print(f"[skip] {model_dir.name} (failed benchmark)")

        return False

    #
    # No predictions
    #

    if not predictions_path.exists():
        print(f"[skip] {model_dir.name} (missing predictions)")

        return False

    print(f"\n=== Evaluating: {model_dir.name} ===")

    try:
        metrics = evaluate_predictions(
            annotations_path,
            predictions_path,
            ignore_partials=ignore_partials,
            partial_threshold=partial_threshold,
            allow_corrupt=allow_corrupt,
        )
    except CorruptPredictionsError as exc:
        # Skip loudly rather than abort the sweep -- but do NOT write metrics,
        # and drop any stale metrics.json so a corrupt run cannot keep passing
        # itself off as evaluated in the results tree.
        print(f"[skip] {model_dir.name}: {exc}")

        metrics_path.unlink(missing_ok=True)

        save_metrics(
            {"status": "corrupt_predictions", "message": str(exc)},
            model_dir / "metrics_invalid.json",
        )

        return False

    save_metrics(
        metrics,
        metrics_path,
    )

    print()

    print(f"AP:   {metrics['AP']:.4f}")

    print(f"AP50: {metrics['AP50']:.4f}")

    print(f"AP75: {metrics['AP75']:.4f}")

    print_per_class(metrics)

    return True


def print_per_class(metrics: dict):
    """
    Print a compact per-class AP / AP50 / AP75 table, if present.

    Only worth showing for multi-class models; a single class is identical to
    the aggregate above.
    """

    per_class = metrics.get("per_class") or {}

    if len(per_class) < 2:
        return

    print()
    print(f"{'class':<16} {'AP':>8} {'AP50':>8} {'AP75':>8}")

    for name, m in per_class.items():
        print(f"{name:<16} {m['AP']:>8.4f} {m['AP50']:>8.4f} {m['AP75']:>8.4f}")
