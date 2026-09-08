from __future__ import annotations

import json
from dataclasses import dataclass
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
from .score_sweep import (
    SCORE_SWEEP_FILENAME,
    compute_score_sweep,
    operating_points,
    save_score_sweep,
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
    """Extract all 12 COCO summary metrics per category from an accumulated evaluator."""

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


@dataclass(frozen=True)
class PreparedEvaluation:
    """
    A predictions/ground-truth pair that is ready to be scored.

    Every consumer must use :func:`prepare_evaluation`. This keeps derived
    analyses and headline metrics on the same filtered detections and ground
    truth.
    """

    #: Ground truth with partial ("do-not-care") annotations removed.
    coco_gt: COCO

    #: Detections loaded into pycocotools' result structure.
    coco_dt: COCO

    #: The detection dicts backing ``coco_dt``, after partial filtering.
    predictions: list[dict]

    #: Whether detections landing on partial plants were dropped.
    ignore_partials: bool = False

    partial_threshold: float = DEFAULT_PARTIAL_THRESHOLD


def prepare_evaluation(
    annotations_path: str | Path,
    predictions_path: str | Path,
    ignore_partials: bool = False,
    partial_threshold: float = DEFAULT_PARTIAL_THRESHOLD,
    allow_corrupt: bool = False,
) -> PreparedEvaluation | None:
    """Load, validate, and partial-filter inputs for scoring.

    Partial ground truth is always unscored; with ``ignore_partials``, detections
    sufficiently contained by a partial box are also dropped. Empty predictions
    return ``None`` without reading the annotations.
    """

    with open(predictions_path) as f:
        predictions = json.load(f)

    #
    # No detections
    #

    if not predictions:
        return None

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

    return PreparedEvaluation(
        coco_gt=coco_gt,
        coco_dt=coco_dt,
        predictions=predictions,
        ignore_partials=ignore_partials,
        partial_threshold=partial_threshold,
    )


def empty_metrics() -> dict:
    """Metrics for a run that produced no detections at all."""

    # Annotated: the aggregate entries are floats but ``per_class`` is a dict.
    metrics: dict = dict.fromkeys(METRIC_NAMES, 0.0)
    metrics["per_class"] = {}

    return metrics


def evaluate_prepared(prepared: PreparedEvaluation) -> dict:
    """
    Score an already-prepared pair with the standard COCO parameters.

    Split out of :func:`evaluate_predictions` so callers that also want the
    confidence sweep prepare once and score the identical detections.
    """

    coco_gt = prepared.coco_gt

    evaluator = COCOeval(
        coco_gt,
        prepared.coco_dt,
        "bbox",
    )

    evaluator.evaluate()
    evaluator.accumulate()
    evaluator.summarize()

    metrics: dict = {
        name: float(value)
        for name, value in zip(
            METRIC_NAMES,
            evaluator.stats,
            strict=False,
        )
    }

    metrics["per_class"] = _per_class_metrics(evaluator, coco_gt)

    return metrics


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

    Partial handling is described on :func:`prepare_evaluation`. Scoring stays
    pycocotools-based, so numbers remain comparable across the pipeline.
    """

    prepared = prepare_evaluation(
        annotations_path,
        predictions_path,
        ignore_partials=ignore_partials,
        partial_threshold=partial_threshold,
        allow_corrupt=allow_corrupt,
    )

    #
    # No detections
    #

    if prepared is None:
        print(f"[warning] no predictions: {predictions_path}")

        return empty_metrics()

    return evaluate_prepared(prepared)


def evaluate_and_sweep(
    annotations_path: str | Path,
    predictions_path: str | Path,
    ignore_partials: bool = False,
    partial_threshold: float = DEFAULT_PARTIAL_THRESHOLD,
    allow_corrupt: bool = False,
    score_sweep: bool = True,
):
    """
    Evaluate a predictions file and, optionally, sweep the confidence threshold.

    Returns ``(metrics, sweep)``; ``sweep`` is ``None`` when disabled or when
    there are no detections. Both come from one :func:`prepare_evaluation` call,
    so the curve and the AP beside it describe the same detections.
    """

    prepared = prepare_evaluation(
        annotations_path,
        predictions_path,
        ignore_partials=ignore_partials,
        partial_threshold=partial_threshold,
        allow_corrupt=allow_corrupt,
    )

    if prepared is None:
        print(f"[warning] no predictions: {predictions_path}")

        return empty_metrics(), None

    metrics = evaluate_prepared(prepared)

    if not score_sweep:
        return metrics, None

    # Keep metrics unchanged because the sweep is a separate artifact.
    return metrics, compute_score_sweep(prepared)


def save_metrics(
    metrics: dict,
    output_path: str | Path,
) -> bool:
    """
    Write metrics, returning whether the file changed.

    An unchanged file is left alone, preserving content and modification time
    while backfilling score_sweep.json.
    """

    payload = json.dumps(metrics, indent=2)

    output_path = Path(output_path)

    if output_path.exists() and output_path.read_text() == payload:
        return False

    output_path.write_text(payload)

    return True


def evaluate_model_dir(
    model_dir: Path,
    annotations_path: Path,
    ignore_partials: bool = False,
    partial_threshold: float = DEFAULT_PARTIAL_THRESHOLD,
    allow_corrupt: bool = False,
    score_sweep: bool = True,
):
    """
    Evaluate one benchmark directory.
    """

    predictions_path = model_dir / "predictions.json"

    error_path = model_dir / "error.json"

    metrics_path = model_dir / "metrics.json"

    sweep_path = model_dir / SCORE_SWEEP_FILENAME

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
        metrics, sweep = evaluate_and_sweep(
            annotations_path,
            predictions_path,
            ignore_partials=ignore_partials,
            partial_threshold=partial_threshold,
            allow_corrupt=allow_corrupt,
            score_sweep=score_sweep,
        )
    except CorruptPredictionsError as exc:
        # Report corrupt runs without aborting the sweep. Remove stale metrics so
        # the results tree cannot treat them as evaluated.
        print(f"[skip] {model_dir.name}: {exc}")

        metrics_path.unlink(missing_ok=True)

        # Same reasoning for the curve: a stale sweep from an earlier, healthy
        # run would otherwise still be picked up by the report.
        sweep_path.unlink(missing_ok=True)

        save_metrics(
            {"status": "corrupt_predictions", "message": str(exc)},
            model_dir / "metrics_invalid.json",
        )

        return False

    save_metrics(
        metrics,
        metrics_path,
    )

    if sweep is None:
        sweep_path.unlink(missing_ok=True)
    else:
        save_score_sweep(sweep, sweep_path)

    print()

    print(f"AP:   {metrics['AP']:.4f}")

    print(f"AP50: {metrics['AP50']:.4f}")

    print(f"AP75: {metrics['AP75']:.4f}")

    print_per_class(metrics)

    print_operating_points(sweep)

    return True


def print_operating_points(sweep):
    """
    Print the best-F1 confidence threshold for each IoU threshold.

    Operating points are derived from the sweep on demand. metrics.json contains
    only threshold-free aggregates.
    """

    if sweep is None:
        return

    points = operating_points(sweep)

    if not points:
        return

    print()
    print(f"{'best F1':<16} {'conf':>8} {'F1':>8} {'P':>8} {'R':>8}")

    for key, point in points.items():
        # "f1_best@0.50" -> "IoU 0.50"
        label = f"IoU {key.split('@')[-1]}"

        print(
            f"{label:<16} {point['score']:>8.3f} {point['f1']:>8.4f} "
            f"{point['precision']:>8.4f} {point['recall']:>8.4f}"
        )


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
