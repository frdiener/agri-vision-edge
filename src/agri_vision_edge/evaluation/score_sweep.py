"""Compute confidence-threshold precision, recall, and F1 from COCO match tables.

Sweeps require untruncated detections, and best-F1 selection excludes thresholds
below the lowest emitted score.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .coco import PreparedEvaluation

#: IoU thresholds the sweep is computed at.
DEFAULT_IOU_THRESHOLDS = (0.5, 0.75)

#: Confidence grid resolution: 0.0, 0.001, ..., 1.0.
DEFAULT_GRID_POINTS = 1001

#: Name used for the class-pooled (micro-averaged) aggregate.
MICRO_CLASS = "all"

#: The single area range the sweep uses. COCO's "all" bucket.
_AREA_RNG_ALL = [0.0, 1e5**2]


def default_score_grid(num: int = DEFAULT_GRID_POINTS) -> np.ndarray:
    """Confidence thresholds the sweep is evaluated at, ascending."""

    return np.linspace(0.0, 1.0, num)


@dataclass(frozen=True)
class ClassSweep:
    """Cumulative detection counts for one category, per IoU threshold."""

    name: str

    category_id: int

    #: Non-ignored ground-truth boxes: the recall denominator.
    n_gt: int

    #: ``[T, G]`` true positives with ``score >= grid[g]``.
    tp: np.ndarray

    #: ``[T, G]`` false positives with ``score >= grid[g]``.
    fp: np.ndarray


@dataclass(frozen=True)
class ScoreSweep:
    """A full confidence sweep for one evaluated run."""

    score_grid: np.ndarray

    iou_thresholds: tuple[float, ...]

    per_class: dict[str, ClassSweep]

    #: Detections per image actually allowed through matching.
    max_dets_used: int

    #: Largest detection count any single image produced.
    max_dets_observed: int

    #: Lowest / highest confidence present in the predictions.
    min_score: float

    max_score: float

    n_images: int

    n_detections: int

    #: Partial-plant convention the sweep was computed under. Curves made with
    #: different settings are not comparable: with it off, a detection on a
    #: partial plant is a false positive.
    ignore_partials: bool = False

    partial_threshold: float = 0.5

    # Derivation

    def counts(self, class_name: str = MICRO_CLASS) -> tuple[np.ndarray, ...]:
        """
        ``(tp, fp, n_gt)`` for one class, or pooled across all for
        ``MICRO_CLASS``.
        """

        if class_name != MICRO_CLASS:
            entry = self.per_class[class_name]
            return entry.tp, entry.fp, entry.n_gt

        if not self.per_class:
            shape = (len(self.iou_thresholds), self.score_grid.size)
            return np.zeros(shape), np.zeros(shape), 0

        tp = sum(entry.tp for entry in self.per_class.values())
        fp = sum(entry.fp for entry in self.per_class.values())
        n_gt = sum(entry.n_gt for entry in self.per_class.values())

        return tp, fp, n_gt

    def curve(
        self,
        iou_threshold: float,
        class_name: str = MICRO_CLASS,
    ) -> dict[str, np.ndarray]:
        """
        Precision / recall / F1 over the confidence grid.

        Undefined points are ``NaN``, not 0: above the highest detection score
        nothing is predicted, so precision has no value.
        """

        index = self._iou_index(iou_threshold)

        tp, fp, n_gt = self.counts(class_name)

        tp = tp[index].astype(np.float64)
        fp = fp[index].astype(np.float64)

        predicted = tp + fp

        with np.errstate(divide="ignore", invalid="ignore"):
            precision = np.where(predicted > 0, tp / predicted, np.nan)
            recall = np.full(tp.shape, np.nan) if n_gt == 0 else tp / float(n_gt)

            denominator = precision + recall
            f1 = np.where(
                denominator > 0,
                2.0 * precision * recall / denominator,
                0.0,
            )

        # F1 is only 0 where P and R are genuinely 0; keep NaN where undefined.
        f1 = np.where(np.isnan(precision) | np.isnan(recall), np.nan, f1)

        return {
            "score": self.score_grid,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "tp": tp,
            "fp": fp,
        }

    def best_f1(
        self,
        iou_threshold: float,
        class_name: str = MICRO_CLASS,
    ) -> dict[str, float]:
        """
        The F1-optimal operating point and the threshold achieving it.

        Thresholds below the score floor are excluded: they all describe the
        same operating point, so a plain argmax would report 0.0 instead of the
        floor. Ties above it resolve to the lowest threshold.
        """

        curve = self.curve(iou_threshold, class_name)

        f1 = np.array(curve["f1"], dtype=np.float64)

        # Mask up to the bin holding the floor, not to `min_score` itself: grid
        # points are not exactly representable, so a detection scoring exactly
        # 0.051 bins one step below `grid == 0.051`. Cutting at the raw value
        # would discard the point that keeps every detection.
        if np.isfinite(self.min_score):
            floor_index = max(
                int(np.searchsorted(self.score_grid, self.min_score, side="right")) - 1,
                0,
            )
            f1[:floor_index] = np.nan

        if np.all(np.isnan(f1)):
            return {
                "score": float("nan"),
                "f1": float("nan"),
                "precision": float("nan"),
                "recall": float("nan"),
            }

        best = int(np.nanargmax(f1))

        return {
            "score": float(curve["score"][best]),
            "f1": float(f1[best]),
            "precision": float(curve["precision"][best]),
            "recall": float(curve["recall"][best]),
        }

    def _iou_index(self, iou_threshold: float) -> int:
        for index, value in enumerate(self.iou_thresholds):
            if abs(value - iou_threshold) < 1e-9:
                return index

        raise KeyError(
            f"IoU threshold {iou_threshold} was not computed; "
            f"available: {list(self.iou_thresholds)}"
        )

    # Serialization

    def to_dict(self) -> dict:
        """
        JSON-ready form.

        Cumulative counts use a fixed grid of about 1,000 points to avoid tens
        of thousands of per-detection rows. They reconstruct each curve at grid
        resolution.
        """

        return {
            "score_grid": {
                "start": float(self.score_grid[0]),
                "stop": float(self.score_grid[-1]),
                "num": int(self.score_grid.size),
            },
            "iou_thresholds": list(self.iou_thresholds),
            "max_dets_used": self.max_dets_used,
            "max_dets_observed": self.max_dets_observed,
            "min_score": self.min_score,
            "max_score": self.max_score,
            "n_images": self.n_images,
            "n_detections": self.n_detections,
            "ignore_partials": self.ignore_partials,
            "partial_threshold": self.partial_threshold,
            "per_class": {
                name: {
                    "category_id": entry.category_id,
                    "n_gt": entry.n_gt,
                    "tp": entry.tp.astype(int).tolist(),
                    "fp": entry.fp.astype(int).tolist(),
                }
                for name, entry in self.per_class.items()
            },
        }

    @classmethod
    def from_dict(cls, data: dict) -> ScoreSweep:
        grid = data["score_grid"]

        return cls(
            score_grid=np.linspace(grid["start"], grid["stop"], grid["num"]),
            iou_thresholds=tuple(data["iou_thresholds"]),
            per_class={
                name: ClassSweep(
                    name=name,
                    category_id=entry["category_id"],
                    n_gt=entry["n_gt"],
                    tp=np.asarray(entry["tp"], dtype=np.int64),
                    fp=np.asarray(entry["fp"], dtype=np.int64),
                )
                for name, entry in data["per_class"].items()
            },
            max_dets_used=data["max_dets_used"],
            max_dets_observed=data["max_dets_observed"],
            min_score=data["min_score"],
            max_score=data["max_score"],
            n_images=data["n_images"],
            n_detections=data["n_detections"],
            ignore_partials=data.get("ignore_partials", False),
            partial_threshold=data.get("partial_threshold", 0.5),
        )


# Computation


def _cumulative_counts(
    scores: np.ndarray,
    flags: np.ndarray,
    grid: np.ndarray,
) -> np.ndarray:
    """
    Count flagged detections with ``score >= grid[g]``, for every ``g``.

    Histogram then reverse cumulative sum: exact at grid resolution, and O(N)
    instead of O(N x G).
    """

    n_thresholds = flags.shape[0]

    counts = np.zeros((n_thresholds, grid.size), dtype=np.int64)

    if scores.size == 0:
        return counts

    # Largest grid index whose threshold the score still satisfies.
    bins = np.searchsorted(grid, np.clip(scores, 0.0, 1.0), side="right") - 1
    bins = np.clip(bins, 0, grid.size - 1)

    for index in range(n_thresholds):
        histogram = np.bincount(
            bins,
            weights=flags[index].astype(np.float64),
            minlength=grid.size,
        )

        # Reverse cumulative sum: "at or above this threshold".
        counts[index] = np.cumsum(histogram[::-1])[::-1].astype(np.int64)

    return counts


def compute_score_sweep(
    prepared: PreparedEvaluation,
    *,
    iou_thresholds=DEFAULT_IOU_THRESHOLDS,
    score_grid: np.ndarray | None = None,
    max_dets: int | None = None,
) -> ScoreSweep:
    """Sweep confidence metrics for a prepared evaluation.

    ``max_dets`` defaults to the largest per-image detection count to avoid
    truncation; an explicit limit may make results threshold-dependent.
    """

    # Imported here so reading a stored sweep never requires the evaluator.
    from pycocotools.cocoeval import COCOeval

    grid = default_score_grid() if score_grid is None else np.asarray(score_grid)

    iou_thresholds = tuple(float(v) for v in iou_thresholds)

    coco_gt = prepared.coco_gt

    image_ids = sorted(coco_gt.getImgIds())
    category_ids = sorted(coco_gt.getCatIds())

    #
    # Maximum detections per image before truncation
    #

    per_image_counts: dict[int, int] = {}

    for prediction in prepared.predictions:
        image_id = prediction["image_id"]
        per_image_counts[image_id] = per_image_counts.get(image_id, 0) + 1

    max_dets_observed = max(per_image_counts.values(), default=0)

    max_dets_used = max_dets_observed if max_dets is None else int(max_dets)

    # COCOeval clamps to maxDets[-1]; 0 would silently discard everything.
    max_dets_used = max(max_dets_used, 1)

    evaluator = COCOeval(coco_gt, prepared.coco_dt, "bbox")

    evaluator.params.imgIds = image_ids
    evaluator.params.catIds = category_ids
    evaluator.params.iouThrs = np.array(iou_thresholds, dtype=np.float64)
    evaluator.params.areaRng = [_AREA_RNG_ALL]
    evaluator.params.areaRngLbl = ["all"]
    evaluator.params.maxDets = [max_dets_used]

    # evaluate() fills evalImgs; accumulate() would discard the score axis.
    evaluator.evaluate()

    #
    # evalImgs is laid out [catId][areaRng][imgId]; areaRng has length 1.
    #

    n_images = len(evaluator.params.imgIds)

    id_to_name = {
        category["id"]: category["name"] for category in coco_gt.loadCats(category_ids)
    }

    per_class: dict[str, ClassSweep] = {}

    for k, category_id in enumerate(evaluator.params.catIds):
        entries = [
            entry
            for entry in evaluator.evalImgs[k * n_images : (k + 1) * n_images]
            if entry is not None
        ]

        name = id_to_name.get(category_id, str(category_id))

        if not entries:
            shape = (len(iou_thresholds), grid.size)

            per_class[name] = ClassSweep(
                name=name,
                category_id=int(category_id),
                n_gt=0,
                tp=np.zeros(shape, dtype=np.int64),
                fp=np.zeros(shape, dtype=np.int64),
            )

            continue

        scores = np.concatenate(
            [np.asarray(e["dtScores"][:max_dets_used]) for e in entries]
        )

        matches = np.concatenate(
            [e["dtMatches"][:, :max_dets_used] for e in entries], axis=1
        )

        ignored = np.concatenate(
            [e["dtIgnore"][:, :max_dets_used] for e in entries], axis=1
        )

        gt_ignored = np.concatenate([e["gtIgnore"] for e in entries])

        # As COCOeval.accumulate: a detection matched to an ignored GT box is
        # neither a true nor a false positive.
        matched = matches > 0
        ignored = ignored.astype(bool)

        true_positive = matched & ~ignored
        false_positive = ~matched & ~ignored

        per_class[name] = ClassSweep(
            name=name,
            # COCOeval hands back numpy integers; keep the artifact plain JSON.
            category_id=int(category_id),
            n_gt=int(np.count_nonzero(gt_ignored == 0)),
            tp=_cumulative_counts(scores, true_positive, grid),
            fp=_cumulative_counts(scores, false_positive, grid),
        )

    all_scores = np.array(
        [float(p.get("score", 0.0)) for p in prepared.predictions],
        dtype=np.float64,
    )

    return ScoreSweep(
        score_grid=grid,
        iou_thresholds=iou_thresholds,
        per_class=per_class,
        max_dets_used=max_dets_used,
        max_dets_observed=max_dets_observed,
        min_score=float(all_scores.min()) if all_scores.size else float("nan"),
        max_score=float(all_scores.max()) if all_scores.size else float("nan"),
        n_images=n_images,
        n_detections=int(all_scores.size),
        ignore_partials=prepared.ignore_partials,
        partial_threshold=prepared.partial_threshold,
    )


# Artifact I/O

#: Written beside ``metrics.json`` by ``ave evaluate``.
SCORE_SWEEP_FILENAME = "score_sweep.json"


def save_score_sweep(sweep: ScoreSweep, output_path: str | Path) -> None:
    with open(output_path, "w") as f:
        json.dump(sweep.to_dict(), f)


def load_score_sweep(path: str | Path) -> ScoreSweep | None:
    """Load a ``score_sweep.json``; ``None`` when absent or unreadable."""

    path = Path(path)

    if not path.is_file():
        return None

    try:
        return ScoreSweep.from_dict(json.loads(path.read_text()))
    except (ValueError, KeyError):
        return None


def operating_points(sweep: ScoreSweep) -> dict:
    """Best-F1 operating points for every IoU threshold and class."""

    points: dict = {}

    for iou in sweep.iou_thresholds:
        key = f"f1_best@{iou:.2f}"

        entry = dict(sweep.best_f1(iou))

        entry["per_class"] = {
            name: sweep.best_f1(iou, name) for name in sweep.per_class
        }

        points[key] = entry

    return points


def score_sweep_frame(
    sweep: ScoreSweep,
    *,
    include_micro: bool = True,
    **columns,
):
    """
    Long-format frame: one row per (class, IoU, threshold).

    Extra keyword arguments become constant columns, for carrying run metadata
    when several sweeps are concatenated.
    """

    import pandas as pd

    names = list(sweep.per_class)

    if include_micro:
        names.append(MICRO_CLASS)

    frames = []

    for name in names:
        for iou in sweep.iou_thresholds:
            curve = sweep.curve(iou, name)

            frame = pd.DataFrame(
                {
                    "score": curve["score"],
                    "precision": curve["precision"],
                    "recall": curve["recall"],
                    "f1": curve["f1"],
                }
            )

            frame["class"] = name
            frame["iou"] = iou

            frames.append(frame)

    if not frames:
        return pd.DataFrame()

    out = pd.concat(frames, ignore_index=True)

    for key, value in columns.items():
        out[key] = value

    return out


__all__ = [
    "DEFAULT_IOU_THRESHOLDS",
    "DEFAULT_GRID_POINTS",
    "MICRO_CLASS",
    "SCORE_SWEEP_FILENAME",
    "ClassSweep",
    "ScoreSweep",
    "compute_score_sweep",
    "default_score_grid",
    "load_score_sweep",
    "operating_points",
    "save_score_sweep",
    "score_sweep_frame",
]
