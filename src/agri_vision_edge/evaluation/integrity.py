"""
Sanity checks on COCO predictions before they are scored.

A broken runtime does not always produce *obviously* broken metrics. The failure
this module exists to stop was measured on the i.MX8MP: the Teflon delegate,
handed an **fp32** SSD graph, returned ``NaN`` boxes and a constant garbage score
tensor -- and the pipeline reported **AP 85.8** for it, higher than any healthy
INT8 run.

The mechanism is in ``pycocotools``. ``COCOeval.evaluateImg`` matches a detection
to a ground-truth box with::

    if ious[dind, gind] < iou: continue

With a ``NaN`` IoU that comparison is ``False``, so the branch is *not* taken and
the detection is accepted as a match -- at **every** IoU threshold, because the
comparison never depends on the threshold. Every detection therefore matches
some ground truth, and AP@0.50 = AP@0.55 = ... = AP@0.95.

That gives the fingerprint worth remembering: **``AP == AP50`` (to full float
precision) means the boxes are not real.** A genuine detector's AP is always
well below its AP50.

Scores outside ``[0, 1]`` are the second symptom. They cannot corrupt AP by
themselves (only the ranking matters), but a "score" of 6.0 means the tensor
being read is not a score tensor, so it is treated as corruption too.

Degenerate (non-positive area) boxes are *reported* but not fatal -- a detector
can legitimately emit one.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path


class CorruptPredictionsError(ValueError):
    """Raised when a predictions file cannot be meaningfully scored."""


@dataclass(frozen=True)
class PredictionIntegrity:
    """Counters describing how well-formed a set of COCO predictions is."""

    total: int
    non_finite_boxes: int
    out_of_range_scores: int
    degenerate_boxes: int
    non_finite_scores: int
    score_min: float | None
    score_max: float | None

    @property
    def corrupt(self) -> bool:
        """Whether the predictions would produce meaningless metrics."""

        return bool(
            self.non_finite_boxes or self.out_of_range_scores or self.non_finite_scores
        )

    def to_dict(self) -> dict:
        data = asdict(self)
        data["corrupt"] = self.corrupt
        return data

    def describe(self, source: str | Path | None = None) -> str:
        where = f" in {source}" if source else ""

        parts = [f"{self.total} prediction(s){where}"]

        if self.non_finite_boxes:
            parts.append(f"{self.non_finite_boxes} with non-finite (NaN/inf) boxes")

        if self.non_finite_scores:
            parts.append(f"{self.non_finite_scores} with non-finite scores")

        if self.out_of_range_scores:
            parts.append(
                f"{self.out_of_range_scores} with scores outside [0, 1] "
                f"(range [{self.score_min}, {self.score_max}])"
            )

        if self.degenerate_boxes:
            parts.append(f"{self.degenerate_boxes} with non-positive area")

        return ", ".join(parts)


def prediction_integrity(predictions) -> PredictionIntegrity:
    """Summarize the well-formedness of a list of COCO prediction dicts."""

    non_finite_boxes = 0
    out_of_range_scores = 0
    non_finite_scores = 0
    degenerate_boxes = 0

    score_min: float | None = None
    score_max: float | None = None

    for prediction in predictions:
        box = prediction.get("bbox") or []

        values = [float(v) for v in box]

        if len(values) != 4 or not all(math.isfinite(v) for v in values):
            non_finite_boxes += 1
        elif values[2] <= 0 or values[3] <= 0:
            degenerate_boxes += 1

        score = float(prediction.get("score", 0.0))

        if not math.isfinite(score):
            non_finite_scores += 1
            continue

        if score < 0.0 or score > 1.0:
            out_of_range_scores += 1

        score_min = score if score_min is None else min(score_min, score)
        score_max = score if score_max is None else max(score_max, score)

    return PredictionIntegrity(
        total=len(predictions),
        non_finite_boxes=non_finite_boxes,
        out_of_range_scores=out_of_range_scores,
        degenerate_boxes=degenerate_boxes,
        non_finite_scores=non_finite_scores,
        score_min=score_min,
        score_max=score_max,
    )


#: Appended to every corruption message -- the cause is almost always upstream
#: of the evaluator, in the runtime that produced the predictions.
_HINT = (
    "This is a broken inference run, not a scoring problem: pycocotools turns a "
    "NaN IoU into a match at every IoU threshold, so such a run reports a high "
    "AP with AP == AP50 instead of failing. Benchmark fp32 models with "
    "--cpu. Re-run the benchmark; pass --allow-corrupt-predictions to score "
    "anyway (the numbers will be meaningless)."
)


def check_predictions(
    predictions,
    *,
    source: str | Path | None = None,
    strict: bool = True,
) -> PredictionIntegrity:
    """
    Validate predictions, raising :class:`CorruptPredictionsError` when unusable.

    With ``strict=False`` the same problem is reported as a warning and the
    caller may proceed, which is only useful for inspecting a known-bad run.
    """

    integrity = prediction_integrity(predictions)

    if not integrity.corrupt:
        return integrity

    message = f"Corrupt predictions: {integrity.describe(source)}. {_HINT}"

    if strict:
        raise CorruptPredictionsError(message)

    print(f"[warning] {message}")

    return integrity


__all__ = [
    "CorruptPredictionsError",
    "PredictionIntegrity",
    "check_predictions",
    "prediction_integrity",
]
