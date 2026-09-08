"""Reject malformed COCO predictions before scoring.

``pycocotools`` treats a NaN IoU comparison as a match, so delegated fp32 runs
with NaN boxes can report implausibly high AP and ``AP == AP50`` instead of
failing. Non-finite boxes or scores and scores outside ``[0, 1]`` are therefore
fatal. Degenerate boxes are reported but allowed.
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


#: Appended to corruption messages caused by invalid runtime predictions.
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
