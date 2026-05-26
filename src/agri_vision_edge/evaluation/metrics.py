from __future__ import annotations

from dataclasses import asdict
from dataclasses import dataclass


@dataclass
class CocoMetrics:

    #
    # Average precision
    #

    map: float
    map50: float
    map75: float

    aps: float
    apm: float
    apl: float

    #
    # Average recall
    #

    ar1: float
    ar10: float
    ar100: float

    ars: float
    arm: float
    arl: float

    #
    # Metadata
    #

    step: int | None = None

    evaluator: str | None = None

    selection_metric: str | None = None

    selection_value: float | None = None

    def to_dict(self):

        return asdict(self)
