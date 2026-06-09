"""
Mutable trainer state.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np


@dataclass
class TrainerState:
    """
    Tracks training progress.
    """

    best_metric: float = -np.inf

    patience_counter: int = 0

    metrics_history: list[dict] = field(
        default_factory=list
    )
