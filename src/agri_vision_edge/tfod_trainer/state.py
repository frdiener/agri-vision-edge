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

    # Reduce-LR-on-plateau bookkeeping (independent of the early-stopping
    # `patience_counter`, so LR drops can happen well before early stopping).
    plateau_counter: int = 0
    cooldown_counter: int = 0

    metrics_history: list[dict] = field(
        default_factory=list
    )
