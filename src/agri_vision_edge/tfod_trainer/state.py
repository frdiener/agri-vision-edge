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

    # True strict maximum of the monitored metric -- drives checkpointing only.
    best_metric: float = -np.inf

    # Early-stopping stall counter and its own delta-gated reference. Decoupled
    # from `best_metric`: `es_ref` only advances on a gain past
    # `early_stopping_min_delta`, so sub-noise improvements don't keep the run
    # alive forever.
    patience_counter: int = 0
    es_ref: float = -np.inf

    # Reduce-LR-on-plateau bookkeeping, independent of both the checkpoint best
    # and the early-stopping counter. `plateau_ref` only advances on a gain past
    # `lr_plateau_min_delta`, so an optimizer jittering around a plateau (with
    # occasional microscopic new bests) still triggers an LR drop.
    plateau_counter: int = 0
    plateau_ref: float = -np.inf
    cooldown_counter: int = 0

    metrics_history: list[dict] = field(
        default_factory=list
    )
