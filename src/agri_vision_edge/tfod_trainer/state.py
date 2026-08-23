"""
Mutable trainer state.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, fields
from pathlib import Path

import numpy as np


@dataclass
class TrainerState:
    """
    Tracks training progress.

    Persisted to ``train_dir/trainer_state.json`` on every evaluation so a run
    that is killed mid-training (the Kaggle 12 h limit, typically) can be
    resumed in a later session *with its bookkeeping intact*. The TF checkpoint
    in ``train_dir`` already restores weights, optimizer slots and the global
    step; what it does not carry is everything on this object -- the best metric
    seen so far, the delta-gated stall references, the plateau/cooldown counters
    and the metrics history. Without them a resumed run would re-seed
    ``best_metric`` from ``-inf`` (so the first post-resume eval would be
    checkpointed as "best" even if it is worse than what the earlier session
    reached), restart the plateau schedule's patience from zero, and -- because
    ``metrics_history`` is written by overwriting -- truncate the curves to just
    the second session.

    See :meth:`save` / :meth:`load`.
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

    # LR-exhausted stop: `lr_floored` latches once a reduction reaches
    # `lr_plateau_min_lr`; `min_lr_stall_counter` then counts "floored stall"
    # events (plateau triggers that can no longer lower the LR) so the run can
    # stop early once further annealing is impossible.
    lr_floored: bool = False
    min_lr_stall_counter: int = 0

    # Path of the checkpoint holding `best_metric`'s weights, used by the
    # `lr_plateau_restore_best` warm restart. Absolute, so it is rebased onto
    # the live train dir on load (a resumed session's train dir is rarely at
    # the path the previous one wrote).
    best_checkpoint_path: str | None = None

    metrics_history: list[dict] = field(
        default_factory=list
    )

    # -- persistence ----------------------------------------------------

    #: `-inf` has no portable JSON spelling (`json` emits `-Infinity`, which is
    #: not valid JSON and trips strict readers), so it round-trips as null.
    _NEG_INF_KEYS = ("best_metric", "es_ref", "plateau_ref")

    def to_mapping(self) -> dict:
        data = {
            f.name: getattr(self, f.name)
            for f in fields(self)
        }
        for key in self._NEG_INF_KEYS:
            value = data[key]
            data[key] = None if value == -np.inf else float(value)
        return data

    @classmethod
    def from_mapping(cls, data: dict) -> TrainerState:
        data = dict(data)
        for key in cls._NEG_INF_KEYS:
            if data.get(key) is None:
                data[key] = -np.inf
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in known})

    def save(self, path) -> None:
        Path(path).write_text(
            json.dumps(self.to_mapping(), indent=2)
        )

    @classmethod
    def load(cls, path, train_dir=None, history_path=None) -> TrainerState:
        """
        Restore a state written by :meth:`save`.

        ``train_dir`` rebases :attr:`best_checkpoint_path`: the stored path is
        absolute and points into the *previous* session's train dir, which on
        Kaggle is a different directory (or gone entirely). Only the file name
        is meaningful, and only if the checkpoint was actually carried over --
        otherwise the reference is dropped, which degrades the warm restart to
        "reduce the LR without restoring" rather than crashing on a dead path.

        ``history_path`` guards against losing an evaluation: the history file
        is written before the counters are updated, so a crash in between
        leaves it one record ahead of the state file. The longer of the two
        wins, so the curves stay complete.
        """
        state = cls.from_mapping(json.loads(Path(path).read_text()))

        if state.best_checkpoint_path and train_dir is not None:
            rebased = Path(train_dir) / Path(state.best_checkpoint_path).name
            # TF checkpoints are a prefix, not a file; `.index` is the one
            # component guaranteed to exist under it.
            state.best_checkpoint_path = (
                str(rebased)
                if rebased.with_suffix(".index").exists()
                else None
            )

        if history_path is not None and Path(history_path).is_file():
            history = json.loads(Path(history_path).read_text())
            if len(history) > len(state.metrics_history):
                state.metrics_history = history

        return state
