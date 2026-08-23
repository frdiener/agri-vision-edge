"""
Tests for the trainer bookkeeping that survives a killed session.

The TF checkpoint in ``train_dir`` restores weights, optimizer slots and the
global step, so a resumed run *looks* correct while silently having thrown away
everything on ``TrainerState``. The failures that causes are all quiet ones:

- ``best_metric`` back at ``-inf``, so the first post-resume evaluation is
  checkpointed as "best" even when the earlier session reached a better model,
  and the export then ships the worse one;
- the plateau/cooldown counters back at zero, so the LR schedule restarts its
  patience and the run stops on a different rule than the one it started under
  -- which is exactly the confound the resolution ladder must not have;
- ``metrics_history`` back to empty, which (because the history file is written
  by overwriting) truncates the curves to the resumed session.

None of these raise, so they are guarded here instead.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

# `agri_vision_edge.tfod_trainer` imports the vendored object_detection package,
# which has to be put on the path first.
from agri_vision_edge.third_party import setup_tensorflow_models

setup_tensorflow_models()

try:
    from agri_vision_edge.tfod_trainer import setup as trainer_setup
    from agri_vision_edge.tfod_trainer.config import TrainingControlConfig
    from agri_vision_edge.tfod_trainer.run import FinetuneRunConfig, RunResult
    from agri_vision_edge.tfod_trainer.state import TrainerState
    from agri_vision_edge.tfod_trainer.training import TrainOutcome
except ImportError as exc:  # pragma: no cover - only without the vendored deps
    pytest.skip(f"object_detection unavailable: {exc}", allow_module_level=True)


def _advanced_state(train_dir):
    """A state that has actually been through some training."""
    state = TrainerState()
    state.best_metric = 0.4212
    state.es_ref = 0.4180
    state.plateau_ref = 0.4150
    state.patience_counter = 4
    state.plateau_counter = 7
    state.cooldown_counter = 3
    state.lr_floored = True
    state.min_lr_stall_counter = 1
    state.metrics_history = [{"step": 100}, {"step": 200}]
    state.best_checkpoint_path = str(train_dir / "ckpt-9")
    return state


def _touch_checkpoint(train_dir, name="ckpt-9"):
    """TF checkpoints are a path prefix; `.index` is the component that exists."""
    train_dir.mkdir(parents=True, exist_ok=True)
    (train_dir / f"{name}.index").touch()
    return train_dir / name


def test_round_trip_preserves_every_counter(tmp_path):
    train_dir = tmp_path / "train"
    _touch_checkpoint(train_dir)
    state = _advanced_state(train_dir)

    path = train_dir / "trainer_state.json"
    state.save(path)
    restored = TrainerState.load(path, train_dir=train_dir)

    assert restored.best_metric == pytest.approx(0.4212)
    assert restored.es_ref == pytest.approx(0.4180)
    assert restored.plateau_ref == pytest.approx(0.4150)
    assert restored.patience_counter == 4
    assert restored.plateau_counter == 7
    assert restored.cooldown_counter == 3
    assert restored.lr_floored is True
    assert restored.min_lr_stall_counter == 1
    assert restored.metrics_history == [{"step": 100}, {"step": 200}]


def test_cold_state_round_trips_through_strict_json(tmp_path):
    """
    A state saved before the first eval carries `-inf` in three fields, which
    `json` spells `-Infinity` -- not valid JSON, and not readable by anything
    stricter than Python's own parser.
    """
    path = tmp_path / "trainer_state.json"
    TrainerState().save(path)

    raw = path.read_text()
    assert "Infinity" not in raw
    json.loads(raw)  # strict parse

    restored = TrainerState.load(path)
    assert restored.best_metric == -np.inf
    assert restored.es_ref == -np.inf
    assert restored.plateau_ref == -np.inf


def test_best_checkpoint_path_is_rebased_onto_the_live_train_dir(tmp_path):
    """
    The stored path points into the previous session's train dir, which on
    Kaggle is a different directory. Only the file name carries over.
    """
    old_train_dir = tmp_path / "old" / "train"
    new_train_dir = tmp_path / "new" / "train"
    _touch_checkpoint(new_train_dir)

    state = _advanced_state(old_train_dir)
    path = tmp_path / "trainer_state.json"
    state.save(path)

    restored = TrainerState.load(path, train_dir=new_train_dir)
    assert restored.best_checkpoint_path == str(new_train_dir / "ckpt-9")


def test_missing_best_checkpoint_is_dropped_not_kept(tmp_path):
    """
    If the checkpoint was not carried over, the warm restart must degrade to
    "reduce the LR without restoring" rather than restore a dead path.
    """
    train_dir = tmp_path / "train"
    train_dir.mkdir(parents=True)
    state = _advanced_state(train_dir)

    path = tmp_path / "trainer_state.json"
    state.save(path)

    restored = TrainerState.load(path, train_dir=train_dir)
    assert restored.best_checkpoint_path is None


def test_history_file_wins_when_it_is_ahead(tmp_path):
    """
    The history is written before the counters settle, so a crash in between
    leaves it one record ahead. Losing that record would put a gap in the curve.
    """
    train_dir = tmp_path / "train"
    _touch_checkpoint(train_dir)
    state = _advanced_state(train_dir)

    path = train_dir / "trainer_state.json"
    state.save(path)

    history_path = train_dir / "metrics_history.json"
    history_path.write_text(
        json.dumps([{"step": 100}, {"step": 200}, {"step": 300}])
    )

    restored = TrainerState.load(
        path, train_dir=train_dir, history_path=history_path
    )
    assert restored.metrics_history == [
        {"step": 100},
        {"step": 200},
        {"step": 300},
    ]


def test_state_history_wins_when_the_file_is_behind(tmp_path):
    """The merge is "longest wins", not "the file always wins"."""
    train_dir = tmp_path / "train"
    _touch_checkpoint(train_dir)
    state = _advanced_state(train_dir)

    path = train_dir / "trainer_state.json"
    state.save(path)

    history_path = train_dir / "metrics_history.json"
    history_path.write_text(json.dumps([{"step": 100}]))

    restored = TrainerState.load(
        path, train_dir=train_dir, history_path=history_path
    )
    assert len(restored.metrics_history) == 2


def test_unknown_keys_are_ignored(tmp_path):
    """A state file written by a newer trainer must not crash an older one."""
    path = tmp_path / "trainer_state.json"
    payload = TrainerState().to_mapping()
    payload["some_future_counter"] = 12
    path.write_text(json.dumps(payload))

    restored = TrainerState.load(path)
    assert restored.patience_counter == 0


# ---------------------------------------------------------------------------
# Which checkpoint a resumed run comes back from
# ---------------------------------------------------------------------------


class _FakeCheckpoint:
    def __init__(self):
        self.restored = None

    def restore(self, path):
        self.restored = path


class _FakeManager:
    def __init__(self, latest):
        self.latest_checkpoint = latest


class _FakeStep:
    def numpy(self):
        return 4242


class _FakeRuntime:
    """Just enough of `Runtime` for `restore_weights` without a real graph."""

    use_moving_average = False

    def __init__(self, last, best):
        self.last_manager = _FakeManager(last)
        self.manager = _FakeManager(best)
        self.ckpt = _FakeCheckpoint()
        self.global_step = _FakeStep()
        self.unpad_groundtruth_tensors = False


def test_resume_prefers_the_rolling_snapshot_over_the_best(monkeypatch):
    """
    `manager` only saves on an improvement, so after a long non-improving tail
    its latest checkpoint is far behind. Resuming from it would rewind the run
    and redo that tail -- with the plateau counters already spent, so the
    resumed session would re-derive the same best instead of training further.
    """
    calls = []
    monkeypatch.setattr(
        trainer_setup,
        "maybe_load_fine_tune_checkpoint",
        lambda *a, **k: calls.append("cold"),
    )

    runtime = _FakeRuntime(last="/train/last/ckpt-5", best="/train/ckpt-2")
    resumed = trainer_setup.restore_weights(None, runtime, None)

    assert resumed is True
    assert runtime.ckpt.restored == "/train/last/ckpt-5"
    assert calls == []


def test_resume_falls_back_to_the_best_checkpoint(monkeypatch):
    """Runs from before the rolling snapshot existed still resume."""
    monkeypatch.setattr(
        trainer_setup, "maybe_load_fine_tune_checkpoint", lambda *a, **k: None
    )

    runtime = _FakeRuntime(last=None, best="/train/ckpt-2")
    assert trainer_setup.restore_weights(None, runtime, None) is True
    assert runtime.ckpt.restored == "/train/ckpt-2"


def test_cold_start_reports_no_resume(monkeypatch):
    """
    An empty train dir must report False, so the caller does not apply a stale
    state file to freshly initialised weights.
    """
    calls = []
    monkeypatch.setattr(
        trainer_setup,
        "maybe_load_fine_tune_checkpoint",
        lambda *a, **k: calls.append("cold"),
    )

    runtime = _FakeRuntime(last=None, best=None)
    assert trainer_setup.restore_weights(None, runtime, None) is False
    assert runtime.ckpt.restored is None
    assert calls == ["cold"]


# ---------------------------------------------------------------------------
# Telling a finished run from one that merely ran out of clock
# ---------------------------------------------------------------------------


def test_budget_stop_is_not_convergence():
    assert TrainOutcome().converged is True
    assert TrainOutcome(budget_exhausted=True).converged is False


def test_run_result_without_an_outcome_counts_as_converged():
    """
    `outcome` is optional, so a RunResult built by older code (or by hand in a
    test) must not be mistaken for an unfinished run -- that would make a
    notebook publish its train dir forever.
    """
    result = RunResult(
        pipeline_config="p",
        train_dir="t",
        best_metric_path="b",
        history_path="h",
        detection_model=None,
        configs={},
    )
    assert result.converged is True

    unfinished = RunResult(
        pipeline_config="p",
        train_dir="t",
        best_metric_path="b",
        history_path="h",
        detection_model=None,
        configs={},
        outcome=TrainOutcome(budget_exhausted=True),
    )
    assert unfinished.converged is False


def test_runtime_budget_survives_the_manifest_round_trip():
    """
    The quant notebooks rebuild their config from `stages.finetune.config` in
    the manifest, so a knob that does not survive `to_mapping` silently reverts
    to its default on every downstream run.
    """
    config = FinetuneRunConfig(
        model_path="m",
        dataset_bundle_path="d",
        num_classes=2,
        output_dir="o",
        control=TrainingControlConfig(lr_plateau=True, max_runtime_hours=10.5),
    )

    mapping = config.to_mapping()
    json.dumps(mapping)  # must be manifest-serialisable

    assert mapping["control"]["max_runtime_hours"] == 10.5
    assert (
        FinetuneRunConfig.from_mapping(mapping).control.max_runtime_hours == 10.5
    )


def test_runtime_budget_defaults_to_off():
    """Every rung that fits a session must be unaffected by this feature."""
    assert TrainingControlConfig().max_runtime_hours is None
