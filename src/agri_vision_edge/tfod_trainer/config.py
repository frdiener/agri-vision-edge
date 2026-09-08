"""
Configuration objects for TFOD training.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(slots=True)
class TrainingControlConfig:
    """
    Custom training-loop control policy.

    Defines metric checkpointing, early stopping, the plateau schedule, and
    graph-modification flags for the custom loop. Pipeline semantics belong in
    ``FineTuneConfig``; orchestration and paths belong in ``FinetuneRunConfig``.
    """

    log_every: int = 100

    # Optional override for the training batch size. When set, it overwrites the
    # pipeline's train_config batch_size before rendering, letting the batch size
    # be configured here alongside the other schedule knobs. Because it feeds
    # steps_per_epoch (= ceil(train_samples / batch_size)), it is resolved before
    # the epoch-derived horizons (max_epochs / warmup_epochs). None (the default)
    # leaves FineTuneConfig.batch_size untouched.
    batch_size: int | None = None

    # Epoch-based evaluation cadence. steps_per_epoch is
    # ceil(train_samples / batch_size), using bundle metadata and the pipeline
    # batch size. Evaluation and stopping bookkeeping run every
    # `eval_every_epochs` epochs and once at the final step. Older bundles
    # without train_samples use the `log_every` step cadence.
    eval_every_epochs: float = 1.0

    # Optional training length expressed in epochs. When set, it OVERRIDES the
    # pipeline's num_steps entirely: the horizon becomes exactly
    # max_epochs * steps_per_epoch (a whole-epoch boundary), so num_steps can be
    # left at its large default and ignored. None (the default) leaves num_steps
    # in charge of the horizon. Ignored when train_samples is unavailable (the
    # trainer then falls back to num_steps).
    max_epochs: int | None = None

    # Optional warmup length expressed in epochs. When set (and the epoch
    # geometry is known), it OVERRIDES the pipeline's warmup_steps:
    # warmup_steps = round(warmup_epochs * steps_per_epoch). This drives both the
    # cosine warmup and the plateau schedule's warmup ramp (which reads
    # warmup_steps back from the rendered pipeline). None (the default) leaves
    # warmup_steps as configured. Ignored when train_samples is unavailable.
    warmup_epochs: float | None = None

    checkpoint_max_to_keep: int = 3

    metric_name: str = "DetectionBoxes_Precision/mAP"

    save_metrics_history: bool = True

    # Custom metric-based early stopping (patience counted in eval intervals).
    # 0 disables the stop but continues logging the non-improvement counter as
    # `patience=N/off`.
    early_stopping_patience: int = 0
    early_stopping_min_delta: float = 0.0

    # Evaluate the restored weights once before the first train step, seeding
    # the best-metric tracker with that baseline and checkpointing it. This
    # guarantees the exported "best" checkpoint is never worse than the starting
    # weights: a reduced-schedule refinement (e.g. the PTQ float base resuming a
    # converged finetune) that only ever regresses will export the baseline
    # itself instead of a checkpoint below it.
    initial_eval_checkpoint: bool = False

    # Metric-driven plateau schedule layered on the best-metric and
    # early-stopping tracker. The mutable LR warms from the pipeline value to its
    # base, then drops by `lr_plateau_factor` after `lr_plateau_patience`
    # non-improving evaluations. `lr_plateau_min_lr` sets the floor and
    # `lr_plateau_cooldown` delays subsequent drops. Conservative defaults avoid
    # reducing the LR before each level has stabilized.
    lr_plateau: bool = False
    lr_plateau_factor: float = 0.5
    lr_plateau_patience: int = 15
    lr_plateau_cooldown: int = 5
    lr_plateau_min_lr: float = 1e-6

    # Optional single-source overrides for the base and warmup learning rates.
    # When set they overwrite the pipeline's cosine `learning_rate_base` /
    # `warmup_learning_rate` before rendering, so both the cosine schedule and
    # the plateau schedule (which reads these back from the rendered pipeline via
    # `extract_lr_params`) pick them up. This keeps LR controls together. None
    # (the default) leaves the finetune config's values untouched. Named for the
    # plateau schedule (its usual caller) but they overwrite the pipeline keys
    # unconditionally.
    lr_plateau_base_lr: float | None = None
    lr_plateau_warmup_lr: float | None = None

    # Minimum metric gain that counts as an improvement for the plateau stall
    # counter (absolute, in metric units). Decoupled from checkpointing: the best
    # checkpoint still tracks the true strict maximum, but the plateau counter
    # only resets on a gain larger than this, so an optimizer that jitters around
    # a plateau while occasionally nudging a microscopic new best still triggers
    # an LR drop. Set to roughly the eval noise floor (COCO mAP on a small val
    # set jitters ~1e-3). 0.0 (the default) means "any improvement resets it".
    lr_plateau_min_delta: float = 0.0

    # On each plateau LR drop, restore the best checkpoint first (a "warm
    # restart": resume the best weights + optimizer slots, keep the current step
    # count) before applying the lower LR. Turns each drop into "rewind to the
    # best point, then refine more gently".
    lr_plateau_restore_best: bool = True

    # Stop after this many stalls at `lr_plateau_min_lr`; further reductions are
    # impossible. A value of 0 disables this stop. The
    # global `early_stopping_patience` still applies as a hard cap). Counted in
    # floored-stall events, each ~`lr_plateau_patience` (+cooldown) evals apart,
    # so this fires well before the generous global patience meant to span the
    # LR annealing.
    lr_plateau_exhausted_patience: int = 0

    # Wall-clock budget for the training loop, in hours. When the budget is
    # spent the run stops *gracefully* at the next evaluation boundary, exactly
    # as if a stopping rule had fired, so the notebook goes on to export and
    # publish its artifacts.
    #
    # Hosted sessions discard all output when they exceed the platform limit.
    # Set this budget low enough to publish checkpoints and `trainer_state.json`
    # for a later session. Checks occur at evaluation boundaries, so allow one
    # train-plus-evaluation interval plus export and upload time.
    #
    # None disables the wall-clock limit and leaves termination to metric rules.
    max_runtime_hours: float | None = None

    # Enable quantization-aware training. False = plain finetune (-> PTQ at
    # conversion). True = the full int8 scheme: BatchNorms folded into the convs,
    # backbone + SSD head fake-quantized up to the float postprocess (see
    # agri_vision_edge.tfod.qat). The only QAT variant we keep.
    qat: bool = False

    # Per-channel weight quantization for QAT. Default False = per-tensor
    # (required by the i.MX8M Plus Vivante/Teflon NPU). Set True for targets that
    # accept per-channel weights (i.MX93 Arm Ethos-U65), where it is usually a
    # touch more accurate. Maps to `per_channel` on quantize_backbone (it selects
    # the pin placement; the converter, not the fake-quant, emits per-channel).
    qat_per_channel: bool = False

    # Treat PhenoBench partial ("do-not-care") plants as do-not-care during the
    # continuous eval, matching the official protocol and the `ave evaluate
    # --ignore-partials` final eval. Partials are carried in the val.record as
    # `is_partial` (mirrored to `groundtruth_is_crowd`), which the TFOD COCO
    # evaluator natively treats as ignore regions. When True the markers are
    # honoured (a detection on a partial plant is not a false positive); when
    # False (the default, strict) the markers are cleared so partials are scored
    # like any other ground-truth. Only has an effect on records built with
    # partials (include_partials=True); records without them are unaffected.
    eval_ignore_partials: bool = False


@dataclass(slots=True)
class TrainerConfig:
    """
    High-level training configuration.

    Independent from TFOD protobuf configuration. Only needs a rendered pipeline
    config + a train dir; every other knob lives in ``control``
    (a :class:`TrainingControlConfig`), so this stays a narrow, model-source- and
    UI-agnostic contract the trainer can be driven with directly.
    """

    pipeline_config: Path
    train_dir: Path

    control: TrainingControlConfig = field(default_factory=TrainingControlConfig)

    @property
    def history_path(self) -> Path:
        return self.train_dir / "metrics_history.json"

    @property
    def best_metric_path(self) -> Path:
        return self.train_dir / "best_metric.json"

    @property
    def state_path(self) -> Path:
        """
        Trainer bookkeeping (best metric, stall counters, history), written
        every eval so a killed run can resume without restarting its schedule.
        Lives in ``train_dir`` next to the checkpoints it belongs to, so
        carrying the train dir between sessions carries the state with it.
        """
        return self.train_dir / "trainer_state.json"
