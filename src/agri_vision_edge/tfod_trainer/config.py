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

    The single home for every knob that drives OUR training loop rather than the
    TFOD protobuf pipeline: metric-based checkpointing, early stopping, the
    reduce-LR-on-plateau schedule, and the graph-modification flags (optimizer
    reset + QAT). None of these are pipeline/model semantics (those live in
    ``FineTuneConfig``) nor run orchestration/paths (those live in
    ``FinetuneRunConfig``); they are the trainer's own contract, so they are
    declared here exactly once and consumed via ``TrainerConfig.control``.
    """

    log_every: int = 100

    checkpoint_max_to_keep: int = 3

    metric_name: str = "DetectionBoxes_Precision/mAP"

    save_metrics_history: bool = True

    # Custom metric-based early stopping (patience counted in eval intervals).
    # 0 disables the stop (the default): the non-improvement counter is still
    # advanced and logged (as `patience=N/off`), it just never terminates the
    # run -- so the LR-plateau schedule is left to decide when to stop, while the
    # counter stays visible for diagnostics.
    early_stopping_patience: int = 0
    early_stopping_min_delta: float = 0.0

    # Rebuild a fresh optimizer / LR schedule before training. Tri-state: None
    # ("auto") is resolved by ``FinetuneRunConfig`` to True under QAT or
    # resume_full and False otherwise; explicit True/False wins. When a
    # ``TrainerConfig`` is built directly (no master), None is treated as False.
    reset_optimizer: bool | None = None

    # Evaluate the restored weights once before the first train step, seeding
    # the best-metric tracker with that baseline and checkpointing it. This
    # guarantees the exported "best" checkpoint is never worse than the starting
    # weights: a reduced-schedule refinement (e.g. the PTQ float base resuming a
    # converged finetune) that only ever regresses will export the baseline
    # itself instead of a checkpoint below it.
    initial_eval_checkpoint: bool = False

    # Metric-driven "reduce LR on plateau" schedule, layered on top of the
    # existing best-metric / early-stopping tracker. When enabled the LR becomes
    # a mutable tf.Variable (see tfod_trainer.setup): it warms up from the
    # pipeline's warmup LR to its base, then -- each time the monitored metric
    # fails to improve for `lr_plateau_patience` consecutive evals -- is
    # multiplied by `lr_plateau_factor` (floored at `lr_plateau_min_lr`), after a
    # `lr_plateau_cooldown` grace period following each drop. Decouples LR
    # annealing from a guessed `num_steps` horizon, which is exactly what the
    # cosine schedule cannot do once early stopping cuts the run short.
    # Defaults tuned for accuracy over run time (empirically: aggressive
    # patience/cooldown + a >0 min_delta collapse the LR before the model has
    # exploited each level, and hurt final mAP). Keep each LR level long and only
    # drop on a genuine flat; let num_steps bound the run rather than an
    # aggressive exhausted-stall stop.
    lr_plateau: bool = False
    lr_plateau_factor: float = 0.5
    lr_plateau_patience: int = 15
    lr_plateau_cooldown: int = 5
    lr_plateau_min_lr: float = 1e-6

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

    # Stop once the LR schedule is spent: after the plateau logic tries to reduce
    # but is already at `lr_plateau_min_lr`, that is a "floored stall" -- further
    # drops cannot help. Stop after this many such events (0 disables it; the
    # global `early_stopping_patience` still applies as a hard cap). Counted in
    # floored-stall events, each ~`lr_plateau_patience` (+cooldown) evals apart,
    # so this fires well before the generous global patience meant to span the
    # LR annealing.
    lr_plateau_exhausted_patience: int = 2

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
