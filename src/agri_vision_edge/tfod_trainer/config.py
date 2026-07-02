"""
Configuration objects for TFOD training.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class TrainerConfig:
    """
    High-level training configuration.

    Independent from TFOD protobuf configuration.
    """

    pipeline_config: Path
    train_dir: Path

    log_every: int = 100

    checkpoint_max_to_keep: int = 3

    metric_name: str = "DetectionBoxes_Precision/mAP"

    early_stopping_patience: int = 10
    early_stopping_min_delta: float = 0.0

    save_metrics_history: bool = True

    reset_optimizer: bool = False

    # Evaluate the restored weights once before the first train step, seeding
    # the best-metric tracker with that baseline and checkpointing it. This
    # guarantees the exported "best" checkpoint is never worse than the starting
    # weights: a reduced-schedule refinement (e.g. the PTQ float base resuming a
    # converged finetune) that only ever regresses will export the baseline
    # itself instead of a checkpoint below it.
    initial_eval_checkpoint: bool = False

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

    @property
    def history_path(self) -> Path:
        return self.train_dir / "metrics_history.json"

    @property
    def best_metric_path(self) -> Path:
        return self.train_dir / "best_metric.json"
