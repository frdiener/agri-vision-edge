"""
One-call finetune / QAT run, driven by a single config object (or dict).

This is the reusable core behind ``notebooks/finetuning.py``: the notebook is
a thin marimo UI over ``FinetuneRunConfig`` + ``run_finetune``, and the same
two can be driven head-less from Python::

    from agri_vision_edge.tfod_trainer import FinetuneRunConfig, run_finetune

    run_finetune(FinetuneRunConfig(
        model_path="models/ssd_mobilenet_v2_320x320_coco17_tpu-8",
        dataset_bundle_path="datasets/phenobench_sc_tiled",
        num_classes=1,
        output_dir="runs/finetune",
    ))

QAT is the same call with ``qat=True`` (and optionally ``qat_per_channel`` /
``reset_optimizer``) set.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from agri_vision_edge.experiment import AugmentationConfig, FineTuneConfig

from .config import TrainerConfig


@dataclass
class FinetuneRunConfig:
    """
    Everything needed to run one finetune (or QAT) job.

    Required:
        model_path:
            Pretrained base model directory. Must contain ``pipeline.config``
            and ``checkpoint/ckpt-0.*`` (the TF model-zoo layout).
        dataset_bundle_path:
            Dataset directory containing ``label_map.pbtxt``, ``train.record``
            and ``val.record``.
        num_classes:
            Number of detection classes.
        output_dir:
            Where the as-run pipeline config and checkpoints are written.

    The semantic pipeline tuning lives in ``finetune`` (a ``FineTuneConfig``);
    the remaining fields fill in the ``TrainerConfig`` knobs, including the
    QAT options (``qat`` / ``qat_per_channel`` / ``reset_optimizer``).
    """

    model_path: Path
    dataset_bundle_path: Path
    num_classes: int
    output_dir: Path

    finetune: FineTuneConfig = field(default_factory=FineTuneConfig)

    # TrainerConfig knobs (early-stopping comes from `finetune`).
    log_every: int = 100
    checkpoint_max_to_keep: int = 3
    metric_name: str = "DetectionBoxes_Precision/mAP"
    save_metrics_history: bool = True

    # QAT. qat=False => plain finetune (-> PTQ at conversion). qat=True => the
    # full int8 scheme (fold BN + fake-quant backbone + head). reset_optimizer is
    # tri-state: None ("auto") resolves to True under QAT or resume_full (both
    # want a fresh optimizer/LR schedule) and False otherwise; explicit True/
    # False wins.
    qat: bool = False
    reset_optimizer: bool | None = None

    # Resume OUR OWN converged export (matching num_classes), restoring the
    # box/class prediction heads too (fine_tune_checkpoint_type="full"), as
    # opposed to bootstrapping from a foreign detection checkpoint (e.g. COCO,
    # different num_classes) whose heads must be dropped and reinitialised
    # ("detection"). QAT always resumes our own export, so it implies this; a
    # plain-float PTQ base (qat=False) that resumes the finetune export must set
    # it explicitly, otherwise its heads are dropped and it retrains from cold
    # (near-zero AP, stuck loss). Independent of quantization.
    resume_full: bool = False

    # Per-channel QAT weight quantization. Default False = per-tensor (i.MX8M
    # Plus Vivante/Teflon NPU). Set True for i.MX93 Arm Ethos-U65, which accepts
    # per-channel weights. The conversion + export reproduce this flag.
    qat_per_channel: bool = False

    def __post_init__(self):
        self.model_path = Path(self.model_path)
        self.dataset_bundle_path = Path(self.dataset_bundle_path)
        self.output_dir = Path(self.output_dir)

        if self.reset_optimizer is None:
            self.reset_optimizer = self.qat or self.resume_full

    # --- derived paths -------------------------------------------------

    @property
    def base_pipeline_config(self) -> Path:
        return self.model_path / "pipeline.config"

    @property
    def base_checkpoint(self) -> Path:
        return self.model_path / "checkpoint" / "ckpt-0"

    @property
    def label_map(self) -> Path:
        return self.dataset_bundle_path / "label_map.pbtxt"

    @property
    def train_record(self) -> Path:
        return self.dataset_bundle_path / "train.record"

    @property
    def val_record(self) -> Path:
        return self.dataset_bundle_path / "val.record"

    @property
    def pipeline_config_path(self) -> Path:
        return self.output_dir / "finetune" / "pipeline.config"

    @property
    def train_dir(self) -> Path:
        return self.output_dir / "train"

    # --- construction from a plain mapping (e.g. JSON / UI dict) --------

    @classmethod
    def from_mapping(cls, data) -> FinetuneRunConfig:
        """
        Build from a plain dict, expanding a nested ``finetune`` (and its
        ``augmentation``) sub-dict into the proper dataclasses.
        """
        data = dict(data)

        finetune = data.get("finetune")
        if isinstance(finetune, dict):
            finetune = dict(finetune)
            augmentation = finetune.get("augmentation")
            if isinstance(augmentation, dict):
                finetune["augmentation"] = AugmentationConfig(**augmentation)
            data["finetune"] = FineTuneConfig(**finetune)

        return cls(**data)

    def to_trainer_config(self) -> TrainerConfig:
        """Map onto the lower-level ``TrainerConfig`` the trainer consumes."""
        return TrainerConfig(
            pipeline_config=self.pipeline_config_path,
            train_dir=self.train_dir,
            log_every=self.log_every,
            checkpoint_max_to_keep=self.checkpoint_max_to_keep,
            metric_name=self.metric_name,
            early_stopping_patience=self.finetune.early_stopping_patience,
            early_stopping_min_delta=self.finetune.early_stopping_min_delta,
            save_metrics_history=self.save_metrics_history,
            reset_optimizer=self.reset_optimizer,
            qat=self.qat,
            qat_per_channel=self.qat_per_channel,
        )


@dataclass
class RunResult:
    """Handles to the artifacts and live objects produced by a run."""

    pipeline_config: Path
    train_dir: Path
    best_metric_path: Path
    history_path: Path
    detection_model: object
    configs: dict


def write_pipeline(cfg: FinetuneRunConfig) -> Path:
    """
    Render the as-run TFOD pipeline config for ``cfg`` and return its path.

    Split out so a caller (e.g. the notebook) can inspect / preview the
    pipeline before committing to a full training run.
    """
    from agri_vision_edge.tfod import configure_ssd_pipeline

    cfg.pipeline_config_path.parent.mkdir(parents=True, exist_ok=True)

    # Resuming our own full model (QAT, or a resume_full PTQ base, from a finetune
    # export) must restore the box/class prediction heads too, so use "full". A
    # plain finetune bootstraps from a foreign detection checkpoint (e.g. COCO,
    # different num_classes) where the heads must be dropped and reinitialised, so
    # it stays "detection".
    fine_tune_checkpoint_type = "full" if (cfg.qat or cfg.resume_full) else "detection"

    configure_ssd_pipeline(
        config=cfg.finetune,
        config_path=cfg.base_pipeline_config,
        output_path=cfg.pipeline_config_path,
        train_record=cfg.train_record,
        val_record=cfg.val_record,
        label_map=cfg.label_map,
        checkpoint_path=cfg.base_checkpoint,
        num_classes=cfg.num_classes,
        fine_tune_checkpoint_type=fine_tune_checkpoint_type,
    )

    return cfg.pipeline_config_path


def run_finetune(cfg) -> RunResult:
    """
    Render the pipeline, build the model + runtime, and train.

    ``cfg`` may be a ``FinetuneRunConfig`` or a plain dict (which is passed
    through ``FinetuneRunConfig.from_mapping``).
    """
    from agri_vision_edge.third_party import setup_tensorflow_models

    setup_tensorflow_models()

    from .setup import (
        build_detection_model,
        create_runtime,
        load_pipeline_configs,
    )
    from .training import train

    if not isinstance(cfg, FinetuneRunConfig):
        cfg = FinetuneRunConfig.from_mapping(cfg)

    write_pipeline(cfg)
    cfg.train_dir.mkdir(parents=True, exist_ok=True)

    trainer_cfg = cfg.to_trainer_config()

    configs = load_pipeline_configs(trainer_cfg.pipeline_config)
    detection_model = build_detection_model(configs)
    runtime = create_runtime(
        detection_model,
        configs,
        trainer_cfg.train_dir,
        checkpoint_max_to_keep=cfg.checkpoint_max_to_keep,
    )

    train(detection_model, runtime, trainer_cfg)

    return RunResult(
        pipeline_config=cfg.pipeline_config_path,
        train_dir=cfg.train_dir,
        best_metric_path=trainer_cfg.best_metric_path,
        history_path=trainer_cfg.history_path,
        detection_model=detection_model,
        configs=configs,
    )
