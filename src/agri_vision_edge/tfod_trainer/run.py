"""Configure and run TFOD finetuning or quantization-aware training."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from math import ceil
from pathlib import Path

from agri_vision_edge.experiment import AugmentationConfig, FineTuneConfig

from .config import TrainerConfig, TrainingControlConfig

# Legacy flat keys that used to live directly on ``FinetuneRunConfig`` (or, for
# early stopping, on the nested ``FineTuneConfig``). ``from_mapping`` folds them
# into ``control`` so historical manifests / head-less dicts keep loading.
_LEGACY_CONTROL_KEYS = frozenset(
    f.name
    for f in TrainingControlConfig.__dataclass_fields__.values()  # type: ignore[attr-defined]
)


@dataclass
class FinetuneRunConfig:
    """Master configuration for one finetuning or QAT job.

    ``resume_full`` restores prediction heads from a matching exported model;
    leave it false when bootstrapping from a foreign detection checkpoint. QAT
    always performs a full-model resume.
    """

    model_path: Path
    dataset_bundle_path: Path
    num_classes: int
    output_dir: Path

    finetune: FineTuneConfig = field(default_factory=FineTuneConfig)
    control: TrainingControlConfig = field(default_factory=TrainingControlConfig)

    resume_full: bool = False

    def __post_init__(self):
        self.model_path = Path(self.model_path)
        self.dataset_bundle_path = Path(self.dataset_bundle_path)
        self.output_dir = Path(self.output_dir)

    # QAT convenience properties

    @property
    def qat(self) -> bool:
        return self.control.qat

    @property
    def qat_per_channel(self) -> bool:
        return self.control.qat_per_channel

    # Derived paths

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
    def dataset_metadata(self) -> Path:
        return self.dataset_bundle_path / "dataset_metadata.json"

    @property
    def pipeline_config_path(self) -> Path:
        return self.output_dir / "finetune" / "pipeline.config"

    @property
    def train_dir(self) -> Path:
        return self.output_dir / "train"

    # Manifest serialization

    @classmethod
    def from_mapping(cls, data) -> FinetuneRunConfig:
        """Build from a mapping, expanding nested configuration dataclasses.

        Legacy flat control keys are folded into ``control`` and obsolete
        ``reset_optimizer`` values are ignored.
        """
        data = dict(data)

        # Collect control knobs from an explicit nested "control" plus any legacy
        # flat top-level keys (the flat keys lose to an explicit nested value).
        control_data = dict(data.pop("control", {}) or {})
        for key in list(data):
            if key in _LEGACY_CONTROL_KEYS:
                control_data.setdefault(key, data.pop(key))

        finetune = data.get("finetune")
        if isinstance(finetune, dict):
            finetune = dict(finetune)
            # Legacy: early stopping used to live on FineTuneConfig.
            for key in ("early_stopping_patience", "early_stopping_min_delta"):
                if key in finetune:
                    control_data.setdefault(key, finetune.pop(key))
            augmentation = finetune.get("augmentation")
            if isinstance(augmentation, dict):
                finetune["augmentation"] = AugmentationConfig(**augmentation)
            data["finetune"] = FineTuneConfig(**finetune)

        # `reset_optimizer` became obsolete when PTQ/QAT began resuming from
        # model-only exports. Drop legacy keys so old manifests still load.
        data.pop("reset_optimizer", None)
        control_data.pop("reset_optimizer", None)

        if control_data:
            data["control"] = TrainingControlConfig(**control_data)

        return cls(**data)

    def to_mapping(self) -> dict:
        """
        Serialize the whole config to a plain (JSON-friendly) dict.

        Round-trips through :meth:`from_mapping`; this is exactly what the
        notebooks commit to the experiment manifest so a run is fully
        reconstructable. Paths are stringified.
        """
        return {
            "model_path": str(self.model_path),
            "dataset_bundle_path": str(self.dataset_bundle_path),
            "num_classes": self.num_classes,
            "output_dir": str(self.output_dir),
            "resume_full": self.resume_full,
            "finetune": asdict(self.finetune),
            "control": asdict(self.control),
        }

    def to_trainer_config(self) -> TrainerConfig:
        """
        Project onto the lower-level ``TrainerConfig`` the trainer consumes.

        No field copy: the trainer shares this config's ``control`` instance and
        receives only the derived paths on top.
        """
        return TrainerConfig(
            pipeline_config=self.pipeline_config_path,
            train_dir=self.train_dir,
            control=self.control,
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

    #: Why the training loop returned (see ``training.TrainOutcome``). Notably
    #: ``outcome.budget_exhausted`` tells a notebook whether this run finished
    #: or merely ran out of wall clock and has to be resumed in another session.
    #: Optional so callers constructing a ``RunResult`` by hand keep working.
    outcome: object = None

    @property
    def converged(self) -> bool:
        """
        False only when the run stopped on its wall-clock budget. A run with no
        budget configured cannot end that way, so this is True for every run
        that predates ``max_runtime_hours``.
        """
        return getattr(self.outcome, "converged", True)


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


def read_train_samples(cfg: FinetuneRunConfig) -> int | None:
    """
    Read ``train_samples`` from the bundle's ``dataset_metadata.json``.

    Returns the training-set size the trainer needs to translate the
    epoch-based cadence (``eval_every_epochs`` / ``max_epochs``) into steps, or
    ``None`` when the metadata file is missing or has no ``train_samples`` key
    (older bundles), in which case the trainer falls back to its step-based
    ``log_every`` cadence.
    """
    meta_path = cfg.dataset_metadata
    if not meta_path.exists():
        return None
    try:
        data = json.loads(meta_path.read_text())
    except (json.JSONDecodeError, OSError):
        return None
    value = data.get("train_samples")
    return int(value) if value else None


def compute_steps_per_epoch(cfg: FinetuneRunConfig) -> int | None:
    """Return ``ceil(train_samples / batch_size)``, or ``None`` without metadata."""
    train_samples = read_train_samples(cfg)
    batch_size = int(cfg.finetune.batch_size)
    if not train_samples or batch_size <= 0:
        return None
    return ceil(train_samples / batch_size)


def apply_config_overrides(cfg: FinetuneRunConfig) -> int | None:
    """Apply training-control overrides to ``cfg.finetune`` in place.

    Batch size is resolved before epoch-derived step counts. Returns the resulting
    steps per epoch, or ``None`` when dataset size is unavailable.
    """
    control = cfg.control

    if control.batch_size is not None:
        cfg.finetune.batch_size = control.batch_size
        print(f"Setting batch_size to {cfg.finetune.batch_size}.")

    steps_per_epoch = compute_steps_per_epoch(cfg)

    if steps_per_epoch:
        if control.max_epochs is not None:
            cfg.finetune.num_steps = control.max_epochs * steps_per_epoch
            print(
                f"Setting max step to {cfg.finetune.num_steps} "
                f"(max_epochs={control.max_epochs} * "
                f"steps_per_epoch={steps_per_epoch})."
            )
        if control.warmup_epochs is not None:
            cfg.finetune.warmup_steps = round(control.warmup_epochs * steps_per_epoch)
            print(
                f"Setting warmup steps to {cfg.finetune.warmup_steps} "
                f"(warmup_epochs={control.warmup_epochs} * "
                f"steps_per_epoch={steps_per_epoch})."
            )

    if control.lr_plateau_base_lr is not None:
        cfg.finetune.learning_rate_base = control.lr_plateau_base_lr
        print(f"Setting learning_rate_base to {cfg.finetune.learning_rate_base}.")
    if control.lr_plateau_warmup_lr is not None:
        cfg.finetune.warmup_learning_rate = control.lr_plateau_warmup_lr
        print(f"Setting warmup_learning_rate to {cfg.finetune.warmup_learning_rate}.")

    return steps_per_epoch


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

    # Resolve control overrides before rendering so derived step counts enter the
    # pipeline and keep cosine decay aligned with the run length.
    steps_per_epoch = apply_config_overrides(cfg)

    write_pipeline(cfg)
    cfg.train_dir.mkdir(parents=True, exist_ok=True)

    trainer_cfg = cfg.to_trainer_config()

    configs = load_pipeline_configs(trainer_cfg.pipeline_config)
    detection_model = build_detection_model(configs)
    runtime = create_runtime(
        detection_model,
        configs,
        trainer_cfg.train_dir,
        checkpoint_max_to_keep=cfg.control.checkpoint_max_to_keep,
        lr_plateau=cfg.control.lr_plateau,
        eval_ignore_partials=cfg.control.eval_ignore_partials,
    )

    outcome = train(
        detection_model,
        runtime,
        trainer_cfg,
        steps_per_epoch=steps_per_epoch,
    )

    return RunResult(
        pipeline_config=cfg.pipeline_config_path,
        train_dir=cfg.train_dir,
        best_metric_path=trainer_cfg.best_metric_path,
        history_path=trainer_cfg.history_path,
        detection_model=detection_model,
        configs=configs,
        outcome=outcome,
    )
