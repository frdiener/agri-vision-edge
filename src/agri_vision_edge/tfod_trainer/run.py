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

QAT is the same call with ``qat=True`` (and optionally ``qat_per_channel``) set.
"""

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
    f.name for f in TrainingControlConfig.__dataclass_fields__.values()  # type: ignore[attr-defined]
)


@dataclass
class FinetuneRunConfig:
    """
    The single master configuration for one finetune (or QAT) job.

    This is the whole notebook / Python API and the object committed to the
    experiment manifest (see :meth:`to_mapping` / :meth:`from_mapping`). Its
    fields fall into three cohesive groups:

    Orchestration (top level):
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
        resume_full:
            Resume OUR OWN converged export (matching num_classes), restoring
            the box/class prediction heads too (fine_tune_checkpoint_type=
            "full"), as opposed to bootstrapping from a foreign detection
            checkpoint (e.g. COCO, different num_classes) whose heads must be
            dropped and reinitialised ("detection"). QAT always resumes our own
            export, so it implies this; a plain-float PTQ base (qat=False) that
            resumes the finetune export must set it explicitly, otherwise its
            heads are dropped and it retrains from cold. Independent of
            quantization.

    ``finetune`` (a :class:`FineTuneConfig`):
        The pure pipeline / model semantics rendered into the TFOD protobuf.

    ``control`` (a :class:`TrainingControlConfig`):
        Every custom training-loop knob -- early stopping, reduce-LR-on-plateau,
        logging / checkpointing, and the QAT flags (``qat`` / ``qat_per_channel``).
        These are consumed by the trainer via :meth:`to_trainer_config`.
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

    # --- QAT convenience (read-through to control) ---------------------

    @property
    def qat(self) -> bool:
        return self.control.qat

    @property
    def qat_per_channel(self) -> bool:
        return self.control.qat_per_channel

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
    def dataset_metadata(self) -> Path:
        return self.dataset_bundle_path / "dataset_metadata.json"

    @property
    def pipeline_config_path(self) -> Path:
        return self.output_dir / "finetune" / "pipeline.config"

    @property
    def train_dir(self) -> Path:
        return self.output_dir / "train"

    # --- (de)serialization for the manifest / head-less dicts ----------

    @classmethod
    def from_mapping(cls, data) -> FinetuneRunConfig:
        """
        Build from a plain dict (e.g. a manifest stage config or UI dict),
        expanding the nested ``finetune`` / ``control`` (and ``augmentation``)
        sub-dicts into their dataclasses.

        Backward compatible with the pre-nesting layout: ``early_stopping_*``
        found inside ``finetune`` and any flat control knobs (``qat``,
        ``qat_per_channel``, ``lr_plateau*``, ``log_every``, ...) sitting at the
        top level are folded into ``control``, so historical manifests keep
        loading. A legacy ``reset_optimizer`` (now removed) is dropped.
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

        # `reset_optimizer` was removed (vestigial once PTQ/QAT resume from an
        # exported model-only ckpt-0 rather than the finetune train dir). Drop any
        # legacy occurrence -- top level or nested -- so old manifests still load.
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
    """
    Derive ``steps_per_epoch = ceil(train_samples / batch_size)``.

    ``batch_size`` is the finetune config's batch size (rendered verbatim into
    the pipeline's ``train_config``; each training step consumes exactly one
    batch) and ``train_samples`` comes from the bundle metadata. Both are known
    before the pipeline is rendered, so this can run pre-render -- which is what
    lets ``run_finetune`` fold a ``max_epochs`` horizon into ``num_steps`` before
    the cosine schedule's ``total_steps`` is baked in. Uses ``ceil`` so a
    training set that is not a whole multiple of the batch size still counts its
    trailing partial batch as the epoch's final step, keeping epoch boundaries
    aligned to integer step counts. Returns ``None`` when the sample count is
    unavailable.
    """
    train_samples = read_train_samples(cfg)
    batch_size = int(cfg.finetune.batch_size)
    if not train_samples or batch_size <= 0:
        return None
    return ceil(train_samples / batch_size)


def apply_config_overrides(cfg: FinetuneRunConfig) -> int | None:
    """
    Fold the ``control`` overrides into ``cfg.finetune`` before the pipeline is
    rendered, so a single place (``control``) can drive keys that otherwise live
    on the pipeline config, and return the resulting ``steps_per_epoch``:

    - ``batch_size``          -> ``batch_size``
    - ``max_epochs``          -> ``num_steps`` (also the cosine ``total_steps``)
    - ``warmup_epochs``       -> ``warmup_steps`` (cosine + plateau warmup ramp)
    - ``lr_plateau_base_lr``  -> ``learning_rate_base``
    - ``lr_plateau_warmup_lr``-> ``warmup_learning_rate``

    Order matters: ``batch_size`` is resolved first because it feeds
    ``steps_per_epoch`` (= ceil(train_samples / batch_size)), which in turn
    drives the epoch-derived horizons (``max_epochs`` / ``warmup_epochs``). The
    epoch-derived overrides need a known ``steps_per_epoch`` (skipped when it is
    ``None``); the ``batch_size`` / LR overrides are applied unconditionally when
    set. Each applied override is logged. Mutates ``cfg.finetune`` in place.
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
            cfg.finetune.warmup_steps = round(
                control.warmup_epochs * steps_per_epoch
            )
            print(
                f"Setting warmup steps to {cfg.finetune.warmup_steps} "
                f"(warmup_epochs={control.warmup_epochs} * "
                f"steps_per_epoch={steps_per_epoch})."
            )

    if control.lr_plateau_base_lr is not None:
        cfg.finetune.learning_rate_base = control.lr_plateau_base_lr
        print(
            f"Setting learning_rate_base to {cfg.finetune.learning_rate_base}."
        )
    if control.lr_plateau_warmup_lr is not None:
        cfg.finetune.warmup_learning_rate = control.lr_plateau_warmup_lr
        print(
            "Setting warmup_learning_rate to "
            f"{cfg.finetune.warmup_learning_rate}."
        )

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

    # Resolve all control overrides (folding them into cfg.finetune) BEFORE
    # rendering, so batch_size / max_epochs -> num_steps / warmup_epochs ->
    # warmup_steps land in the pipeline keys -- num_steps also being the cosine
    # schedule's total_steps -- keeping the LR decay aligned with the actual run
    # length. Returns the epoch geometry the trainer needs.
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

    train(
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
    )
