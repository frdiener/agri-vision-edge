"""
TFOD setup helpers.

Responsible for:
    - pipeline loading
    - model construction
    - optimizer creation
    - checkpoint management
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import tensorflow as tf
from google.protobuf import text_format
from object_detection import eval_util
from object_detection.builders import model_builder, optimizer_builder
from object_detection.model_lib_v2 import (
    _ensure_model_is_built,
    load_fine_tune_checkpoint,
)
from object_detection.protos import pipeline_pb2
from object_detection.utils import config_util, label_map_util, variables_helper


@dataclass(slots=True)
class Runtime:
    """
    Runtime objects needed during training.
    """

    configs: dict

    optimizer: tf.keras.optimizers.Optimizer

    learning_rate: object

    global_step: tf.Variable

    ckpt: tf.train.Checkpoint

    manager: tf.train.CheckpointManager

    #: Rolling snapshot of the *latest* training state, in a `last/`
    #: subdirectory of the train dir. `manager` only ever saves on a new best
    #: (see `save_best_checkpoint`), so it is the wrong thing to resume from: a
    #: session killed well after its best would rewind to that best and re-tread
    #: the ground between, with its plateau counters already spent. This one is
    #: written every evaluation so an interrupted run continues where it
    #: stopped. Kept in a subdirectory so `export_run`, which opens its own
    #: manager on the train dir, still finds only best checkpoints and exports
    #: the best model rather than the newest one.
    last_manager: tf.train.CheckpointManager

    evaluators: list

    add_regularization_loss: bool

    unpad_groundtruth_tensors: bool

    clip_gradients_value: float | None

    use_moving_average: bool

    # Mutable learning-rate variable for the reduce-on-plateau schedule. None
    # when plateau scheduling is disabled (the LR is then the proto's step-keyed
    # schedule owned by the optimizer). `lr_base` / `lr_warmup` / `lr_warmup_steps`
    # are the warmup parameters extracted from the pipeline LR config, used by the
    # training loop to ramp `lr_var` before plateau reductions take over.
    lr_var: object = None
    lr_base: float = 0.0
    lr_warmup: float = 0.0
    lr_warmup_steps: int = 0

    # When True, honour partial ("do-not-care") ground-truth markers
    # (groundtruth_is_crowd, mirrored from is_partial in the record) during eval
    # so detections on partial plants are not penalized. When False (default,
    # strict) the markers are cleared and partials are scored normally.
    eval_ignore_partials: bool = False


def load_pipeline_configs(
    pipeline_path,
) -> dict:
    """
    Load TFOD pipeline config.
    """

    pipeline_config = pipeline_pb2.TrainEvalPipelineConfig()

    text_format.Merge(
        pipeline_path.read_text(),
        pipeline_config,
    )

    return {
        "model": pipeline_config.model,
        "train_config": pipeline_config.train_config,
        "train_input_config": pipeline_config.train_input_reader,
        "eval_input_configs": pipeline_config.eval_input_reader,
        "eval_input_config": pipeline_config.eval_input_reader[0],
        "eval_config": pipeline_config.eval_config,
    }


def build_detection_model(
    configs: dict,
):
    """
    Create TFOD model.
    """

    return model_builder.build(
        model_config=configs["model"],
        is_training=True,
    )


def create_evaluators(
    configs: dict,
):
    category_index = label_map_util.create_category_index_from_labelmap(
        configs["eval_input_config"].label_map_path
    )

    return eval_util.get_evaluators(
        configs["eval_config"],
        list(category_index.values()),
        eval_util.evaluator_options_from_eval_config(configs["eval_config"]),
    )


def extract_lr_params(optimizer_config):
    """
    Read (base_lr, warmup_lr, warmup_steps) from a pipeline optimizer config.

    The plateau schedule replaces the proto's step-keyed LR schedule with a
    mutable variable, but still honours the configured warmup ramp and uses the
    configured base LR as the variable's starting (post-warmup) value. Handles
    the cosine / exponential / manual-step / constant learning-rate types;
    unknown types fall back to (0, 0, 0).
    """
    opt_type = optimizer_config.WhichOneof("optimizer")
    lr = getattr(optimizer_config, opt_type).learning_rate
    lr_type = lr.WhichOneof("learning_rate")

    if lr_type == "cosine_decay_learning_rate":
        c = lr.cosine_decay_learning_rate
        return (
            float(c.learning_rate_base),
            float(c.warmup_learning_rate),
            int(c.warmup_steps),
        )
    if lr_type == "exponential_decay_learning_rate":
        c = lr.exponential_decay_learning_rate
        return (
            float(c.initial_learning_rate),
            float(c.burnin_learning_rate),
            int(c.burnin_steps),
        )
    if lr_type == "manual_step_learning_rate":
        c = lr.manual_step_learning_rate
        return (float(c.initial_learning_rate), 0.0, 0)
    if lr_type == "constant_learning_rate":
        c = lr.constant_learning_rate
        return (float(c.learning_rate), 0.0, 0)

    return (0.0, 0.0, 0)


class _VariableLearningRate(
    tf.keras.optimizers.schedules.LearningRateSchedule
):
    """
    A ``LearningRateSchedule`` that simply returns a mutable variable's current
    value, ignoring the step.

    Passing a bare ``tf.Variable`` as ``learning_rate`` is unreliable across the
    TF 2.11 legacy/new Keras optimizers (the new optimizer copies a Variable
    into a fresh internal one, so later assignments would not propagate). A
    schedule, by contrast, is always held by reference and invoked each
    ``apply_gradients`` on BOTH optimizer implementations -- so reading ``var``
    inside ``__call__`` gives the optimizer the live LR we control from the
    training loop.
    """

    def __init__(self, var):
        super().__init__()
        self.var = var

    def __call__(self, step):
        return self.var

    def get_config(self):
        # Never serialised (checkpointing goes through tf.train.Checkpoint), but
        # required by the abstract base.
        return {}


def build_optimizer_with_lr_var(optimizer_config, lr_var):
    """
    Build the pipeline's optimizer but with ``lr_var`` (a mutable tf.Variable)
    as the learning rate, so the training loop can adjust it on plateau.

    Mirrors ``object_detection.builders.optimizer_builder`` (momentum / adam /
    rms_prop + optional EMA wrapper), substituting the step-keyed LR schedule
    with a ``_VariableLearningRate`` wrapper around ``lr_var``. The optimizer
    invokes the schedule each ``apply_gradients``, reading the live variable.
    """
    opt_type = optimizer_config.WhichOneof("optimizer")

    lr_schedule = _VariableLearningRate(lr_var)

    if opt_type == "rms_prop_optimizer":
        c = optimizer_config.rms_prop_optimizer
        optimizer = tf.keras.optimizers.RMSprop(
            lr_schedule,
            decay=c.decay,
            momentum=c.momentum_optimizer_value,
            epsilon=c.epsilon,
        )
    elif opt_type == "momentum_optimizer":
        c = optimizer_config.momentum_optimizer
        optimizer = tf.keras.optimizers.SGD(
            lr_schedule,
            momentum=c.momentum_optimizer_value,
        )
    elif opt_type == "adam_optimizer":
        c = optimizer_config.adam_optimizer
        optimizer = tf.keras.optimizers.Adam(lr_schedule, epsilon=c.epsilon)
    else:
        raise ValueError(
            f"Optimizer {opt_type!r} not supported for plateau scheduling."
        )

    if optimizer_config.use_moving_average:
        # Same EMA wrapper the stock optimizer_builder uses.
        from official.modeling.optimization import ema_optimizer

        optimizer = ema_optimizer.ExponentialMovingAverage(
            optimizer=optimizer,
            average_decay=optimizer_config.moving_average_decay,
        )

    return optimizer


def create_runtime(
    detection_model,
    configs,
    train_dir,
    checkpoint_max_to_keep=3,
    lr_plateau=False,
    eval_ignore_partials=False,
) -> Runtime:

    # Resolve `fine_tune_checkpoint_type` from the deprecated
    # `from_detection_checkpoint` field when it is not set explicitly.
    config_util.update_fine_tune_checkpoint_type(configs["train_config"])

    global_step = tf.Variable(
        0,
        trainable=False,
        dtype=tf.int64,
        name="global_step",
    )

    lr_var = None
    lr_base = lr_warmup = 0.0
    lr_warmup_steps = 0

    if lr_plateau:
        # Replace the proto's step-keyed schedule with a mutable variable driven
        # by the training loop (warmup ramp, then reduce-on-plateau). Start at
        # the warmup LR (or the base LR when no warmup is configured).
        lr_base, lr_warmup, lr_warmup_steps = extract_lr_params(
            configs["train_config"].optimizer
        )
        initial_lr = lr_warmup if lr_warmup_steps > 0 else lr_base
        lr_var = tf.Variable(
            initial_lr,
            trainable=False,
            dtype=tf.float32,
            name="learning_rate",
        )
        optimizer = build_optimizer_with_lr_var(
            configs["train_config"].optimizer,
            lr_var,
        )
        learning_rate = lr_var
    else:
        optimizer, (learning_rate,) = optimizer_builder.build(
            configs["train_config"].optimizer,
            global_step=global_step,
        )

    clip_gradients_value = None

    if configs["train_config"].gradient_clipping_by_norm > 0:
        clip_gradients_value = configs["train_config"].gradient_clipping_by_norm

    ckpt = tf.train.Checkpoint(
        step=global_step,
        model=detection_model,
        optimizer=optimizer,
    )

    manager = tf.train.CheckpointManager(
        ckpt,
        train_dir,
        max_to_keep=checkpoint_max_to_keep,
    )

    # Same Checkpoint object (so model + optimizer + step are all captured),
    # separate directory and state file. Only one is kept: this exists to answer
    # "where had the run got to", not to provide history.
    last_manager = tf.train.CheckpointManager(
        ckpt,
        str(Path(train_dir) / "last"),
        max_to_keep=1,
    )

    # NOTE: weights are restored later in `restore_weights` (called from
    # `train`), once the train dataset is available. With EMA enabled the
    # optimizer's shadow variables must be created before any restore, and
    # creating them requires building the model on a real input batch.

    return Runtime(
        configs=configs,
        optimizer=optimizer,
        learning_rate=learning_rate,
        global_step=global_step,
        ckpt=ckpt,
        manager=manager,
        last_manager=last_manager,
        evaluators=create_evaluators(configs),
        add_regularization_loss=(configs["train_config"].add_regularization_loss),
        unpad_groundtruth_tensors=(configs["train_config"].unpad_groundtruth_tensors),
        clip_gradients_value=clip_gradients_value,
        use_moving_average=(configs["train_config"].optimizer.use_moving_average),
        lr_var=lr_var,
        lr_base=lr_base,
        lr_warmup=lr_warmup,
        lr_warmup_steps=lr_warmup_steps,
        eval_ignore_partials=eval_ignore_partials,
    )


def maybe_load_fine_tune_checkpoint(
    detection_model,
    runtime,
    train_dataset,
):
    """
    Restore pretrained weights from the pipeline's fine-tune checkpoint.

    Mirrors ``object_detection.model_lib_v2.train_loop``: on a cold start
    (no checkpoint in the train directory yet) the pretrained
    detection/classification checkpoint referenced by the pipeline config is
    loaded, so training fine-tunes from those weights instead of starting
    from random initialization. Skipped when resuming an existing train-dir
    checkpoint (handled by ``restore_weights``).

    ``train_dataset`` is required to build the model variables (via a dummy
    forward pass) before the object-based restore.
    """

    if runtime.manager.latest_checkpoint:
        # Resuming is handled by restore_weights via ckpt.restore.
        return

    train_config = runtime.configs["train_config"]

    if not train_config.fine_tune_checkpoint:
        print("No fine_tune_checkpoint set; training from scratch.")
        return

    print(
        "Loading fine-tune checkpoint "
        f"({train_config.fine_tune_checkpoint_type}): "
        f"{train_config.fine_tune_checkpoint}"
    )

    variables_helper.ensure_checkpoint_supported(
        train_config.fine_tune_checkpoint,
        train_config.fine_tune_checkpoint_type,
        runtime.manager.directory,
    )

    load_fine_tune_checkpoint(
        detection_model,
        train_config.fine_tune_checkpoint,
        train_config.fine_tune_checkpoint_type,
        train_config.fine_tune_checkpoint_version,
        # Force the dummy forward pass so all model variables exist before
        # the object-based restore matches them.
        True,
        train_dataset,
        runtime.unpad_groundtruth_tensors,
    )


def restore_weights(
    detection_model,
    runtime,
    train_dataset,
):
    """
    Initialize model weights before training.

    Order mirrors ``object_detection.model_lib_v2.train_loop``:

      1. When EMA (``optimizer.use_moving_average``) is enabled, build the
         model and create the optimizer's shadow variables *first*, so they
         exist before any restore and are themselves restored on resume.
      2. If a checkpoint already exists in the train directory, resume from
         it (this restores model, optimizer, shadow variables and step).
         The rolling ``last/`` snapshot wins over the best-checkpoint history:
         it is never older (both are written at the same evaluation, and only
         one of them is conditional on an improvement), and resuming from the
         *best* would silently rewind the run to it and redo everything since.
      3. Otherwise, load the pretrained fine-tune checkpoint.

    ``train_dataset`` is required to build the model on a real input batch.

    Returns:
        bool: True if training resumed from a checkpoint already in the train
        directory (so the caller should also restore the trainer's own
        bookkeeping, see ``TrainerState.load``), False if the weights came from
        the pretrained fine-tune checkpoint (a cold start).
    """

    model_built = False

    if runtime.use_moving_average:
        print("EMA enabled: creating optimizer shadow variables...")
        _ensure_model_is_built(
            detection_model,
            train_dataset,
            runtime.unpad_groundtruth_tensors,
        )
        model_built = True
        runtime.optimizer.shadow_copy(detection_model)

    resume_from = (
        runtime.last_manager.latest_checkpoint
        or runtime.manager.latest_checkpoint
    )
    if resume_from:
        # Build the variables before restoring. `ckpt.restore` is object-based
        # and therefore lazy: with nothing built it quietly defers every value
        # instead of failing, and the caller then trips over a model whose
        # `trainable_variables` is empty (ValueError out of
        # `ensure_optimizer_state_created`). The cold-start path below gets this
        # for free -- `load_fine_tune_checkpoint` forces the same dummy forward
        # pass -- which is why only resuming hit it.
        if not model_built:
            _ensure_model_is_built(
                detection_model,
                train_dataset,
                runtime.unpad_groundtruth_tensors,
            )

        print(f"Resuming from checkpoint: {resume_from}")
        runtime.ckpt.restore(resume_from)
        print(f"Resumed at step {int(runtime.global_step.numpy())}.")
        return True

    maybe_load_fine_tune_checkpoint(
        detection_model,
        runtime,
        train_dataset,
    )
    return False


def apply_graph_modifications(
    detection_model,
    runtime,
    trainer_cfg,
    train_dataset,
):
    """
    Apply BatchNorm folding and backbone QAT.

    Mirrors the graph-modification block in
    ``object_detection.model_lib_v2.train_loop``: when ``qat`` is set, fold
    BatchNorms into the convs, then fake-quantize the backbone + SSD head (the
    full int8 scheme, ``agri_vision_edge.tfod.qat``).

    Must run after ``restore_weights`` (so the loaded weights are folded /
    quantized) and before the train step is traced (so the modified backbone is
    captured). Mutates ``runtime`` and ``detection_model`` in place.

    Note: there is deliberately no optimizer reset here. PTQ/QAT resume from an
    exported model-only ``ckpt-0`` (``load_fine_tune_checkpoint`` restores model
    weights only, never the optimizer or step), so the optimizer built in
    ``create_runtime`` is already fresh and starts at step 0. Keeping that single
    optimizer instance also keeps ``runtime.ckpt`` coherent, so the
    ``lr_plateau_restore_best`` warm restart actually restores the live model +
    optimizer under QAT/PTQ.

    Returns:
        bool: True if the backbone graph was modified (QAT), so the caller can
        run an initial evaluation of the new configuration.
    """

    if not trainer_cfg.control.qat:
        return False

    # Folding and cloning require the backbone variables to exist.
    _ensure_model_is_built(
        detection_model,
        train_dataset,
        runtime.unpad_groundtruth_tensors,
    )

    # Imported lazily to mirror train_loop and avoid import cycles.
    from agri_vision_edge.tfod.qat import quantize_detection_model

    # quantize_detection_model is self-contained: it folds BatchNorms and inserts
    # the fake-quant nodes for the WHOLE model. FPN folds+quantizes the backbone
    # as its own graph then the combined head; plain SSD inlines the backbone with
    # the head into ONE combined functional graph (so the dual-use relu6 tap is
    # interior and no stray requant is left at the backbone/head boundary).
    print("Folding + quantizing the full model (backbone + detection head)...")
    image_size = runtime.configs["model"].ssd.image_resizer.fixed_shape_resizer.height
    quantize_detection_model(
        detection_model,
        image_size,
        per_channel=trainer_cfg.control.qat_per_channel,
    )

    return True
