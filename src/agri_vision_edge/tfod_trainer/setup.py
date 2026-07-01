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

    evaluators: list

    add_regularization_loss: bool

    unpad_groundtruth_tensors: bool

    clip_gradients_value: float | None

    use_moving_average: bool


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


def create_runtime(
    detection_model,
    configs,
    train_dir,
    checkpoint_max_to_keep=3,
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
        evaluators=create_evaluators(configs),
        add_regularization_loss=(configs["train_config"].add_regularization_loss),
        unpad_groundtruth_tensors=(configs["train_config"].unpad_groundtruth_tensors),
        clip_gradients_value=clip_gradients_value,
        use_moving_average=(configs["train_config"].optimizer.use_moving_average),
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
      3. Otherwise, load the pretrained fine-tune checkpoint.

    ``train_dataset`` is required to build the model on a real input batch.
    """

    if runtime.use_moving_average:
        print("EMA enabled: creating optimizer shadow variables...")
        _ensure_model_is_built(
            detection_model,
            train_dataset,
            runtime.unpad_groundtruth_tensors,
        )
        runtime.optimizer.shadow_copy(detection_model)

    if runtime.manager.latest_checkpoint:
        print(f"Resuming from checkpoint: {runtime.manager.latest_checkpoint}")
        runtime.ckpt.restore(runtime.manager.latest_checkpoint)
        return

    maybe_load_fine_tune_checkpoint(
        detection_model,
        runtime,
        train_dataset,
    )


def apply_graph_modifications(
    detection_model,
    runtime,
    trainer_cfg,
    train_dataset,
):
    """
    Apply optimizer reset, BatchNorm folding and backbone QAT.

    Functionally mirrors the corresponding block in
    ``object_detection.model_lib_v2.train_loop``, in order:

      1. ``reset_optimizer``: zero the global step and rebuild the optimizer
         from the pipeline config.
      2. ``qat``: fold BatchNorms into the convs, then fake-quantize the
         backbone + SSD head (the full int8 scheme,
         ``agri_vision_edge.tfod.qat``).

    Must run after ``restore_weights`` (so the loaded weights are folded /
    quantized) and before the train step is traced (so the rebuilt optimizer
    and modified backbone are captured). Mutates ``runtime`` and
    ``detection_model`` in place.

    Returns:
        bool: True if the backbone graph was modified (QAT), so the caller can
        run an initial evaluation of the new configuration.
    """

    train_config = runtime.configs["train_config"]

    if trainer_cfg.reset_optimizer:
        print("Resetting the optimizer...")
        runtime.global_step.assign(0)
        optimizer, (learning_rate,) = optimizer_builder.build(
            train_config.optimizer,
            global_step=runtime.global_step,
        )
        runtime.optimizer = optimizer
        runtime.learning_rate = learning_rate

    if not trainer_cfg.qat:
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
        per_channel=trainer_cfg.qat_per_channel,
    )

    return True
