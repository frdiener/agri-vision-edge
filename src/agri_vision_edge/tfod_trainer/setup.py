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

from object_detection.protos import pipeline_pb2
from object_detection.builders import model_builder
from object_detection.builders import optimizer_builder
from object_detection.utils import label_map_util
from object_detection import eval_util


@dataclass(slots=True)
class Runtime:
    """
    Runtime objects needed during training.
    """

    configs: dict

    optimizer: tf.keras.optimizers.Optimizer

    learning_rate: object

    global_step: tf.Variable

    manager: tf.train.CheckpointManager

    evaluators: list

    add_regularization_loss: bool

    unpad_groundtruth_tensors: bool

    clip_gradients_value: float | None


def load_pipeline_configs(
    pipeline_path,
) -> dict:
    """
    Load TFOD pipeline config.
    """

    pipeline_config = (
        pipeline_pb2.TrainEvalPipelineConfig()
    )

    text_format.Merge(
        pipeline_path.read_text(),
        pipeline_config,
    )

    return {
        "model": pipeline_config.model,
        "train_config": pipeline_config.train_config,
        "train_input_config":
            pipeline_config.train_input_reader,
        "eval_input_configs":
            pipeline_config.eval_input_reader,
        "eval_input_config":
            pipeline_config.eval_input_reader[0],
        "eval_config":
            pipeline_config.eval_config,
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
    category_index = (
        label_map_util.create_category_index_from_labelmap(
            configs["eval_input_config"].label_map_path
        )
    )

    return eval_util.get_evaluators(
        configs["eval_config"],
        list(category_index.values()),
        eval_util.evaluator_options_from_eval_config(
            configs["eval_config"]
        ),
    )


def create_runtime(
    detection_model,
    configs,
    train_dir,
    checkpoint_max_to_keep=3,
) -> Runtime:

    global_step = tf.Variable(
        0,
        trainable=False,
        dtype=tf.int64,
        name="global_step",
    )

    optimizer, (learning_rate,) = (
        optimizer_builder.build(
            configs["train_config"].optimizer,
            global_step=global_step,
        )
    )

    clip_gradients_value = None

    if (
        configs["train_config"]
        .gradient_clipping_by_norm
        > 0
    ):
        clip_gradients_value = (
            configs["train_config"]
            .gradient_clipping_by_norm
        )

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

    ckpt.restore(manager.latest_checkpoint)

    return Runtime(
        configs=configs,
        optimizer=optimizer,
        learning_rate=learning_rate,
        global_step=global_step,
        manager=manager,
        evaluators=create_evaluators(configs),
        add_regularization_loss=(
            configs["train_config"]
            .add_regularization_loss
        ),
        unpad_groundtruth_tensors=(
            configs["train_config"]
            .unpad_groundtruth_tensors
        ),
        clip_gradients_value=clip_gradients_value,
    )
