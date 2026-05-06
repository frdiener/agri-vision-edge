"""
TensorFlow Object Detection pipeline configuration utilities.

Provides helpers for modifying TF-OD pipeline.config files
programmatically for custom datasets and training setups.
"""

from pathlib import Path
from typing import Union

import tensorflow as tf
from google.protobuf import text_format

from object_detection.protos import pipeline_pb2


PathLike = Union[str, Path]


def load_pipeline_config(config_path: PathLike):
    """
    Load a TF-OD pipeline config protobuf.

    Args:
        config_path:
            Path to pipeline.config.

    Returns:
        pipeline_pb2.TrainEvalPipelineConfig
    """
    config_path = Path(config_path)

    pipeline_config = pipeline_pb2.TrainEvalPipelineConfig()

    with tf.io.gfile.GFile(str(config_path), "r") as f:
        text = f.read()

    text_format.Merge(text, pipeline_config)

    return pipeline_config


def save_pipeline_config(
    pipeline_config,
    output_path: PathLike,
) -> None:
    """
    Save a TF-OD pipeline config protobuf.

    Args:
        pipeline_config:
            TrainEvalPipelineConfig protobuf.
        output_path:
            Destination path.
    """
    output_path = Path(output_path)

    config_text = text_format.MessageToString(pipeline_config)

    output_path.write_text(config_text)


def configure_ssd_pipeline(
    config_path: PathLike,
    output_path: PathLike,
    train_record: PathLike,
    val_record: PathLike,
    label_map: PathLike,
    checkpoint_path: PathLike,
    num_classes: int = 2,
    batch_size: int = 4,
    num_steps: int = 10_000,
    learning_rate_base: float = 0.004,
    warmup_learning_rate: float = 0.001,
    warmup_steps: int = 500,
) -> None:
    """
    Configure an SSD-based TF-OD pipeline config.

    This utility updates:
    - dataset paths
    - label map paths
    - checkpoint paths
    - class count
    - batch size
    - learning rate schedule
    - training steps

    Args:
        config_path:
            Input template pipeline.config.
        output_path:
            Destination pipeline.config.
        train_record:
            TFRecord for training.
        val_record:
            TFRecord for evaluation.
        label_map:
            label_map.pbtxt path.
        checkpoint_path:
            Fine-tuning checkpoint path (ckpt-0).
        num_classes:
            Number of detection classes.
        batch_size:
            Training batch size.
        num_steps:
            Total training steps.
        learning_rate_base:
            Base cosine decay learning rate.
        warmup_learning_rate:
            Warmup learning rate.
        warmup_steps:
            Warmup schedule length.
    """
    pipeline_config = load_pipeline_config(config_path)

    #
    # Model
    #

    pipeline_config.model.ssd.num_classes = num_classes

    #
    # Train config
    #

    pipeline_config.train_config.batch_size = batch_size

    pipeline_config.train_config.fine_tune_checkpoint = str(
        checkpoint_path
    )

    pipeline_config.train_config.fine_tune_checkpoint_type = (
        "detection"
    )

    pipeline_config.train_config.num_steps = num_steps

    pipeline_config.train_config.sync_replicas = False
    pipeline_config.train_config.replicas_to_aggregate = 1

    #
    # Learning rate
    #

    lr_config = (
        pipeline_config
        .train_config
        .optimizer
        .momentum_optimizer
        .learning_rate
        .cosine_decay_learning_rate
    )

    lr_config.learning_rate_base = learning_rate_base
    lr_config.warmup_learning_rate = warmup_learning_rate
    lr_config.total_steps = num_steps
    lr_config.warmup_steps = warmup_steps

    #
    # Train input
    #

    pipeline_config.train_input_reader.label_map_path = str(
        label_map
    )

    pipeline_config.train_input_reader.tf_record_input_reader.input_path[:] = [
        str(train_record)
    ]

    #
    # Eval input
    #

    pipeline_config.eval_input_reader[0].label_map_path = str(
        label_map
    )

    pipeline_config.eval_input_reader[0].tf_record_input_reader.input_path[:] = [
        str(val_record)
    ]

    save_pipeline_config(
        pipeline_config,
        output_path,
    )
