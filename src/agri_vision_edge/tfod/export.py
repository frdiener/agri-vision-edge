"""
TensorFlow Object Detection export utilities.

Provides helpers for exporting TF-OD checkpoints
to TensorFlow SavedModel and TFLite-compatible
SavedModel formats.

These wrap the stock ``object_detection`` export scripts. The forked,
in-process QAT exporters (``qat_backbone`` / ``fold_bn``) were removed when the
vendored ``object_detection`` tree was restored to upstream; use
``agri_vision_edge.tfod_trainer.export.export_run`` for QAT / folded exports.
"""

from pathlib import Path

import tensorflow as tf
from google.protobuf import text_format
from object_detection.protos import pipeline_pb2

from .common import (
    get_tf_models_research_dir,
    run_tfod_command,
)

PathLike = str | Path


def _load_pipeline_config(
    pipeline_config_path,
    config_override="",
):
    """
    Load and optionally override pipeline config.
    """

    pipeline_config = pipeline_pb2.TrainEvalPipelineConfig()

    with tf.io.gfile.GFile(
        pipeline_config_path,
        "r",
    ) as f:
        text_format.Parse(
            f.read(),
            pipeline_config,
        )

    if config_override:
        override_config = pipeline_pb2.TrainEvalPipelineConfig()

        text_format.Parse(
            config_override,
            override_config,
        )

        pipeline_config.MergeFrom(override_config)

    return pipeline_config


def export_saved_model(
    pipeline_config_path: PathLike,
    trained_checkpoint_dir: PathLike,
    output_directory: PathLike,
    input_type: str = "image_tensor",
    log_file: PathLike | None = None,
):
    """
    Export a TensorFlow Object Detection model
    to TensorFlow SavedModel format.

    This uses TF-OD's generic exporter and produces
    a standard TensorFlow SavedModel suitable for:

    - TensorFlow inference
    - further graph manipulation
    - generic SavedModel workflows

    Args:
        pipeline_config_path:
            Path to pipeline.config.
        trained_checkpoint_dir:
            Directory containing training checkpoints.
        output_directory:
            Export destination directory.
        checkpoint_path:
            Optional specific checkpoint path
            (e.g. ckpt-12).
        input_type:
            TF-OD exporter input type.
        log_file:
            Optional export log file.

    Returns:
        Completed subprocess handle.
    """
    research_dir = get_tf_models_research_dir()

    script = research_dir / "object_detection" / "exporter_main_v2.py"

    args = [
        "python",
        str(script),
        "--input_type",
        input_type,
        "--pipeline_config_path",
        str(pipeline_config_path),
        "--trained_checkpoint_dir",
        str(trained_checkpoint_dir),
        "--output_directory",
        str(output_directory),
    ]

    return run_tfod_command(
        args,
        log_file=log_file,
        background=False,
    )


def export_tflite_graph(
    pipeline_config_path: PathLike,
    trained_checkpoint_dir: PathLike,
    output_directory: PathLike,
    max_detections: int = 100,
    use_regular_nms: bool = False,
    log_file: PathLike | None = None,
):
    """
    Export a TF-OD model using the dedicated
    TensorFlow Lite export pipeline.

    This exporter rewrites the graph specifically
    for TFLite compatibility and should be preferred
    when the final deployment target is:

    - TensorFlow Lite
    - embedded inference
    - NPU delegates
    - Edge accelerators

    Compared to the generic SavedModel exporter,
    this export path typically produces graphs with:

    - fewer dynamic ops
    - fewer TensorList ops
    - reduced control flow
    - better quantization compatibility

    Args:
        pipeline_config_path:
            Path to pipeline.config.
        trained_checkpoint_dir:
            Directory containing training checkpoints.
        output_directory:
            Export destination directory.
        checkpoint_path:
            Optional specific checkpoint path.
        max_detections:
            Maximum detections per image.
        use_regular_nms:
            Use regular NMS instead of fast NMS.
        log_file:
            Optional export log file.

    Returns:
        Completed subprocess handle.
    """
    research_dir = get_tf_models_research_dir()

    script = research_dir / "object_detection" / "export_tflite_graph_tf2.py"

    args = [
        "python",
        str(script),
        "--pipeline_config_path",
        str(pipeline_config_path),
        "--trained_checkpoint_dir",
        str(trained_checkpoint_dir),
        "--output_directory",
        str(output_directory),
        "--max_detections",
        str(max_detections),
    ]

    if use_regular_nms:
        args.append("--use_regular_nms")

    return run_tfod_command(
        args,
        log_file=log_file,
        background=False,
    )
