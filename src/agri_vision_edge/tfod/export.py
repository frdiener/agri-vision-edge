"""
TensorFlow Object Detection export utilities.

Provides helpers for exporting TF-OD checkpoints
to TensorFlow SavedModel format.
"""

from pathlib import Path
from typing import Optional, Union

from .common import (
    get_tf_models_research_dir,
    run_tfod_command,
)


PathLike = Union[str, Path]


def export_saved_model(
    pipeline_config_path: PathLike,
    trained_checkpoint_dir: PathLike,
    output_directory: PathLike,
    checkpoint_path: Optional[PathLike] = None,
    input_type: str = "image_tensor",
    log_file: Optional[PathLike] = None,
):
    """
    Export a TensorFlow Object Detection model
    to TensorFlow SavedModel format.

    By default, TensorFlow Object Detection exports the
    latest checkpoint found inside `trained_checkpoint_dir`.

    Optionally, a specific checkpoint can be exported
    by providing `checkpoint_path`.

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

    script = (
        research_dir
        / "object_detection"
        / "exporter_main_v2.py"
    )

    #
    # TF-OD exporter expects checkpoint path WITHOUT suffix
    #
    # Example:
    #   ckpt-12
    # NOT:
    #   ckpt-12.index
    #

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

    if checkpoint_path is not None:
        args.extend(
            [
                "--checkpoint_path",
                str(checkpoint_path),
            ]
        )

    return run_tfod_command(
        args,
        log_file=log_file,
        background=False,
    )
