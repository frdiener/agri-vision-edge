"""
TensorFlow Object Detection export utilities.

Provides helpers for exporting TF-OD checkpoints
to SavedModel format.
"""

from pathlib import Path
from typing import Optional, Union

from .train import (
    get_tf_models_research_dir,
    run_tfod_command,
)


PathLike = Union[str, Path]


def export_saved_model(
    pipeline_config_path: PathLike,
    trained_checkpoint_dir: PathLike,
    output_directory: PathLike,
    input_type: str = "image_tensor",
    log_file: Optional[PathLike] = None,
):
    """
    Export a TF-OD model to TensorFlow SavedModel format.

    Args:
        pipeline_config_path:
            pipeline.config path.
        trained_checkpoint_dir:
            Directory containing training checkpoints.
        output_directory:
            Export destination directory.
        input_type:
            TF-OD exporter input type.
        log_file:
            Optional export log file.
    """
    research_dir = get_tf_models_research_dir()

    script = (
        research_dir
        / "object_detection"
        / "exporter_main_v2.py"
    )

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
