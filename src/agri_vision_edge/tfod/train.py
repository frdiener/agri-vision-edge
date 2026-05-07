"""
TensorFlow Object Detection training utilities.

Provides helpers for launching TF-OD training jobs
using vendored TensorFlow Models code.
"""

from pathlib import Path
from typing import Optional, Union

from .common import (
    get_tf_models_research_dir,
    run_tfod_command,
)


PathLike = Union[str, Path]


def launch_training(
    pipeline_config_path: PathLike,
    model_dir: PathLike,
    checkpoint_every_n: int = 1000,
    checkpoint_max_to_keep: int = 100,
    log_file: Optional[PathLike] = None,
    background: bool = False,
):
    """
    Launch TensorFlow Object Detection training.

    Args:
        pipeline_config_path:
            Path to pipeline.config.
        model_dir:
            Output training directory.
        checkpoint_every_n:
            Checkpoint frequency.
        log_file:
            Optional training log file.
        background:
            Run asynchronously.

    Returns:
        subprocess.Popen or subprocess.CompletedProcess
    """
    research_dir = get_tf_models_research_dir()

    script = (
        research_dir
        / "object_detection"
        / "model_main_tf2.py"
    )

    args = [
        "python",
        str(script),
        "--pipeline_config_path",
        str(pipeline_config_path),
        "--model_dir",
        str(model_dir),
        "--alsologtostderr",
        "--checkpoint_every_n",
        str(checkpoint_every_n),
        "--checkpoint_max_to_keep",
        str(checkpoint_max_to_keep),
    ]

    return run_tfod_command(
        args,
        log_file=log_file,
        background=background,
    )
