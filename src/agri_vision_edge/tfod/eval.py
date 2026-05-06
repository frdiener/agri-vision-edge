"""
TensorFlow Object Detection evaluation utilities.

Provides helpers for:

- launching TF-OD evaluation
- enumerating checkpoints
- evaluating multiple checkpoints
- extracting evaluation metrics
- selecting best checkpoints
"""

from pathlib import Path
from typing import Iterable, Optional, Sequence, Union

import re
import subprocess
import tempfile
import shutil

import pandas as pd

from .common import (
    get_tf_models_research_dir,
    run_tfod_command,
)


PathLike = Union[str, Path]


#
# Checkpoint discovery
#


def list_checkpoints(
    checkpoint_dir: PathLike,
):
    """
    List available TF checkpoints.

    Args:
        checkpoint_dir:
            Training checkpoint directory.

    Returns:
        Sorted list of checkpoint prefixes.
    """
    checkpoint_dir = Path(checkpoint_dir)

    checkpoints = []

    for path in checkpoint_dir.glob("ckpt-*.index"):
        stem = path.stem

        match = re.match(r"ckpt-(\d+)", stem)

        if match is None:
            continue

        step = int(match.group(1))

        checkpoints.append(
            (
                step,
                checkpoint_dir / stem,
            )
        )

    checkpoints.sort(key=lambda x: x[0])

    return [p for _, p in checkpoints]


def checkpoint_step(
    checkpoint_path: PathLike,
):
    """
    Extract numeric checkpoint step.

    Args:
        checkpoint_path:
            Checkpoint prefix.

    Returns:
        Integer step number.
    """
    checkpoint_path = Path(checkpoint_path)

    match = re.search(
        r"ckpt-(\d+)",
        checkpoint_path.name,
    )

    if match is None:
        raise ValueError(
            f"Invalid checkpoint name: {checkpoint_path}"
        )

    return int(match.group(1))


#
# Single evaluation
#


def launch_eval(
    pipeline_config_path: PathLike,
    checkpoint_dir: PathLike,
    model_dir: PathLike,
    eval_timeout: int = 1,
    log_file: Optional[PathLike] = None,
):
    """
    Launch TF-OD evaluation.

    Args:
        pipeline_config_path:
            pipeline.config path.
        checkpoint_dir:
            Directory containing checkpoints.
        model_dir:
            Evaluation output directory.
        eval_timeout:
            Timeout in seconds.
        log_file:
            Optional log file.
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
        "--checkpoint_dir",
        str(checkpoint_dir),
        "--eval_timeout",
        str(eval_timeout),
        "--alsologtostderr",
    ]

    return run_tfod_command(
        args,
        log_file=log_file,
    )
