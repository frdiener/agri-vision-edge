"""
TensorFlow Object Detection training utilities.

Provides helpers for launching TF-OD training, evaluation,
and export scripts from vendored TensorFlow Models code.
"""

from pathlib import Path
from typing import Optional, Sequence, Union
import subprocess
import os

import agri_vision_edge


PathLike = Union[str, Path]


def get_tf_models_research_dir() -> Path:
    """
    Get the vendored TensorFlow Models research directory.

    Returns:
        Path to tensorflow_models/research
    """
    ave_root = Path(agri_vision_edge.__file__).resolve().parent

    return (
        ave_root
        / "third_party"
        / "tensorflow_models"
        / "research"
    )


def build_tfod_env():
    """
    Build environment variables for TF-OD subprocesses.

    Returns:
        dict: environment with proper PYTHONPATH
    """
    env = os.environ.copy()

    research_dir = get_tf_models_research_dir()
    slim_dir = research_dir / "slim"

    pythonpath = env.get("PYTHONPATH", "")

    paths = [
        str(research_dir),
        str(slim_dir),
    ]

    if pythonpath:
        paths.append(pythonpath)

    env["PYTHONPATH"] = ":".join(paths)

    return env


def run_tfod_command(
    args: Sequence[str],
    log_file: Optional[PathLike] = None,
    background: bool = False,
):
    """
    Run a TF-OD subprocess command.

    Args:
        args:
            Command arguments.
        log_file:
            Optional log file path.
        background:
            If True, launch process asynchronously.

    Returns:
        subprocess.Popen or subprocess.CompletedProcess
    """
    env = build_tfod_env()

    #
    # Interactive background mode
    #

    if background:
        stdout = None
        stderr = None

        if log_file is not None:
            log_handle = open(log_file, "w")

            stdout = log_handle
            stderr = subprocess.STDOUT

        return subprocess.Popen(
            args,
            env=env,
            stdout=stdout,
            stderr=stderr,
        )

    #
    # Blocking streaming mode
    #

    process = subprocess.Popen(
        args,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    log_handle = None

    if log_file is not None:
        log_handle = open(log_file, "w")

    try:
        for line in process.stdout:
            print(line, end="")

            if log_handle is not None:
                log_handle.write(line)

        process.wait()

    finally:
        if log_handle is not None:
            log_handle.close()

    if process.returncode != 0:
        raise subprocess.CalledProcessError(
            process.returncode,
            args,
        )

    return process


def launch_training(
    pipeline_config_path: PathLike,
    model_dir: PathLike,
    checkpoint_every_n: int = 1000,
    log_file: Optional[PathLike] = None,
    background: bool = False,
):
    """
    Launch TF-OD training.

    Args:
        pipeline_config_path:
            pipeline.config path.
        model_dir:
            Output training directory.
        checkpoint_every_n:
            Checkpoint frequency.
        log_file:
            Optional training log file.
        background:
            Run asynchronously.
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
    ]

    return run_tfod_command(
        args,
        log_file=log_file,
        background=background,
    )


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
            Training checkpoint directory.
        model_dir:
            Evaluation output directory.
        eval_timeout:
            Eval timeout in seconds.
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
