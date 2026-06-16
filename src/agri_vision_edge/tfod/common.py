"""
Shared TensorFlow Object Detection infrastructure utilities.

Provides:

- TensorFlow Models path discovery
- TF-OD subprocess environment setup
- generic subprocess execution helpers
"""

import os
import subprocess
from collections.abc import Sequence
from pathlib import Path

import agri_vision_edge

PathLike = str | Path


def get_tf_models_research_dir() -> Path:
    """
    Get vendored TensorFlow Models research directory.

    Returns:
        Path to tensorflow_models/research
    """
    ave_root = Path(
        agri_vision_edge.__file__
    ).resolve().parent

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
        Environment dictionary with correct PYTHONPATH.
    """
    env = os.environ.copy()

    research_dir = get_tf_models_research_dir()
    slim_dir = research_dir / "slim"

    existing = env.get("PYTHONPATH", "")

    paths = [
        str(research_dir),
        str(slim_dir),
    ]

    if existing:
        paths.append(existing)

    env["PYTHONPATH"] = ":".join(paths)

    return env


def run_tfod_command(
    args: Sequence[str],
    log_file: PathLike | None = None,
    background: bool = False,
):
    """
    Run TensorFlow Object Detection subprocess.

    Args:
        args:
            Command arguments.
        log_file:
            Optional log file path.
        background:
            Run asynchronously.

    Returns:
        subprocess.Popen or subprocess.CompletedProcess
    """
    env = build_tfod_env()

    #
    # Background mode
    #

    if background:
        stdout = None
        stderr = None

        log_handle = None

        if log_file is not None:
            log_handle = open(log_file, "w")

            stdout = log_handle
            stderr = subprocess.STDOUT

        process = subprocess.Popen(
            args,
            env=env,
            stdout=stdout,
            stderr=stderr,
        )

        #
        # Keep handle alive
        #
        process._log_handle = log_handle

        return process

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
