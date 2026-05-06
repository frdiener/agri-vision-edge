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

from agri_vision_edge.evaluation import load_event_scalars


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


#
# Batch checkpoint evaluation
#


def evaluate_checkpoints(
    pipeline_config_path: PathLike,
    checkpoint_dir: PathLike,
    output_dir: PathLike,
):
    """
    Evaluate all checkpoints sequentially.

    Each checkpoint is copied into a temporary directory
    and evaluated independently using TensorFlow Object
    Detection evaluation mode.

    Args:
        pipeline_config_path:
            Path to pipeline.config.
        checkpoint_dir:
            Directory containing training checkpoints.
        output_dir:
            Root evaluation output directory.

    Returns:
        pd.DataFrame containing aggregated checkpoint metrics.
    """
    checkpoint_dir = Path(checkpoint_dir)
    output_dir = Path(output_dir)

    checkpoints = list_checkpoints(
        checkpoint_dir,
    )

    all_metrics = []

    for checkpoint in checkpoints:
        step = checkpoint_step(checkpoint)

        print(f"Evaluating checkpoint {step}")

        #
        # Temporary isolated checkpoint directory
        #
        with tempfile.TemporaryDirectory() as tmpdir_raw:
            tmpdir = Path(tmpdir_raw)

            #
            # Copy TF checkpoint files
            #
            for suffix in [
                ".index",
                ".data-00000-of-00001",
            ]:
                src = Path(str(checkpoint) + suffix)

                if not src.exists():
                    raise FileNotFoundError(
                        f"Missing checkpoint file: {src}"
                    )

                dst = tmpdir / src.name

                shutil.copy2(src, dst)

            #
            # TensorFlow checkpoint state file
            #
            (tmpdir / "checkpoint").write_text(
                f'model_checkpoint_path: "{checkpoint.name}"\n'
            )

            #
            # Eval output directory
            #
            eval_dir = output_dir / f"ckpt-{step}"

            eval_dir.mkdir(
                parents=True,
                exist_ok=True,
            )

            #
            # Run evaluation
            #
            launch_eval(
                pipeline_config_path=pipeline_config_path,
                checkpoint_dir=tmpdir,
                model_dir=eval_dir,
                eval_timeout=1,
            )

            #
            # TensorBoard event files are written into:
            #   eval_dir / "eval"
            #
            event_dir = eval_dir / "eval"

            print("Reading metrics from:", event_dir)

            if not event_dir.exists():
                print(
                    f"No eval directory found for checkpoint {step}"
                )
                continue

            event_files = list(
                event_dir.glob("events.out.tfevents.*")
            )

            if not event_files:
                print(
                    f"No TensorBoard event files found for checkpoint {step}"
                )
                continue

            #
            # Parse TensorBoard metrics
            #
            df = load_event_scalars(
                event_dir,
            )

            if df.empty:
                print(
                    f"No metrics found for checkpoint {step}"
                )
                continue

            #
            # Keep latest value per metric tag
            #
            latest = (
                df.sort_values("wall_time")
                .groupby("tag")
                .tail(1)
            )

            metrics = {
                row["tag"]: row["value"]
                for _, row in latest.iterrows()
            }

            metrics["checkpoint"] = str(checkpoint)
            metrics["step"] = step

            all_metrics.append(metrics)

    if not all_metrics:
        return pd.DataFrame()

    result = pd.DataFrame(all_metrics)

    result = result.sort_values(
        "step"
    ).reset_index(drop=True)

    return result


#
# Best checkpoint selection
#


def find_best_checkpoint(
    metrics_df: pd.DataFrame,
    metric: str = "DetectionBoxes_Precision/mAP",
    maximize: bool = True,
):
    """
    Select best checkpoint from metrics table.

    Args:
        metrics_df:
            Evaluation metrics dataframe.
        metric:
            Metric column.
        maximize:
            Whether larger is better.

    Returns:
        pd.Series row for best checkpoint.
    """
    if metric not in metrics_df.columns:
        raise ValueError(
            f"Metric not found: {metric}"
        )

    idx = (
        metrics_df[metric].idxmax()
        if maximize
        else metrics_df[metric].idxmin()
    )

    return metrics_df.loc[idx]
