"""
Checkpoint evaluation and metric aggregation utilities.

Provides helpers for evaluating multiple TensorFlow Object
Detection checkpoints and summarizing validation metrics.
"""

from pathlib import Path
from typing import Optional, Union, List

import re
import tempfile
import shutil

import pandas as pd

from .tensorboard import load_event_scalars
from ..tfod.eval import launch_eval


PathLike = Union[str, Path]


def find_checkpoints(
    checkpoint_dir: PathLike,
) -> List[Path]:
    """
    Discover TensorFlow checkpoints.

    Args:
        checkpoint_dir:
            Directory containing ckpt-* files.

    Returns:
        Sorted checkpoint base paths.
    """
    checkpoint_dir = Path(checkpoint_dir)

    checkpoints = []

    for path in checkpoint_dir.glob("ckpt-*.index"):
        stem = path.with_suffix("")
        checkpoints.append(stem)

    checkpoints = sorted(
        checkpoints,
        key=lambda p: int(p.name.split("-")[-1]),
    )

    return checkpoints


def summarize_checkpoint_metrics(
    metrics_df: pd.DataFrame,
    metric: str = "DetectionBoxes_Precision/mAP",
) -> pd.Series:
    """
    Select best checkpoint according to metric.

    Args:
        metrics_df:
            Aggregated checkpoint metrics.
        metric:
            Metric used for ranking.

    Returns:
        Best checkpoint row.
    """
    if metric not in metrics_df.columns:
        raise ValueError(
            f"Metric '{metric}' not found."
        )

    idx = metrics_df[metric].idxmax()

    return metrics_df.loc[idx]

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
