"""
Training curve visualization utilities.

Provides plots for:
- training losses
- learning rate schedules
- TensorFlow Object Detection API metrics
"""

from pathlib import Path
from typing import Iterable, Optional, Sequence, Union

import matplotlib.pyplot as plt
import pandas as pd


DEFAULT_FIGSIZE = (8, 5)
DEFAULT_DPI = 300


def _prepare_axis(ax):
    """
    Apply consistent styling to matplotlib axes.
    """
    ax.grid(True, alpha=0.3)
    ax.set_axisbelow(True)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _smooth_series(
    values: pd.Series,
    smoothing: float,
) -> pd.Series:
    """
    Apply exponential moving average smoothing.

    Args:
        values:
            Input values.
        smoothing:
            Smoothing factor in [0, 1).

    Returns:
        Smoothed series.
    """
    if smoothing <= 0:
        return values

    alpha = 1.0 - smoothing

    return values.ewm(alpha=alpha).mean()


def available_tags(
    df: pd.DataFrame,
) -> list[str]:
    """
    Return available metric tags.

    Args:
        df:
            Metrics dataframe.

    Returns:
        Sorted list of tags.
    """
    if "tag" not in df.columns:
        return []

    return sorted(df["tag"].unique())


def plot_metric_curves(
    df: pd.DataFrame,
    tags: Sequence[str],
    title: str,
    ylabel: str,
    smoothing: float = 0.0,
    figsize=DEFAULT_FIGSIZE,
    save_path: Optional[Union[str, Path]] = None,
    dpi: int = DEFAULT_DPI,
):
    """
    Generic multi-metric plotting utility.

    Args:
        df:
            DataFrame from `load_event_scalars`.
        tags:
            Metric tags to plot.
        title:
            Figure title.
        ylabel:
            Y-axis label.
        smoothing:
            Exponential smoothing factor in [0, 1).
        figsize:
            Figure size.
        save_path:
            Optional export path.
        dpi:
            Export DPI for raster formats.

    Returns:
        Tuple[Figure, Axes]
    """
    if "tag" not in df.columns:
        raise ValueError(
            "Input dataframe does not contain a 'tag' column."
        )

    fig, ax = plt.subplots(figsize=figsize)

    plotted = False

    for tag in tags:
        subset = df[df["tag"] == tag]

        if subset.empty:
            continue

        subset = subset.sort_values("step")

        values = _smooth_series(
            subset["value"],
            smoothing=smoothing,
        )

        label = tag.split("/")[-1]

        ax.plot(
            subset["step"],
            values,
            label=label,
            linewidth=2,
        )

        plotted = True

    if not plotted:
        raise ValueError(
            f"No matching tags found. Available tags:\n"
            f"{available_tags(df)}"
        )

    ax.set_title(title)
    ax.set_xlabel("Training Step")
    ax.set_ylabel(ylabel)

    _prepare_axis(ax)

    ax.legend(frameon=False)

    fig.tight_layout()

    if save_path is not None:
        fig.savefig(
            save_path,
            bbox_inches="tight",
            dpi=dpi,
        )

    return fig, ax


def plot_loss_curves(
    df: pd.DataFrame,
    smoothing: float = 0.6,
    save_path: Optional[Union[str, Path]] = None,
    dpi: int = DEFAULT_DPI,
):
    """
    Plot TensorFlow Object Detection loss curves.

    Args:
        df:
            DataFrame from `load_event_scalars`.
        smoothing:
            EMA smoothing factor in [0, 1).
        save_path:
            Optional export path.
        dpi:
            Export DPI.

    Returns:
        Tuple[Figure, Axes]
    """
    tags = [
        "Loss/total_loss",
        "Loss/classification_loss",
        "Loss/localization_loss",
        "Loss/regularization_loss",
    ]

    return plot_metric_curves(
        df=df,
        tags=tags,
        title="Training Loss",
        ylabel="Loss",
        smoothing=smoothing,
        save_path=save_path,
        dpi=dpi,
    )


def plot_learning_rate(
    df: pd.DataFrame,
    smoothing: float = 0.0,
    save_path: Optional[Union[str, Path]] = None,
    dpi: int = DEFAULT_DPI,
):
    """
    Plot learning rate schedule.

    Args:
        df:
            DataFrame from `load_event_scalars`.
        smoothing:
            EMA smoothing factor.
        save_path:
            Optional export path.
        dpi:
            Export DPI.

    Returns:
        Tuple[Figure, Axes]
    """
    return plot_metric_curves(
        df=df,
        tags=["learning_rate"],
        title="Learning Rate Schedule",
        ylabel="Learning Rate",
        smoothing=smoothing,
        figsize=(8, 4),
        save_path=save_path,
        dpi=dpi,
    )


def plot_steps_per_second(
    df: pd.DataFrame,
    smoothing: float = 0.5,
    save_path: Optional[Union[str, Path]] = None,
    dpi: int = DEFAULT_DPI,
):
    """
    Plot training throughput.

    Args:
        df:
            DataFrame from `load_event_scalars`.
        smoothing:
            EMA smoothing factor.
        save_path:
            Optional export path.
        dpi:
            Export DPI.

    Returns:
        Tuple[Figure, Axes]
    """
    return plot_metric_curves(
        df=df,
        tags=["steps_per_sec"],
        title="Training Throughput",
        ylabel="Steps / Second",
        smoothing=smoothing,
        figsize=(8, 4),
        save_path=save_path,
        dpi=dpi,
    )


def plot_map_curves(
    df: pd.DataFrame,
    smoothing: float = 0.0,
    save_path: Optional[Union[str, Path]] = None,
    dpi: int = DEFAULT_DPI,
):
    """
    Plot validation mAP metrics.
    """
    tags = [
        "DetectionBoxes_Precision/mAP",
        "DetectionBoxes_Precision/mAP@.50IOU",
        "DetectionBoxes_Precision/mAP@.75IOU",
    ]

    return plot_metric_curves(
        df=df,
        tags=tags,
        title="Validation mAP",
        ylabel="mAP",
        smoothing=smoothing,
        save_path=save_path,
        dpi=dpi,
    )


def plot_recall_curves(
    df: pd.DataFrame,
    smoothing: float = 0.0,
    save_path: Optional[Union[str, Path]] = None,
    dpi: int = DEFAULT_DPI,
):
    """
    Plot validation recall metrics.
    """
    tags = [
        "DetectionBoxes_Recall/AR@1",
        "DetectionBoxes_Recall/AR@10",
        "DetectionBoxes_Recall/AR@100",
    ]

    return plot_metric_curves(
        df=df,
        tags=tags,
        title="Validation Recall",
        ylabel="Average Recall",
        smoothing=smoothing,
        save_path=save_path,
        dpi=dpi,
    )
