"""
Training curve visualization utilities.
"""

from pathlib import Path
from typing import Optional, Union

import matplotlib.pyplot as plt
import pandas as pd


def _prepare_axis(ax):
    ax.grid(True, alpha=0.3)
    ax.set_axisbelow(True)


def plot_loss_curves(
    df: pd.DataFrame,
    save_path: Optional[Union[str, Path]] = None,
):
    """
    Plot TensorFlow Object Detection loss curves.

    Args:
        df:
            DataFrame from `load_event_scalars`.
        save_path:
            Optional output path.
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    tags = [
        "Loss/total_loss",
        "Loss/classification_loss",
        "Loss/localization_loss",
    ]

    for tag in tags:
        subset = df[df["tag"] == tag]

        if subset.empty:
            continue

        ax.plot(
            subset["step"],
            subset["value"],
            label=tag.split("/")[-1],
        )

    ax.set_title("Training Loss")
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Loss")

    _prepare_axis(ax)

    ax.legend()

    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, bbox_inches="tight")

    return fig, ax


def plot_learning_rate(
    df: pd.DataFrame,
    save_path: Optional[Union[str, Path]] = None,
):
    """
    Plot learning rate schedule.

    Args:
        df:
            DataFrame from `load_event_scalars`.
        save_path:
            Optional output path.
    """
    subset = df[df["tag"] == "learning_rate"]

    fig, ax = plt.subplots(figsize=(8, 4))

    ax.plot(
        subset["step"],
        subset["value"],
    )

    ax.set_title("Learning Rate")
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Learning Rate")

    _prepare_axis(ax)

    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, bbox_inches="tight")

    return fig, ax
