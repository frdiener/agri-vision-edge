"""
Evaluation utilities for TensorFlow Object Detection experiments.

Provides:
- TensorBoard event parsing
- Training curve visualization
- Experiment summaries
- Confusion matrix generation
"""

from .tensorboard import load_event_scalars
from .curves import (
    plot_loss_curves,
    plot_learning_rate,
    plot_steps_per_second,
    available_tags,
    plot_map_curves,
    plot_recall_curves,
    plot_checkpoint_metrics,
)
from .checkpoint import (
    evaluate_checkpoints,
    summarize_checkpoint_metrics,
    find_best_checkpoint,
    list_checkpoints,
    checkpoint_step,
)

__all__ = [
    "load_event_scalars",
    "plot_loss_curves",
    "plot_learning_rate",
    "plot_steps_per_second",
    "available_tags",
    "plot_map_curves",
    "plot_recall_curves",
    "plot_checkpoint_metrics",
    "evaluate_checkpoints",
    "summarize_checkpoint_metrics",
    "evaluate_checkpoints",
    "find_best_checkpoint",
    "list_checkpoints",
    "checkpoint_step",

]
