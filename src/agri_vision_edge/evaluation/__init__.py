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
)

__all__ = [
    "load_event_scalars",
    "plot_loss_curves",
    "plot_learning_rate",
    "plot_steps_per_second",
    "available_tags",
]

