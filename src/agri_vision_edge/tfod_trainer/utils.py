"""
Miscellaneous helpers.
"""

from __future__ import annotations

import json
import pprint
import time


def current_learning_rate(lr):
    """
    Resolve LR schedule or scalar.
    """

    return lr() if callable(lr) else lr


def metrics_to_float(metrics):
    """
    Convert tensors to JSON-safe values.
    """

    return {
        k: (
            float(v.numpy())
            if hasattr(v, "numpy")
            else float(v)
        )
        for k, v in metrics.items()
    }


def pretty_print_metrics(
    step: int,
    metrics: dict,
    time_taken: float,
):
    print(
        f"Step {step} "
        f"({time_taken:.3f}s/step)"
    )

    print(
        pprint.pformat(
            metrics,
            width=120,
        )
    )


def write_json(
    path,
    payload,
):
    with open(path, "w") as f:
        json.dump(
            payload,
            f,
            indent=2,
        )
