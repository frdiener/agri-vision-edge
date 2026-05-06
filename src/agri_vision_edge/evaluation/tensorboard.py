"""
TensorBoard event parsing utilities.

Converts TensorBoard event files into tidy pandas DataFrames
for downstream analysis and plotting.
"""

from pathlib import Path
from typing import Union

import pandas as pd
from tensorboard.backend.event_processing.event_accumulator import (
    EventAccumulator,
)


def load_event_scalars(
    logdir: Union[str, Path],
) -> pd.DataFrame:
    """
    Load scalar TensorBoard metrics into a DataFrame.

    Args:
        logdir:
            TensorBoard log directory.

    Returns:
        pd.DataFrame:
            Columns:
                - wall_time
                - step
                - tag
                - value
    """
    logdir = Path(logdir)

    accumulator = EventAccumulator(str(logdir))
    accumulator.Reload()

    rows = []

    for tag in accumulator.Tags()["scalars"]:
        events = accumulator.Scalars(tag)

        for event in events:
            rows.append({
                "wall_time": event.wall_time,
                "step": event.step,
                "tag": tag,
                "value": event.value,
            })

    return pd.DataFrame(rows)
