"""
TensorBoard event parsing utilities.

Parses TensorFlow event files into tidy pandas DataFrames.
Compatible with TensorFlow 2 Object Detection API logs.
"""

from pathlib import Path
from typing import Union

import pandas as pd
import tensorflow as tf


def load_event_scalars(
    logdir: Union[str, Path],
) -> pd.DataFrame:
    """
    Load TensorBoard scalar metrics from event files.

    Args:
        logdir:
            Directory containing TensorBoard event files.

    Returns:
        pd.DataFrame with columns:
            - wall_time
            - step
            - tag
            - value
    """
    logdir = Path(logdir)

    event_files = sorted(
        logdir.glob("events.out.tfevents.*")
    )

    rows = []

    for event_file in event_files:

        for event in tf.compat.v1.train.summary_iterator(
            str(event_file)
        ):

            step = event.step
            wall_time = event.wall_time

            if not event.summary.value:
                continue

            for value in event.summary.value:

                tag = value.tag

                # --- scalar summary ---
                if value.HasField("simple_value"):

                    rows.append({
                        "wall_time": wall_time,
                        "step": step,
                        "tag": tag,
                        "value": value.simple_value,
                    })

                # --- tensor summary ---
                elif value.HasField("tensor"):

                    tensor = tf.make_ndarray(value.tensor)

                    if tensor.shape == ():
                        scalar = float(tensor)

                    elif tensor.size == 1:
                        scalar = float(tensor.reshape(-1)[0])

                    else:
                        continue

                    rows.append({
                        "wall_time": wall_time,
                        "step": step,
                        "tag": tag,
                        "value": scalar,
                    })

    df = pd.DataFrame(rows)

    if not df.empty:
        df = df.sort_values(
            ["tag", "step"]
        ).reset_index(drop=True)

    return df
