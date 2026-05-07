"""
Kaggle runtime metadata collection utilities.

This module provides lightweight provenance capture for experiments
executed inside Kaggle notebook environments.

The implementation intentionally avoids:
- Kaggle API dependencies
- authentication requirements
- internet access
- undocumented internal APIs

All metadata collection is environment-variable based and designed
to fail gracefully outside Kaggle.

Examples
--------
>>> from agri_vision_edge.experiment.kaggle import (
...     capture_kaggle_metadata,
... )
>>>
>>> metadata = capture_kaggle_metadata()
>>>
>>> if metadata is not None:
...     print(metadata["accelerator"])

Notes
-----
Sensitive runtime secrets are intentionally excluded from exported
metadata to avoid accidental credential leakage.
"""

from __future__ import annotations

import os
from typing import Any


#: Kaggle environment variables considered useful for experiment
#: provenance and reproducibility tracking.
KAGGLE_ENV_KEYS: tuple[str, ...] = (
    "KAGGLE_KERNEL_ID",
    "KAGGLE_KERNEL_VERSION_ID",
    "KAGGLE_KERNEL_OWNER",
    "KAGGLE_KERNEL_TITLE",
    "KAGGLE_KERNEL_RUN_TYPE",
    "KAGGLE_URL_BASE",
    "HOSTNAME",
    "TPU_NAME",
)


#: Environment variables that should never be exported because
#: they may contain authentication credentials or sensitive tokens.
EXCLUDED_ENV_KEYS: set[str] = {
    "KAGGLE_DATA_PROXY_TOKEN",
    "KAGGLE_USER_SECRETS_TOKEN",
}


def is_kaggle_environment() -> bool:
    """
    Determine whether execution is occurring inside Kaggle.

    Returns
    -------
    bool
        True if Kaggle runtime indicators are detected.
    """

    return (
        "KAGGLE_KERNEL_RUN_TYPE" in os.environ
        or "KAGGLE_URL_BASE" in os.environ
    )


def capture_kaggle_metadata() -> dict[str, Any] | None:
    """
    Capture Kaggle runtime metadata.

    The returned structure is fully JSON-serializable and intended
    for inclusion in experiment manifests.

    Returns
    -------
    dict[str, Any] or None
        Kaggle runtime metadata if running inside Kaggle,
        otherwise None.

    Notes
    -----
    Sensitive environment variables are intentionally excluded.
    """

    if not is_kaggle_environment():
        return None

    metadata: dict[str, Any] = {}

    for key in KAGGLE_ENV_KEYS:

        if key in EXCLUDED_ENV_KEYS:
            continue

        value = os.environ.get(key)

        if value is not None:
            metadata[key.lower()] = value

    metadata["accelerator"] = detect_kaggle_accelerator()

    return metadata


def detect_kaggle_accelerator() -> str:
    """
    Infer the active Kaggle accelerator type.

    Returns
    -------
    str
        One of:
        - "TPU"
        - "GPU"
        - "CPU"

    Notes
    -----
    GPU detection uses TensorFlow if available.
    If TensorFlow is unavailable or GPU detection fails,
    the function safely falls back to CPU.
    """

    if os.environ.get("TPU_NAME"):
        return "TPU"

    try:
        import tensorflow as tf

        gpus = tf.config.list_physical_devices("GPU")

        if gpus:
            return "GPU"

    except Exception:
        pass

    return "CPU"
