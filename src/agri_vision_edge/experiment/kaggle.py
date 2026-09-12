"""Capture Kaggle experiment provenance from non-secret environment variables.

Collection requires no Kaggle API, authentication, or network access and
returns no metadata outside Kaggle.
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
    """Return whether Kaggle runtime indicators are present."""

    return "KAGGLE_KERNEL_RUN_TYPE" in os.environ or "KAGGLE_URL_BASE" in os.environ


def capture_kaggle_metadata() -> dict[str, Any] | None:
    """Return JSON-safe, non-secret Kaggle metadata, or ``None`` outside Kaggle."""

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
    """Return ``TPU``, ``GPU``, or ``CPU``; detection failures fall back to CPU."""

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
