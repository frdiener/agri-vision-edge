from __future__ import annotations

import os
import platform
import sys
from typing import Any


def capture_environment() -> dict[str, Any]:
    """
    Capture runtime environment information useful for
    experiment reproducibility.

    Returns
    -------
    dict
        JSON-serializable environment metadata.
    """

    environment = {
        "python": capture_python_environment(),
        "system": capture_system_environment(),
        "cuda_visible_devices":
            os.environ.get("CUDA_VISIBLE_DEVICES"),
    }

    # optional collectors
    tensorflow_info = capture_tensorflow_environment()

    if tensorflow_info is not None:
        environment["tensorflow"] = tensorflow_info

    return environment


# ============================================================
# Python
# ============================================================

def capture_python_environment() -> dict[str, Any]:

    return {
        "version": platform.python_version(),
        "implementation":
            platform.python_implementation(),
        "executable": sys.executable,
    }


# ============================================================
# System
# ============================================================

def capture_system_environment() -> dict[str, Any]:

    return {
        "platform": platform.platform(),
        "system": platform.system(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "hostname": platform.node(),
    }


# ============================================================
# TensorFlow (optional)
# ============================================================

def capture_tensorflow_environment() -> dict[str, Any] | None:
    """
    Capture TensorFlow runtime details if TensorFlow
    is installed.
    """

    try:
        import tensorflow as tf
        import google.protobuf

    except Exception:
        return None

    gpus = []

    try:
        gpus = [
            {
                "name": gpu.name,
                "device_type": gpu.device_type,
            }
            for gpu in tf.config.list_physical_devices("GPU")
        ]

    except Exception:
        pass

    return {
        "version": tf.__version__,
        "protobuf_version":
            google.protobuf.__version__,
        "gpu_devices": gpus,
    }
