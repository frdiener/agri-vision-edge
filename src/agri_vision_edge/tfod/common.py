"""
Shared TensorFlow Object Detection infrastructure utilities.

Provides:

- TensorFlow Models path discovery
"""

from pathlib import Path

import agri_vision_edge


def get_tf_models_research_dir() -> Path:
    """
    Get vendored TensorFlow Models research directory.

    Returns:
        Path to tensorflow_models/research
    """
    ave_root = Path(agri_vision_edge.__file__).resolve().parent

    return ave_root / "third_party" / "tensorflow_models" / "research"
