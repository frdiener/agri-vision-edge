"""
Utilities for vendored third-party libraries.
"""

from pathlib import Path
import sys


def setup_tensorflow_models():
    """
    Add TensorFlow Models research directories to PYTHONPATH.
    """
    root = Path(__file__).resolve().parents[2]

    research_dir = root / "third_party" / "models" / "research"

    paths = [
        research_dir,
        research_dir / "slim",
    ]

    for path in paths:
        path_str = str(path)

        if path_str not in sys.path:
            sys.path.insert(0, path_str)
