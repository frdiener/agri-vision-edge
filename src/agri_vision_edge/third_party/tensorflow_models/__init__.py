"""
TensorFlow Models integration utilities.
"""

from pathlib import Path
import sys


def setup_tensorflow_models():
    """
    Add TensorFlow Models research directories to PYTHONPATH.

    Enables imports such as:

        from object_detection.builders import model_builder
    """
    root = Path(__file__).resolve().parent

    research_dir = root / "research"
    slim_dir = research_dir / "slim"

    for path in [research_dir, slim_dir]:
        path_str = str(path.resolve())

        if path_str not in sys.path:
            sys.path.insert(0, path_str)
