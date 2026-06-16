"""
Utilities and integrations for vendored third-party libraries.

This subpackage contains lightweight wrappers and helpers for
external code vendored into agri_vision_edge.

Vendored projects:
- TensorFlow Models (object_detection, slim)

(PhenoBench is no longer vendored — use the `phenobench` PyPI package directly.)
"""

from .tensorflow_models import setup_tensorflow_models

__all__ = [
    "setup_tensorflow_models",
]
