"""
Utilities and integrations for vendored third-party libraries.

This subpackage contains lightweight wrappers and helpers for
external code vendored into agri_vision_edge.

Vendored projects:
- TensorFlow Models (object_detection, slim)
- PhenoBench
"""

from .tensorflow_models import setup_tensorflow_models

__all__ = [
    "setup_tensorflow_models",
]
