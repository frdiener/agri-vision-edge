"""
Shared TensorFlow Object Detection infrastructure utilities.

Provides:

- TensorFlow Models path discovery
- FPN native-resize upsampling override (NPU-delegatable top-down)
"""

import functools
from collections.abc import Iterator
from contextlib import contextmanager
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


# Keras SSD-FPN feature extractors whose top-down upsampling can be switched to
# the native resize op. Their constructors all accept ``use_native_resize_op``.
_FPN_KERAS_EXTRACTORS = (
    "ssd_mobilenet_v2_fpn_keras",
    "ssd_mobilenet_v1_fpn_keras",
)


@contextmanager
def fpn_native_resize_upsampling(enabled: bool = True) -> Iterator[None]:
    """Temporarily make Keras SSD-FPN extractors use native nearest-neighbor resize.

    This preserves checkpoint semantics while avoiding delegate-incompatible
    ``PACK`` upsampling; when ``enabled`` is false, the context is a no-op.
    """
    if not enabled:
        yield
        return

    from object_detection.builders import model_builder

    cmap = model_builder.SSD_KERAS_FEATURE_EXTRACTOR_CLASS_MAP
    saved = {k: cmap[k] for k in _FPN_KERAS_EXTRACTORS if k in cmap}
    try:
        for name, cls in saved.items():
            cmap[name] = functools.partial(cls, use_native_resize_op=True)
        yield
    finally:
        cmap.update(saved)
