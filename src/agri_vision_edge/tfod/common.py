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
    """
    Force keras SSD-FPN extractors to upsample with the native resize op.

    By default the FPNLite top-down pathway upsamples via a reshape/tile trick
    (``use_native_resize_op=False``), which TFLite emits as ``PACK`` (builtin op
    83) + ``RESHAPE``. The Teflon/etnaviv NPU delegate (i.MX8M Plus, i.MX93) has
    no ``PACK`` kernel, so every upsample becomes a CPU island that fragments the
    FPN neck into many partitions and forces the surrounding tower convs onto the
    CPU. Enabling ``use_native_resize_op`` replaces each 4-op ``PACK+RESHAPE``
    island with a single ``RESIZE_NEAREST_NEIGHBOR`` (builtin op 97), which the
    delegate *does* support -- the whole graph then runs on the NPU. The two
    upsamplings are mathematically identical (nearest-neighbour 2x), so existing
    checkpoints convert unchanged.

    ``model_builder`` only wires ``use_native_resize_op`` for BiFPN, not the
    regular FPN ``fpn`` branch, so this context manager temporarily overrides the
    relevant ``SSD_KERAS_FEATURE_EXTRACTOR_CLASS_MAP`` entries for the duration of
    ``model_builder.build``. The vendored ``object_detection`` is left untouched.

    Args:
        enabled: When False this is a no-op (default delegate-unfriendly upsample).

    Usage:
        with fpn_native_resize_upsampling(enabled):
            detection_model = model_builder.build(config.model, is_training=False)
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
