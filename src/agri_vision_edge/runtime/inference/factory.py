"""
Detector-runtime factory.

Picks the right :class:`~.base.BaseRuntime` for a model, so callers
(``ave infer`` / ``ave benchmark``) stay model-agnostic:

- **a SavedModel directory** → the pre-conversion TF reference:
  :class:`~.saved_model.SavedModelRuntime`.
- **4 outputs** → SSD MobileNetV2 (post-NMS boxes/scores/classes/count):
  :class:`~.tflite.TFLiteRuntime`.
- **3 outputs** → YOLOv7-tiny raw grids: :class:`~.yolo.YoloTFLiteRuntime`.

TFLite models are dispatched on output shape; the SavedModel is dispatched on
the path first, because peeking at it with ``Interpreter`` would raise.

Imports are deferred into the function so importing this module stays light
(neither runtime — nor TensorFlow — is pulled in until a runtime is built).
"""

from __future__ import annotations

from pathlib import Path

# Mirror tflite.py's interpreter import for the cheap output-count peek.
try:
    from tflite_runtime.interpreter import Interpreter
except ImportError:
    import tensorflow as tf

    Interpreter = tf.lite.Interpreter

from .base import BaseRuntime
from .tflite import DEFAULT_TEFLON_LIB


def build_runtime(
    model_path: str | Path,
    *,
    delegate_path: str | None = DEFAULT_TEFLON_LIB,
    score_threshold: float | None = None,
    iou_threshold: float | None = None,
    size: int | None = None,
) -> BaseRuntime:
    """
    Build the detector runtime matching the model's output layout.

    ``score_threshold`` / ``iou_threshold`` left as ``None`` are resolved from
    the model's embedded metadata by the runtime (falling back to built-in
    defaults); pass explicit values to override.
    """

    # Checked before the peek: `Interpreter` raises on a SavedModel directory,
    # so the reference runtime has to be dispatched on the path shape instead of
    # the output layout.
    from .saved_model import is_saved_model_dir

    if is_saved_model_dir(model_path):
        from .saved_model import SavedModelRuntime

        return SavedModelRuntime(
            model_path=model_path,
            score_threshold=score_threshold,
        )

    peek = Interpreter(model_path=str(model_path))
    peek.allocate_tensors()
    num_outputs = len(peek.get_output_details())
    del peek

    if num_outputs == 4:
        from .tflite import TFLiteRuntime

        return TFLiteRuntime(
            model_path=model_path,
            delegate_path=delegate_path,
            score_threshold=score_threshold,
        )

    from .yolo import YoloTFLiteRuntime

    return YoloTFLiteRuntime(
        model_path=model_path,
        delegate_path=delegate_path,
        score_threshold=score_threshold,
        iou_threshold=iou_threshold,
        size=size,
    )
