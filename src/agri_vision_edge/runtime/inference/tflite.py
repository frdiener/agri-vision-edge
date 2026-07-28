"""
TensorFlow Lite runtime wrapper.

Supports:

- CPU execution
- optional delegates
- quantized models
- float-output models
- configuration read from the model's embedded metadata
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

# Prefer the standalone tflite-runtime (the on-device path, installed via the
# `device` extra). When it is absent — e.g. the preparation/conversion env,
# which installs the `prep` extra instead — fall back to full TensorFlow's
# tf.lite so conversion never depends on the standalone interpreter.
try:
    from tflite_runtime.interpreter import Interpreter, load_delegate
except ImportError:
    import tensorflow as tf

    Interpreter = tf.lite.Interpreter
    load_delegate = tf.lite.experimental.load_delegate

from .base import (
    BaseRuntime,
    Detection,
)
from .model_metadata import ModelMetadata

DEFAULT_TEFLON_LIB = "/usr/lib/libteflon.so"


def load_delegates_with_status(delegate_path) -> tuple[list, str | None]:
    """
    Load the optional TFLite delegate, reporting whether it actually loaded.

    Returns ``(delegates, active_path)`` where ``delegates`` is suitable for
    ``Interpreter(experimental_delegates=...)`` and ``active_path`` is the
    delegate that is really in use, or ``None`` for plain CPU execution.

    The fallback to CPU is deliberate -- a missing or unloadable delegate must
    not abort a sweep -- but it is silent in the results unless the *effective*
    delegate is recorded: a run that asked for the NPU and quietly got the CPU
    otherwise looks exactly like a successful NPU run in the artifacts.
    """

    if delegate_path is None:
        return [], None

    delegate_path = Path(delegate_path)

    if not delegate_path.exists():
        print(f"[runtime] delegate not found: {delegate_path}")

        return [], None

    try:
        delegate = load_delegate(str(delegate_path))

        print(f"[runtime] loaded delegate: {delegate_path}")

        return [delegate], str(delegate_path)

    except Exception as e:
        print(f"[runtime] failed to load delegate: {e}")

        return [], None


def load_delegates(delegate_path):
    """
    Load the optional TFLite delegate (delegate list only).

    Thin wrapper over :func:`load_delegates_with_status` for callers that do not
    care which delegate ended up being used.
    """

    delegates, _ = load_delegates_with_status(delegate_path)

    return delegates


class TFLiteRuntime(BaseRuntime):
    """
    TensorFlow Lite inference runtime.
    """

    def __init__(
        self,
        model_path: str | Path,
        *,
        delegate_path: str | None = DEFAULT_TEFLON_LIB,
        score_threshold: float | None = None,
    ):

        self.model_path = Path(model_path)

        # Labels, input normalization and post-processing defaults all come from
        # the model's embedded metadata (see model_metadata.ModelMetadata).
        self.metadata = ModelMetadata.load(self.model_path)

        self.labels = self.metadata.labels

        # An explicit score_threshold always wins; otherwise use the model's own
        # embedded default, falling back to 0.0 (keep every detection).
        if score_threshold is None:
            score_threshold = (
                self.metadata.score_threshold
                if self.metadata.score_threshold is not None
                else 0.0
            )

        self.score_threshold = score_threshold

        # `active_delegate` is the delegate actually in use (None = CPU), which
        # is what the benchmark artifacts have to record: `delegate_path` is
        # only what was *asked for*.
        delegates, self.active_delegate = self._load_delegates(delegate_path)

        self.requested_delegate = (
            str(delegate_path) if delegate_path is not None else None
        )

        self.interpreter = Interpreter(
            model_path=str(self.model_path),
            experimental_delegates=delegates,
        )

        self.interpreter.allocate_tensors()

        self.input_details = self.interpreter.get_input_details()

        self.output_details = self.interpreter.get_output_details()

        print("\n=== OUTPUT DETAILS ===")

        for detail in self.output_details:
            print(
                detail["name"],
                detail["dtype"],
                detail["quantization"],
                detail["shape"],
            )

        if len(self.output_details) != 4:
            output_names = [d.get("name", "<unknown>") for d in self.output_details]

            raise RuntimeError(
                "Unsupported TFLite detector.\n"
                "Expected SSD-style outputs "
                f"(boxes,scores,num,classes),\n"
                f"got {len(self.output_details)} "
                f"output tensor(s): {output_names}"
            )

        self._input_size = int(self.input_details[0]["shape"][1])

        # Input normalization (applied as (pixels - mean) / std for float models)
        # comes from the embedded metadata, defaulting to the SSD MobileNetV2
        # [0, 255] -> [-1, 1] mapping (mean = std = 127.5).
        self._norm_mean = np.array(self.metadata.norm_mean, dtype=np.float32)
        self._norm_std = np.array(self.metadata.norm_std, dtype=np.float32)

    @property
    def input_size(self) -> int:

        return self._input_size

    #
    # Delegates
    #

    def _load_delegates(
        self,
        delegate_path,
    ):

        return load_delegates_with_status(delegate_path)

    #
    # Preprocessing
    #

    def preprocess(
        self,
        image,
    ):

        image = cv2.resize(
            image,
            (
                self.input_size,
                self.input_size,
            ),
        )

        input_detail = self.input_details[0]

        dtype = input_detail["dtype"]

        #
        # INT8
        #

        if dtype == np.int8:
            image = image.astype(np.float32)

            image = image - 128.0

            image = np.round(image)

            image = np.clip(
                image,
                -128,
                127,
            )

            image = image.astype(np.int8)

        #
        # FP32
        #
        # A float input has no quantization params to encode the expected domain,
        # so the normalization must be applied explicitly. The graph expects
        # already-normalized input (e.g. [-1, 1] = (px - 127.5) / 127.5); feeding
        # raw [0, 255] silently wrecks detections. mean/std come from the model's
        # embedded metadata (see ModelMetadata / self.metadata.norm_*).
        #

        else:
            image = image.astype(np.float32)

            image = (image - self._norm_mean) / self._norm_std

        image = np.expand_dims(
            image,
            axis=0,
        )

        return image

    #
    # Quantization helpers
    #

    @staticmethod
    def dequantize(
        tensor,
        quantization,
    ):

        scale, zero_point = quantization

        return scale * (tensor.astype(np.float32) - zero_point)

    #
    # Inference
    #

    def predict(
        self,
        image,
    ):

        input_tensor = self.preprocess(image)

        self.interpreter.set_tensor(
            self.input_details[0]["index"],
            input_tensor,
        )

        self.interpreter.invoke()

        raw_scores = self.interpreter.get_tensor(self.output_details[0]["index"])

        raw_boxes = self.interpreter.get_tensor(self.output_details[1]["index"])

        raw_num = self.interpreter.get_tensor(self.output_details[2]["index"])

        raw_classes = self.interpreter.get_tensor(self.output_details[3]["index"])

        # Auto-detect quantized outputs from the score tensor's dtype rather than
        # relying on external metadata. The SSD models export float outputs
        # (inference_output_type = tf.float32), but a genuinely INT8-output graph
        # is handled correctly without configuration.
        dequantize_outputs = self.output_details[0]["dtype"] in (np.int8, np.uint8)

        #
        # Quantized outputs
        #

        if dequantize_outputs:
            scores = self.dequantize(
                raw_scores,
                self.output_details[0]["quantization"],
            )

            boxes = self.dequantize(
                raw_boxes,
                self.output_details[1]["quantization"],
            )

            classes = self.dequantize(
                raw_classes,
                self.output_details[3]["quantization"],
            )

            num = int(
                np.squeeze(
                    self.dequantize(
                        raw_num,
                        self.output_details[2]["quantization"],
                    )
                )
            )

        #
        # Float outputs
        #

        else:
            scores = raw_scores

            boxes = raw_boxes

            classes = raw_classes

            num = int(np.squeeze(raw_num))

        scores = scores[0]
        boxes = boxes[0]

        classes_float = classes[0]

        num = max(
            0,
            min(
                num,
                len(scores),
                len(boxes),
                len(classes_float),
            ),
        )

        #
        # COCO compatibility
        #
        # The SSD category tensor is 0-based; shifting by 1 yields the 1-based
        # category ids of the COCO/label-map convention (and of the labels dict
        # in ModelMetadata: line 0 -> category_id 1).
        class_offset = 1

        detections = []

        kept = 0

        rejected_score = 0
        rejected_nan_score = 0
        rejected_nan_class = 0

        for box, cls_id, score in zip(
            boxes[:num],
            classes_float[:num],
            scores[:num],
            strict=False,
        ):
            if np.isnan(score):
                rejected_nan_score += 1
                continue

            if np.isnan(cls_id):
                rejected_nan_class += 1
                continue

            score = float(score)

            if score < self.score_threshold:
                rejected_score += 1
                continue

            cls_id = int(np.round(cls_id))

            kept += 1

            detections.append(
                Detection(
                    category_id=(cls_id + class_offset),
                    score=score,
                    bbox=box.tolist(),
                )
            )

        return detections
