"""
TensorFlow Lite runtime wrapper.

Supports:

- CPU execution
- optional delegates
- quantized models
- float-output models
- metadata sidecars
"""

from __future__ import annotations

import json
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

DEFAULT_TEFLON_LIB = "/usr/lib/libteflon.so"


def load_delegates(delegate_path):
    """
    Load the optional TFLite delegate.

    Returns a list suitable for ``Interpreter(experimental_delegates=...)`` —
    empty (CPU) when ``delegate_path`` is ``None``, missing, or fails to load.
    Shared by every runtime so delegate handling stays uniform.
    """

    if delegate_path is None:
        return []

    delegate_path = Path(delegate_path)

    if not delegate_path.exists():
        print(f"[runtime] delegate not found: {delegate_path}")

        return []

    try:
        delegate = load_delegate(str(delegate_path))

        print(f"[runtime] loaded delegate: {delegate_path}")

        return [delegate]

    except Exception as e:
        print(f"[runtime] failed to load delegate: {e}")

        return []


class TFLiteRuntime(BaseRuntime):
    """
    TensorFlow Lite inference runtime.
    """

    def __init__(
        self,
        model_path: str | Path,
        *,
        delegate_path: str | None = DEFAULT_TEFLON_LIB,
        score_threshold: float = 0.0,
    ):

        self.model_path = Path(model_path)

        self.score_threshold = score_threshold

        self.metadata = self._load_metadata()

        delegates = self._load_delegates(delegate_path)

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

        self._norm_mean, self._norm_std = self._load_input_normalization()

    @property
    def input_size(self) -> int:

        return self._input_size

    #
    # Metadata
    #

    def _load_metadata(self):

        metadata_path = self.model_path.with_suffix(".runtime.json")

        if not metadata_path.exists():
            print("[runtime] metadata sidecar not found")

            return {}

        with open(metadata_path) as f:
            metadata = json.load(f)

        print(f"[runtime] loaded metadata: {metadata_path.name}")

        return metadata

    def _load_input_normalization(self):
        """
        Resolve the input normalization ``(mean, std)`` applied as
        ``(pixels - mean) / std`` before inference.

        Priority:

        1. ``preprocessing`` block in the ``.runtime.json`` sidecar
           (dependency-free, works on-device).
        2. ``NormalizationOptions`` embedded in the TFLite metadata, read via
           ``tflite_support`` when that package is importable (the prep/eval env).
        3. Default ``mean=std=127.5`` — the SSD MobileNetV2 ``[0, 255] -> [-1, 1]``
           mapping every model in this project uses.

        Returned as float32 arrays so they broadcast over an ``HxWx3`` image.
        """

        mean, std = [127.5], [127.5]
        source = "default"

        preprocessing = self.metadata.get("preprocessing") or {}

        if "mean" in preprocessing and "std" in preprocessing:
            mean = preprocessing["mean"]
            std = preprocessing["std"]
            source = "sidecar"

        else:
            try:
                from tflite_support import metadata as _metadata

                displayer = _metadata.MetadataDisplayer.with_model_file(
                    str(self.model_path)
                )

                meta = json.loads(displayer.get_metadata_json())

                units = meta["subgraph_metadata"][0]["input_tensor_metadata"][0][
                    "process_units"
                ]

                for unit in units:
                    if unit.get("options_type") == "NormalizationOptions":
                        options = unit["options"]
                        mean = options["mean"]
                        std = options["std"]
                        source = "embedded metadata"
                        break

            except Exception as exc:
                print(
                    f"[runtime] no embedded normalization metadata "
                    f"({type(exc).__name__}); using default"
                )

        print(f"[runtime] input normalization ({source}): mean={mean} std={std}")

        return (
            np.array(mean, dtype=np.float32),
            np.array(std, dtype=np.float32),
        )

    #
    # Delegates
    #

    def _load_delegates(
        self,
        delegate_path,
    ):

        return load_delegates(delegate_path)

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
        # raw [0, 255] silently wrecks detections. mean/std come from the model
        # metadata (see _load_input_normalization).
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

        dequantize_outputs = self.metadata.get("runtime", {}).get(
            "dequantize_outputs",
            False,
        )

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

        class_offset = self.metadata.get("runtime", {}).get(
            "class_index_offset",
            1,
        )

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
