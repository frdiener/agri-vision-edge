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

from tflite_runtime.interpreter import (
    Interpreter,
    load_delegate,
)

from .base import (
    BaseRuntime,
    Detection,
)


DEFAULT_TEFLON_LIB = (
    "/usr/lib/libteflon.so"
)


class TFLiteRuntime(BaseRuntime):
    """
    TensorFlow Lite inference runtime.
    """

    def __init__(
        self,
        model_path: str | Path,
        *,
        delegate_path: str | None = DEFAULT_TEFLON_LIB,
        score_threshold: float = 0.3,
    ):

        self.model_path = Path(model_path)

        self.score_threshold = score_threshold

        self.metadata = self._load_metadata()

        delegates = self._load_delegates(
            delegate_path
        )

        self.interpreter = Interpreter(
            model_path=str(self.model_path),

            experimental_delegates=delegates,
        )

        self.interpreter.allocate_tensors()

        self.input_details = (
            self.interpreter.get_input_details()
        )

        self.output_details = (
            self.interpreter.get_output_details()
        )

        print("\n=== OUTPUT DETAILS ===")

        for detail in self.output_details:

            print(
                detail["name"],
                detail["dtype"],
                detail["quantization"],
                detail["shape"],
            )

        if len(self.output_details) != 4:

            output_names = [
                d.get("name", "<unknown>")
                for d in self.output_details
            ]

            raise RuntimeError(
                "Unsupported TFLite detector.\n"
                "Expected SSD-style outputs "
                f"(boxes,scores,num,classes),\n"
                f"got {len(self.output_details)} "
                f"output tensor(s): {output_names}"
            )
        
        self._input_size = int(
            self.input_details[0]["shape"][1]
        )

    @property
    def input_size(self) -> int:

        return self._input_size

    #
    # Metadata
    #

    def _load_metadata(self):

        metadata_path = (
            self.model_path.with_suffix(
                ".runtime.json"
            )
        )

        if not metadata_path.exists():

            print(
                "[runtime] metadata sidecar "
                "not found"
            )

            return {}

        with open(metadata_path) as f:

            metadata = json.load(f)

        print(
            f"[runtime] loaded metadata: "
            f"{metadata_path.name}"
        )

        return metadata

    #
    # Delegates
    #

    def _load_delegates(
        self,
        delegate_path,
    ):

        delegates = []

        if delegate_path is None:
            return delegates

        delegate_path = Path(delegate_path)

        if not delegate_path.exists():

            print(
                "[runtime] delegate not found:"
                f" {delegate_path}"
            )

            return delegates

        try:

            delegate = load_delegate(
                str(delegate_path)
            )

            delegates.append(delegate)

            print(
                "[runtime] loaded delegate:"
                f" {delegate_path}"
            )

        except Exception as e:

            print(
                "[runtime] failed to load "
                f"delegate: {e}"
            )

        return delegates

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

            scale, zero_point = (
                input_detail["quantization"]
            )

            image = image.astype(np.float32)

            image = (
                image / scale
            ) + zero_point

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

        else:

            image = image.astype(np.float32)

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

        return scale * (
            tensor.astype(np.float32)
            - zero_point
        )

    #
    # Inference
    #

    def predict(
        self,
        image,
    ):

        input_tensor = self.preprocess(
            image
        )

        self.interpreter.set_tensor(
            self.input_details[0]["index"],
            input_tensor,
        )

        self.interpreter.invoke()

        raw_scores = (
            self.interpreter.get_tensor(
                self.output_details[0]["index"]
            )
        )

        raw_boxes = (
            self.interpreter.get_tensor(
                self.output_details[1]["index"]
            )
        )

        raw_num = (
            self.interpreter.get_tensor(
                self.output_details[2]["index"]
            )
        )

        raw_classes = (
            self.interpreter.get_tensor(
                self.output_details[3]["index"]
            )
        )

        dequantize_outputs = (
            self.metadata
            .get("runtime", {})
            .get(
                "dequantize_outputs",
                False,
            )
        )

        #
        # Quantized outputs
        #

        if dequantize_outputs:

            scores = self.dequantize(
                raw_scores,
                self.output_details[0][
                    "quantization"
                ],
            )

            boxes = self.dequantize(
                raw_boxes,
                self.output_details[1][
                    "quantization"
                ],
            )

            classes = self.dequantize(
                raw_classes,
                self.output_details[3][
                    "quantization"
                ],
            )

            num = int(
                np.squeeze(
                    self.dequantize(
                        raw_num,
                        self.output_details[2][
                            "quantization"
                        ],
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

            num = int(
                np.squeeze(raw_num)
            )

        scores = scores[0]
        boxes = boxes[0]

        classes = np.round(
            classes[0]
        ).astype(np.int32)

        #
        # COCO compatibility
        #

        class_offset = (
            self.metadata
            .get("runtime", {})
            .get(
                "class_index_offset",
                1,
            )
        )

        detections = []

        for box, cls_id, score in zip(
            boxes[:num],
            classes[:num],
            scores[:num],
        ):

            score = float(score)

            if score < self.score_threshold:
                continue

            detections.append(
                Detection(
                    category_id=(
                        int(cls_id)
                        + class_offset
                    ),

                    score=score,

                    bbox=box.tolist(),
                )
            )

        return detections
