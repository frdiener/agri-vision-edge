from __future__ import annotations

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


class TFLiteRuntime(BaseRuntime):

    def __init__(
        self,
        model_path: str | Path,
        *,
        delegate_path: str | None = None,
        score_threshold: float = 0.3,
    ):

        self.score_threshold = score_threshold

        delegates = []

        if delegate_path is not None:

            delegate = load_delegate(
                delegate_path
            )

            delegates.append(delegate)

        self.interpreter = Interpreter(
            model_path=str(model_path),

            experimental_delegates=delegates,
        )

        self.interpreter.allocate_tensors()

        self.input_details = (
            self.interpreter.get_input_details()
        )

        self.output_details = (
            self.interpreter.get_output_details()
        )

        self._input_size = int(
            self.input_details[0]["shape"][1]
        )

    @property
    def input_size(self) -> int:

        return self._input_size

    def preprocess(
        self,
        image: np.ndarray,
    ) -> np.ndarray:

        image = cv2.resize(
            image,
            (self.input_size, self.input_size),
        )

        input_detail = self.input_details[0]

        dtype = input_detail["dtype"]

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

        else:

            image = image.astype(np.float32)

        image = np.expand_dims(
            image,
            axis=0,
        )

        return image

    def dequantize(
        self,
        tensor,
        quantization,
    ):

        scale, zero_point = quantization

        return scale * (
            tensor.astype(np.float32)
            - zero_point
        )

    def predict(
        self,
        image: np.ndarray,
    ) -> list[Detection]:

        input_tensor = self.preprocess(
            image
        )

        self.interpreter.set_tensor(
            self.input_details[0]["index"],
            input_tensor,
        )

        self.interpreter.invoke()

        scores = self.interpreter.get_tensor(
            self.output_details[0]["index"]
        )

        boxes = self.interpreter.get_tensor(
            self.output_details[1]["index"]
        )

        classes = self.interpreter.get_tensor(
            self.output_details[3]["index"]
        )

        #
        # Dequantize if needed
        #

        if (
            self.output_details[0]["dtype"]
            == np.int8
        ):

            scores = self.dequantize(
                scores,
                self.output_details[0][
                    "quantization"
                ],
            )

            boxes = self.dequantize(
                boxes,
                self.output_details[1][
                    "quantization"
                ],
            )

            classes = self.dequantize(
                classes,
                self.output_details[3][
                    "quantization"
                ],
            )

        scores = scores[0]
        boxes = boxes[0]

        classes = np.round(
            classes[0]
        ).astype(np.int32)

        detections = []

        for box, cls_id, score in zip(
            boxes,
            classes,
            scores,
        ):

            score = float(score)

            if score < self.score_threshold:
                continue

            detections.append(
                Detection(
                    class_id=int(cls_id),
                    score=score,
                    bbox=box.tolist(),
                )
            )

        return detections
