"""
YOLOv7-tiny TFLite runtime.

The phenobench yolov7-tiny export keeps the three detection heads' **raw
pre-sigmoid grid logits** (``[1, 3, gh, gw, 5 + num_classes]``, no baked-in
decode/NMS), unlike the SSD models whose graph already emits post-NMS boxes. So
this runtime reconstructs detections in Python — sigmoid, grid/anchor box
assembly (yolov5/yolov7 convention), objectness × class confidence, then
per-class NMS — and returns the same canonical :class:`Detection` list as
:class:`~agri_vision_edge.runtime.inference.tflite.TFLiteRuntime`, so ``ave
infer`` and ``ave benchmark`` treat both families identically.

Input is 512×512 with ``[0, 1]`` normalization (vs the SSD 320×320 ``[-1, 1]``);
both are read from the model, so no per-model flags are needed.

The full YOLO training → tflite build path is not yet integrated in this package
(see ``notebooks/yolov7_phenobench.ipynb`` + a sibling repo); this runtime only
*runs* an already-exported tflite.
"""

from __future__ import annotations

import cv2
import numpy as np

# Mirror tflite.py's interpreter import (standalone tflite_runtime on-device,
# tf.lite in the prep env). Imported locally so the symbol resolves statically.
try:
    from tflite_runtime.interpreter import Interpreter
except ImportError:
    import tensorflow as tf

    Interpreter = tf.lite.Interpreter

from .base import (
    BaseRuntime,
    Detection,
)
from .tflite import (
    DEFAULT_TEFLON_LIB,
    load_delegates,
)

# Stock yolov7-tiny anchors (cfg/training/yolov7-tiny.yaml — the phenobench-tiny
# notebook trains the unmodified cfg), keyed by output stride. Verified to decode
# the exported tflite's raw grid logits onto plants.
YOLO_ANCHORS = {
    8: [(10, 13), (16, 30), (33, 23)],
    16: [(30, 61), (62, 45), (59, 119)],
    32: [(116, 90), (156, 198), (373, 326)],
}

# Minimum candidate confidence kept before NMS, independent of (and never above)
# the caller's score_threshold. Mirrors yolov5/yolov7 mAP eval (conf_thres=1e-3):
# it bounds the candidate set so per-class NMS stays tractable when the caller
# wants a permissive threshold (e.g. benchmark's 0.0), with negligible mAP effect.
CANDIDATE_FLOOR = 1e-3


class YoloTFLiteRuntime(BaseRuntime):
    """
    TensorFlow Lite runtime for raw-grid YOLOv7-tiny detectors.
    """

    def __init__(
        self,
        model_path,
        *,
        delegate_path: str | None = DEFAULT_TEFLON_LIB,
        score_threshold: float = 0.0,
        iou_threshold: float = 0.65,
        max_detections: int = 100,
        size: int | None = None,
    ):

        self.interpreter = Interpreter(
            model_path=str(model_path),
            experimental_delegates=load_delegates(delegate_path),
        )

        self.interpreter.allocate_tensors()

        self.input_details = self.interpreter.get_input_details()

        self.output_details = self.interpreter.get_output_details()

        self.score_threshold = score_threshold

        self.iou_threshold = iou_threshold

        self.max_detections = max_detections

        self._model_size = int(self.input_details[0]["shape"][1])

        self._input_size = size or self._model_size

    @property
    def input_size(self) -> int:

        return self._input_size

    #
    # Preprocessing
    #

    def _quantize_input(self, image):
        """``[0, 1]`` normalization, then the model's input quantization."""

        normalized = image.astype(np.float32) / 255.0

        input_detail = self.input_details[0]

        dtype = input_detail["dtype"]

        if dtype not in (np.int8, np.uint8):
            return normalized

        scale, zero_point = input_detail["quantization"]

        quantized = np.round(normalized / scale + zero_point)

        info = np.iinfo(dtype)

        return np.clip(quantized, info.min, info.max).astype(dtype)

    #
    # Inference
    #

    def predict(self, image):

        resized = cv2.resize(
            image,
            (self._input_size, self._input_size),
        )

        self.interpreter.set_tensor(
            self.input_details[0]["index"],
            self._quantize_input(resized)[None],
        )

        self.interpreter.invoke()

        floor = max(self.score_threshold, CANDIDATE_FLOOR)

        boxes, scores, classes = [], [], []

        for detail in self.output_details:
            logits = self.interpreter.get_tensor(detail["index"]).astype(np.float32)

            if detail["dtype"] in (np.int8, np.uint8):
                scale, zero_point = detail["quantization"]
                logits = (logits - zero_point) * scale

            # [1, num_anchors, gh, gw, 5 + num_classes]
            _, num_anchors, grid_h, grid_w, _ = logits.shape

            stride = self._input_size // grid_h
            anchors = YOLO_ANCHORS[stride]

            pred = 1.0 / (1.0 + np.exp(-logits[0]))  # sigmoid

            grid_y, grid_x = np.meshgrid(
                np.arange(grid_h),
                np.arange(grid_w),
                indexing="ij",
            )

            for a in range(num_anchors):
                cell = pred[a]  # [gh, gw, 5 + num_classes]

                cx = (cell[..., 0] * 2 - 0.5 + grid_x) * stride
                cy = (cell[..., 1] * 2 - 0.5 + grid_y) * stride
                bw = (cell[..., 2] * 2) ** 2 * anchors[a][0]
                bh = (cell[..., 3] * 2) ** 2 * anchors[a][1]

                confidence = cell[..., 4:5] * cell[..., 5:]  # obj × class

                class_id = confidence.argmax(-1)
                class_score = confidence.max(-1)

                keep = class_score >= floor

                if not keep.any():
                    continue

                size = self._input_size

                ymin = (cy - bh / 2) / size
                xmin = (cx - bw / 2) / size
                ymax = (cy + bh / 2) / size
                xmax = (cx + bw / 2) / size

                boxes.extend(
                    np.stack(
                        [ymin[keep], xmin[keep], ymax[keep], xmax[keep]],
                        axis=-1,
                    )
                )

                scores.extend(class_score[keep])
                classes.extend(class_id[keep])

        return _per_class_nms(
            boxes,
            scores,
            classes,
            self.iou_threshold,
            self.max_detections,
        )


def _per_class_nms(boxes, scores, classes, iou_threshold, max_detections):
    """Greedy per-class NMS over ``[ymin, xmin, ymax, xmax]`` boxes."""

    if not boxes:
        return []

    boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
    scores = np.asarray(scores, dtype=np.float32)
    classes = np.asarray(classes, dtype=np.int32)

    y1, x1, y2, x2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (y2 - y1).clip(min=0) * (x2 - x1).clip(min=0)

    kept = []

    for cls in np.unique(classes):
        idx = np.where(classes == cls)[0]
        order = idx[scores[idx].argsort()[::-1]]

        while order.size:
            i = order[0]
            kept.append(i)

            if order.size == 1:
                break

            rest = order[1:]

            inter_h = (np.minimum(y2[i], y2[rest]) - np.maximum(y1[i], y1[rest])).clip(
                min=0
            )
            inter_w = (np.minimum(x2[i], x2[rest]) - np.maximum(x1[i], x1[rest])).clip(
                min=0
            )
            inter = inter_h * inter_w

            iou = inter / (areas[i] + areas[rest] - inter + 1e-9)

            order = rest[iou <= iou_threshold]

    # Keep the globally highest-scoring detections (COCO evaluates maxDets=100).
    kept.sort(key=lambda i: scores[i], reverse=True)
    kept = kept[:max_detections]

    return [
        Detection(
            category_id=int(classes[i] + 1),
            score=float(scores[i]),
            bbox=boxes[i].tolist(),
        )
        for i in kept
    ]
