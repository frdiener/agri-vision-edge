"""
TensorFlow SavedModel runtime — the pre-conversion reference.

Every other runtime here reads a ``.tflite``. This one reads the SavedModel the
trainer exported, which is the rung above TFLite in the deployment chain::

    SavedModel (TF float, TFOD post-processing)   <- this module
      -> TFLite fp32          conversion + TFLite_Detection_PostProcess
      -> TFLite int8 (CPU)    quantization
      -> TFLite int8 (NPU)    delegation

Without it the first two losses are folded together and both get attributed to
quantization, even though the TFLite export swaps TFOD's post-processing for a
different NMS implementation.

It implements :class:`~.base.BaseRuntime` unchanged, which is the point: ``ave
benchmark`` then produces the same ``predictions.json`` through the same COCO
export, and ``ave evaluate`` scores it against the same annotations with the
same pycocotools call. The reference is commensurable with the device numbers
by construction rather than by careful reimplementation.

Two differences from the TFLite path are inherent and worth knowing when
reading the delta:

* **Resampling.** The serving signature takes ``uint8 [1, None, None, 3]`` and
  resizes *inside* the graph (``fixed_shape_resizer``), so the image is fed at
  native resolution. The TFLite runtimes resize externally with ``cv2``. Part
  of any SavedModel-vs-TFLite gap is therefore resampling, not post-processing.
* **Score floor.** The pipeline's ``batch_non_max_suppression.score_threshold``
  is baked into the graph and cannot be overridden here, unlike the TFLite
  runtimes where ``ave benchmark`` pins it to 0. Use a
  ``saved_model_nms0`` export (see
  :func:`agri_vision_edge.tfod_trainer.export.export_scoring_saved_model`) when
  the floor binds -- it does for the single-class models, whose detections stop
  dead at 0.05, though not for the multi-class ones, which hit the
  100-detection cap first.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .base import BaseRuntime, Detection

#: Serving signature the TFOD exporter writes.
SIGNATURE_KEY = "serving_default"

#: Marks a SavedModel directory.
SAVED_MODEL_PROTO = "saved_model.pb"


def is_saved_model_dir(path: str | Path) -> bool:
    """Whether ``path`` is a SavedModel directory rather than a ``.tflite``."""
    path = Path(path)
    return path.is_dir() and (path / SAVED_MODEL_PROTO).exists()


def _resizer_size(model_dir: Path) -> int:
    """
    The graph's fixed input resolution, read from a neighbouring pipeline.config.

    Informational only -- unlike the TFLite runtimes nothing is resized against
    it, because the graph does its own resizing. Returns 0 when no config is
    found, which is not an error: the model still runs.
    """
    for candidate in (
        model_dir / "pipeline.config",
        model_dir.parent / "pipeline.config",
    ):
        if not candidate.exists():
            continue
        try:
            import re

            text = candidate.read_text()
            block = re.search(r"fixed_shape_resizer\s*{[^}]*}", text)
            if block:
                height = re.search(r"height:\s*(\d+)", block.group(0))
                if height:
                    return int(height.group(1))
        except OSError:
            continue
    return 0


def decode_detections(
    boxes: np.ndarray,
    scores: np.ndarray,
    classes: np.ndarray,
    count: int,
    *,
    score_threshold: float = 0.0,
    max_detections: int | None = None,
) -> list[Detection]:
    """
    Turn the serving signature's arrays into :class:`~.base.Detection` objects.

    Split out from :meth:`SavedModelRuntime.predict` so the decode -- the part
    with the off-by-one risks (``num_detections`` slicing, the 1-based class
    ids) -- is testable without loading TensorFlow or a real model.
    """
    count = int(count)

    if max_detections is not None:
        count = min(count, max_detections)

    if count <= 0:
        return []

    detections = []

    for box, score, category in zip(
        boxes[:count], scores[:count], classes[:count], strict=True
    ):
        if score < score_threshold:
            continue

        detections.append(
            Detection(
                category_id=int(category),
                score=float(score),
                bbox=[float(v) for v in box],
            )
        )

    return detections


class SavedModelRuntime(BaseRuntime):
    """
    Run a TFOD-exported SavedModel behind the common runtime interface.

    The serving signature already emits post-NMS detections in exactly
    :class:`~.base.Detection`'s layout -- ``detection_boxes`` are normalized
    ``[ymin, xmin, ymax, xmax]`` -- so no box conversion is involved.
    ``detection_classes`` carry the exporter's ``label_id_offset``, i.e. they
    are 1-based and line up with our COCO ``category_id`` directly.
    """

    #: Recorded in runtime.json so the report can hold the reference rows apart
    #: from the device ones (their latency is not comparable).
    runtime_format = "savedmodel"

    #: No delegate is involved; `save_benchmark_artifacts` reads this.
    active_delegate = None

    def __init__(
        self,
        model_path: str | Path,
        *,
        score_threshold: float | None = None,
        max_detections: int | None = None,
    ):
        import tensorflow as tf

        self.model_path = Path(model_path)

        if not is_saved_model_dir(self.model_path):
            raise FileNotFoundError(
                f"{self.model_path} is not a SavedModel directory "
                f"(no {SAVED_MODEL_PROTO})"
            )

        self._model = tf.saved_model.load(str(self.model_path))

        if SIGNATURE_KEY not in self._model.signatures:
            raise KeyError(
                f"{self.model_path} has no {SIGNATURE_KEY!r} signature "
                f"(found {sorted(self._model.signatures)})"
            )

        self._fn = self._model.signatures[SIGNATURE_KEY]
        self._tf = tf

        # The graph applies its own NMS score threshold; this only trims further.
        self.score_threshold = 0.0 if score_threshold is None else score_threshold
        self.max_detections = max_detections

        self._input_size = _resizer_size(self.model_path)

        inputs = self._fn.structured_input_signature[1]
        self.input_details = [
            {"name": name, "shape": str(spec.shape.as_list()), "dtype": spec.dtype.name}
            for name, spec in sorted(inputs.items())
        ]
        self.output_details = [
            {"name": name, "shape": str(spec.shape.as_list()), "dtype": spec.dtype.name}
            for name, spec in sorted(self._fn.structured_outputs.items())
        ]

    @property
    def input_size(self) -> int:
        return self._input_size

    def predict(self, image: np.ndarray) -> list[Detection]:
        # Fed at native resolution on purpose: the graph's fixed_shape_resizer
        # is what the model was trained with, so resizing here would apply a
        # second, different resampling.
        outputs = self._fn(
            input_tensor=self._tf.constant(image[None], dtype=self._tf.uint8)
        )

        return decode_detections(
            outputs["detection_boxes"][0].numpy(),
            outputs["detection_scores"][0].numpy(),
            outputs["detection_classes"][0].numpy(),
            outputs["num_detections"][0].numpy(),
            score_threshold=self.score_threshold,
            max_detections=self.max_detections,
        )


__all__ = [
    "SAVED_MODEL_PROTO",
    "SIGNATURE_KEY",
    "SavedModelRuntime",
    "decode_detections",
    "is_saved_model_dir",
]
