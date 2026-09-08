"""Run the pre-conversion TFOD SavedModel reference through the common API.

This separates conversion and TFLite post-processing loss from quantization
loss while keeping prediction export and scoring identical to device runs.
Two differences remain. SavedModel resizes native-resolution input inside the
graph, and its NMS score floor is baked in. Use a ``saved_model_nms0`` export
when the floor must be removed.
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
    """Return whether ``path`` is a SavedModel directory."""
    path = Path(path)
    return path.is_dir() and (path / SAVED_MODEL_PROTO).exists()


def _resizer_size(model_dir: Path) -> int:
    """
    The graph's fixed input resolution, read from a neighbouring pipeline.config.

    This value is informational because the graph performs its own resizing.
    Returns 0 when no configuration is found.
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

    This isolates the off-by-one risks in ``num_detections`` slicing and
    1-based class IDs for tests that do not load TensorFlow.
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

    The serving signature emits normalized ``[ymin, xmin, ymax, xmax]`` boxes
    in :class:`~.base.Detection` layout. ``detection_classes`` includes the
    exporter's 1-based ``label_id_offset`` and matches COCO ``category_id``.
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
