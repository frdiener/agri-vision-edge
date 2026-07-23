"""
Unit tests for reading deployment configuration back from a converted model.

These pin the behaviour of
:class:`agri_vision_edge.runtime.inference.model_metadata.ModelMetadata`:

- labels are read from the ``labels.txt`` associated file that the converter
  appends to the ``.tflite`` as a ZIP archive (stdlib :mod:`zipfile`);
- input normalization and the ``DETECTOR_POSTPROCESSING`` parameters are read
  from the ``<model>.metadata.json`` sidecar (stdlib :mod:`json`);
- everything degrades to clean defaults when the metadata is absent, which is
  the case for the separately-prepared YOLO artifacts (a plain flatbuffer with
  no appended zip and no sidecar).

The tests synthesize a model file (arbitrary flatbuffer-like prefix bytes with a
ZIP appended) plus a sidecar, so they need neither TensorFlow nor real model
artifacts.
"""

from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path

from agri_vision_edge.runtime.inference.model_metadata import ModelMetadata


def _write_model_with_labels(path: Path, labels: list[str]) -> None:
    """Emulate a converted model: flatbuffer prefix + appended labels ZIP."""

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("labels.txt", "".join(f"{name}\n" for name in labels))

    # A ZIP reader locates its central directory from the end of the file, so a
    # non-zip prefix (standing in for the TFLite flatbuffer) is ignored.
    path.write_bytes(b"TFL3" + b"\x00" * 32 + buffer.getvalue())


def _write_sidecar(
    path: Path,
    *,
    mean: list[float],
    std: list[float],
    postprocessing: dict | None,
) -> None:
    """Write a ``<model>.metadata.json`` sidecar like the converter emits."""

    subgraph: dict = {
        "input_tensor_metadata": [
            {
                "process_units": [
                    {
                        "options_type": "NormalizationOptions",
                        "options": {"mean": mean, "std": std},
                    }
                ]
            }
        ]
    }

    if postprocessing is not None:
        blob = json.dumps(postprocessing, sort_keys=True).encode("utf-8")
        subgraph["custom_metadata"] = [
            {"name": "DETECTOR_POSTPROCESSING", "data": list(blob)}
        ]

    sidecar = path.with_suffix(".metadata.json")
    sidecar.write_text(json.dumps({"subgraph_metadata": [subgraph]}))


def test_reads_labels_normalization_and_postprocessing(tmp_path):
    model = tmp_path / "ssd.tflite"
    _write_model_with_labels(model, ["crop", "weed"])
    _write_sidecar(
        model,
        mean=[127.5],
        std=[127.5],
        postprocessing={
            "iou_threshold": 0.5,
            "nms": "fast",
            "max_detections": 100,
            "score_threshold": 0.05,
        },
    )

    meta = ModelMetadata.load(model, verbose=False)

    # Labels are 1-based (labels.txt line 0 -> category_id 1).
    assert meta.labels == {1: "crop", 2: "weed"}
    assert meta.norm_mean == [127.5]
    assert meta.norm_std == [127.5]
    assert meta.score_threshold == 0.05
    assert meta.iou_threshold == 0.5
    assert meta.max_detections == 100
    assert meta.nms == "fast"
    assert meta.sources == {
        "labels": "embedded",
        "normalization": "sidecar",
        "postprocessing": "sidecar",
    }


def test_single_class_label_offset(tmp_path):
    model = tmp_path / "ssd_sc.tflite"
    _write_model_with_labels(model, ["plant"])
    _write_sidecar(model, mean=[127.5], std=[127.5], postprocessing=None)

    meta = ModelMetadata.load(model, verbose=False)

    assert meta.labels == {1: "plant"}
    # No custom blob -> post-processing stays unset for the caller to default.
    assert meta.score_threshold is None
    assert meta.iou_threshold is None


def test_missing_metadata_degrades_to_defaults(tmp_path):
    # A plain (non-zip) flatbuffer with no sidecar, like the YOLO artifacts.
    model = tmp_path / "yolo.tflite"
    model.write_bytes(b"TFL3" + b"\x00" * 256)

    meta = ModelMetadata.load(model, verbose=False)

    assert meta.labels == {}
    assert meta.norm_mean == [127.5]
    assert meta.norm_std == [127.5]
    assert meta.score_threshold is None
    assert meta.iou_threshold is None
    assert meta.max_detections is None
    assert meta.nms is None
    assert meta.sources == {}


def test_labels_without_sidecar(tmp_path):
    # Embedded labels present, but no sidecar (normalization/postproc absent).
    model = tmp_path / "labels_only.tflite"
    _write_model_with_labels(model, ["crop", "weed"])

    meta = ModelMetadata.load(model, verbose=False)

    assert meta.labels == {1: "crop", 2: "weed"}
    assert meta.sources.get("labels") == "embedded"
    assert "normalization" not in meta.sources
    assert meta.norm_mean == [127.5]
