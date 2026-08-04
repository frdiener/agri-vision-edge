"""
Tests for the SavedModel reference runtime and its plumbing.

This runtime exists to measure the rung above TFLite in the deployment chain,
and it is only useful if its output is *commensurable* with the TFLite runs —
same COCO export, same category ids, same box layout. These pin the parts where
that could silently drift.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from agri_vision_edge.cli.benchmark import collect_models
from agri_vision_edge.conversion.tflite import stage_graph_flags
from agri_vision_edge.evaluation.benchmark_report import _input_dtype, _runtime_fields
from agri_vision_edge.runtime.inference.saved_model import (
    SAVED_MODEL_PROTO,
    decode_detections,
    is_saved_model_dir,
)


def _saved_model(root, name="saved_model"):
    d = root / name
    d.mkdir(parents=True)
    (d / SAVED_MODEL_PROTO).write_bytes(b"")
    return d


#
# Detection decoding
#


def _arrays(n=4):
    boxes = np.array([[0.1 * i, 0.2, 0.1 * i + 0.05, 0.3] for i in range(1, n + 1)])
    scores = np.array([0.9, 0.5, 0.04, 0.01])[:n]
    classes = np.array([1, 2, 1, 2], dtype=np.float32)[:n]
    return boxes, scores, classes


def test_num_detections_slices_the_padded_arrays():
    # The signature always returns 100 slots; only `num_detections` are real.
    boxes, scores, classes = _arrays()

    out = decode_detections(boxes, scores, classes, count=2)

    assert len(out) == 2
    assert [d.score for d in out] == [pytest.approx(0.9), pytest.approx(0.5)]


def test_class_ids_are_passed_through_as_written():
    """
    The exporter applies ``label_id_offset``, so classes arrive 1-based and
    already match our COCO ``category_id`` (crop=1, weed=2). Re-basing them
    here is exactly the bug that once scored every weed as a crop.
    """
    boxes, scores, classes = _arrays()

    out = decode_detections(boxes, scores, classes, count=4)

    assert [d.category_id for d in out] == [1, 2, 1, 2]


def test_boxes_keep_the_normalized_ymin_xmin_ymax_xmax_layout():
    boxes, scores, classes = _arrays()

    out = decode_detections(boxes, scores, classes, count=1)

    assert out[0].bbox == [pytest.approx(v) for v in boxes[0]]


def test_score_threshold_trims_and_defaults_to_keeping_everything():
    boxes, scores, classes = _arrays()

    assert len(decode_detections(boxes, scores, classes, count=4)) == 4
    assert (
        len(decode_detections(boxes, scores, classes, count=4, score_threshold=0.05))
        == 2
    )


def test_max_detections_caps_before_the_threshold():
    boxes, scores, classes = _arrays()

    out = decode_detections(boxes, scores, classes, count=4, max_detections=1)

    assert len(out) == 1


def test_zero_and_negative_counts_yield_nothing():
    boxes, scores, classes = _arrays()

    assert decode_detections(boxes, scores, classes, count=0) == []
    assert decode_detections(boxes, scores, classes, count=-1) == []


#
# Path dispatch
#


def test_is_saved_model_dir(tmp_path):
    assert is_saved_model_dir(_saved_model(tmp_path))

    plain = tmp_path / "plain"
    plain.mkdir()
    assert not is_saved_model_dir(plain)

    f = tmp_path / "m.tflite"
    f.write_bytes(b"")
    assert not is_saved_model_dir(f)


def test_collect_models_accepts_a_saved_model_directory(tmp_path):
    d = _saved_model(tmp_path)

    # A SavedModel is itself a directory, so it must not be walked as a folder.
    assert collect_models(d) == [d]


def test_collect_models_finds_saved_models_side_by_side(tmp_path):
    a = _saved_model(tmp_path, "saved_model")
    b = _saved_model(tmp_path, "saved_model_nms0")

    assert collect_models(tmp_path) == [a, b]


def test_collect_models_still_prefers_tflite_in_a_mixed_directory(tmp_path):
    _saved_model(tmp_path)
    model = tmp_path / "m.tflite"
    model.write_bytes(b"")

    assert collect_models(tmp_path) == [model]


def test_collect_models_single_file(tmp_path):
    model = tmp_path / "m.tflite"
    model.write_bytes(b"")

    assert collect_models(model) == [model]


#
# Report plumbing
#


def test_runtime_fields_reads_the_format_and_defaults_to_tflite():
    assert _runtime_fields({"format": "savedmodel"})["format"] == "savedmodel"
    # Artifacts written before `format` existed are all TFLite.
    assert _runtime_fields({})["format"] == "tflite"


def test_input_dtype_handles_both_artifact_flavours():
    # TFLite records a repr; the SavedModel runtime records a bare name.
    tflite = {"input_details": [{"dtype": "<class 'numpy.float32'>"}]}
    saved = {"input_details": [{"dtype": "uint8"}]}

    assert _input_dtype(tflite) == "numpy.float32"
    assert _input_dtype(saved) == "uint8"
    assert _input_dtype({}) is None
    assert _input_dtype({"input_details": [{}]}) is None


def test_runtime_fields_does_not_raise_on_a_bare_dtype():
    # Splitting on quotes unconditionally used to IndexError here.
    fields = _runtime_fields({"input_details": [{"dtype": "uint8"}], "backend": "cpu"})

    assert fields["input_dtype"] == "uint8"


#
# Stage -> graph flags
#


@pytest.mark.parametrize(
    ("stage", "expected"),
    [
        ("finetune", (False, False)),
        ("ptq", (False, False)),
        ("qat_per-tensor", (True, False)),
        ("qat_per-channel", (True, True)),
    ],
)
def test_stage_graph_flags(stage, expected):
    # A QAT checkpoint stores folded/fake-quantized variables; the stage name is
    # the only record of how to rebuild the graph before restoring.
    assert stage_graph_flags(stage) == expected


def test_benchmark_artifacts_record_the_format(tmp_path):
    from agri_vision_edge.evaluation.artifacts import save_benchmark_artifacts
    from agri_vision_edge.evaluation.benchmark import BenchmarkResult

    class _Runtime:
        runtime_format = "savedmodel"
        active_delegate = None
        input_details = [{"name": "input_tensor", "dtype": "uint8"}]
        output_details: list = []

    save_benchmark_artifacts(
        output_dir=tmp_path / "run",
        benchmark_result=BenchmarkResult(predictions=[], latencies_ms=[1.0]),
        runtime=_Runtime(),
        model_name="saved_model_nms0",
        delegate=None,
    )

    runtime_json = json.loads((tmp_path / "run" / "runtime.json").read_text())

    assert runtime_json["format"] == "savedmodel"
    assert runtime_json["backend"] == "cpu"
