"""
Tests for representative-dataset (calibration) resolution during conversion.

``rep_dataset.json`` stores *positions*, so a calibration dataset built from the
wrong bundle or the wrong tiling still indexes fine and still yields plausible
field images -- it just calibrates on the wrong ones, silently. These cover the
two ways that happened.
"""

from __future__ import annotations

import json

import pytest

from agri_vision_edge.conversion.tflite import (
    _check_calibration_dataset,
    _dataset_dir,
)


def _bundle(root, name, **metadata):
    path = root / name
    path.mkdir(parents=True)
    if metadata:
        (path / "dataset_metadata.json").write_text(json.dumps(metadata))
    return path


def test_no_partials_bundle_is_preferred(tmp_path):
    # Both exist; the models are trained on the no-partials bundle and the two
    # have different sample counts, so picking the legacy one miscalibrates.
    _bundle(tmp_path, "phenobench_mc_tiled")
    _bundle(tmp_path, "phenobench_mc_tiled_no-partials")

    resolved = _dataset_dir("ssd-mn2-fpnlite_mc_phenobench-tiled_320", tmp_path)

    assert resolved.name == "phenobench_mc_tiled_no-partials"


def test_legacy_bundle_is_the_fallback(tmp_path):
    _bundle(tmp_path, "phenobench_sc")

    resolved = _dataset_dir("ssd-mn2_sc_phenobench_320", tmp_path)

    assert resolved.name == "phenobench_sc"


def test_missing_bundle_is_reported(tmp_path):
    with pytest.raises(FileNotFoundError, match="phenobench_mc"):
        _dataset_dir("ssd-mn2_mc_phenobench_320", tmp_path)


def test_sample_count_mismatch_is_rejected(tmp_path):
    # The regression: tiling an already-tiled dataset multiplied the sample
    # count, so every rep_dataset.json index addressed a different image.
    bundle = _bundle(tmp_path, "phenobench_mc_tiled_no-partials", train_samples=12663)

    with pytest.raises(ValueError, match="wrong images"):
        _check_calibration_dataset(range(50652), bundle, [0, 1, 2])


def test_matching_sample_count_passes(tmp_path):
    bundle = _bundle(tmp_path, "phenobench_mc_tiled_no-partials", train_samples=12663)

    _check_calibration_dataset(range(12663), bundle, [0, 12662])


def test_out_of_range_index_is_rejected(tmp_path):
    bundle = _bundle(tmp_path, "phenobench_mc_no-partials", train_samples=1407)

    with pytest.raises(ValueError, match="indexes up to"):
        _check_calibration_dataset(range(1407), bundle, [1407])


def test_bundle_without_metadata_is_not_checked(tmp_path):
    bundle = _bundle(tmp_path, "phenobench_mc")

    _check_calibration_dataset(range(10), bundle, [0, 1])
