"""
Tests for the preprocessing (``cv2.resize``) correction.

Every benchmark latency contains a resize whose cost depends on the pair
(source resolution, model input size), and the sweep varies both -- so the
correction is what keeps the resolution ladder and the tiling study from
comparing preprocessing instead of detectors.

The failure modes worth pinning down are all silent ones:

* the correction subtracting the *wrong* pair, which looks entirely plausible
  because both pairs are small numbers;
* a run whose host was never measured coming out uncorrected but sitting in a
  column that claims it was;
* the ``_cpu`` / ``_unpatched`` trees not finding their board's measurement,
  which would leave two thirds of the sweep uncorrected for no reason -- resize
  runs on the CPU whatever the delegate is doing.
"""

from __future__ import annotations

import json

import pandas as pd
import pytest

from agri_vision_edge.evaluation.benchmark_report import (
    RESIZE_ARTIFACT,
    add_resize_cost,
    load_resize_costs,
    resize_cost_table,
)

BOARD = "frdm-imx93"

#: (source, target) -> median ms. Deliberately spread far apart so a join that
#: picks the wrong cell cannot pass by rounding.
COSTS = {
    (1024, 320): 8.0,
    (1024, 512): 6.0,
    (1024, 1024): 1.0,
    (512, 320): 2.0,
    (512, 512): 0.5,
    (512, 1024): 4.0,
}

SOURCE_PX = {"untiled": 1024, "tiled": 512}


def _artifact(platform: str) -> dict:
    return {
        "schema": 1,
        "kind": "resize",
        "platform": platform,
        "opencv": {"version": "4.9.0", "threads": 2},
        "resizes": [
            {
                "eval_tiling": tiling,
                "source_px": source,
                "target_px": target,
                "identity": source == target,
                "latency": {
                    "count": 640,
                    "mean_latency_ms": ms * 1.1,
                    "median_latency_ms": ms,
                    "p95_latency_ms": ms * 1.2,
                },
            }
            for tiling, source in SOURCE_PX.items()
            for target in (320, 512, 1024)
            for ms in [COSTS[(source, target)]]
        ],
    }


def _tree(tmp_path, platforms=(BOARD,)):
    for platform in platforms:
        directory = tmp_path / platform
        directory.mkdir(parents=True, exist_ok=True)
        (directory / RESIZE_ARTIFACT).write_text(json.dumps(_artifact(platform)))
    return tmp_path


def _run(platform, *, eval_tiling, size, median_ms):
    return {
        "run": f"{eval_tiling}_ssd-mn2_mc_phenobench_{size}_int8_ptq_"
        f"per-tensor_fastnms",
        "platform": platform,
        "eval_tiling": eval_tiling,
        "size": size,
        "median_latency_ms": median_ms,
        "mean_latency_ms": median_ms + 1.0,
        "p95_latency_ms": median_ms + 2.0,
    }


def test_costs_load_one_row_per_source_and_target(tmp_path):
    costs = load_resize_costs(_tree(tmp_path))

    assert len(costs) == 6
    assert set(costs["platform"]) == {BOARD}
    # `size` is the run-name token, not the integer, so it joins without a cast.
    assert set(costs["size"]) == {"320", "512", "1024"}


def test_the_artifact_is_not_mistaken_for_a_run(tmp_path):
    """It is a file, not a directory -- the run scanner must never see it."""
    root = _tree(tmp_path)

    assert (root / BOARD / RESIZE_ARTIFACT).is_file()
    assert not any(p.is_dir() for p in (root / BOARD).iterdir())


def test_correction_uses_the_source_resolution_of_the_eval_regime(tmp_path):
    """
    The same model on the same board pays a different resize per regime.

    This is the whole reason the correction exists: 1024 -> 320 against
    512 -> 320 is a four-fold difference here, and nothing in ``latency.json``
    records which one a run paid.
    """
    df = pd.DataFrame(
        [
            _run(BOARD, eval_tiling="untiled", size="320", median_ms=30.0),
            _run(BOARD, eval_tiling="tiled", size="320", median_ms=30.0),
        ]
    )

    out = add_resize_cost(df, load_resize_costs(_tree(tmp_path)))

    untiled = out[out["eval_tiling"] == "untiled"].iloc[0]
    tiled = out[out["eval_tiling"] == "tiled"].iloc[0]

    assert untiled["resize_ms"] == 8.0
    assert tiled["resize_ms"] == 2.0
    assert untiled["resize_source_px"] == 1024
    assert tiled["resize_source_px"] == 512

    assert untiled["median_latency_ms_net"] == 22.0
    assert tiled["median_latency_ms_net"] == 28.0

    # The uncorrected figures are equal; the corrected ones differ by the
    # preprocessing alone. That gap is what the tiling study would otherwise
    # have reported as a property of the model.
    assert untiled["median_latency_ms"] == tiled["median_latency_ms"]


def test_every_latency_statistic_is_corrected_and_fps_follows(tmp_path):
    df = pd.DataFrame([_run(BOARD, eval_tiling="untiled", size="512", median_ms=50.0)])

    row = add_resize_cost(df, load_resize_costs(_tree(tmp_path))).iloc[0]

    assert row["median_latency_ms_net"] == 44.0
    assert row["mean_latency_ms_net"] == 45.0
    assert row["p95_latency_ms_net"] == 46.0
    assert row["fps_net"] == pytest.approx(1000.0 / 44.0)
    assert row["resize_share"] == pytest.approx(6.0 / 50.0)


def test_cpu_and_unpatched_trees_share_their_board_measurement(tmp_path):
    """
    Resize runs on the CPU whatever the delegate is doing or is built from, so
    one measurement per board covers all three of its results trees.
    """
    df = pd.DataFrame(
        [
            _run(BOARD, eval_tiling="untiled", size="320", median_ms=30.0),
            _run(f"{BOARD}_cpu", eval_tiling="untiled", size="320", median_ms=90.0),
            _run(
                f"{BOARD}_unpatched", eval_tiling="untiled", size="320", median_ms=40.0
            ),
        ]
    )

    out = add_resize_cost(df, load_resize_costs(_tree(tmp_path)))

    assert list(out["resize_ms"]) == [8.0, 8.0, 8.0]
    assert set(out["resize_platform"]) == {BOARD}


def test_a_trees_own_measurement_wins_over_its_boards(tmp_path):
    """
    Exact match first. A tree that was measured in its own right -- a different
    OpenCV build, a different governor -- must not be silently overwritten by
    the board-level fallback.
    """
    root = _tree(tmp_path, platforms=(BOARD,))

    own = root / f"{BOARD}_cpu"
    own.mkdir()
    artifact = _artifact(f"{BOARD}_cpu")
    for entry in artifact["resizes"]:
        entry["latency"]["median_latency_ms"] = 99.0
    (own / RESIZE_ARTIFACT).write_text(json.dumps(artifact))

    df = pd.DataFrame(
        [_run(f"{BOARD}_cpu", eval_tiling="untiled", size="320", median_ms=200.0)]
    )

    row = add_resize_cost(df, load_resize_costs(root)).iloc[0]

    assert row["resize_platform"] == f"{BOARD}_cpu"
    assert row["resize_ms"] == 99.0


def test_an_unmeasured_host_is_left_missing_not_uncorrected(tmp_path):
    """
    A row that quietly carries its uncorrected latency in a corrected column is
    a wrong number that no reader can detect. NaN is a missing measurement and
    says so -- which is also the right answer for the SavedModel reference
    trees, whose graphs resize internally and pay no cv2.resize at all.
    """
    df = pd.DataFrame(
        [
            _run("tf-savedmodel", eval_tiling="untiled", size="320", median_ms=30.0),
            _run("some-new-board", eval_tiling="tiled", size="512", median_ms=30.0),
        ]
    )

    out = add_resize_cost(df, load_resize_costs(_tree(tmp_path)))

    assert out["resize_ms"].isna().all()
    assert out["median_latency_ms_net"].isna().all()
    assert (out["median_latency_ms"] == 30.0).all()


def test_correction_never_reports_negative_compute(tmp_path):
    """
    A resize larger than the run it is subtracted from means the two were not
    measured under comparable conditions. Clipping keeps the frame printable;
    the zero is the tell.
    """
    df = pd.DataFrame([_run(BOARD, eval_tiling="untiled", size="320", median_ms=1.0)])

    row = add_resize_cost(df, load_resize_costs(_tree(tmp_path))).iloc[0]

    assert row["median_latency_ms_net"] == 0.0
    assert row["resize_share"] > 1.0


def test_missing_results_root_yields_an_empty_frame_not_an_error(tmp_path):
    assert load_resize_costs(tmp_path / "nope").empty
    assert resize_cost_table(tmp_path / "nope").empty


def test_cost_table_is_one_row_per_host_and_source(tmp_path):
    table = resize_cost_table(load_resize_costs(_tree(tmp_path)))

    assert len(table) == 2
    assert list(table["Host"]) == [BOARD, BOARD]
    assert set(table["Source"]) == {"1024x1024", "512x512"}
    assert "-> 320 (ms)" in table.columns
