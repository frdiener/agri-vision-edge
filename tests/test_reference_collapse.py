"""
Tests for collapsing the sweep onto the reference configuration.

Outside the tiling and resolution studies, single-class, tiled-trained and
tiled-evaluated runs are folded into the untiled multi-class reference. That is
only defensible for *cost* metrics, and only with the residual stated: the
dimensions are individually small but all point the same way, because the
reference cell is the one fed full-frame images and therefore the one paying
the largest resize cost.
"""

from __future__ import annotations

import pandas as pd
import pytest

from agri_vision_edge.evaluation.benchmark_report import (
    CPU_REFERENCE_PLATFORM,
    DEFAULT_NMS,
    REFERENCE_CONFIG,
    collapse_bias_table,
    collapse_divergence,
    collapse_holds,
    collapsed_latency_table,
    drop_failed_deployments,
    reference_config_slice,
)

NPU = "frdm-imx93"


def _run(
    *,
    classes="mc",
    dataset="phenobench",
    eval_tiling="untiled",
    arch="ssd-mn2",
    scheme_granularity="per-tensor",
    platform=NPU,
    ap=0.38,
    latency=30.0,
):
    return {
        "run": f"{eval_tiling}_{arch}_{classes}_{dataset}_320_int8_ptq_"
        f"{scheme_granularity}_{DEFAULT_NMS}",
        "platform": platform,
        "backend": "delegate" if platform == NPU else "cpu",
        "arch": arch,
        "arch_label": arch,
        "classes": classes,
        "dataset": dataset,
        "size": "320",
        "eval_tiling": eval_tiling,
        "precision": "int8",
        "quant": "ptq",
        "granularity": scheme_granularity,
        "nms": DEFAULT_NMS,
        "AP": ap,
        "median_latency_ms": latency,
        "p95_latency_ms": None if latency is None else latency * 1.05,
    }


def _matrix(values):
    """One run per (classes, dataset, eval_tiling) cell, latency from `values`."""
    rows = []
    for (classes, dataset, eval_tiling), latency in values.items():
        rows.append(
            _run(
                classes=classes,
                dataset=dataset,
                eval_tiling=eval_tiling,
                latency=latency,
            )
        )
    return pd.DataFrame(rows)


# ------------------------------------------------------- the reference slice


def test_reference_slice_pins_all_three_dimensions():
    df = _matrix(
        {
            ("mc", "phenobench", "untiled"): 30.0,
            ("sc", "phenobench", "untiled"): 29.0,
            ("mc", "phenobench-tiled", "untiled"): 28.0,
            ("mc", "phenobench", "tiled"): 26.0,
        }
    )

    assert list(reference_config_slice(df)["median_latency_ms"]) == [30.0]


def test_a_dimension_can_be_left_open_for_the_studies_that_need_it():
    df = _matrix(
        {
            ("mc", "phenobench", "untiled"): 30.0,
            ("mc", "phenobench", "tiled"): 26.0,
            ("sc", "phenobench", "untiled"): 29.0,
        }
    )

    open_tiling = reference_config_slice(df, eval_tiling=None)

    assert sorted(open_tiling["median_latency_ms"]) == [26.0, 30.0]


def test_the_reference_config_is_untiled_multiclass():
    # Pinned by a test because everything else is defined relative to it.
    assert REFERENCE_CONFIG == {
        "classes": "mc",
        "dataset": "phenobench",
        "eval_tiling": "untiled",
        # Pinned too: without it the other three stop identifying one cell as
        # soon as the resolution ladder lands.
        "size": "320",
    }


# ------------------------------------------------------------ the guard


def _two_archs(spread_within, spread_between):
    """Runs where `classes` moves latency a little and `arch` a lot."""
    rows = []
    for arch, base in (("ssd-mn2", 30.0), ("ssd-mn2-fpnlite", 30.0 + spread_between)):
        for classes, delta in (("mc", 0.0), ("sc", -spread_within)):
            rows.append(_run(arch=arch, classes=classes, latency=base + delta))
    return pd.DataFrame(rows)


def test_the_collapse_holds_when_the_folded_axis_is_the_smaller_effect():
    df = _two_archs(spread_within=0.5, spread_between=15.0)

    divergence = collapse_divergence(df)

    assert collapse_holds(divergence)


def test_the_collapse_is_refused_when_the_folded_axis_rivals_the_kept_one():
    # This is the accuracy case: folding something that moves the metric as
    # much as the effect under study destroys the effect.
    df = _two_archs(spread_within=15.0, spread_between=2.0)

    divergence = collapse_divergence(df)

    assert not collapse_holds(divergence)


def test_the_guard_is_per_metric():
    # Same runs, two metrics: cost collapses, accuracy does not.
    rows = []
    for arch, lat_base, ap_base in (("ssd-mn2", 30.0, 0.38), ("fpnlite", 45.0, 0.40)):
        for classes, ap in (("mc", 0.0), ("sc", -0.20)):
            rows.append(
                _run(
                    arch=arch,
                    classes=classes,
                    latency=lat_base,
                    ap=ap_base + ap,
                )
            )
    df = pd.DataFrame(rows)

    assert collapse_holds(collapse_divergence(df, metric="median_latency_ms"))
    assert not collapse_holds(collapse_divergence(df, metric="AP"))


def test_an_empty_or_invariant_frame_does_not_pass_the_guard():
    # "Nothing to compare" must not read as "collapse verified".
    assert not collapse_holds(collapse_divergence(pd.DataFrame()))
    assert not collapse_holds(pd.DataFrame())


# ------------------------------------------------------------ the bias


def test_the_bias_reports_direction_not_just_magnitude():
    df = _matrix(
        {
            ("mc", "phenobench", "untiled"): 30.0,
            ("sc", "phenobench", "untiled"): 29.1,
            ("mc", "phenobench-tiled", "untiled"): 28.2,
            ("mc", "phenobench", "tiled"): 25.5,
        }
    )

    bias = collapse_bias_table(df).set_index("dimension")

    assert bias.loc["classes", "median change %"] == pytest.approx(-3.0)
    assert bias.loc["dataset", "median change %"] == pytest.approx(-6.0)
    assert bias.loc["eval_tiling", "median change %"] == pytest.approx(-15.0)


def test_symmetric_residuals_report_as_no_bias():
    df = _matrix(
        {
            ("mc", "phenobench", "untiled"): 30.0,
            ("sc", "phenobench", "untiled"): 33.0,
            ("mc", "phenobench-tiled", "untiled"): 27.0,
        }
    )

    bias = collapse_bias_table(df).set_index("dimension")

    assert bias.loc["classes", "median change %"] == pytest.approx(10.0)
    assert bias.loc["dataset", "median change %"] == pytest.approx(-10.0)


# ------------------------------------------------- the collapsed latency table


def test_the_collapsed_row_carries_the_reference_cell_and_the_bias():
    df = _matrix(
        {
            ("mc", "phenobench", "untiled"): 30.0,
            ("sc", "phenobench", "untiled"): 30.0,
            ("mc", "phenobench-tiled", "untiled"): 30.0,
            ("mc", "phenobench", "tiled"): 20.0,
        }
    )

    row = collapsed_latency_table(df).iloc[0]

    assert row["Runs"] == 4
    assert row["Lat med (ms)"] == pytest.approx(27.5)
    assert row["Ref cell (ms)"] == pytest.approx(30.0)
    # The collapsed figure is optimistic by a sixth, and says so.
    assert row["bias %"] == pytest.approx(-8.3, abs=0.1)
    assert row["spread %"] == pytest.approx(36.4, abs=0.1)


def test_dimensions_left_out_of_the_collapse_are_pinned_not_averaged():
    df = _matrix(
        {
            ("mc", "phenobench", "untiled"): 30.0,
            ("sc", "phenobench", "untiled"): 30.0,
            ("mc", "phenobench", "tiled"): 20.0,
        }
    )

    row = collapsed_latency_table(df, collapse=("classes", "dataset")).iloc[0]

    assert row["Runs"] == 2
    assert row["Lat med (ms)"] == pytest.approx(30.0)
    assert row["bias %"] == pytest.approx(0.0)


def test_failed_deployments_never_enter_the_mean():
    # A collapsed run has entirely real latency that is nonetheless not the cost
    # of a working detector; averaging is exactly where that does damage.
    df = pd.concat(
        [
            _matrix({("mc", "phenobench", "untiled"): 30.0}),
            pd.DataFrame(
                [
                    _run(platform=CPU_REFERENCE_PLATFORM, ap=0.38, latency=100.0),
                    _run(
                        classes="sc",
                        platform=CPU_REFERENCE_PLATFORM,
                        ap=0.38,
                        latency=100.0,
                    ),
                    # Same file, delegated, AP collapsed to ~0 and "fast".
                    _run(classes="sc", ap=0.001, latency=5.0),
                ]
            ),
        ],
        ignore_index=True,
    )

    assert len(drop_failed_deployments(df)) == 3

    npu_row = collapsed_latency_table(df).set_index("Platform").loc[NPU]
    assert npu_row["Runs"] == 1
    assert npu_row["Lat med (ms)"] == pytest.approx(30.0)


def test_empty_frames_are_handled():
    assert collapsed_latency_table(pd.DataFrame()).empty
    assert collapse_bias_table(pd.DataFrame()).empty
    assert collapse_divergence(pd.DataFrame()).empty
    assert drop_failed_deployments(pd.DataFrame()).empty
