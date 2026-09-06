"""
Tests for the post-processing substitution analysis.

Every deployable is converted twice -- once with the fused fast NMS that ships
and once with the per-class pass the training checkpoint runs -- from one
checkpoint, one graph and one calibration set. The pair therefore prices the
substitution and nothing else, but only if the two rows are actually kept
apart: they are identical in every field a table or figure groups on except
``nms``, so a frame that is not scoped collapses them into a mean.
"""

from __future__ import annotations

import pandas as pd
import pytest

from agri_vision_edge.evaluation.benchmark_report import (
    DEFAULT_NMS,
    REGULAR_NMS,
    latency_table,
    nms_latency_tradeoff_table,
    nms_substitution_table,
    sanity_checks,
    scheme_comparison_table,
    select_nms,
)


def _run(
    nms,
    *,
    ap,
    crop=None,
    weed=None,
    classes="mc",
    latency=20.0,
    platform="gaia",
    dataset="phenobench",
):
    return {
        "run": f"untiled_ssd-mn2_{classes}_{dataset}_320_int8_ptq_per-tensor_{nms}",
        "platform": platform,
        "arch_label": "SSD MobileNetV2",
        "arch": "ssd-mn2",
        "classes": classes,
        "dataset": dataset,
        "size": "320",
        "eval_tiling": "untiled",
        "precision": "int8",
        "quant": "ptq",
        "granularity": "per-tensor",
        "nms": nms,
        "backend": "cpu",
        "AP": ap,
        "crop_AP": crop,
        "weed_AP": weed,
        "AR100": ap,
        "median_latency_ms": latency,
        "p95_latency_ms": latency,
        "fps": 1000.0 / latency,
    }


def _pair(**kwargs):
    """A deployed/control pair differing only in the post-processing."""
    fast = kwargs.pop("fast")
    control = kwargs.pop("control")
    return pd.DataFrame(
        [
            _run(DEFAULT_NMS, **fast, **kwargs),
            _run(REGULAR_NMS, **control, **kwargs),
        ]
    )


# ---------------------------------------------------------------- select_nms


def test_select_nms_keeps_one_variant():
    df = _pair(fast={"ap": 0.38}, control={"ap": 0.39})

    assert list(select_nms(df, DEFAULT_NMS)["AP"]) == [0.38]
    assert list(select_nms(df, REGULAR_NMS)["AP"]) == [0.39]


def test_select_nms_keeps_runs_that_have_no_variant():
    # The SavedModel reference and the YOLO exports carry no NMS token, and
    # they are rungs the ladder needs -- dropping them would remove the very
    # comparison the filter exists to make possible.
    df = pd.DataFrame([_run(None, ap=0.40), _run(REGULAR_NMS, ap=0.39)])

    assert list(select_nms(df, DEFAULT_NMS)["AP"]) == [0.40]


def test_select_nms_tolerates_frames_without_the_column():
    df = pd.DataFrame([{"AP": 0.4}])

    assert len(select_nms(df, DEFAULT_NMS)) == 1


# ------------------------------------------------------- scoping regressions


def test_scheme_comparison_reports_one_row_per_scheme():
    # The regression this guards: with both variants in the frame every scheme
    # appeared twice, identical in every visible column.
    table = scheme_comparison_table(_pair(fast={"ap": 0.38}, control={"ap": 0.39}))

    assert len(table) == 1
    assert table["mAP"].iloc[0] == pytest.approx(0.38)


def test_latency_is_not_pooled_across_the_two_graphs():
    df = pd.DataFrame(
        [
            _run(DEFAULT_NMS, ap=0.38, latency=18.0),
            _run(REGULAR_NMS, ap=0.39, latency=20.0),
        ]
    )

    table = latency_table(df)

    assert len(table) == 2
    assert sorted(table["Lat med (ms)"]) == [18.0, 20.0]


# -------------------------------------------------------------- the analysis


def test_the_pair_is_reported_as_one_row_of_differences():
    df = _pair(
        fast={"ap": 0.38, "crop": 0.55, "weed": 0.21},
        control={"ap": 0.39, "crop": 0.57, "weed": 0.21},
    )

    row = nms_substitution_table(df).iloc[0]

    assert row["dAP"] == pytest.approx(-0.01)
    assert row["dCrop AP"] == pytest.approx(-0.02)
    # SSD shares one box across classes, so the class-agnostic pass drops the
    # lower-scoring hypothesis -- the loss lands on crop, not on weed.
    assert row["dWeed AP"] == pytest.approx(0.0)


def test_unpaired_runs_are_not_reported():
    df = pd.DataFrame([_run(DEFAULT_NMS, ap=0.38)])

    assert nms_substitution_table(df).empty


def test_single_class_pairs_are_the_null_control():
    df = _pair(fast={"ap": 0.23, "weed": 0.23}, control={"ap": 0.23, "weed": 0.23}, classes="sc")

    assert nms_substitution_table(df)["dAP"].iloc[0] == pytest.approx(0.0)
    assert sanity_checks(df).empty


def test_a_non_zero_single_class_pair_is_an_error():
    df = _pair(
        fast={"ap": 0.23, "weed": 0.23},
        control={"ap": 0.25, "weed": 0.25},
        classes="sc",
    )

    issues = sanity_checks(df)

    assert "nms-control-broken" in set(issues["check"])
    assert (issues[issues["check"] == "nms-control-broken"]["severity"] == "error").all()


# ------------------------------------------------------------ latency payoff


def test_the_latency_saving_is_net_of_the_single_class_drift():
    # sc must be a null by construction, so whatever it shows is drift between
    # the two separately-benchmarked runs and has to come off the mc figure.
    df = pd.concat(
        [
            pd.DataFrame(
                [
                    _run(DEFAULT_NMS, ap=0.38, latency=19.0),
                    _run(REGULAR_NMS, ap=0.39, latency=20.0),
                ]
            ),
            pd.DataFrame(
                [
                    _run(DEFAULT_NMS, ap=0.23, classes="sc", latency=20.2),
                    _run(REGULAR_NMS, ap=0.23, classes="sc", latency=20.0),
                ]
            ),
        ],
        ignore_index=True,
    )

    row = nms_latency_tradeoff_table(df).iloc[0]

    assert row["dLatency mc (ms)"] == pytest.approx(-1.0)
    assert row["sc drift (ms)"] == pytest.approx(0.2)
    assert row["NMS saving (ms)"] == pytest.approx(-1.2)


def test_no_tradeoff_row_without_the_control():
    df = pd.DataFrame(
        [
            _run(DEFAULT_NMS, ap=0.38, latency=19.0),
            _run(REGULAR_NMS, ap=0.39, latency=20.0),
        ]
    )

    assert nms_latency_tradeoff_table(df).empty


def test_empty_frames_are_handled():
    assert nms_substitution_table(pd.DataFrame()).empty
    assert nms_latency_tradeoff_table(pd.DataFrame()).empty
