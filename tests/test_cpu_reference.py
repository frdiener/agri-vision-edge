"""
Tests for the CPU-reference guard.

The benchmark report shows a single CPU curve and treats it as the
unaccelerated reference for every board, which is only legitimate while the
per-board ``<board>_cpu`` trees agree with the reference host. These pin the
check that licenses that collapse.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from agri_vision_edge.evaluation.benchmark_report import (
    CPU_REFERENCE_PLATFORM,
    cpu_reference_divergence,
    cpu_reference_holds,
    cpu_reference_summary,
)


def _runs(platform, backend, aps, *, granularity=("per-tensor", None)):
    """Two runs per platform: one INT8 (granularity) and one FP32 (no granularity)."""
    rows = []
    for ap, gran in zip(aps, granularity, strict=True):
        rows.append(
            {
                "platform": platform,
                "backend": backend,
                "arch_label": "SSD MobileNetV2",
                "class_label": "Multi-class (crop+weed)",
                "dataset": "phenobench",
                "eval_tiling": "untiled",
                "precision": "int8" if gran else "fp32",
                "quant": "ptq",
                "granularity": gran if gran else np.nan,
                "AP": ap,
                "AP50": ap + 0.2,
            }
        )
    return rows


def _frame(*groups):
    return pd.DataFrame([r for g in groups for r in g])


def test_identical_cpu_trees_hold():
    df = _frame(
        _runs(CPU_REFERENCE_PLATFORM, "cpu", [0.30, 0.33]),
        _runs("frdm-imx8mp_cpu", "cpu", [0.30, 0.33]),
    )

    div = cpu_reference_divergence(df, metrics=("AP", "AP50"))

    assert set(div["platform"]) == {"frdm-imx8mp_cpu"}
    assert div["max_abs_diff"].max() == 0.0
    assert cpu_reference_holds(div)


def test_fp32_rows_are_compared_despite_a_nan_granularity():
    """
    The subtle one: fp32 carries no granularity.

    NaN never equals NaN, so a raw join on the granularity key silently drops
    every float config -- the guard would then "pass" while comparing only the
    INT8 half, which is exactly the half that is bit-identical anyway.
    """
    df = _frame(
        _runs(CPU_REFERENCE_PLATFORM, "cpu", [0.30, 0.33]),
        _runs("frdm-imx8mp_cpu", "cpu", [0.30, 0.99]),  # fp32 diverges
    )

    div = cpu_reference_divergence(df, metrics=("AP",))

    assert div["configs"].sum() == 2, "both the int8 and the fp32 config"
    assert set(div["precision"]) == {"int8", "fp32"}
    assert div.loc[div["precision"] == "fp32", "max_abs_diff"].iloc[0] > 0.6
    assert not cpu_reference_holds(div)


def test_small_kernel_noise_still_holds_but_is_not_bit_identical():
    df = _frame(
        _runs(CPU_REFERENCE_PLATFORM, "cpu", [0.30, 0.33]),
        _runs("frdm-imx8mp_cpu", "cpu", [0.30, 0.33 + 2.2e-7]),
    )

    div = cpu_reference_divergence(df, metrics=("AP",))

    assert cpu_reference_holds(div)
    assert not cpu_reference_holds(div, tolerance=1e-9)
    assert div.loc[div["precision"] == "int8", "bit_identical"].iloc[0] == 1
    assert div.loc[div["precision"] == "fp32", "bit_identical"].iloc[0] == 0


def test_delegate_runs_are_not_compared():
    # An NPU tree is expected to differ; only CPU trees are being collapsed.
    df = _frame(
        _runs(CPU_REFERENCE_PLATFORM, "cpu", [0.30, 0.33]),
        _runs("frdm-imx8mp", "delegate", [0.0001, 0.33]),
    )

    assert cpu_reference_divergence(df).empty


def test_missing_reference_is_unverified_not_passing():
    df = _frame(_runs("frdm-imx8mp_cpu", "cpu", [0.30, 0.33]))

    div = cpu_reference_divergence(df)

    assert div.empty
    # Empty means "nothing was checked" -- it must not read as a pass.
    assert not cpu_reference_holds(div)


def test_empty_frame_is_handled():
    assert cpu_reference_divergence(pd.DataFrame()).empty


def test_int8_topk_drift_is_tolerated_but_the_same_drift_in_fp32_is_not():
    """
    The reason the tolerance is per precision.

    INT8 scores are quantised to multiples of 1/256, so detections tied at the
    per-image top-k cutoff are ordered differently by the x86 and ARM kernels.
    The surviving set differs slightly and ``ignore_partials`` amplifies that
    into AP at the 1e-4 scale -- far above float noise, far below anything the
    report concludes. The same excursion in an FP32 tree has no such
    explanation and means the trees are not running the same computation.
    """
    drift = 1.6e-4

    int8_drifts = _frame(
        _runs(CPU_REFERENCE_PLATFORM, "cpu", [0.30, 0.33]),
        _runs("frdm-imx8mp_cpu", "cpu", [0.30 + drift, 0.33]),
    )
    fp32_drifts = _frame(
        _runs(CPU_REFERENCE_PLATFORM, "cpu", [0.30, 0.33]),
        _runs("frdm-imx8mp_cpu", "cpu", [0.30, 0.33 + drift]),
    )

    assert cpu_reference_holds(cpu_reference_divergence(int8_drifts, metrics=("AP",)))
    assert not cpu_reference_holds(
        cpu_reference_divergence(fp32_drifts, metrics=("AP",))
    )


def test_an_unrecognised_precision_gets_the_strictest_bound():
    # A new export scheme must not inherit the INT8 allowance by default.
    df = _frame(
        _runs(CPU_REFERENCE_PLATFORM, "cpu", [0.30, 0.33]),
        _runs("frdm-imx8mp_cpu", "cpu", [0.30 + 1e-4, 0.33]),
    )
    df.loc[df["precision"] == "int8", "precision"] = "int4"

    div = cpu_reference_divergence(df, metrics=("AP",))

    assert not cpu_reference_holds(div)


def test_summary_names_each_precision_and_its_bound():
    df = _frame(
        _runs(CPU_REFERENCE_PLATFORM, "cpu", [0.30, 0.33]),
        _runs("frdm-imx8mp_cpu", "cpu", [0.30 + 1.6e-4, 0.33]),
    )

    summary = cpu_reference_summary(cpu_reference_divergence(df, metrics=("AP",)))

    assert "int8 1.6e-04 of 2e-03" in summary
    assert "fp32 0.0e+00 of 1e-05" in summary
