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

    assert div["configs"].iloc[0] == 2, "both the int8 and the fp32 config"
    assert div["max_abs_diff"].iloc[0] > 0.6
    assert not cpu_reference_holds(div)


def test_small_kernel_noise_still_holds_but_is_not_bit_identical():
    df = _frame(
        _runs(CPU_REFERENCE_PLATFORM, "cpu", [0.30, 0.33]),
        _runs("frdm-imx8mp_cpu", "cpu", [0.30, 0.33 + 2.2e-7]),
    )

    div = cpu_reference_divergence(df, metrics=("AP",))

    assert cpu_reference_holds(div)
    assert not cpu_reference_holds(div, tolerance=1e-9)
    assert div["bit_identical"].iloc[0] == 1


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
