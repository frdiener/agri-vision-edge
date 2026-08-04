"""
Tests for the four-rung degradation ladder.

Each delta isolates one transformation by holding the others fixed. The rung
that matters is the first: without the SavedModel reference the conversion loss
is invisible and gets folded into "quantization", which for at least one real
config understates quantization's share by more than a full AP point.
"""

from __future__ import annotations

import pandas as pd
import pytest

from agri_vision_edge.evaluation.benchmark_report import (
    CPU_REFERENCE_PLATFORM,
    REFERENCE_PLATFORM,
    degradation_ladder_table,
)

NPU = "frdm-imx93"


def _run(platform, precision, quant, granularity, ap, *, arch="SSD MobileNetV2"):
    return {
        "platform": platform,
        "arch_label": arch,
        "classes": "mc",
        "dataset": "phenobench",
        "eval_tiling": "untiled",
        "precision": precision,
        "quant": quant,
        "granularity": granularity,
        "AP": ap,
    }


def _chain(*, reference=0.40, fp32=0.39, int8_cpu=0.37, int8_npu=0.371):
    rows = [
        _run(CPU_REFERENCE_PLATFORM, "fp32", "ptq", None, fp32),
        _run(CPU_REFERENCE_PLATFORM, "int8", "ptq", "per-tensor", int8_cpu),
        _run(NPU, "int8", "ptq", "per-tensor", int8_npu),
    ]
    if reference is not None:
        rows.append(_run(REFERENCE_PLATFORM, "fp32", "ptq", None, reference))
    return pd.DataFrame(rows)


def test_the_three_losses_are_successive_differences():
    table = degradation_ladder_table(_chain(), npu_platform=NPU)

    row = table.iloc[0]
    assert row["conversion"] == pytest.approx(0.39 - 0.40)
    assert row["quantization"] == pytest.approx(0.37 - 0.39)
    assert row["delegation"] == pytest.approx(0.371 - 0.37)


def test_all_four_rungs_are_reported():
    table = degradation_ladder_table(_chain(), npu_platform=NPU)

    assert list(table.columns) == [
        "SavedModel",
        "TFLite fp32",
        "int8 CPU",
        f"int8 NPU ({NPU})",
        "conversion",
        "quantization",
        "delegation",
    ]


def test_a_missing_reference_empties_the_column_without_dropping_the_row():
    # An incomplete reference sweep must stay visible; silently narrowing the
    # table would hide exactly the rung this function exists to expose.
    table = degradation_ladder_table(_chain(reference=None), npu_platform=NPU)

    assert len(table) == 1
    assert table["SavedModel"].isna().all()
    assert table["conversion"].isna().all()
    assert table["quantization"].notna().all()


def test_a_missing_npu_leaves_delegation_empty():
    df = _chain()
    df = df[df["platform"] != NPU]

    table = degradation_ladder_table(df, npu_platform=NPU)

    assert len(table) == 1
    assert table["delegation"].isna().all()
    assert table["conversion"].notna().all()


def test_configs_without_the_deployed_chain_are_dropped():
    # Only the reference exists -- there is no ladder to report.
    df = pd.DataFrame([_run(REFERENCE_PLATFORM, "fp32", "ptq", None, 0.4)])

    assert degradation_ladder_table(df, npu_platform=NPU).empty


def test_eval_tiling_is_honoured():
    df = _chain()
    df.loc[df["platform"] == CPU_REFERENCE_PLATFORM, "eval_tiling"] = "tiled"

    # The fp32/int8 CPU rungs moved to the other regime, so nothing pairs up.
    assert degradation_ladder_table(df, npu_platform=NPU).empty


def test_qat_path_selects_the_qat_int8_exports():
    df = pd.concat(
        [
            _chain(),
            pd.DataFrame(
                [
                    _run(CPU_REFERENCE_PLATFORM, "int8", "qat", "per-tensor", 0.38),
                    _run(NPU, "int8", "qat", "per-tensor", 0.381),
                    _run(REFERENCE_PLATFORM, "fp32", "qat", None, 0.41),
                ]
            ),
        ],
        ignore_index=True,
    )

    table = degradation_ladder_table(df, npu_platform=NPU, quant="qat")

    row = table.iloc[0]
    assert row["SavedModel"] == pytest.approx(0.41)
    assert row["int8 CPU"] == pytest.approx(0.38)
    assert row["quantization"] == pytest.approx(0.38 - 0.39)


def test_empty_frame_is_handled():
    assert degradation_ladder_table(pd.DataFrame(), npu_platform=NPU).empty
