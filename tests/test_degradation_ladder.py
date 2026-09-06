"""
Tests for the degradation ladder.

Each delta isolates one transformation by holding the others fixed. Two rungs
carry the weight:

* the **SavedModel** rung -- without it the conversion loss is invisible and
  gets folded into "quantization", which for at least one real config
  understates quantization's share by more than a full AP point;
* the **NMS** rung -- the export swaps TFOD's per-class suppression for the
  fused ``TFLite_Detection_PostProcess`` fast pass, and that substitution, not
  the format change, is most of what "conversion" used to measure.
"""

from __future__ import annotations

import pandas as pd
import pytest

from agri_vision_edge.evaluation.benchmark_report import (
    CPU_REFERENCE_PLATFORM,
    DEFAULT_NMS,
    REFERENCE_PLATFORM,
    REGULAR_NMS,
    degradation_ladder_table,
)

NPU = "frdm-imx93"

CONTROL_COL = f"TFLite fp32 ({REGULAR_NMS})"
DEPLOYED_COL = f"TFLite fp32 ({DEFAULT_NMS})"
NPU_COL = f"int8 NPU ({NPU})"


def _run(platform, precision, quant, granularity, ap, *, nms=None, classes="mc"):
    return {
        "platform": platform,
        "arch_label": "SSD MobileNetV2",
        "classes": classes,
        "dataset": "phenobench",
        "size": "320",
        "eval_tiling": "untiled",
        "precision": precision,
        "quant": quant,
        "granularity": granularity,
        "nms": nms,
        "AP": ap,
    }


def _chain(
    *,
    reference=0.40,
    fp32_control=0.395,
    fp32=0.39,
    int8_cpu=0.37,
    int8_npu=0.371,
):
    """The full deployed chain, one row per rung."""
    rows = [
        _run(CPU_REFERENCE_PLATFORM, "fp32", "ptq", None, fp32, nms=DEFAULT_NMS),
        _run(
            CPU_REFERENCE_PLATFORM,
            "int8",
            "ptq",
            "per-tensor",
            int8_cpu,
            nms=DEFAULT_NMS,
        ),
        _run(NPU, "int8", "ptq", "per-tensor", int8_npu, nms=DEFAULT_NMS),
    ]
    if fp32_control is not None:
        rows.append(
            _run(
                CPU_REFERENCE_PLATFORM,
                "fp32",
                "ptq",
                None,
                fp32_control,
                nms=REGULAR_NMS,
            )
        )
    if reference is not None:
        # The SavedModel tree has no NMS token: it *is* the per-class
        # implementation the control reproduces.
        rows.append(_run(REFERENCE_PLATFORM, "fp32", "ptq", None, reference))
    return pd.DataFrame(rows)


def test_the_four_losses_are_successive_differences():
    row = degradation_ladder_table(_chain(), npu_platform=NPU).iloc[0]

    assert row["conversion"] == pytest.approx(0.395 - 0.40)
    assert row["nms-swap"] == pytest.approx(0.39 - 0.395)
    assert row["quantization"] == pytest.approx(0.37 - 0.39)
    assert row["delegation"] == pytest.approx(0.371 - 0.37)


def test_the_losses_sum_to_the_end_to_end_drop():
    row = degradation_ladder_table(_chain(), npu_platform=NPU).iloc[0]

    total = row[["conversion", "nms-swap", "quantization", "delegation"]].sum()
    assert total == pytest.approx(row[NPU_COL] - row["SavedModel"])


def test_all_rungs_are_reported():
    table = degradation_ladder_table(_chain(), npu_platform=NPU)

    assert list(table.columns) == [
        "SavedModel",
        CONTROL_COL,
        DEPLOYED_COL,
        "int8 CPU",
        NPU_COL,
        "conversion",
        "nms-swap",
        "quantization",
        "delegation",
    ]


def test_the_deployed_rungs_never_pick_up_the_control():
    # The control is a different graph at the same rung; if it leaked into the
    # deployed columns every downstream delta would be measured against a model
    # that is not shipped.
    table = degradation_ladder_table(_chain(fp32_control=0.9), npu_platform=NPU)

    assert table[DEPLOYED_COL].iloc[0] == pytest.approx(0.39)
    assert table["quantization"].iloc[0] == pytest.approx(0.37 - 0.39)


def test_a_missing_reference_empties_the_column_without_dropping_the_row():
    # An incomplete reference sweep must stay visible; silently narrowing the
    # table would hide exactly the rung this function exists to expose.
    table = degradation_ladder_table(_chain(reference=None), npu_platform=NPU)

    assert len(table) == 1
    assert table["SavedModel"].isna().all()
    assert table["conversion"].isna().all()
    assert table["quantization"].notna().all()


def test_a_missing_control_empties_only_the_nms_column():
    table = degradation_ladder_table(_chain(fp32_control=None), npu_platform=NPU)

    assert len(table) == 1
    assert table["nms-swap"].isna().all()
    assert table["conversion"].isna().all()
    assert table["quantization"].notna().all()
    assert table["delegation"].notna().all()


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
                    _run(
                        CPU_REFERENCE_PLATFORM,
                        "int8",
                        "qat",
                        "per-tensor",
                        0.38,
                        nms=DEFAULT_NMS,
                    ),
                    _run(NPU, "int8", "qat", "per-tensor", 0.381, nms=DEFAULT_NMS),
                    _run(REFERENCE_PLATFORM, "fp32", "qat", None, 0.41),
                ]
            ),
        ],
        ignore_index=True,
    )

    row = degradation_ladder_table(df, npu_platform=NPU, quant="qat").iloc[0]

    assert row["SavedModel"] == pytest.approx(0.41)
    assert row["int8 CPU"] == pytest.approx(0.38)
    assert row["quantization"] == pytest.approx(0.38 - 0.39)


def test_frames_without_an_nms_column_still_build_a_ladder():
    # Older result trees predate the paired export; they must degrade to a
    # ladder with an empty NMS rung, not to an empty table.
    df = _chain(fp32_control=None).drop(columns=["nms"])

    table = degradation_ladder_table(df, npu_platform=NPU)

    assert len(table) == 1
    assert table["quantization"].notna().all()
    assert table["nms-swap"].isna().all()


def test_empty_frame_is_handled():
    assert degradation_ladder_table(pd.DataFrame(), npu_platform=NPU).empty
