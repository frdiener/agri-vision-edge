"""
Tests for the six-step story tables of Chapter 6.

Each step narrows the sweep itself, so the tests are mostly about scope: a step
that quietly reads the wrong slice produces a plausible table, which is the
failure mode worth guarding.
"""

from __future__ import annotations

import pandas as pd
import pytest

from agri_vision_edge.evaluation.benchmark_report import (
    CPU_REFERENCE_PLATFORM,
    DEFAULT_NMS,
    REFERENCE_PLATFORM,
    REGULAR_NMS,
    baseline_table,
    device_latency_table,
    discover_board_pairs,
    preparation_ladder_table,
    qat_reclaim_table,
    story_ablation_table,
)

NPU = "frdm-imx93"
CPU = "frdm-imx93_cpu"


def _row(
    platform,
    *,
    ap,
    precision="fp32",
    quant="ptq",
    granularity=None,
    nms=DEFAULT_NMS,
    arch="ssd-mn2",
    arch_label="SSD MobileNetV2",
    classes="mc",
    dataset="phenobench",
    eval_tiling="untiled",
    latency=None,
    backend=None,
):
    return {
        "run": "synthetic",
        "platform": platform,
        "backend": backend or ("delegate" if platform == NPU else "cpu"),
        "arch": arch,
        "arch_label": arch_label,
        "classes": classes,
        "dataset": dataset,
        "eval_tiling": eval_tiling,
        "size": "320",
        "precision": precision,
        "quant": quant,
        "granularity": granularity,
        "nms": nms,
        "AP": ap,
        "AP50": None if ap is None else ap + 0.28,
        "AP75": None if ap is None else ap + 0.01,
        "crop_AP": None if ap is None else ap + 0.19,
        "weed_AP": None if ap is None else ap - 0.19,
        "APS": None if ap is None else ap - 0.24,
        # Step 1 is the one table that sits beside published numbers, so it
        # reads the official evaluator rather than pycocotools. The two stacks
        # do not agree -- upstream scores a little higher here -- which is
        # exactly why they must not be mixed, so the fixture keeps them apart.
        "faithful_mAP": None if ap is None else ap + 0.0064,
        "faithful_mAP50": None if ap is None else ap + 0.2942,
        "faithful_mAP75": None if ap is None else ap + 0.0161,
        "faithful_crop_AP": None if ap is None else ap + 0.2005,
        "faithful_weed_AP": None if ap is None else ap - 0.1877,
        "faithful_stale": False,
        "median_latency_ms": latency,
        "p95_latency_ms": None if latency is None else latency * 1.05,
    }


def _chain(**overrides):
    """The full preparation chain for one configuration."""
    return [
        _row(REFERENCE_PLATFORM, ap=0.3966, nms=None, **overrides),
        _row(CPU_REFERENCE_PLATFORM, ap=0.3940, nms=REGULAR_NMS, **overrides),
        _row(CPU_REFERENCE_PLATFORM, ap=0.3890, **overrides),
        _row(
            CPU_REFERENCE_PLATFORM,
            ap=0.3846,
            precision="int8",
            granularity="per-channel",
            **overrides,
        ),
        _row(
            CPU_REFERENCE_PLATFORM,
            ap=0.3769,
            precision="int8",
            granularity="per-tensor",
            **overrides,
        ),
        _row(
            CPU_REFERENCE_PLATFORM,
            ap=0.3840,
            precision="int8",
            quant="qat",
            granularity="per-channel",
            **overrides,
        ),
        _row(
            CPU_REFERENCE_PLATFORM,
            ap=0.3875,
            precision="int8",
            quant="qat",
            granularity="per-tensor",
            **overrides,
        ),
    ]


# ------------------------------------------------------------ step 1


def test_baseline_reports_the_float_savedmodel_not_the_converted_model():
    # The baseline has to be measured before TensorFlow Lite is involved,
    # otherwise every later cost is quoted against something already degraded.
    table = baseline_table(pd.DataFrame(_chain()))
    ours = table[table["Source"].str.startswith("this work")]

    assert len(ours) == 1
    assert ours["Input"].iloc[0] == "320x320"


def test_baseline_uses_the_official_evaluator_not_pycocotools():
    # It is the only table putting our numbers next to someone else's, so it
    # has to use the metric stack those numbers came from. Reporting the
    # pycocotools value (39.66) here would silently compare across stacks.
    table = baseline_table(pd.DataFrame(_chain()))
    ours = table[table["Source"].str.startswith("this work")]

    assert ours["AP"].iloc[0] == pytest.approx(40.30)


def test_stale_official_metrics_are_dropped_rather_than_shown():
    rows = _chain()
    for row in rows:
        row["faithful_stale"] = True

    table = baseline_table(pd.DataFrame(rows))
    ours = (
        table[table["Source"].str.startswith("this work")] if not table.empty else table
    )

    assert ours.empty


def test_baseline_includes_the_published_rows_and_marks_their_split():
    table = baseline_table(pd.DataFrame(_chain()))

    assert "Faster R-CNN" in set(table["Detector"])
    assert (
        table[table["Detector"] == "Faster R-CNN"]["Source"].iloc[0]
        == "published (test split)"
    )


def test_baseline_ignores_off_reference_configurations():
    df = pd.DataFrame(_chain() + _chain(classes="sc"))

    ours = baseline_table(df)
    ours = ours[ours["Source"].str.startswith("this work")]

    assert len(ours) == 1


# ------------------------------------------------------------ step 2 / 3


def test_the_float_rung_prices_conversion_at_the_requested_nms():
    # The ladder runs at one post-processing variant throughout, so its float
    # step is the format conversion alone. Asking for the other variant moves
    # that step by exactly the substitution's cost: the two decompose
    # additively (-0.26 conversion, -0.50 substitution, -0.76 together), which
    # is what lets the swap be priced on its own axis instead of inside this
    # table.
    rows = pd.DataFrame(_chain())
    regular = preparation_ladder_table(rows, nms="regnms").set_index("Stage")
    fast = preparation_ladder_table(rows, nms="fastnms").set_index("Stage")

    assert regular.loc["Float TFLite, Per-class NMS", "MNv2 d"] == pytest.approx(-0.26)
    assert fast.loc[
        "Float TFLite, Fast NMS (default export)", "MNv2 d"
    ] == pytest.approx(-0.76)


def test_int8_rows_are_quoted_against_the_deployed_float_not_each_other():
    # They are alternative exports of the same model, not a chain; quoting
    # per-tensor against per-channel would invent a cost that nobody pays.
    table = preparation_ladder_table(pd.DataFrame(_chain())).set_index("Stage")

    assert table.loc["INT8 PTQ, per-channel", "MNv2 d"] == pytest.approx(-0.44)
    assert table.loc["INT8 PTQ, per-tensor", "MNv2 d"] == pytest.approx(-1.21)
    assert table.loc["INT8 QAT, per-tensor", "MNv2 d"] == pytest.approx(-0.15)


def test_the_ptq_only_table_stops_before_qat():
    table = preparation_ladder_table(pd.DataFrame(_chain()), include_qat=False)

    assert not any("QAT" in stage for stage in table["Stage"])
    assert len(table) == 4


def test_the_float_rung_reads_the_variant_it_was_asked_for():
    # Regression, in its current form: the float rung has to be the file that
    # actually ships at the presented post-processing. Reading the other
    # variant's run here would charge the substitution to conversion, which is
    # the confusion the single-variant ladder exists to prevent.
    rows = pd.DataFrame(_chain())
    regular = preparation_ladder_table(rows, nms="regnms").set_index("Stage")
    fast = preparation_ladder_table(rows, nms="fastnms").set_index("Stage")

    assert regular.loc["Float TFLite, Per-class NMS", "MNv2 AP"] == pytest.approx(39.40)
    assert fast.loc[
        "Float TFLite, Fast NMS (default export)", "MNv2 AP"
    ] == pytest.approx(38.90)


def test_the_ablation_control_rung_survives_the_same_way():
    row = story_ablation_table(pd.DataFrame(_chain())).iloc[0]

    assert row["Conversion"] is not None and not pd.isna(row["Conversion"])
    assert row["NMS swap"] is not None and not pd.isna(row["NMS swap"])


def test_the_ladder_is_not_confused_by_other_input_resolutions():
    # Regression, and it bit for real: the resolution ladder landed in the
    # sweep and every rung silently picked the 1024 run, reporting a +8.4 AP
    # "conversion gain". Class regime, training set and evaluation regime stop
    # identifying one cell as soon as a second resolution exists.
    other = [dict(row, size="1024", AP=(row["AP"] or 0) + 0.09) for row in _chain()]
    df = pd.DataFrame(_chain() + other)

    table = preparation_ladder_table(df).set_index("Stage")

    assert table.loc["Float SavedModel (reference)", "MNv2 AP"] == pytest.approx(39.66)
    assert table.loc[
        "Float TFLite, Fast NMS (default export)", "MNv2 d"
    ] == pytest.approx(-0.76)


def test_the_ladder_stays_at_the_reference_configuration():
    df = pd.DataFrame(_chain() + _chain(classes="sc", eval_tiling="tiled"))

    table = preparation_ladder_table(df).set_index("Stage")

    assert table.loc["Float SavedModel (reference)", "MNv2 AP"] == pytest.approx(39.66)


# ------------------------------------------------------------ step 3


def test_qat_reclaim_expresses_the_repair_as_a_share_of_the_deficit():
    table = qat_reclaim_table(pd.DataFrame(_chain())).set_index("Granularity")

    per_tensor = table.loc["per-tensor"]
    assert per_tensor["PTQ cost"] == pytest.approx(-1.21)
    assert per_tensor["Reclaimed"] == pytest.approx(1.06)
    assert per_tensor["Reclaimed %"] == pytest.approx(88.0, abs=1.0)


def test_no_reclaim_percentage_where_there_was_no_deficit():
    # per-channel loses 0.44 here; a "percentage repaired" of a deficit that
    # barely exists is noise amplified into a headline number.
    rows = _chain()
    for row in rows:
        if row["granularity"] == "per-channel" and row["quant"] == "ptq":
            row["AP"] = 0.3889  # -0.01 against the float model

    table = qat_reclaim_table(pd.DataFrame(rows)).set_index("Granularity")

    assert pd.isna(table.loc["per-channel", "Reclaimed %"])


# ------------------------------------------------------------ step 5


def test_board_pairs_exclude_control_builds():
    df = pd.DataFrame(
        [
            _row(NPU, ap=0.38),
            _row(CPU, ap=0.38),
            _row(f"{NPU}_unpatched", ap=0.38),
            _row(f"{NPU}_unpatched_cpu", ap=0.38),
        ]
    )

    assert discover_board_pairs(df) == [(NPU, CPU)]


def test_speedup_compares_the_same_board_with_the_delegate_off():
    df = pd.DataFrame(
        [
            _row(
                CPU_REFERENCE_PLATFORM,
                ap=0.377,
                precision="int8",
                granularity="per-tensor",
            ),
            _row(
                CPU, ap=0.377, precision="int8", granularity="per-tensor", latency=106.7
            ),
            _row(
                NPU, ap=0.378, precision="int8", granularity="per-tensor", latency=29.8
            ),
        ]
    )

    row = device_latency_table(df).iloc[0]

    assert row["Speedup"] == pytest.approx(3.6, abs=0.05)
    assert row["NPU FPS"] == pytest.approx(33.6, abs=0.1)
    assert row["dAP"] == pytest.approx(0.001)


def test_a_broken_delegated_run_never_reaches_the_latency_table():
    # It is fast precisely because it is not computing anything; unfiltered,
    # it sorts to the top of a latency ranking.
    df = pd.DataFrame(
        [
            _row(
                CPU_REFERENCE_PLATFORM,
                ap=0.377,
                precision="int8",
                granularity="per-tensor",
            ),
            _row(
                CPU, ap=0.377, precision="int8", granularity="per-tensor", latency=106.7
            ),
            _row(
                NPU, ap=0.001, precision="int8", granularity="per-tensor", latency=4.0
            ),
        ]
    )

    assert device_latency_table(df).empty
    assert not device_latency_table(df, deployable_only=False).empty


# ------------------------------------------------------------ step 6


def test_each_deviation_is_a_single_axis_from_the_reference():
    df = pd.DataFrame(
        _chain() + _chain(classes="sc") + _chain(dataset="phenobench-tiled")
    )

    table = story_ablation_table(df)

    assert set(table["Variant"]) == {
        "Reference (mc, trained full, eval full)",
        "Single-class",
        "Trained tiled",
    }


def test_the_ablation_carries_the_per_step_costs():
    table = story_ablation_table(pd.DataFrame(_chain())).iloc[0]

    assert table["Float AP"] == pytest.approx(39.66)
    assert table["Conversion"] == pytest.approx(-0.26)
    assert table["NMS swap"] == pytest.approx(-0.50)
    assert table["PTQ"] == pytest.approx(-1.21)
    assert table["QAT reclaim"] == pytest.approx(1.06)


def test_a_missing_rung_empties_the_cell_without_dropping_the_row():
    # The SavedModel reference was never swept for tiled evaluation; that row
    # still has to appear, with the conversion columns blank.
    rows = [
        r for r in _chain(eval_tiling="tiled") if r["platform"] != REFERENCE_PLATFORM
    ]
    df = pd.DataFrame(_chain() + rows)

    table = story_ablation_table(df).set_index("Variant")

    assert "Evaluated tiled" in table.index
    assert pd.isna(table.loc["Evaluated tiled", "Float AP"])
    assert pd.isna(table.loc["Evaluated tiled", "Conversion"])
    assert table.loc["Evaluated tiled", "PTQ"] == pytest.approx(-1.21)


def test_empty_frames_are_handled():
    for fn in (
        baseline_table,
        preparation_ladder_table,
        qat_reclaim_table,
        device_latency_table,
        story_ablation_table,
    ):
        assert fn(pd.DataFrame()).empty
