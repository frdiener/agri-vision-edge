"""
Tests for the deployability matrix.

The three ways a deployment fails here do not look alike, and only one of them
raises anything. The matrix exists to make the other two visible:

* a run that produces ``NaN`` boxes is *rewarded* by pycocotools, so the
  integrity gate removes its ``metrics.json`` -- which means the evidence lives
  in the skip list, not in the results frame;
* a run that loads and produces plausible boxes can still score near zero under
  a delegate while the same file is fine on the same board's CPU.
"""

from __future__ import annotations

import pandas as pd
import pytest

from agri_vision_edge.evaluation.benchmark_report import (
    CPU_REFERENCE_PLATFORM,
    DEFAULT_NMS,
    deployability_matrix,
    deployability_summary,
)

NPU = "frdm-imx8mp"


def _run(platform, ap, *, precision="int8", granularity="per-tensor"):
    # Only quantized exports carry a weight granularity -- every fp32 run in
    # the real sweep has it unset, and the run name has no granularity token
    # either. Keeping one here would build a row whose `scheme` no other code
    # path can reproduce from its name.
    if precision == "fp32":
        granularity = None

    suffix = f"{granularity}_" if granularity else ""
    return {
        "run": f"untiled_ssd-mn2_mc_phenobench_320_{precision}_ptq_"
        f"{suffix}{DEFAULT_NMS}",
        "platform": platform,
        "arch_label": "SSD MobileNetV2",
        "classes": "mc",
        "dataset": "phenobench",
        "size": "320",
        "eval_tiling": "untiled",
        "precision": precision,
        "quant": "ptq",
        "granularity": granularity,
        "nms": DEFAULT_NMS,
        "AP": ap,
    }


def _cell(matrix, platform, scheme="int8_ptq_per-tensor"):
    return matrix.loc[matrix["scheme"] == scheme, platform].iloc[0]


def test_a_delegate_that_reproduces_the_cpu_reference_is_ok():
    df = pd.DataFrame([_run(CPU_REFERENCE_PLATFORM, 0.38), _run(NPU, 0.379)])

    assert _cell(deployability_matrix(df), NPU) == "ok"


def test_a_collapsed_delegated_run_is_not_reported_as_a_quantization_cost():
    # Same file, same board, delegate on: 0.003 against 0.38 on the CPU. That
    # is a broken deployment, and it must not be able to enter an accuracy
    # table as a very bad INT8 number.
    df = pd.DataFrame([_run(CPU_REFERENCE_PLATFORM, 0.38), _run(NPU, 0.003)])

    assert _cell(deployability_matrix(df), NPU) == "collapsed"


def test_a_mild_disagreement_is_degraded_not_collapsed():
    df = pd.DataFrame([_run(CPU_REFERENCE_PLATFORM, 0.38), _run(NPU, 0.30)])

    assert _cell(deployability_matrix(df), NPU) == "degraded"


def test_a_weak_export_is_not_blamed_on_the_accelerator():
    # Low AP everywhere is a quantization result, not a deployment failure:
    # the verdict is relative to the CPU reference for exactly this reason.
    df = pd.DataFrame([_run(CPU_REFERENCE_PLATFORM, 0.02), _run(NPU, 0.0199)])

    assert _cell(deployability_matrix(df), NPU) == "ok"


def test_unscoreable_runs_are_recovered_from_the_skip_list():
    # The run happened; its output was refused. Without the skip list this cell
    # is indistinguishable from one that was never benchmarked, which inverts
    # the finding.
    df = pd.DataFrame([_run(CPU_REFERENCE_PLATFORM, 0.39, precision="fp32")])
    skipped = [
        f"{NPU}/untiled_ssd-mn2_mc_phenobench_320_fp32_ptq_{DEFAULT_NMS} "
        "(corrupt predictions)"
    ]

    matrix = deployability_matrix(df, skipped)

    assert _cell(matrix, NPU, "fp32_ptq") == "unscoreable"


def test_a_never_run_cell_is_blank():
    df = pd.DataFrame([_run(CPU_REFERENCE_PLATFORM, 0.38), _run(NPU, 0.38)])
    df = pd.concat([df, pd.DataFrame([_run("frdm-imx93", 0.38)])], ignore_index=True)
    df = df[df["platform"] != "frdm-imx93"]

    matrix = deployability_matrix(df)

    assert "frdm-imx93" not in matrix.columns


def test_the_skip_list_respects_the_nms_scope():
    df = pd.DataFrame([_run(CPU_REFERENCE_PLATFORM, 0.39, precision="fp32")])
    skipped = [f"{NPU}/untiled_ssd-mn2_mc_phenobench_320_fp32_ptq_regnms (failed)"]

    # The control is not the deployable, so it must not fill the deployable's
    # cell.
    assert deployability_matrix(df, skipped).get(NPU) is None


def test_narrowing_the_columns_keeps_the_reference_scoring():
    df = pd.DataFrame([_run(CPU_REFERENCE_PLATFORM, 0.38), _run(NPU, 0.003)])

    # `drop_constant_keys=False`: this asserts on the full config identity, and
    # a one-config fixture would otherwise have most of it dropped as constant.
    matrix = deployability_matrix(df, platforms=[NPU], drop_constant_keys=False)

    assert list(matrix.columns) == [
        "arch_label",
        "classes",
        "dataset",
        "size",
        "scheme",
        NPU,
    ]
    assert _cell(matrix, NPU) == "collapsed"


def test_summary_counts_verdicts_per_platform():
    df = pd.DataFrame(
        [
            _run(CPU_REFERENCE_PLATFORM, 0.38),
            _run(NPU, 0.003),
            _run(CPU_REFERENCE_PLATFORM, 0.39, granularity="per-channel"),
            _run(NPU, 0.389, granularity="per-channel"),
        ]
    )

    summary = deployability_summary(deployability_matrix(df)).set_index("platform")

    assert summary.loc[NPU, "collapsed"] == 1
    assert summary.loc[NPU, "ok"] == 1


def test_empty_frame_is_handled():
    assert deployability_matrix(pd.DataFrame()).empty
    assert deployability_summary(pd.DataFrame()).empty


def test_the_skip_list_respects_the_callers_config_scope():
    # Regression: the frame is how a caller states its scope, but the skip list
    # was not filtered by it -- so scoping to multi-class still resurrected the
    # single-class runs here as `unscoreable`, inventing rows the caller had
    # deliberately excluded.
    df = pd.DataFrame([_run(CPU_REFERENCE_PLATFORM, 0.39, precision="fp32")])
    skipped = [
        f"{NPU}/untiled_ssd-mn2_sc_phenobench_320_fp32_ptq_{DEFAULT_NMS} "
        "(corrupt predictions)"
    ]

    matrix = deployability_matrix(df, skipped, drop_constant_keys=False)

    assert "sc" not in set(matrix["classes"])


def test_an_out_of_scope_skip_does_not_invent_a_platform_column():
    # The same leak in its worst form: a config absent from the frame entirely
    # would add a board column made only of skip-list entries.
    df = pd.DataFrame([_run(CPU_REFERENCE_PLATFORM, 0.39, precision="fp32")])
    skipped = [
        "frdm-imx93/untiled_ssd-mn2_sc_phenobench-tiled_320_fp32_ptq_"
        f"{DEFAULT_NMS} (corrupt predictions)"
    ]

    matrix = deployability_matrix(df, skipped)

    assert "frdm-imx93" not in matrix.columns


def test_keys_the_caller_pinned_are_dropped_from_the_table():
    # A caller scoped to one class regime and training set gets three columns
    # repeating one value beside the verdicts they came for. What a row *is*
    # -- architecture and scheme -- always stays.
    df = pd.DataFrame([_run(CPU_REFERENCE_PLATFORM, 0.38), _run(NPU, 0.38)])

    matrix = deployability_matrix(df)

    assert "arch_label" in matrix.columns
    assert "scheme" in matrix.columns
    assert not {"classes", "dataset", "size"} & set(matrix.columns)


def test_a_missing_reference_tree_warns_instead_of_scoring_everything_ok():
    """
    `platforms=` narrows the columns; filtering `runs_df` narrows the *scoring*
    too. Without the reference every cell falls through to `ok`, which reads as
    "the accelerator broke nothing" -- the exact inverse of the finding. That
    is silent, so it has to be loud.
    """
    df = pd.DataFrame([_run(CPU_REFERENCE_PLATFORM, 0.38), _run(NPU, 0.003)])

    with pytest.warns(RuntimeWarning, match="reference platform"):
        matrix = deployability_matrix(df[df["platform"] != CPU_REFERENCE_PLATFORM])

    # The fallback itself is unchanged -- it is correct per cell, and only
    # wrong in bulk. The warning is what makes the bulk case visible.
    assert _cell(matrix, NPU) == "ok"

    # With the reference in scope the same run is correctly condemned.
    assert _cell(deployability_matrix(df), NPU) == "collapsed"
