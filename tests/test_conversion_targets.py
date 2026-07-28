"""
Tests for how a conversion target locates its source checkpoint.
"""

from __future__ import annotations

from agri_vision_edge.conversion.tflite import (
    STANDARD_TARGETS,
    ConversionTarget,
)


def test_ptq_targets_share_one_checkpoint():
    # Granularity is only a converter flag for PTQ.
    for per_channel in (False, True):
        target = ConversionTarget("int8", "ptq", per_channel=per_channel)
        assert target.stage_candidates == ("ptq",)


def test_qat_targets_prefer_the_shared_run():
    # Granularity does not change the QAT training graph, so both targets are
    # exported from the same run.
    for per_channel in (False, True):
        target = ConversionTarget("int8", "qat", per_channel=per_channel)
        assert target.stage_candidates[0] == "qat_per-tensor"


def test_qat_per_channel_falls_back_to_its_own_stage():
    target = ConversionTarget("int8", "qat", per_channel=True)
    assert target.stage_candidates[-1] == "qat_per-channel"


def test_every_standard_target_names_a_stage():
    for target in STANDARD_TARGETS:
        assert target.stage_candidates
        assert all(isinstance(name, str) for name in target.stage_candidates)


def test_int8_targets_carry_their_granularity_in_the_filename():
    assert (
        ConversionTarget("int8", "qat", per_channel=True).suffix
        == "int8_qat_per-channel_fastnms"
    )
    assert ConversionTarget("fp32", "ptq", per_channel=False).suffix == (
        "fp32_ptq_fastnms"
    )
