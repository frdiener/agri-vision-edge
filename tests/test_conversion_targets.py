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
        assert target.stage_subdir == "ptq"


def test_qat_targets_use_their_own_granularity_stage():
    assert (
        ConversionTarget("int8", "qat", per_channel=False).stage_subdir
        == "qat_per-tensor"
    )
    assert (
        ConversionTarget("int8", "qat", per_channel=True).stage_subdir
        == "qat_per-channel"
    )


def test_every_standard_target_names_a_stage():
    for target in STANDARD_TARGETS:
        assert isinstance(target.stage_subdir, str)
        assert target.stage_subdir


def test_int8_targets_carry_their_granularity_in_the_filename():
    assert (
        ConversionTarget("int8", "qat", per_channel=True).suffix
        == "int8_qat_per-channel_fastnms"
    )
    assert ConversionTarget("fp32", "ptq", per_channel=False).suffix == (
        "fp32_ptq_fastnms"
    )
