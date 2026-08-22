"""
Tests for how a conversion target locates its source checkpoint.
"""

from __future__ import annotations

from agri_vision_edge.conversion.tflite import (
    FAST_NMS_TARGETS,
    REGULAR_NMS_TARGETS,
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


def test_fast_nms_is_the_default_so_shipped_filenames_are_unchanged():
    # The deployed artifacts, the benchmark_results tree and the report's run
    # names all key off `_fastnms`; adding the NMS dimension must not rename it.
    assert not ConversionTarget("fp32", "ptq", per_channel=False).regular_nms
    for target in FAST_NMS_TARGETS:
        assert target.suffix.endswith("_fastnms")
        assert target.nms == "fast"


def test_regular_nms_targets_only_differ_in_the_nms_token():
    target = ConversionTarget("fp32", "ptq", per_channel=False, regular_nms=True)
    assert target.suffix == "fp32_ptq_regnms"

    for fast, regular in zip(FAST_NMS_TARGETS, REGULAR_NMS_TARGETS, strict=True):
        # Same checkpoint and same graph: the pair is only usable as a control
        # if nothing but the post-processing op differs.
        assert regular.stage_candidates == fast.stage_candidates
        assert (regular.precision, regular.quantization, regular.per_channel) == (
            fast.precision,
            fast.quantization,
            fast.per_channel,
        )
        assert regular.label == fast.label
        assert regular.nms == "regular"
        assert regular.suffix == f"{fast.label}_regnms"


def test_standard_targets_pair_every_deployable_with_its_control():
    assert STANDARD_TARGETS == FAST_NMS_TARGETS + REGULAR_NMS_TARGETS

    # Filenames must stay unique, or one flavour silently overwrites the other.
    suffixes = [target.suffix for target in STANDARD_TARGETS]
    assert len(set(suffixes)) == len(suffixes)
