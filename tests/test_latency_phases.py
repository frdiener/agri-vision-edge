"""
Tests for the ``predict()`` phase breakdown.

A raw ``predict()`` latency cannot answer "what does this model cost": the call
begins by resizing the source frame to the model's input, which is priced by
the input resolution rather than by the network. ``ave resources`` therefore
records where each call went, and reports the average net of the resize beside
the end-to-end one.

Two properties are load-bearing and neither is obvious from reading the code:

* timing is **off** unless asked for, because ``ave benchmark``'s timed region
  has to remain the same call it has always been or ``benchmark_results`` stops
  being one comparable series;
* the phase dict is per-*instance* state despite living on the class, which is
  the classic mutable-class-attribute trap -- one runtime enabling timing must
  not switch it on for every other runtime in the process.
"""

from __future__ import annotations

import numpy as np
import pytest

from agri_vision_edge.cli.resources import PHASES, phase_breakdown
from agri_vision_edge.runtime.inference.base import BaseRuntime, Detection


class StubRuntime(BaseRuntime):
    """A runtime whose 'work' is a fixed, known number of phases."""

    @property
    def input_size(self) -> int:
        return 320

    def predict(self, image: np.ndarray) -> list[Detection]:
        resize_start = self._mark()
        self._phase("resize", resize_start)

        invoke_start = self._mark()
        self._phase("invoke", invoke_start)

        return [Detection(category_id=1, score=1.0, bbox=[0.0, 0.0, 1.0, 1.0])]


def test_timing_is_off_until_asked_for():
    runtime = StubRuntime()

    assert runtime.timing_enabled is False

    runtime.predict(None)

    assert runtime.phase_timings_ms == {}


def test_enabling_records_each_phase():
    runtime = StubRuntime()
    runtime.enable_phase_timing()
    runtime.predict(None)

    assert set(runtime.phase_timings_ms) == {"resize", "invoke"}
    assert all(value >= 0.0 for value in runtime.phase_timings_ms.values())


def test_one_runtime_enabling_timing_does_not_enable_it_globally():
    """The class-level default must not become shared mutable state."""
    enabled = StubRuntime()
    enabled.enable_phase_timing()
    enabled.predict(None)

    other = StubRuntime()
    other.predict(None)

    assert other.timing_enabled is False
    assert other.phase_timings_ms == {}
    assert enabled.phase_timings_ms != {}


def test_phase_dict_is_overwritten_per_call_not_appended():
    """
    Callers must copy what they need out each iteration. Stated as a test
    because the alternative -- accumulating -- would look identical for one
    call and silently grow unbounded over a 120 s loop.
    """
    runtime = StubRuntime()
    runtime.enable_phase_timing()

    runtime.predict(None)
    first = dict(runtime.phase_timings_ms)

    runtime.predict(None)

    assert set(runtime.phase_timings_ms) == set(first)
    assert len(runtime.phase_timings_ms) == 2


# ---------------------------------------------------------------------------
# summarising
# ---------------------------------------------------------------------------


def _samples(resize, preprocess, invoke, postprocess):
    return {
        "resize": list(resize),
        "preprocess": list(preprocess),
        "invoke": list(invoke),
        "postprocess": list(postprocess),
    }


def test_net_of_resize_is_computed_per_iteration():
    """
    Not as a difference of medians. Here the slow resize lands on the *fast*
    inference, so the two disagree: median(total) - median(resize) = 18.0,
    while the honest per-iteration median is 15.0.
    """
    latencies = [20.0, 20.0, 30.0]
    samples = _samples([10.0, 5.0, 2.0], [11.0, 6.0, 3.0], [8.0, 13.0, 26.0], [1.0] * 3)

    breakdown = phase_breakdown(latencies, samples)

    assert breakdown["net_of_resize"]["median_latency_ms"] == 15.0
    assert breakdown["net_of_resize"]["mean_latency_ms"] == pytest.approx(
        (10.0 + 15.0 + 28.0) / 3
    )


def test_resize_share_is_reported():
    latencies = [10.0] * 4
    samples = _samples([2.0] * 4, [3.0] * 4, [6.0] * 4, [1.0] * 4)

    breakdown = phase_breakdown(latencies, samples)

    assert breakdown["resize_share"] == pytest.approx(0.2)


def test_only_the_net_figure_carries_a_frame_rate():
    """
    A phase has a duration, not a throughput. "4144 fps" for the resize step is
    arithmetically true and describes nothing; the net figure is a rate the
    machine could actually sustain if frames arrived pre-scaled.
    """
    latencies = [10.0] * 4
    samples = _samples([2.0] * 4, [3.0] * 4, [6.0] * 4, [1.0] * 4)

    breakdown = phase_breakdown(latencies, samples)

    for phase in PHASES:
        assert "throughput_fps" not in breakdown[phase]

    assert breakdown["net_of_resize"]["throughput_fps"] == pytest.approx(125.0)


def test_a_runtime_that_reports_no_phases_yields_no_breakdown():
    """
    The SavedModel runtime resizes inside its graph and has nothing to report.
    An empty breakdown is the correct answer; zeros would read as "free".
    """
    assert phase_breakdown([10.0, 10.0], _samples([], [], [], [])) == {}
    assert phase_breakdown([], _samples([1.0], [1.0], [1.0], [1.0])) == {}
