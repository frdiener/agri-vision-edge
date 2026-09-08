"""Canonical detection types and runtime interfaces."""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


@dataclass
class Detection:
    """Detection with a normalized ``[ymin, xmin, ymax, xmax]`` bounding box."""

    category_id: int

    score: float

    bbox: list[float]


class BaseRuntime(ABC):
    """Runtime interface returning canonical :class:`Detection` objects."""

    #: Phase timing is off unless a caller asks for it, and it is a class-level
    #: default so no runtime has to remember to initialise it. ``ave benchmark``
    #: never turns it on: its timed region has to stay the same call it has
    #: always been, or the whole ``benchmark_results`` tree stops being one
    #: series of comparable measurements.
    timing_enabled: bool = False

    #: Phase durations (ms) of the most recent ``predict()``. Empty until
    #: timing is enabled; overwritten per call, so a caller that wants a series
    #: copies it out each iteration.
    phase_timings_ms: dict[str, float] = {}  # noqa: RUF012

    def enable_phase_timing(self) -> None:
        """
        Start recording where each ``predict()`` call spends its time.

        ``predict()`` includes resizing the source frame to the model input.
        Phase timing separates this source-resolution cost from network cost in
        the same loop and thermal state as the power measurement.

        Enabling timing adds several ``time.perf_counter()`` calls per inference.
        Disabled timing performs one attribute lookup per phase.
        """
        self.timing_enabled = True
        self.phase_timings_ms = {}

    def _mark(self) -> float:
        """Phase start stamp, or a placeholder when timing is off."""
        return time.perf_counter() if self.timing_enabled else 0.0

    def _phase(self, name: str, start: float) -> None:
        """Close the phase opened by :meth:`_mark`."""
        if self.timing_enabled:
            self.phase_timings_ms[name] = (time.perf_counter() - start) * 1000.0

    @property
    @abstractmethod
    def input_size(self) -> int:
        """Square input resolution in pixels."""
        pass

    @abstractmethod
    def predict(
        self,
        image: np.ndarray,
    ) -> list[Detection]:
        """Run inference on an RGB uint8 image."""
        pass
