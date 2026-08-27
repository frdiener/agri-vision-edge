"""
Common runtime abstractions.

Defines canonical detection structures and
runtime interfaces shared across TFLite,
ExecuTorch, and future runtimes.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


@dataclass
class Detection:
    """
    Canonical object detection result.

    Bounding boxes use normalized coordinates:

        [ymin, xmin, ymax, xmax]

    compatible with TensorFlow SSD outputs.
    """

    category_id: int

    score: float

    bbox: list[float]


class BaseRuntime(ABC):
    """
    Abstract runtime interface.

    All runtimes should expose a common
    prediction API returning canonical
    Detection objects.

    Optionally, a runtime can also say *where* a ``predict()`` call spent its
    time -- see :meth:`enable_phase_timing`.
    """

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

        The motivating question is what a *model* costs, and a raw
        ``predict()`` figure does not answer it: the call begins by resizing
        the source frame to the model's input, which is real deployment work
        whose size is set by the input resolution rather than by the network.
        Recording the phases in the same loop that measures power means the
        breakdown comes from the run being characterised, not from a separate
        measurement taken at a different die temperature.

        Cost when enabled: a handful of ``time.perf_counter()`` calls per
        inference, sub-microsecond against a millisecond-scale one. Cost when
        disabled: one attribute lookup per phase and no clock read at all,
        which is why this is a flag rather than a subclass.
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
        """
        Square input resolution.
        """
        pass

    @abstractmethod
    def predict(
        self,
        image: np.ndarray,
    ) -> list[Detection]:
        """
        Run inference on an RGB uint8 image.

        Args:
            image:
                RGB image array.

        Returns:
            List of detections.
        """
        pass
