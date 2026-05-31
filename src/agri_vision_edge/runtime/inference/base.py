"""
Common runtime abstractions.

Defines canonical detection structures and
runtime interfaces shared across TFLite,
ExecuTorch, and future runtimes.
"""

from __future__ import annotations

from abc import ABC
from abc import abstractmethod

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
    """

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
