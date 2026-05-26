from __future__ import annotations

from abc import ABC
from abc import abstractmethod

from dataclasses import dataclass

import numpy as np


@dataclass
class Detection:

    class_id: int

    score: float

    #
    # COCO-style normalized bbox:
    #
    # [ymin, xmin, ymax, xmax]
    #

    bbox: list[float]


class BaseRuntime(ABC):

    @abstractmethod
    def predict(
        self,
        image: np.ndarray,
    ) -> list[Detection]:
        """
        Run inference on an RGB image.

        Args:
            image:
                RGB uint8 image.

        Returns:
            List of detections.
        """
        pass

    @property
    @abstractmethod
    def input_size(self) -> int:
        pass
