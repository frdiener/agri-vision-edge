"""
Configuration objects for TFOD training.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path


class QATScheme(str, Enum):
    FULL = "full"
    WEIGHTS = "weights"
    FOLDED = "folded"

@dataclass(slots=True)
class TrainerConfig:
    """
    High-level training configuration.

    Independent from TFOD protobuf configuration.
    """

    pipeline_config: Path
    train_dir: Path

    log_every: int = 100

    checkpoint_max_to_keep: int = 3

    metric_name: str = (
        "DetectionBoxes_Precision/mAP"
    )

    early_stopping_patience: int = 10
    early_stopping_min_delta: float = 0.0

    save_metrics_history: bool = True

    reset_optimizer: bool = False
    qat_scheme: QATScheme = QATScheme.FOLDED

    def __post_init__(self):
        if isinstance(self.qat_scheme, str):
            self.qat_scheme = QATScheme(
                self.qat_scheme.lower()
            )

        if (
            self.qat_scheme is not None
            and not isinstance(
                self.qat_scheme,
                QATScheme,
            )
        ):
            raise TypeError(
                "qat_scheme must be "
                "QATScheme, str, or None"
            )

    @property
    def qat_enabled(self) -> bool:
        return self.qat_scheme is not None

    @property
    def history_path(self) -> Path:
        return self.train_dir / "metrics_history.json"

    @property
    def best_metric_path(self) -> Path:
        return self.train_dir / "best_metric.json"
