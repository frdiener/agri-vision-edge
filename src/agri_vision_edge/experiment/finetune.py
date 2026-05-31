from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class AugmentationConfig:
    """
    Data augmentation settings.

    Defaults are tuned for top-down agricultural imagery
    such as drone, gantry, and overhead weed-detection
    cameras.
    """

    #
    # Geometry
    #

    # Enable random cropping.
    random_crop: bool = True

    # Random crop parameters.
    crop_min_object_covered: float = 0.5
    crop_min_area: float = 0.5
    crop_max_area: float = 1.0
    crop_overlap_thresh: float = 0.3

    # Horizontal flip.
    horizontal_flip: bool = True
    horizontal_flip_probability: float = 0.5

    # Vertical flip.
    vertical_flip: bool = True
    vertical_flip_probability: float = 0.5

    # Random 90° rotations.
    rotation90: bool = True
    rotation90_probability: float = 0.5

    #
    # Scale / zoom invariance
    #

    # Simulated zoom range.
    #
    # Recommended:
    #
    #   (0.8, 1.2)
    #
    # None disables zoom augmentation.
    zoom_range: tuple[float, float] | None = (
        0.8,
        1.2,
    )

    #
    # Photometric augmentation
    #

    # Brightness adjustment.
    #
    # None disables brightness augmentation.
    brightness_max_delta: float | None = 0.15

    # Contrast scaling range.
    #
    # Recommended:
    #
    #   (0.8, 1.2)
    #
    # None disables contrast augmentation.
    contrast_range: tuple[float, float] | None = (
        0.8,
        1.2,
    )

    # Saturation scaling range.
    #
    # Recommended:
    #
    #   (0.8, 1.2)
    #
    # None disables saturation augmentation.
    saturation_range: tuple[float, float] | None = (
        0.8,
        1.2,
    )

    #
    # Compression robustness
    #

    # Simulate JPEG compression artifacts.
    #
    # Useful when deployment images may come from:
    #
    # - IP cameras
    # - embedded devices
    # - compressed storage
    #
    # Recommended:
    #
    #   (50, 100)
    #
    # None disables JPEG augmentation.
    jpeg_quality_range: tuple[int, int] | None = (
        50,
        100,
    )


@dataclass
class FineTuneConfig:
    """
    High-level TensorFlow Object Detection fine-tuning
    configuration.

    Defaults are tuned for small-object agricultural
    detection workloads.
    """

    #
    # Training
    #

    # Upstreams:
    # - SSD MobileNet V2 300x300: 512
    # - SSD MobileNet V2 FPN 320x320: 128
    #
    # Reduced for practical single/few-GPU fine-tuning.
    batch_size: int = 16

    # Upstreams:
    # - SSD MobileNet V2 300x300: 0.8
    # - SSD MobileNet V2 FPN 320x320: 0.08
    #
    # Scaled down for small-batch transfer learning.
    learning_rate_base: float = 0.004

    # Upstreams:
    # - SSD MobileNet V2 300x300: 0.13333
    # - SSD MobileNet V2 FPN 320x320: 0.026666
    #
    # Scaled proportionally to reduced LR.
    warmup_learning_rate: float = 0.001

    # Both upstreams use 50k.
    #
    # Reduced for fine-tuning workloads.
    num_steps: int = 20_000

    # Upstreams:
    # - SSD MobileNet V2 300x300: 2000
    # - SSD MobileNet V2 FPN 320x320: 1000
    warmup_steps: int = 1000

    #
    # Custom early stopping
    #

    early_stopping_patience: int = 50
    early_stopping_min_delta: float = 0.0

    #
    # Image sizing
    #

    # Upstream SSD MobileNet V2 config says 300x300,
    # but the released checkpoint is actually trained
    # and exported for 320x320 inputs.
    #
    # SSD MobileNet V2 FPN also uses 320x320.
    image_size: int = 320

    #
    # Anchor tuning
    #

    # Used only for classic SSD anchor generators.
    #
    # Upstream SSD MobileNet V2:
    #   min_scale = 0.2
    #   max_scale = 0.95
    #
    # Reduced for smaller agricultural objects.
    anchor_min_scale: Optional[float] = 0.03
    anchor_max_scale: Optional[float] = 0.35

    # Common upstream aspect ratios.
    anchor_aspect_ratios: tuple[float, ...] = (
        1.0,
        2.0,
        0.5,
    )

    #
    # Matcher thresholds
    #

    # Both upstreams use 0.5.
    #
    # Relaxed slightly for small-object matching.
    matched_threshold: float = 0.4
    unmatched_threshold: float = 0.4

    #
    # Data augmentation
    #

    augmentation: AugmentationConfig = field(
        default_factory=AugmentationConfig
    )

    #
    # NMS
    #

    # Upstreams use effectively zero threshold.
    #
    # Raised to reduce low-confidence detections.
    nms_score_threshold: float = 0.05

    # Both upstreams use 0.6.
    #
    # Slightly stricter suppression.
    nms_iou_threshold: float = 0.5

    # Upstreams use 100.
    #
    # Reduced because agricultural scenes usually
    # contain fewer valid objects per image.
    max_detections_per_class: int = 60
    max_total_detections: int = 60

    #
    # Multiscale/FPN anchor tuning
    #

    # Upstream SSD MobileNet V2 FPN:
    #   anchor_scale = 4.0
    #
    # Reduced for tiny agricultural objects.
    fpn_anchor_scale: float = 1.5

    # Upstream:
    #   scales_per_octave = 2
    #
    # Reduced anchor density.
    fpn_scales_per_octave: int = 1
