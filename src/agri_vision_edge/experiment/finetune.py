from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class FineTuneConfig:

    #
    # Training
    #

    # upstreams:
    # - SSD MobileNet V2 300x300: 512
    # - SSD MobileNet V2 FPN 320x320: 128
    #
    # reduced for practical single/few-GPU fine-tuning
    batch_size: int = 16

    # upstreams:
    # - SSD MobileNet V2 300x300: 0.8
    # - SSD MobileNet V2 FPN 320x320: 0.08
    #
    # scaled down for small-batch transfer learning
    learning_rate_base: float = 0.004

    # upstreams:
    # - SSD MobileNet V2 300x300: 0.13333
    # - SSD MobileNet V2 FPN 320x320: 0.026666
    #
    # scaled proportionally to reduced LR
    warmup_learning_rate: float = 0.001

    # both upstreams use 50k
    #
    # reduced for fine-tuning workloads
    num_steps: int = 20_000

    # upstreams:
    # - SSD MobileNet V2 300x300: 2000
    # - SSD MobileNet V2 FPN 320x320: 1000
    warmup_steps: int = 1000

    #
    # Custom early stopping
    #
    # not present in upstream TF-OD configs
    #

    early_stopping_patience: int = 50
    early_stopping_min_delta: float = 0.0

    #
    # Image sizing
    #

    # upstream SSD MobileNet V2 config says 300x300,
    # but the released checkpoint is actually trained/exported
    # for 320x320 inputs
    #
    # FPN upstream also uses 320x320
    image_size: int = 320

    #
    # Anchor tuning
    #

    #
    # Used only for classic SSD anchor generators.
    #
    # upstream SSD MobileNet V2 300x300:
    #   min_scale = 0.2
    #   max_scale = 0.95
    #
    # reduced for smaller agricultural objects
    #
    anchor_min_scale: Optional[float] = 0.03
    anchor_max_scale: Optional[float] = 0.35

    #
    # common upstream ratios merged from both models
    #
    anchor_aspect_ratios: tuple[float, ...] = (
        1.0,
        2.0,
        0.5,
    )

    #
    # Matcher thresholds
    #

    # both upstreams use 0.5
    #
    # relaxed slightly for small-object matching
    matched_threshold: float = 0.4
    unmatched_threshold: float = 0.4

    #
    # Augmentation
    #

    # both upstreams enable random crop
    #
    # disabled by default because aggressive cropping
    # can remove tiny target objects entirely
    use_random_crop: bool = False

    #
    # NMS
    #

    # upstreams use effectively zero threshold (1e-8)
    #
    # raised to reduce low-confidence detections
    nms_score_threshold: float = 0.05

    # both upstreams use 0.6
    #
    # slightly stricter suppression
    nms_iou_threshold: float = 0.5

    # upstreams use 100
    #
    # reduced because agricultural scenes usually
    # contain fewer valid objects per image
    max_detections_per_class: int = 60
    max_total_detections: int = 60    

    #
    # Multiscale/FPN anchor tuning
    #

    # upstream SSD MobileNet V2 FPN:
    #   anchor_scale = 4.0
    #
    # reduced for tiny agricultural objects
    fpn_anchor_scale: float = 1.5

    # upstream:
    #   scales_per_octave = 2
    #
    # reduced anchor density
    fpn_scales_per_octave: int = 1
