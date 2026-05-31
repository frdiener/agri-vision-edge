"""
TensorFlow Object Detection pipeline configuration utilities.

Provides helpers for modifying TF-OD pipeline.config files
programmatically for custom datasets and training setups.
"""

from __future__ import annotations

from pathlib import Path
from typing import Union

import tensorflow as tf
from google.protobuf import text_format

from object_detection.protos import pipeline_pb2

from ..experiment.finetune import (
    AugmentationConfig,
    FineTuneConfig,
)


PathLike = Union[str, Path]


def _clear_augmentations(
    pipeline_config: pipeline_pb2.TrainEvalPipelineConfig,
) -> None:

    del (
        pipeline_config
        .train_config
        .data_augmentation_options[:]
    )


def _apply_augmentations(
    pipeline_config: pipeline_pb2.TrainEvalPipelineConfig,
    config: FineTuneConfig,
) -> None:

    augmentations = (
        pipeline_config
        .train_config
        .data_augmentation_options
    )

    aug = config.augmentation

    #
    # Random crop
    #

    if aug.random_crop:

        step = augmentations.add()

        crop = (
            step.random_crop_image
        )

        crop.min_object_covered = (
            aug.crop_min_object_covered
        )

        crop.min_area = (
            aug.crop_min_area
        )

        crop.max_area = (
            aug.crop_max_area
        )

        crop.overlap_thresh = (
            aug.crop_overlap_thresh
        )

    #
    # Horizontal flip
    #

    if aug.horizontal_flip:

        step = augmentations.add()

        step.random_horizontal_flip.probability = (
            aug.horizontal_flip_probability
        )

    #
    # Vertical flip
    #

    if aug.vertical_flip:

        step = augmentations.add()

        step.random_vertical_flip.probability = (
            aug.vertical_flip_probability
        )

    #
    # 90° rotation
    #

    if aug.rotation90:

        step = augmentations.add()

        step.random_rotation90.probability = (
            aug.rotation90_probability
        )

    #
    # Zoom simulation
    #

    if aug.zoom_range is not None:

        zoom_min, zoom_max = (
            aug.zoom_range
        )

        step = augmentations.add()

        zoom = (
            step.random_scale_crop_and_pad_to_square
        )

        zoom.output_size = (
            config.image_size
        )

        zoom.scale_min = zoom_min
        zoom.scale_max = zoom_max

    #
    # Brightness
    #

    if aug.brightness_max_delta is not None:

        step = augmentations.add()

        step.random_adjust_brightness.max_delta = (
            aug.brightness_max_delta
        )

    #
    # Contrast
    #

    if aug.contrast_range is not None:

        contrast_min, contrast_max = (
            aug.contrast_range
        )

        step = augmentations.add()

        contrast = (
            step.random_adjust_contrast
        )

        contrast.min_delta = contrast_min
        contrast.max_delta = contrast_max

    #
    # Saturation
    #

    if aug.saturation_range is not None:

        saturation_min, saturation_max = (
            aug.saturation_range
        )

        step = augmentations.add()

        saturation = (
            step.random_adjust_saturation
        )

        saturation.min_delta = saturation_min
        saturation.max_delta = saturation_max

    #
    # JPEG robustness
    #

    if aug.jpeg_quality_range is not None:

        quality_min, quality_max = (
            aug.jpeg_quality_range
        )

        step = augmentations.add()

        jpeg = (
            step.random_jpeg_quality
        )

        jpeg.random_coef = 0.5

        jpeg.min_jpeg_quality = quality_min
        jpeg.max_jpeg_quality = quality_max


def load_pipeline_config(
    config_path: PathLike,
) -> pipeline_pb2.TrainEvalPipelineConfig:

    config_path = Path(config_path)

    pipeline_config = (
        pipeline_pb2.TrainEvalPipelineConfig()
    )

    with tf.io.gfile.GFile(
        str(config_path),
        "r",
    ) as f:
        text = f.read()

    text_format.Merge(
        text,
        pipeline_config,
    )

    return pipeline_config


def save_pipeline_config(
    pipeline_config,
    output_path: PathLike,
) -> None:

    output_path = Path(output_path)

    config_text = text_format.MessageToString(
        pipeline_config
    )

    output_path.write_text(config_text)


def configure_ssd_pipeline(
    *,
    config: FineTuneConfig,

    config_path: PathLike,
    output_path: PathLike,

    train_record: PathLike,
    val_record: PathLike,

    label_map: PathLike,

    checkpoint_path: PathLike,

    num_classes: int = 2,

    qat_delay: int | None = None,
) -> None:
    """
    Configure SSD TF-OD pipeline from a semantic
    FineTuneConfig object.
    """

    pipeline_config = load_pipeline_config(
        config_path
    )

    #
    # Data augmentation
    #

    _clear_augmentations(
        pipeline_config
    )

    _apply_augmentations(
        pipeline_config,
        config,
    )

    #
    # Model
    #

    pipeline_config.model.ssd.num_classes = (
        num_classes
    )

    #
    # Train config
    #

    pipeline_config.train_config.batch_size = (
        config.batch_size
    )

    pipeline_config.train_config.fine_tune_checkpoint = (
        str(checkpoint_path)
    )

    pipeline_config.train_config.fine_tune_checkpoint_type = (
        "detection"
    )

    pipeline_config.train_config.num_steps = (
        config.num_steps
    )

    pipeline_config.train_config.sync_replicas = (
        False
    )

    pipeline_config.train_config.replicas_to_aggregate = (
        1
    )

    #
    # Learning rate
    #

    lr_config = (
        pipeline_config
        .train_config
        .optimizer
        .momentum_optimizer
        .learning_rate
        .cosine_decay_learning_rate
    )

    lr_config.learning_rate_base = (
        config.learning_rate_base
    )

    lr_config.warmup_learning_rate = (
        config.warmup_learning_rate
    )

    lr_config.total_steps = (
        config.num_steps
    )

    lr_config.warmup_steps = (
        config.warmup_steps
    )

    #
    # Train input
    #

    pipeline_config.train_input_reader.label_map_path = (
        str(label_map)
    )

    (
        pipeline_config
        .train_input_reader
        .tf_record_input_reader
        .input_path[:]
    ) = [str(train_record)]

    #
    # Eval input
    #

    pipeline_config.eval_input_reader[
        0
    ].label_map_path = str(label_map)

    (
        pipeline_config
        .eval_input_reader[0]
        .tf_record_input_reader
        .input_path[:]
    ) = [str(val_record)]

    #
    # QAT
    #

    if qat_delay is not None:

        pipeline_config.graph_rewriter.quantization.delay = (
            qat_delay
        )

    #
    # Image sizing
    #

    (
        pipeline_config
        .model
        .ssd
        .image_resizer
        .fixed_shape_resizer
        .height
    ) = config.image_size

    (
        pipeline_config
        .model
        .ssd
        .image_resizer
        .fixed_shape_resizer
        .width
    ) = config.image_size

    #
    # Anchor tuning
    #

    anchor_generator = (
        pipeline_config
        .model
        .ssd
        .anchor_generator
    )

    anchor_generator_type = (
        anchor_generator.WhichOneof(
            "anchor_generator_oneof"
        )
    )

    #
    # Classic SSD anchors
    #

    if anchor_generator_type == (
        "ssd_anchor_generator"
    ):

        ssd_anchor_gen = (
            anchor_generator
            .ssd_anchor_generator
        )

        if config.anchor_min_scale is not None:
            ssd_anchor_gen.min_scale = (
                config.anchor_min_scale
            )

        if config.anchor_max_scale is not None:
            ssd_anchor_gen.max_scale = (
                config.anchor_max_scale
            )

        del ssd_anchor_gen.aspect_ratios[:]

        ssd_anchor_gen.aspect_ratios.extend(
            config.anchor_aspect_ratios
        )

    #
    # FPN / RetinaNet-style anchors
    #

    elif anchor_generator_type == (
        "multiscale_anchor_generator"
    ):

        multiscale_anchor_gen = (
            anchor_generator
            .multiscale_anchor_generator
        )

        #
        # Reduced for tiny agricultural objects
        #

        multiscale_anchor_gen.anchor_scale = (
            config.fpn_anchor_scale
        )

        multiscale_anchor_gen.scales_per_octave = (
            config.fpn_scales_per_octave
        )

        del multiscale_anchor_gen.aspect_ratios[:]

        multiscale_anchor_gen.aspect_ratios.extend(
            config.anchor_aspect_ratios
        )

    else:

        raise ValueError(
            f"Unsupported anchor generator: "
            f"{anchor_generator_type}"
        )

    #
    # Matcher thresholds
    #

    matcher = (
        pipeline_config
        .model
        .ssd
        .matcher
        .argmax_matcher
    )

    matcher.matched_threshold = (
        config.matched_threshold
    )

    matcher.unmatched_threshold = (
        config.unmatched_threshold
    )

    #
    # NMS
    #

    nms = (
        pipeline_config
        .model
        .ssd
        .post_processing
        .batch_non_max_suppression
    )

    nms.score_threshold = (
        config.nms_score_threshold
    )

    nms.iou_threshold = (
        config.nms_iou_threshold
    )

    nms.max_detections_per_class = (
        config.max_detections_per_class
    )

    nms.max_total_detections = (
        config.max_total_detections
    )

    #
    # Save
    #

    save_pipeline_config(
        pipeline_config,
        output_path,
    )
