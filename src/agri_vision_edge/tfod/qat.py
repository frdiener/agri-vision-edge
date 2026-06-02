import tensorflow as tf
import tensorflow_model_optimization as tfmot

from object_detection.core.freezable_batch_norm import (
    FreezableBatchNorm,
)


def _annotate_layer(layer):
    if isinstance(
        layer,
        (
            tf.keras.layers.Conv2D,
            tf.keras.layers.DepthwiseConv2D,
        ),
    ):
        return tfmot.quantization.keras.quantize_annotate_layer(
            layer
        )

    return layer


def quantize_backbone(backbone):
    """
    Convert a TFOD MobileNetV2 backbone into a
    TF-MOT QAT backbone while preserving weights.
    """

    with tfmot.quantization.keras.quantize_scope(
        {
            "FreezableBatchNorm": FreezableBatchNorm,
        }
    ):
        annotated = tf.keras.models.clone_model(
            backbone,
            clone_function=_annotate_layer,
        )

        qat_backbone = (
            tfmot.quantization.keras.quantize_apply(
                annotated
            )
        )

    return qat_backbone


def ensure_model_is_built_for_qat(
    detection_model,
    pipeline_config
):
    ssd_config = pipeline_config.model.ssd

    h = (
        ssd_config.image_resizer
        .fixed_shape_resizer
        .height
    )

    w = (
        ssd_config.image_resizer
        .fixed_shape_resizer
        .width
    )

    dummy = tf.zeros(
        [1, h, w, 3],
        dtype=tf.float32
    )

    image, shapes = detection_model.preprocess(dummy)

    detection_model.predict(
        image,
        shapes
    )
