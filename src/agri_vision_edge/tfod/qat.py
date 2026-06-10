"""
Quantization-aware training utilities for folded TFOD MobileNetV2 backbones.

Three quantization schemes are supported:

    * "legacy"
        TFMOT's default MobileNetV2 quantization strategy.

    * "weights"
        Explicit per-tensor quantization of convolution weights.

    * "full"
        Explicit per-tensor quantization of convolution weights
        and activations.

The custom schemes preserve quantization metadata for all
Conv2D and DepthwiseConv2D layers, enabling deployment on
accelerators that do not support per-channel quantization.
"""

import tensorflow as tf
import tensorflow_model_optimization as tfmot
from object_detection.core.freezable_batch_norm import FreezableBatchNorm
from tensorflow.keras.utils import register_keras_serializable
from tensorflow_model_optimization.quantization.keras import default_8bit


@register_keras_serializable()
class BaseQuantConfig(
    tfmot.quantization.keras.QuantizeConfig,
):
    """
    Shared quantization configuration for convolutional layers.
    """

    def __init__(
        self,
        *,
        per_axis: bool,
        symmetric: bool,
    ):
        self.per_axis = per_axis
        self.symmetric = symmetric

    def _weight_quantizer(self):
        return (
            tfmot.quantization.keras.quantizers
            .LastValueQuantizer(
                num_bits=8,
                per_axis=self.per_axis,
                symmetric=self.symmetric,
                narrow_range=True,
            )
        )

    def _kernel(self, layer):
        if isinstance(
            layer,
            tf.keras.layers.DepthwiseConv2D,
        ):
            return layer.depthwise_kernel

        if isinstance(
            layer,
            tf.keras.layers.Conv2D,
        ):
            return layer.kernel

        raise TypeError(
            f"Unsupported layer: {type(layer)}"
        )

    def get_weights_and_quantizers(
        self,
        layer,
    ):
        return [
            (
                self._kernel(layer),
                self._weight_quantizer(),
            )
        ]

    def set_quantize_weights(
        self,
        layer,
        quantize_weights,
    ):
        if isinstance(
            layer,
            tf.keras.layers.DepthwiseConv2D,
        ):
            layer.depthwise_kernel = (
                quantize_weights[0]
            )
        else:
            layer.kernel = (
                quantize_weights[0]
            )

    def get_config(self):
        return {
            "per_axis": self.per_axis,
            "symmetric": self.symmetric,
        }

    @classmethod
    def from_config(cls, config):
        return cls(**config)


@register_keras_serializable()
class WeightOnlyQuantConfig(
    BaseQuantConfig,
):
    """
    Quantize convolution weights only.

    Activations remain unquantized except for any
    additional TFMOT wrappers inserted by graph
    transforms.
    """

    def get_activations_and_quantizers(
        self,
        layer,
    ):
        return []

    def set_quantize_activations(
        self,
        layer,
        quantize_activations,
    ):
        pass

    def get_output_quantizers(
        self,
        layer,
    ):
        return []


@register_keras_serializable()
class FullQuantConfig(
    BaseQuantConfig,
):
    """
    Quantize convolution weights and activations.
    """

    def _activation_quantizer(self):
        return (
            tfmot.quantization.keras.quantizers
            .MovingAverageQuantizer(
                num_bits=8,
                per_axis=False,
                symmetric=False,
                narrow_range=False,
            )
        )

    def get_activations_and_quantizers(
        self,
        layer,
    ):
        if (
            layer.activation
            is tf.keras.activations.linear
        ):
            return []

        return [
            (
                layer.activation,
                self._activation_quantizer(),
            )
        ]

    def set_quantize_activations(
        self,
        layer,
        quantize_activations,
    ):
        if quantize_activations:
            layer.activation = (
                quantize_activations[0]
            )

    def get_output_quantizers(
        self,
        layer,
    ):
        return [
            self._activation_quantizer(),
        ]


def annotate_conv_layers(
    layer,
    quantize_config,
):
    """
    Annotate convolution layers for QAT.
    """

    if isinstance(
        layer,
        (
            tf.keras.layers.Conv2D,
            tf.keras.layers.DepthwiseConv2D,
        ),
    ):
        return (
            tfmot.quantization.keras
            .quantize_annotate_layer(
                layer,
                quantize_config=quantize_config,
            )
        )

    return layer


def _quantize_backbone_legacy(
    backbone,
    *,
    per_axis: bool,
):
    annotated = (
        tfmot.quantization.keras
        .quantize_annotate_model(
            backbone
        )
    )

    with tfmot.quantization.keras.quantize_scope(
        {
            "FreezableBatchNorm":
                FreezableBatchNorm,
        }
    ):
        return (
            tfmot.quantization.keras
            .quantize_apply(
                annotated,
                scheme=default_8bit
                .Default8BitQuantizeScheme(
                    disable_per_axis=not per_axis,
                ),
            )
        )

    
def _quantize_backbone_tfmot(
    backbone,
    *,
    per_axis: bool,
):
    annotated = tfmot.quantization.keras.quantize_annotate_model(
        backbone
    )
    return tfmot.quantization.keras.quantize_apply(
        annotated,
        scheme=default_8bit.Default8BitQuantizeScheme(
            disable_per_axis=per_axis
        )
    )


def quantize_backbone(
    backbone,
    *,
    scheme: str = "weights",
    per_axis: bool = False,
    symmetric: bool = True,
):
    """
    Convert a backbone model to a QAT-enabled model.

    Supported schemes:

        weights
            Custom weight-only quantization.

        full
            Custom weight and activation quantization.

        default_8bit
            TFMOT Default8BitQuantizeScheme.

        annotate_all
            TFMOT default annotation and quantization.
    """

    tfmot_schemes = {
        "default_8bit": _quantize_backbone_legacy,
        "annotate_all": _quantize_backbone_tfmot,
    }

    if scheme in tfmot_schemes:
        return tfmot_schemes[scheme](
            backbone,
            per_axis=per_axis,
        )

    custom_configs = {
        "weights": WeightOnlyQuantConfig,
        "full": FullQuantConfig,
    }

    try:
        config_cls = custom_configs[scheme]
    except KeyError as exc:
        valid = [
            *tfmot_schemes,
            *custom_configs,
        ]
        raise ValueError(
            f"Unknown quantization scheme '{scheme}'. "
            f"Expected one of: {', '.join(valid)}."
        ) from exc

    quantize_config = config_cls(
        per_axis=per_axis,
        symmetric=symmetric,
    )

    with tfmot.quantization.keras.quantize_scope(
        {
            "FreezableBatchNorm": FreezableBatchNorm,
            "WeightOnlyQuantConfig": WeightOnlyQuantConfig,
            "FullQuantConfig": FullQuantConfig,
        }
    ):
        annotated = tf.keras.models.clone_model(
            backbone,
            clone_function=lambda layer: annotate_conv_layers(
                layer,
                quantize_config,
            ),
        )

        return tfmot.quantization.keras.quantize_apply(
            annotated
        )


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
