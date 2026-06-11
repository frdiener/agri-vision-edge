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
class FixedRelu6Quantizer(
    tfmot.quantization.keras.quantizers.Quantizer
):
    def build(self, tensor_shape, name, layer):
        return {}

    def __call__(self, inputs, training, weights, **kwargs):
        return tf.quantization.fake_quant_with_min_max_vars(
            inputs,
            min=0.0,
            max=6.0,
            num_bits=8,
            narrow_range=False,
        )

    def get_config(self):
        return {}

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
class FullQuantFixedReLUConfig(
    BaseQuantConfig,
):
    """
    Quantize convolution weights and activations, pinning activation /
    output ranges to a fixed [0, 6].

    Why this exists: the deployment delegate (teflon/mesa) accepts a ReLU6
    op only when the output tensor's effective dequantized range,
    (quantized_max - zero_point) * scale, stays <= 6.0 + RELU6_EPSILON. The
    learned-range schemes (MovingAverage in ``FullQuantConfig``, AllValues in
    ``FullQuantAllValuesConfig``) are asymmetric (symmetric=False), so on a
    post-ReLU6 tensor they learn a min that dips slightly *below* 0. The
    zero-point is then nudged to keep 0 exactly representable, which pushes
    the dequantized max (quantized_max - zero_point) * scale just *above* 6
    and the op is rejected. A hard [0, 6] range pins min=0 exactly, so the
    dequantized range is exactly 6 and the op is accepted by any delegate.
    (The current target runs a patched mesa with a larger RELU6_EPSILON, so
    the learned-range schemes also deploy there - but "fixed" remains the
    portable, stock-delegate-safe option.)

    KNOWN PROBLEM: the same fixed [0, 6] range is also applied as the OUTPUT
    quantizer of *linear* (signed) convolutions - e.g. the inverted-residual
    projection / bottleneck convs, which have no ReLU and produce negative
    values. Those negatives are clamped to 0, discarding half the signal.
    Deployment-safe but lossy; expect reduced accuracy versus a signed
    quantizer.
    """

    def _activation_quantizer(self):
        return FixedRelu6Quantizer()

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


@register_keras_serializable()
class FullQuantConfig(
    BaseQuantConfig,
):
    """
    Quantize convolution weights and activations with a MovingAverage
    (EMA) range for activations / outputs.

    KNOWN PROBLEM: MovingAverageQuantizer initialises its range to [-6, 6]
    and expands it through a slow EMA. Over a typical QAT fine-tune
    (hundreds to low-thousands of steps) the range barely moves off [-6, 6],
    so every quantized activation / output - including the signed backbone
    feature maps feeding the SSD head - is effectively hard-clamped to ~±6.
    That squashes the logits and detection scores saturate near
    sigmoid(0) = 0.5. It only calibrates properly after many thousands of
    steps. Prefer ``FullQuantAllValuesConfig`` ("full_av"), which calibrates
    in ~1 step. See that class for the deployment caveat.
    """

    def _activation_quantizer(self):
        return (
            tfmot.quantization.keras.quantizers
            .MovingAverageQuantizer(
                num_bits=8,
                per_axis=False,
                symmetric=False,
                narrow_range=False,
            ))

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


@register_keras_serializable()
class FullQuantAllValuesConfig(
    FullQuantConfig,
):
    """
    Full quantization (weights + activations) using an AllValues range for
    activations / outputs instead of a MovingAverage EMA.

    Motivation: ``FullQuantConfig`` clamps activations to ~±6 during normal
    fine-tunes (see its docstring), collapsing detection scores to ~0.5.
    AllValuesQuantizer tracks the true observed min/max and calibrates within
    ~1 step, so the signed dynamic range of the backbone features is
    preserved and the scores are no longer capped.

    KNOWN CAVEAT: this quantizer is asymmetric (symmetric=False), like
    MovingAverage and unlike ``FullQuantFixedReLUConfig``. On a post-ReLU6
    tensor it learns a min slightly below 0; the zero-point nudge then pushes
    the dequantized max (quantized_max - zero_point) * scale just above 6. A
    stock delegate rejects the ReLU6 op when that exceeds 6 + RELU6_EPSILON
    (the exact reason the "fixed" scheme was added). The current target runs
    a PATCHED mesa with a larger RELU6_EPSILON, so this scheme is expected to
    deploy there; on a stock/unpatched delegate, fall back to "fixed". Note
    the overshoot is only legitimately roundable to [0, 6] when it is within
    one int8 step (RELU6_EPSILON < scale ~= 6/255).

    Inherits the activation / output wiring from ``FullQuantConfig`` and only
    swaps the quantizer.
    """

    def _activation_quantizer(self):
        return (
            tfmot.quantization.keras.quantizers
            .AllValuesQuantizer(
                num_bits=8,
                per_axis=False,
                symmetric=False,
                narrow_range=False,
            )
        )


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
            Custom weight and activation quantization (MovingAverage
            activation range). NOTE: clamps activations to ~±6 during
            normal-length fine-tunes -> scores collapse to ~0.5. See
            FullQuantConfig.

        fixed
            Like "full" but pins activation ranges to a fixed [0, 6] so
            conv+ReLU6 ops stay fusible on the deployment delegate. Clamps
            signed conv outputs. See FullQuantFixedReLUConfig.

        full_av
            Like "full" but uses AllValues activation ranges, which
            calibrate immediately and preserve signed dynamic range (fixes
            the ~0.5 score collapse). Being asymmetric, its post-ReLU6 min
            undershoots 0 and the zero-point nudge pushes the dequantized max
            just above 6, so a stock delegate rejects the ReLU6 op (ok on the
            patched mesa). See FullQuantAllValuesConfig.

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
        "fixed": FullQuantFixedReLUConfig,
        "full_av": FullQuantAllValuesConfig,
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
            "FullQuantFixedReLUConfig": FullQuantFixedReLUConfig,
            "FullQuantAllValuesConfig": FullQuantAllValuesConfig,
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
