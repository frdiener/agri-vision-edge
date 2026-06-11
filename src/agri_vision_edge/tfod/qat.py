"""
Quantization-aware training utilities for folded TFOD MobileNetV2 backbones.

The deployment accelerator only supports PER-TENSOR int8 quantization, so the
core job here is to emit per-tensor weights reliably. TFMOT's built-in
Default8BitQuantizeScheme ("default_8bit" / "annotate_all" below) cannot be
trusted for this: annotating the whole model inserts fake-quant no-ops at some
points, and those make the TFLite converter ignore the per-tensor
(disable_per_channel) request and silently emit per-channel weights. To avoid
that, the custom schemes annotate ONLY Conv2D / DepthwiseConv2D layers and pin
per-tensor quantization explicitly via LastValueQuantizer(per_axis=False).

Custom schemes - see ``quantize_backbone`` for details:

    * "full"     - RECOMMENDED. Per-layer full int8: conv->ReLU6 convs are
                   weight-only and the ReLU6 layer pins its output to a fixed
                   [0, 6] (the tensor that survives fusion); signed convs use
                   weights + AllValues so their range is not clamped.
    * "weights"  - per-tensor weights only.

This single "full" scheme replaces the earlier full / fixed / full_av / mixed
experiments: it takes AllValues for the signed half (their only real benefit)
and places the [0, 6] pin on the ReLU6 *layer* output (which the old schemes
pinned on the wrong, pre-fusion tensor). With correct [-1, 1] calibration,
plain PTQ already lands near fp32, so this exists to make per-tensor int8
deployment robust rather than to recover accuracy.

Legacy/comparison schemes ("default_8bit", "annotate_all") use the TFMOT
default scheme and suffer the per-channel leakage described above.
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
class SignedConvQuantConfig(
    BaseQuantConfig,
):
    """
    Per-tensor weights + an AllValues output quantizer.

    For convolutions whose output is *signed* (the inverted-residual
    projection / bottleneck convs, which have no following ReLU and produce
    negative values). AllValues tracks the true observed min/max and
    calibrates in ~1 step, so the signed dynamic range is preserved - no ±6
    clamp, so the features feeding the SSD head are not squashed.

    These convs are linear (activation is applied by a separate layer), so
    there is no activation function to wrap - only the layer output is
    quantized.
    """

    def _output_quantizer(self):
        return (
            tfmot.quantization.keras.quantizers
            .AllValuesQuantizer(
                num_bits=8,
                per_axis=False,
                symmetric=False,
                narrow_range=False,
            )
        )

    def get_activations_and_quantizers(self, layer):
        return []

    def set_quantize_activations(self, layer, quantize_activations):
        pass

    def get_output_quantizers(self, layer):
        return [self._output_quantizer()]


@register_keras_serializable()
class ReLU6OutputConfig(
    tfmot.quantization.keras.QuantizeConfig,
):
    """
    Pin a ReLU6 *layer's* output to a fixed [0, 6] (no weights).

    This is the tensor that matters: in the folded backbone the conv and
    ReLU6 are separate layers, and after TFLite fuses conv+ReLU6 the fused
    op's output IS the ReLU6 layer's output. Annotating the ReLU6 layer (not
    the preceding conv) is therefore what makes the deployed dequantized
    range exactly 6 (scale = 6/255, zero_point = -128 ->
    (127 - (-128)) * 6/255 = 6.0), which a stock delegate accepts.

    The preceding conv is quantized weight-only (see the "full" scheme), so
    there is no second fake-quant on the pre-ReLU6 tensor to fight with.
    """

    def get_weights_and_quantizers(self, layer):
        return []

    def set_quantize_weights(self, layer, quantize_weights):
        pass

    def get_activations_and_quantizers(self, layer):
        return []

    def set_quantize_activations(self, layer, quantize_activations):
        pass

    def get_output_quantizers(self, layer):
        return [FixedRelu6Quantizer()]

    def get_config(self):
        return {}


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
    """
    TFMOT Default8BitQuantizeScheme path (disable_per_axis controls
    per-tensor vs per-channel weight quantization).

    KNOWN PROBLEM: this scheme annotates the whole model, which inserts
    fake-quant no-ops at some points (e.g. around add / passthrough ops).
    Those extra nodes make the TFLite converter ignore the per-tensor
    (disable_per_channel) request, so weights silently come out per-channel -
    which the deployment accelerator does not support. The custom
    QuantizeConfig schemes ("full" / "weights") were introduced to avoid this:
    they annotate only Conv/DepthwiseConv (and, for "full", the ReLU6 layers)
    and pin per-tensor quantization explicitly via
    LastValueQuantizer(per_axis=False). Prefer those; keep this only for
    comparison.
    """

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
    """
    Plain TFMOT default annotation + apply. Same KNOWN PROBLEM as
    ``_quantize_backbone_legacy``: whole-model annotation inserts fake-quant
    no-ops that make the converter ignore the per-tensor request and emit
    per-channel weights (unsupported on the target). Kept for comparison;
    prefer the custom QuantizeConfig schemes.
    """

    annotated = tfmot.quantization.keras.quantize_annotate_model(
        backbone
    )
    return tfmot.quantization.keras.quantize_apply(
        annotated,
        scheme=default_8bit.Default8BitQuantizeScheme(
            disable_per_axis=per_axis
        )
    )


def _relu6_fed_conv_names(backbone):
    """
    Names of Conv2D / DepthwiseConv2D layers whose output feeds a ReLU6
    (max_value == 6), following through BatchNorm / ZeroPadding.

    These convs fuse into a conv+ReLU6 op whose output is genuinely [0, 6]
    (non-negative), so a min-pinned [0, 6] quantizer is appropriate. Every
    other conv (e.g. the inverted-residual projection / bottleneck convs)
    produces a signed output and wants a signed quantizer instead.
    """

    def is_relu6(layer):
        return (
            isinstance(layer, tf.keras.layers.ReLU)
            and getattr(layer, "max_value", None) == 6
        )

    passthrough = (
        tf.keras.layers.BatchNormalization,
        FreezableBatchNorm,
        tf.keras.layers.ZeroPadding2D,
    )

    consumers = {l.name: [] for l in backbone.layers}
    for layer in backbone.layers:
        for node in layer._outbound_nodes:
            consumers[layer.name].append(node.outbound_layer)

    def feeds_relu6(layer):
        for consumer in consumers[layer.name]:
            nxt = consumer
            while isinstance(nxt, passthrough):
                downstream = consumers[nxt.name]
                if not downstream:
                    break
                nxt = downstream[0]
            if is_relu6(nxt):
                return True
        return False

    return {
        layer.name
        for layer in backbone.layers
        if isinstance(
            layer,
            (
                tf.keras.layers.Conv2D,
                tf.keras.layers.DepthwiseConv2D,
            ),
        )
        and feeds_relu6(layer)
    }


def _is_relu6(layer):
    return (
        isinstance(layer, tf.keras.layers.ReLU)
        and getattr(layer, "max_value", None) == 6
    )


def _quantize_backbone_full(
    backbone,
    *,
    per_axis: bool,
    symmetric: bool,
):
    """
    The single, optimized full-int8 scheme. Each layer is quantized according
    to what its output actually is, so nothing is over-restricted and the only
    fixed [0, 6] pin lands where it survives TFLite fusion:

      * conv feeding a ReLU6      -> weights only (no output quantizer). The
        following ReLU6 layer owns the output quant, so conv+ReLU6 fuse into a
        single op with one output range and no double fake-quant.
      * ReLU6 layer               -> ReLU6OutputConfig: output pinned to a
        fixed [0, 6]. This IS the fused op's output tensor, so the deployed
        dequantized range is exactly 6 -> stock-delegate safe (no mesa patch).
      * signed (linear) conv      -> SignedConvQuantConfig: weights + AllValues
        output. Keeps the negative dynamic range; no ±6 clamp, so the features
        feeding the SSD head are not squashed.

    Replaces the earlier full / fixed / full_av / mixed schemes: AllValues for
    the signed half (their only real benefit) and a correctly-placed [0, 6] pin
    for the ReLU6 half (which the old schemes pinned on the wrong - pre-fusion -
    tensor). With correct [-1, 1] calibration, plain PTQ already lands near
    fp32, so this exists mainly to make per-tensor int8 deployment robust.
    """

    relu6_fed = _relu6_fed_conv_names(backbone)

    weights_only = WeightOnlyQuantConfig(
        per_axis=per_axis,
        symmetric=symmetric,
    )
    signed_cfg = SignedConvQuantConfig(
        per_axis=per_axis,
        symmetric=symmetric,
    )
    relu6_cfg = ReLU6OutputConfig()

    def clone_function(layer):
        if isinstance(
            layer,
            (
                tf.keras.layers.Conv2D,
                tf.keras.layers.DepthwiseConv2D,
            ),
        ):
            config = (
                weights_only
                if layer.name in relu6_fed
                else signed_cfg
            )
            return (
                tfmot.quantization.keras
                .quantize_annotate_layer(
                    layer,
                    quantize_config=config,
                )
            )

        if _is_relu6(layer):
            return (
                tfmot.quantization.keras
                .quantize_annotate_layer(
                    layer,
                    quantize_config=relu6_cfg,
                )
            )

        return layer

    with tfmot.quantization.keras.quantize_scope(
        {
            "FreezableBatchNorm": FreezableBatchNorm,
            "WeightOnlyQuantConfig": WeightOnlyQuantConfig,
            "SignedConvQuantConfig": SignedConvQuantConfig,
            "ReLU6OutputConfig": ReLU6OutputConfig,
        }
    ):
        annotated = tf.keras.models.clone_model(
            backbone,
            clone_function=clone_function,
        )

        return tfmot.quantization.keras.quantize_apply(
            annotated
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

        full
            The recommended full-int8 scheme. Per-layer: conv->ReLU6 convs are
            weight-only and the ReLU6 layer pins its output to a fixed [0, 6]
            (exactly where TFLite fusion keeps it -> stock-delegate safe);
            signed (linear) convs use weights + AllValues (no ±6 clamp). See
            _quantize_backbone_full.

        weights
            Weight-only quantization (activations left to graph transforms /
            the converter). See WeightOnlyQuantConfig.

        default_8bit
            TFMOT Default8BitQuantizeScheme. LEGACY/comparison only - its
            whole-model annotation inserts fake-quant no-ops that make the
            converter ignore the per-tensor request and emit per-channel
            weights (unsupported on the target). See _quantize_backbone_legacy.

        annotate_all
            TFMOT default annotation and quantization. Same per-channel
            leakage problem as "default_8bit". See _quantize_backbone_tfmot.

    NOTE: with correct [-1, 1] calibration, plain PTQ already lands near fp32
    on this model, so QAT ("full") is robustness insurance rather than an
    accuracy requirement.
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

    if scheme == "full":
        return _quantize_backbone_full(
            backbone,
            per_axis=per_axis,
            symmetric=symmetric,
        )

    custom_configs = {
        "weights": WeightOnlyQuantConfig,
    }

    try:
        config_cls = custom_configs[scheme]
    except KeyError as exc:
        valid = [
            "full",
            *custom_configs,
            *tfmot_schemes,
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
