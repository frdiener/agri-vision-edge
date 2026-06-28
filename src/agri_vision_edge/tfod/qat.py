"""
Quantization-aware training utilities for folded TFOD MobileNetV2 backbones.

The deployment accelerators want a fully int8 graph, so the single ``full``
scheme here annotates ONLY Conv2D / DepthwiseConv2D layers (plus, for the
per-channel target, the ReLU6 / residual-Add layers) and pins the activation
ranges explicitly. It has two variants chosen by ``per_channel`` (the target's
weight granularity); both quantize activations and use per-TENSOR weight
fake-quant -- per-channel weights, when wanted, are produced by the CONVERTER,
never by per-channel fake-quant (which breaks the converter's int8 calibration):

    * per_channel=False (i.MX8M Plus): pin [0,6] ON THE CONV (self-contained op)
      so weights stay per-tensor.
    * per_channel=True  (i.MX93 Ethos-U): convs feeding ReLU6 are weight-only and
      the ReLU6 *layer* + residual Add carry the pins, so TFLite is free to emit
      per-channel weights.

With correct [-1, 1] calibration plain PTQ already lands near fp32, so QAT
exists to make int8 deployment robust rather than to recover accuracy.
"""

import collections

import numpy as np
import tensorflow as tf
import tensorflow_model_optimization as tfmot
from object_detection.core.freezable_batch_norm import FreezableBatchNorm
from tensorflow.keras.utils import register_keras_serializable


@register_keras_serializable()
class FixedRelu6Quantizer(tfmot.quantization.keras.quantizers.Quantizer):
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

    The weight fake-quant is ALWAYS per-tensor (symmetric, narrow range): baking
    per-channel weight fake-quant into the graph breaks the TFLite converter's
    int8 calibration. Per-channel weights, when wanted, are emitted by the
    CONVERTER (weight-only convs + ``_experimental_disable_per_channel=False``),
    not here -- see ``_quantize_backbone_full``.
    """

    def _weight_quantizer(self):
        return tfmot.quantization.keras.quantizers.LastValueQuantizer(
            num_bits=8,
            per_axis=False,
            symmetric=True,
            narrow_range=True,
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

        raise TypeError(f"Unsupported layer: {type(layer)}")

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
            layer.depthwise_kernel = quantize_weights[0]
        else:
            layer.kernel = quantize_weights[0]

    def get_config(self):
        return {}

    @classmethod
    def from_config(cls, config):
        return cls()


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
        return tfmot.quantization.keras.quantizers.AllValuesQuantizer(
            num_bits=8,
            per_axis=False,
            symmetric=False,
            narrow_range=False,
        )

    def get_activations_and_quantizers(self, layer):
        return []

    def set_quantize_activations(self, layer, quantize_activations):
        pass

    def get_output_quantizers(self, layer):
        return [self._output_quantizer()]


@register_keras_serializable()
class ReLU6ConvQuantConfig(
    BaseQuantConfig,
):
    """
    Per-tensor weights + a fixed [0, 6] output quantizer, applied to a conv
    whose output feeds a ReLU6.

    The [0, 6] pin lives on the CONV's OWN output (not on a separate ReLU6
    layer). TFLite then fuses the following ReLU6 into the conv, and because the
    conv is a self-contained quantized op (per-tensor weights + its own output
    range) the fused op keeps PER-TENSOR weights. Pinning a *separate* ReLU6
    layer instead (the ReLU6OutputConfig path) leaves the conv weight-only,
    which makes TFLite emit a fully-int8 fused conv with PER-CHANNEL weights that
    `_experimental_disable_per_channel` does NOT override -- the per-channel
    regression. Clamping the conv output to [0, 6] is equivalent to ReLU6
    (negatives -> 0, > 6 -> 6), and the delegate fuses + accepts the resulting
    conv+ReLU6 op (dequantized range exactly 6).
    """

    def _output_quantizer(self):
        return FixedRelu6Quantizer()

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

    This is the PER-CHANNEL counterpart to ReLU6ConvQuantConfig. In the folded
    backbone the conv and ReLU6 are separate layers, and after TFLite fuses
    conv+ReLU6 the fused op's output IS the ReLU6 layer's output - so pinning
    HERE (not on the conv) makes the deployed dequantized range exactly 6
    (scale = 6/255, zero_point = -128) which a stock delegate accepts. Crucially,
    because the preceding conv is left WEIGHT-ONLY (no self-contained output
    range), TFLite is free to emit PER-CHANNEL weights for it - which is exactly
    what we want for the per-channel (i.MX93 Ethos-U) target. ReLU6ConvQuantConfig
    does the opposite (pins on the conv) to FORCE per-tensor for i.MX8M Plus.
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


@register_keras_serializable()
class AddOutputConfig(
    tfmot.quantization.keras.QuantizeConfig,
):
    """
    Quantize the output of a residual ``Add`` (no weights), signed AllValues.

    MobileNetV2 inverted-residual blocks end in ``project_conv -> Add(skip)``.
    With weight-only relu6 convs (the per-channel scheme) the ``Add`` output is
    otherwise un-fake-quantized, leaving a coverage gap that lets the converter
    fall back to dynamic-range weights downstream. Pinning the Add output
    (AllValues, signed -- residual sums are signed) closes the gap.
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
        return [
            tfmot.quantization.keras.quantizers.AllValuesQuantizer(
                num_bits=8,
                per_axis=False,
                symmetric=False,
                narrow_range=False,
            )
        ]

    def get_config(self):
        return {}


def _relu6_fed_conv_names(backbone):
    """
    Names of Conv2D / DepthwiseConv2D layers whose output feeds a ReLU6
    (max_value == 6), following through BatchNorm / ZeroPadding.

    These convs fuse into a conv+ReLU6 op whose output is genuinely [0, 6]
    (non-negative), so a min-pinned [0, 6] quantizer is appropriate. Every
    other conv (e.g. the inverted-residual projection / bottleneck convs)
    produces a signed output and wants a signed quantizer instead.
    """

    is_relu6 = _is_relu6

    passthrough = (
        tf.keras.layers.BatchNormalization,
        FreezableBatchNorm,
        tf.keras.layers.ZeroPadding2D,
    )

    consumers = {layer.name: [] for layer in backbone.layers}
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
    """
    True for a ReLU6 activation, whether it is a ``keras.layers.ReLU(max_value=6)``
    (the MobileNetV2 backbone) or a ``Lambda(tf.nn.relu6)`` (the SSD feature-map
    generator / head). The Lambda carries no metadata identifying it as ReLU6, so
    we probe it: a layer is ReLU6 iff it clips its input to [0, 6].
    """
    if (
        isinstance(layer, tf.keras.layers.ReLU)
        and getattr(layer, "max_value", None) == 6
    ):
        return True

    if isinstance(layer, tf.keras.layers.Lambda):
        try:
            probe = tf.constant([-6.0, -1.0, 0.0, 3.0, 6.0, 9.0], dtype=tf.float32)
            out = np.asarray(layer(probe))
            return np.allclose(out, np.clip(probe.numpy(), 0.0, 6.0))
        except Exception:
            return False

    return False


def _quantize_backbone_full(
    backbone,
    *,
    per_channel: bool,
):
    """
    Full-int8 scheme. Each conv is quantized by what its output actually is, and
    the only fixed [0, 6] pin lands where it survives TFLite fusion. There are
    two variants, selected by ``per_channel`` (the deployment target's weight
    granularity), because forcing per-tensor and allowing per-channel need the
    [0, 6] pin in DIFFERENT places:

    PER-TENSOR target (``per_channel=False``, i.MX8M Plus / stock delegate):
      * conv feeding a ReLU6 -> ReLU6ConvQuantConfig: weights + a fixed [0, 6]
        output quantizer ON THE CONV, making it a self-contained quantized op so
        TFLite keeps PER-TENSOR weights through the conv+ReLU6 fusion.
      * signed (linear) conv -> SignedConvQuantConfig (weights + AllValues).
      * ReLU6 / Add layers   -> left float (fused into the conv).

    PER-CHANNEL target (``per_channel=True``, i.MX93 Ethos-U):
      * conv feeding a ReLU6 -> WeightOnlyQuantConfig (NO conv-output pin), so
        TFLite is free to emit PER-CHANNEL weights for it.
      * ReLU6 layer          -> ReLU6OutputConfig: fixed [0, 6] on the ReLU6
        LAYER output (the tensor that survives fusion -> deployed range exact 6).
      * signed (linear) conv -> SignedConvQuantConfig (weights + AllValues).
      * residual Add         -> AddOutputConfig (signed AllValues): closes the
        fake-quant coverage gap left by the weight-only convs.

    IMPORTANT: the weight FAKE-QUANT is per-tensor in BOTH variants (see
    BaseQuantConfig). Per-channel weights are produced by the CONVERTER
    (weight-only convs + ``_experimental_disable_per_channel=False``), NOT by
    per-channel fake-quant nodes -- baking
    ``fake_quant_with_min_max_vars_per_channel`` into the graph makes the TFLite
    int8 calibration collect ~0 activation ranges and collapses AP (fp32 stays
    fine). So ``per_channel`` only chooses the pin placement; it never changes
    the fake-quant granularity.
    """

    relu6_fed = _relu6_fed_conv_names(backbone)

    signed_cfg = SignedConvQuantConfig()

    if per_channel:
        relu6_conv_cfg = WeightOnlyQuantConfig()
        relu6_layer_cfg = ReLU6OutputConfig()
        add_cfg = AddOutputConfig()
    else:
        relu6_conv_cfg = ReLU6ConvQuantConfig()
        relu6_layer_cfg = None
        add_cfg = None

    def clone_function(layer):
        if isinstance(
            layer,
            (
                tf.keras.layers.Conv2D,
                tf.keras.layers.DepthwiseConv2D,
            ),
        ):
            config = relu6_conv_cfg if layer.name in relu6_fed else signed_cfg
            return tfmot.quantization.keras.quantize_annotate_layer(
                layer,
                quantize_config=config,
            )

        # Per-channel scheme only: pin the ReLU6 layer output and the residual
        # Add output (the per-tensor scheme leaves these float -- the conv owns
        # the pin there).
        if per_channel and _is_relu6(layer):
            return tfmot.quantization.keras.quantize_annotate_layer(
                layer,
                quantize_config=relu6_layer_cfg,
            )

        if per_channel and isinstance(layer, tf.keras.layers.Add):
            return tfmot.quantization.keras.quantize_annotate_layer(
                layer,
                quantize_config=add_cfg,
            )

        return layer

    with tfmot.quantization.keras.quantize_scope(
        {
            "FreezableBatchNorm": FreezableBatchNorm,
            "ReLU6ConvQuantConfig": ReLU6ConvQuantConfig,
            "SignedConvQuantConfig": SignedConvQuantConfig,
            "WeightOnlyQuantConfig": WeightOnlyQuantConfig,
            "ReLU6OutputConfig": ReLU6OutputConfig,
            "AddOutputConfig": AddOutputConfig,
        }
    ):
        annotated = tf.keras.models.clone_model(
            backbone,
            clone_function=clone_function,
        )

        return tfmot.quantization.keras.quantize_apply(annotated)


def quantize_backbone(
    backbone,
    *,
    per_channel: bool = False,
):
    """
    Convert a backbone model to a QAT-enabled model (the full int8 scheme).

    ``per_channel`` selects the deployment target's weight granularity:
    False pins [0,6] on the conv (forces per-tensor weights, i.MX8M Plus); True
    leaves relu6-fed convs weight-only and pins the ReLU6 layer + residual Add
    (lets the converter emit per-channel weights, i.MX93 Ethos-U). It only
    chooses the pin placement -- the weight fake-quant is per-tensor either way.
    See _quantize_backbone_full.

    NOTE: with correct [-1, 1] calibration plain PTQ already lands near fp32 on
    this model, so QAT is robustness insurance rather than an accuracy
    requirement.
    """

    return _quantize_backbone_full(backbone, per_channel=per_channel)


def ensure_model_is_built_for_qat(detection_model, pipeline_config):
    ssd_config = pipeline_config.model.ssd

    h = ssd_config.image_resizer.fixed_shape_resizer.height

    w = ssd_config.image_resizer.fixed_shape_resizer.width

    dummy = tf.zeros([1, h, w, 3], dtype=tf.float32)

    image, shapes = detection_model.preprocess(dummy)

    detection_model.predict(image, shapes)


# =========================================================
# Whole-model QAT: weight-preserving functional rebuild of the SSD head.
#
# object_detection's feature_map_generator (KerasMultiResolutionFeatureMaps)
# and box-predictor heads are *subclassed* Keras models. Swapping folded /
# quantize-wrapped layers into them in place breaks TFLite conversion (the
# swapped layers are not tracked sublayers, so the SavedModel trace prunes the
# graph to empty). Rebuilding them as FUNCTIONAL models -- reusing the converged
# layers, so weights are preserved exactly -- and wrapping them in tracked
# adapter Layers lets the same clone_model + quantize_apply path used for the
# backbone quantize them. A QAT model can then cover the whole graph up to the
# (float) TFLite_Detection_PostProcess.
#
# Specific to the plain SSD MobileNetV2 head (KerasMultiResolutionFeatureMaps +
# ConvolutionalBoxPredictor). FPNLite / other heads would need their own rebuild.
# =========================================================

_QAT_CONV = (tf.keras.layers.Conv2D, tf.keras.layers.DepthwiseConv2D)


def _clone_conv_unique(layer, name):
    """Clone a Conv2D/DepthwiseConv2D with a unique name (weights copied)."""
    cfg = layer.get_config()
    cfg["name"] = name
    new = type(layer).from_config(cfg)
    kernel = (
        layer.depthwise_kernel
        if isinstance(layer, tf.keras.layers.DepthwiseConv2D)
        else layer.kernel
    )
    new.build((None, None, None, int(kernel.shape[2])))
    new.set_weights(layer.get_weights())
    return new


def fold_functional(model):
    """
    Fold every conv->BN pair in a FUNCTIONAL model by rebuilding the graph with
    the BatchNorm dropped (BN params baked into a bias-enabled conv). Generic;
    unlike ``fold_mobilenetv2_backbone`` it makes no MobileNetV2 topology
    assumptions, so it works on the rebuilt feature-map generator.
    """
    consumers = collections.defaultdict(list)
    for layer in model.layers:
        for node in layer.inbound_nodes:
            for parent in tf.nest.flatten(node.inbound_layers):
                consumers[parent.name].append(layer)

    conv_bn, drop = {}, set()
    for layer in model.layers:
        if isinstance(layer, _QAT_CONV):
            cs = consumers.get(layer.name, [])
            if len(cs) == 1 and "BatchNorm" in type(cs[0]).__name__:
                conv_bn[layer.name] = cs[0]
                drop.add(cs[0].name)

    out, new_inputs = {}, []
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.InputLayer):
            ni = tf.keras.Input(shape=layer.output.shape[1:], name=layer.name)
            out[layer.name] = ni
            new_inputs.append(ni)
            continue
        parents = list(tf.nest.flatten(layer.inbound_nodes[0].inbound_layers))
        x = [out[p.name] for p in parents]
        x = x[0] if len(x) == 1 else x
        if layer.name in drop:  # BN -> its conv's folded output
            out[layer.name] = out[parents[0].name]
        elif layer.name in conv_bn:
            folded = _fold_conv_bn_functional(layer, conv_bn[layer.name])
            out[layer.name] = folded(x)
        else:
            out[layer.name] = layer(x)  # reuse (Lambda/ReLU6/...)
    outputs = [out[o._keras_history.layer.name] for o in model.outputs]
    return tf.keras.Model(new_inputs, outputs)


def _fold_conv_bn_functional(conv, bn):
    """BN-fold helper that builds from the kernel shape (subclassed-layer convs
    don't expose ``input_shape``)."""
    k = conv.kernel.numpy()
    b = conv.bias.numpy() if conv.use_bias else np.zeros(k.shape[-1], np.float32)
    g, be = bn.gamma.numpy(), bn.beta.numpy()
    mu, var = bn.moving_mean.numpy(), bn.moving_variance.numpy()
    scale = g / np.sqrt(var + bn.epsilon)
    folded = tf.keras.layers.Conv2D(
        conv.filters,
        conv.kernel_size,
        strides=conv.strides,
        padding=conv.padding,
        dilation_rate=conv.dilation_rate,
        activation=None,
        use_bias=True,
        name=conv.name + "_folded",
    )
    folded.build((None, None, None, int(k.shape[2])))
    folded.set_weights([k * scale.reshape(1, 1, 1, -1), be + (b - mu) * scale])
    return folded


def rebuild_feature_map_generator_functional(fmg, feature_specs):
    """
    Functionally reconstruct a KerasMultiResolutionFeatureMaps, reusing its
    converged layers (weights preserved). ``feature_specs`` is an OrderedDict
    {backbone_feature_key: TensorShape} for the inputs it consumes.
    """
    inp = collections.OrderedDict(
        (k, tf.keras.Input(shape=tuple(v.as_list()[1:]), name=k.replace("/", "__")))
        for k, v in feature_specs.items()
    )
    fmaps = []
    for index, from_layer in enumerate(fmg.feature_map_layout["from_layer"]):
        if from_layer:
            fm = inp[from_layer]
        else:
            fm = fmaps[-1]
            for layer in fmg.convolutions[index]:
                # Mirror the backbone fold (_clone_or_replace_layer): swap the
                # ReLU6 Lambda for an explicit keras.layers.ReLU (same name, no
                # weights). TFLite fuses conv + keras-ReLU6 into one op, but a
                # conv + Lambda(relu6) does NOT fuse -- it leaves a
                # dequant->relu6->quant sandwich in the head that the NPU
                # delegate cannot consume.
                if isinstance(layer, tf.keras.layers.Lambda) and _is_relu6(layer):
                    layer = tf.keras.layers.ReLU(max_value=6.0, name=layer.name)
                fm = layer(fm)
        fmaps.append(fm)
    return tf.keras.Model(inputs=inp, outputs=fmaps)


class FMGAdapter(tf.keras.layers.Layer):
    """Drop-in for ``feature_extractor.feature_map_generator``: wraps the
    quantized functional generator (tracked sublayer, so it serializes) and
    adapts the meta-arch's dict-in / OrderedDict-out contract."""

    def __init__(self, func_model, output_keys, **kw):
        super().__init__(**kw)
        self.func = func_model
        self.output_keys = list(output_keys)

    def call(self, image_features):
        san = {k.replace("/", "__"): v for k, v in image_features.items()}
        ordered = [san[name] for name in self.func.input_names]
        outs = self.func(ordered)
        outs = outs if isinstance(outs, (list, tuple)) else [outs]
        # NB: no `strict=` — under @tf.function autograph rewrites `zip` into its
        # own `zip_`, which does not accept Python 3.10's strict keyword (works
        # eagerly, TypeErrors in graph mode = the training/eval forward path).
        # noqa note: see FMGAdapter -- autograph rewrites zip and rejects strict=
        return collections.OrderedDict(zip(self.output_keys, outs))  # noqa: B905  # noqa: B905


def rebuild_box_predictor_functional(box_predictor, feature_shapes):
    """
    Functionally reconstruct a ConvolutionalBoxPredictor, preserving weights.
    Box/class heads are one 1x1 conv each (no BN) followed by a reshape; the
    convs share names across heads so each is cloned to a unique name. Outputs
    are the box-encoding tensors followed by the class tensors.
    """
    from object_detection.core.box_predictor import (
        BOX_ENCODINGS,
        CLASS_PREDICTIONS_WITH_BACKGROUND,
    )

    inputs = [
        tf.keras.Input(shape=tuple(s.as_list()[1:]), name=f"bp_in_{i}")
        for i, s in enumerate(feature_shapes)
    ]
    box_out, cls_out = [], []
    for i, x0 in enumerate(inputs):
        x = x0
        for layer in box_predictor._shared_nets[
            i
        ]:  # empty unless a tower is configured
            x = layer(x)
        bh = box_predictor._prediction_heads[BOX_ENCODINGS][i]
        b = x
        for layer in bh._box_encoder_layers:
            b = (
                _clone_conv_unique(layer, f"BoxEncodingPredictor_{i}")
                if isinstance(layer, _QAT_CONV)
                else layer
            )(b)
        box_out.append(
            tf.keras.layers.Reshape(
                (-1, 1, bh._box_code_size), name=f"box_reshape_{i}"
            )(b)
        )
        ch = box_predictor._prediction_heads[CLASS_PREDICTIONS_WITH_BACKGROUND][i]
        c = x
        for layer in ch._class_predictor_layers:
            c = (
                _clone_conv_unique(layer, f"ClassPredictor_{i}")
                if isinstance(layer, _QAT_CONV)
                else layer
            )(c)
        cls_out.append(
            tf.keras.layers.Reshape((-1, ch._num_class_slots), name=f"cls_reshape_{i}")(
                c
            )
        )
    return tf.keras.Model(inputs, box_out + cls_out)


class BoxPredictorAdapter(tf.keras.layers.Layer):
    """Drop-in for ``_box_predictor``: wraps the quantized functional predictor
    and returns the {BOX_ENCODINGS, CLASS_PREDICTIONS_WITH_BACKGROUND} dict the
    meta-arch consumes."""

    def __init__(self, func_model, num_feature_maps, **kw):
        super().__init__(**kw)
        self.func = func_model
        self.n = num_feature_maps
        self.is_keras_model = True

    def call(self, image_features):
        from object_detection.core.box_predictor import (
            BOX_ENCODINGS,
            CLASS_PREDICTIONS_WITH_BACKGROUND,
        )

        outs = self.func(list(image_features))
        outs = outs if isinstance(outs, (list, tuple)) else [outs]
        return {
            BOX_ENCODINGS: list(outs[: self.n]),
            CLASS_PREDICTIONS_WITH_BACKGROUND: list(outs[self.n :]),
        }


def _quantize_weights_only(model):
    """
    Annotate every conv weight-only (no output fake-quant) and quantize_apply.

    Used for the SSD box predictor: its 1x1 box/class convs feed reshape ->
    CONCAT (across feature maps) -> the float TFLite_Detection_PostProcess.
    Pinning each conv's output scale (the full scheme's AllValues) gives the six
    feature maps SIX different scales, so the converter has to insert a requant
    QUANTIZE on five of them before the concat -- stray ops that the NPU delegate
    cannot consume (it trips). Leaving the output scale FREE (weight-only) lets
    the converter align all concat inputs to one scale, exactly like PTQ, so the
    graph stays fully fused (no stray QUANTIZE). Weights are still QAT-trained;
    per-tensor vs per-channel weight emission is the converter's call
    (``_experimental_disable_per_channel``).
    """
    cfg = WeightOnlyQuantConfig()

    def clone_function(layer):
        if isinstance(layer, (tf.keras.layers.Conv2D, tf.keras.layers.DepthwiseConv2D)):
            return tfmot.quantization.keras.quantize_annotate_layer(
                layer, quantize_config=cfg
            )
        return layer

    with tfmot.quantization.keras.quantize_scope(
        {"WeightOnlyQuantConfig": WeightOnlyQuantConfig}
    ):
        return tfmot.quantization.keras.quantize_apply(
            tf.keras.models.clone_model(model, clone_function=clone_function)
        )


def quantize_detection_head(detection_model, image_size, *, per_channel=False):
    """
    Quantize the SSD head in place via weight-preserving functional rebuilds, so
    QAT covers the whole graph up to the postprocess. Call AFTER the backbone has
    been folded + quantized.

    ``per_channel`` must match the backbone (it selects the same pin placement
    for the feature generator's full scheme). The box predictor is always
    quantized weight-only (see _quantize_weights_only -- pinning its output
    scales would force stray requant QUANTIZE ops at the concat that trip the
    NPU delegate).

    Dispatches on the head architecture:
      * plain SSD MobileNetV2 (KerasMultiResolutionFeatureMaps +
        ConvolutionalBoxPredictor) -- below.
      * FPNLite (KerasFpnTopDownFeatureMaps + WeightSharedConvolutionalBox
        Predictor) -- see _quantize_fpn_detection_head.
    """
    fe = detection_model.feature_extractor
    if hasattr(fe, "_fpn_features_generator"):
        return _quantize_fpn_detection_head(
            detection_model, image_size, per_channel=per_channel
        )

    from object_detection.utils import ops as od_ops

    fmg = fe.feature_map_generator
    backbone = fe.classification_backbone

    keys = [fl for fl in fmg.feature_map_layout["from_layer"] if fl]
    pp, _ = detection_model.preprocess(
        tf.zeros([1, image_size, image_size, 3], dtype=tf.float32)
    )
    feats = backbone(od_ops.pad_to_multiple(pp, fe._pad_to_multiple))
    feature_specs = collections.OrderedDict(
        (k, feats[i].shape) for i, k in enumerate(keys)
    )
    fmg_outputs = fmg({k: tf.zeros(v) for k, v in feature_specs.items()})
    out_keys = list(fmg_outputs.keys())
    feature_shapes = [t.shape for t in fmg_outputs.values()]

    qfmg = quantize_backbone(
        fold_functional(rebuild_feature_map_generator_functional(fmg, feature_specs)),
        per_channel=per_channel,
    )
    fe.feature_map_generator = FMGAdapter(qfmg, out_keys)

    qbp = _quantize_weights_only(
        rebuild_box_predictor_functional(detection_model._box_predictor, feature_shapes)
    )
    detection_model._box_predictor = BoxPredictorAdapter(qbp, len(feature_shapes))

    return detection_model


# =========================================================
# FPNLite head QAT (SSD MobileNetV2 FPN).
#
# The FPN head is structurally different from the plain SSD head:
#   * feature generation is a top-down FPN (KerasFpnTopDownFeatureMaps:
#     projections + nearest-neighbour upsample + residual ADD + smoothing convs)
#     followed by extra "coarse" stride-2 layers, instead of a flat chain of
#     extra conv blocks;
#   * the box predictor is a WeightSharedConvolutionalBoxPredictor (one shared
#     tower + shared box/class heads applied to every feature map, with per-level
#     BatchNorm), instead of one conv per feature map.
# Both use SeparableConv2D everywhere, which our fold/quant only handle once split
# into DepthwiseConv2D + Conv2D(1x1). The functional rebuilds below mirror the
# plain-SSD ones (reuse converged layers, weights preserved) but replay those
# FPN-specific forwards layer-by-layer so fold + quantize + TFLite tracing work.
# =========================================================


def _split_separable_conv(sep, tag):
    """
    Split a SeparableConv2D into (DepthwiseConv2D no-bias, Conv2D 1x1 +bias),
    weights copied. Our fold/quant primitives are Conv2D/DepthwiseConv2D only
    (TFLite splits separables the same way). ``tag`` makes the names unique --
    the FPN/weight-shared graphs reuse one separable across feature maps, and the
    per-map BatchNorm folds into a distinct Conv2D each, so they cannot share.
    """
    cfg = sep.get_config()
    dw = tf.keras.layers.DepthwiseConv2D(
        kernel_size=cfg["kernel_size"],
        strides=cfg["strides"],
        padding=cfg["padding"],
        depth_multiplier=cfg["depth_multiplier"],
        dilation_rate=cfg["dilation_rate"],
        use_bias=False,
        name=f"{sep.name}_{tag}_dw",
    )
    pw = tf.keras.layers.Conv2D(
        filters=cfg["filters"],
        kernel_size=1,
        use_bias=cfg["use_bias"],
        name=f"{sep.name}_{tag}_pw",
    )
    w = sep.get_weights()  # [depthwise_kernel, pointwise_kernel, (bias)]
    in_ch = w[0].shape[2] * w[0].shape[3]
    dw.build((None, None, None, w[0].shape[2]))
    dw.set_weights([w[0]])
    pw.build((None, None, None, in_ch))
    pw.set_weights(w[1:] if cfg["use_bias"] else [w[1]])
    return dw, pw


def _quantize_fpn_detection_head(detection_model, image_size, *, per_channel=False):
    """
    FPNLite head QAT: quantize the FPN feature generator + coarse layers (full
    scheme) and the weight-shared box predictor (weight-only), in place. Mirrors
    ``quantize_detection_head`` for the plain SSD head; see the section banner.

    BOTH schemes route through the single combined functional model (FPN
    generator + coarse blocks + weight-shared box predictor in ONE
    quantize_apply); see ``qat_fpn_combined``. One graph is required for the
    per-TENSOR scheme: it leaves the ReLU6 *layer* float (it pins the conv, to
    keep weights per-tensor), so a feature-map ReLU6 on a SEPARATE-model boundary
    (a top-down map fanning out to the internal RESIZE and to the box predictor)
    cannot fuse and TFLite emits a stray float-ReLU6 island (DEQUANTIZE -> ReLU6
    -> QUANTIZE per consumer). Making that ReLU6 interior to one graph lets
    conv+ReLU6 fuse: 0 stray quant/dequant AND 0 per-channel weight tensors.
    Per-CHANNEL converts equally cleanly through the same graph (0 stray,
    per-channel weights) -- verified with experiments/fpn_qat_probe.
    """
    from agri_vision_edge.tfod.qat_fpn_combined import (
        quantize_fpn_detection_head_combined,
    )

    return quantize_fpn_detection_head_combined(
        detection_model, image_size, per_channel=per_channel
    )
