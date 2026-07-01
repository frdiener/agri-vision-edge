"""
Quantization-aware training for folded TFOD SSD / SSD-FPNLite MobileNetV2.

The deployment accelerators want a fully int8 graph. Every model -- the
MobileNetV2 backbone and the functionally-rebuilt detection heads alike -- is
first **folded** (``folding.fold_model``: BN folded into the conv, ReLU6
pre-folded into the conv's intrinsic ``activation``) and then quantized by the
single "full" scheme below (``quantize_backbone``). Only Conv2D /
DepthwiseConv2D (plus residual ``Add`` in the per-channel target) are annotated;
activation ranges are pinned explicitly where it matters.

Two variants are selected by ``per_channel`` -- the deployment target's WEIGHT
granularity. The [0, 6] ReLU6 range is what both variants must deliver on the
tensor that survives TFLite's conv+ReLU6 fusion; pre-folding the ReLU6 into the
conv makes the conv's OWN output that surviving tensor, so:

    * per_channel=False (i.MX8M Plus / stock delegate): the relu6-fed conv gets
      a fixed [0, 6] output quantizer (``ReLU6ConvQuantConfig``). The conv is
      then a self-contained quantized op, so TFLite keeps PER-TENSOR weights
      through the conv+ReLU6 fusion. This is what forces a strictly per-tensor
      int8 graph.

    * per_channel=True (i.MX93 Ethos-U): the relu6-fed conv is left WEIGHT-ONLY
      (``FreeOutputConvQuantConfig``), so TFLite is free to emit PER-CHANNEL weights
      for it. The [0, 6] range still lands exactly (scale 6/255, zero_point
      -128) because the conv carries an intrinsic ``tf.nn.relu6`` and TFLite
      pins the fused activation to relu6's known range.

In BOTH variants the weight FAKE-QUANT is per-tensor (see ``BaseQuantConfig``).
Per-channel weights, when wanted, are produced by the CONVERTER (weight-only
convs + ``_experimental_disable_per_channel=False``), never by per-channel
fake-quant nodes -- baking ``fake_quant_with_min_max_vars_per_channel`` into the
graph makes the TFLite int8 calibration collect ~0 activation ranges and
collapses AP. So ``per_channel`` only chooses the pin placement.

With correct [-1, 1] calibration plain PTQ already lands near fp32, so QAT
exists to make int8 deployment robust rather than to recover accuracy.
"""

import collections
import itertools

import numpy as np
import tensorflow as tf
import tensorflow_model_optimization as tfmot
from object_detection.core.freezable_batch_norm import FreezableBatchNorm
from tensorflow.keras.utils import register_keras_serializable

from agri_vision_edge.tfod.folding import fold_model, is_relu6

_CONV = (tf.keras.layers.Conv2D, tf.keras.layers.DepthwiseConv2D)


# =========================================================
# Quantization configs
# =========================================================


@register_keras_serializable()
class BaseQuantConfig(tfmot.quantization.keras.QuantizeConfig):
    """
    Shared configuration for convolutional layers: PER-TENSOR weight fake-quant.

    The weight fake-quant is ALWAYS per-tensor (symmetric, narrow range): baking
    per-channel weight fake-quant into the graph breaks the TFLite converter's
    int8 calibration. Per-channel weights, when wanted, are emitted by the
    CONVERTER (weight-only convs + ``_experimental_disable_per_channel=False``),
    not here.
    """

    def _weight_quantizer(self):
        return tfmot.quantization.keras.quantizers.LastValueQuantizer(
            num_bits=8,
            per_axis=False,
            symmetric=True,
            narrow_range=True,
        )

    def _kernel(self, layer):
        if isinstance(layer, tf.keras.layers.DepthwiseConv2D):
            return layer.depthwise_kernel
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer.kernel
        raise TypeError(f"Unsupported layer: {type(layer)}")

    def get_weights_and_quantizers(self, layer):
        return [(self._kernel(layer), self._weight_quantizer())]

    def set_quantize_weights(self, layer, quantize_weights):
        if isinstance(layer, tf.keras.layers.DepthwiseConv2D):
            layer.depthwise_kernel = quantize_weights[0]
        else:
            layer.kernel = quantize_weights[0]

    def get_config(self):
        return {}

    @classmethod
    def from_config(cls, config):
        return cls()


@register_keras_serializable()
class FreeOutputConvQuantConfig(BaseQuantConfig):
    """
    Quantize convolution weights only; leave the OUTPUT scale FREE (calibrated
    by the converter). No output/activation fake-quant node is inserted.

    Two uses, both relying on the converter -- not a fixed pin -- to set the
    output scale:

      (a) per-channel relu6-fed convs: a free output range lets TFLite emit
          PER-CHANNEL weights, while the intrinsic ``tf.nn.relu6`` still pins the
          fused output MIN at 0 (zp -128) and the converter calibrates the MAX
          (<= 6). A fixed [0,6] output pin cannot be used here: TFMOT cannot
          quantize tf.nn.relu6 as an activation, and a fixed OUTPUT pin combined
          with per-channel weights crashes the legacy converter on the FPN head.

      (b) box-predictor convs (linear, feeding a concat): a free scale lets the
          converter align all concat inputs to one scale -- no stray requant
          QUANTIZE the NPU delegate would trip on.
    """

    def get_activations_and_quantizers(self, layer):
        return []

    def set_quantize_activations(self, layer, quantize_activations):
        pass

    def get_output_quantizers(self, layer):
        return []


@register_keras_serializable()
class SignedConvQuantConfig(BaseQuantConfig):
    """
    Per-tensor weights + an AllValues (signed) output quantizer.

    For convolutions whose output is *signed* (the inverted-residual projection /
    bottleneck convs, which have no following ReLU and produce negative values).
    AllValues tracks the true observed min/max and calibrates in ~1 step, so the
    signed dynamic range feeding the head is preserved -- no ±6 clamp.
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
class FixedRelu6Quantizer(tfmot.quantization.keras.quantizers.Quantizer):
    """A fixed [0, 6] fake-quant: scale = 6/255, zero_point = -128."""

    def build(self, tensor_shape, name, layer):
        return {}

    def __call__(self, inputs, training, weights, **kwargs):
        return tf.quantization.fake_quant_with_min_max_args(
            inputs,
            min=0.0,
            max=6.0,
            num_bits=8,
            narrow_range=False,
        )

    def get_config(self):
        return {}


@register_keras_serializable()
class ReLU6ConvQuantConfig(BaseQuantConfig):
    """
    Per-tensor weights + a fixed [0, 6] output quantizer, for a conv that carries
    an intrinsic ``tf.nn.relu6`` (the pre-folded relu6-fed convs).

    The [0, 6] pin lives on the CONV's OWN output. Because the ReLU6 was folded
    into the conv, that output IS the post-ReLU6 tensor -- the one that survives
    TFLite's conv+ReLU6 fusion. And because the conv is a self-contained
    quantized op (per-tensor weights + its own fixed output range), the fused op
    keeps PER-TENSOR weights. This is the per-TENSOR target's mechanism for a
    strictly per-tensor int8 graph with an exact [0, 6] ReLU6 range.
    """

    def get_activations_and_quantizers(self, layer):
        return []

    def set_quantize_activations(self, layer, quantize_activations):
        pass

    def get_output_quantizers(self, layer):
        # The fixed [0,6] pin must be an OUTPUT quantizer: TFMOT's activation
        # quantizer path rejects tf.nn.relu6 (QuantizeAwareActivation whitelists
        # only a few Keras activations), so the ReLU6 range can only be pinned on
        # the (post-fold) conv output tensor -- which, thanks to pre-folding, is
        # the tensor that survives conv+ReLU6 fusion.
        return [FixedRelu6Quantizer()]


@register_keras_serializable()
class AddOutputConfig(tfmot.quantization.keras.QuantizeConfig):
    """
    Quantize the output of a residual ``Add`` (no weights), signed AllValues.

    MobileNetV2 inverted-residual blocks (and the FPN top-down merges) end in
    ``project_conv -> Add(skip)``. In the PER-CHANNEL scheme the relu6-fed convs
    are weight-only, so the ``Add`` output would otherwise be un-fake-quantized,
    leaving a coverage gap that lets the converter fall back to dynamic-range
    weights downstream. Pinning the Add output (AllValues, signed) closes it.
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


_QUANT_SCOPE = {
    "FreezableBatchNorm": FreezableBatchNorm,
    "BaseQuantConfig": BaseQuantConfig,
    "FreeOutputConvQuantConfig": FreeOutputConvQuantConfig,
    "SignedConvQuantConfig": SignedConvQuantConfig,
    "ReLU6ConvQuantConfig": ReLU6ConvQuantConfig,
    "AddOutputConfig": AddOutputConfig,
}


def _has_intrinsic_relu6(layer: tf.keras.layers.Layer) -> bool:
    """
    True for a Conv2D / DepthwiseConv2D carrying an intrinsic ``tf.nn.relu6``
    activation -- i.e. a relu6-fed conv after ``folding.fold_model`` pre-folds
    the ReLU6 into it.
    """
    if not isinstance(layer, _CONV):
        return False

    if layer.activation is tf.nn.relu6:
        return True

    # Defensive fallback for serialized/deserialized activation callables.
    try:
        probe = tf.constant([-6.0, -1.0, 0.0, 3.0, 6.0, 9.0], dtype=tf.float32)
        output = np.asarray(layer.activation(probe))
        return np.allclose(output, np.clip(probe.numpy(), 0.0, 6.0))
    except Exception:
        return False


# =========================================================
# The "full" int8 scheme -- shared by backbone and every detection head.
# =========================================================


def _quantize_full(model, *, per_channel: bool, weight_only_names=frozenset()):
    """
    One ``quantize_apply`` over a folded functional model, applying the full
    int8 scheme. Each conv is classified by what its folded output is:

      * intrinsic relu6 (absorbed by ``fold_model``)
          -> per-tensor: ``ReLU6ConvQuantConfig`` -- weights + fixed [0, 6] OUTPUT
             pin. Conv is self-contained, so TFLite keeps PER-TENSOR weights.
          -> per-channel: ``FreeOutputConvQuantConfig`` -- weight-only, free output
             scale. TFLite emits PER-CHANNEL weights; intrinsic relu6 still clamps
             the fused activation so the converter calibrates a range bounded by 6.
      * name in ``weight_only_names`` (box-predictor convs feeding a concat)
          -> ``FreeOutputConvQuantConfig`` -- free output for shared concat scale.
      * otherwise (signed / linear) -> ``SignedConvQuantConfig``.

    Residual ``Add`` layers are pinned (``AddOutputConfig``) per-channel only.
    ``weight_only_names`` lets the FPN combined graph force box-predictor convs
    weight-only; intrinsic-relu6 detection takes precedence for tower convs that
    happen to feed a relu6.
    """
    signed_cfg = SignedConvQuantConfig()
    weight_only_cfg = FreeOutputConvQuantConfig()
    relu6_conv_cfg = FreeOutputConvQuantConfig() if per_channel else ReLU6ConvQuantConfig()
    add_cfg = AddOutputConfig() if per_channel else None

    def clone_function(layer):
        if isinstance(layer, _CONV):
            if _has_intrinsic_relu6(layer):
                config = relu6_conv_cfg
            elif layer.name in weight_only_names:
                config = weight_only_cfg
            else:
                config = signed_cfg
            return tfmot.quantization.keras.quantize_annotate_layer(
                layer, quantize_config=config
            )

        if add_cfg is not None and isinstance(layer, tf.keras.layers.Add):
            return tfmot.quantization.keras.quantize_annotate_layer(
                layer, quantize_config=add_cfg
            )

        return layer

    with tfmot.quantization.keras.quantize_scope(_QUANT_SCOPE):
        annotated = tf.keras.models.clone_model(model, clone_function=clone_function)
        return tfmot.quantization.keras.quantize_apply(annotated)


def quantize_backbone(backbone, *, per_channel: bool):
    """
    Apply the full int8 scheme to a FOLDED MobileNetV2 backbone (or any folded
    functional feature graph). See the module docstring for the per-tensor vs
    per-channel pin placement. Input must already be folded
    (``folding.fold_model`` / ``fold_mobilenetv2_backbone``).
    """
    return _quantize_full(backbone, per_channel=per_channel)


def ensure_model_is_built_for_qat(detection_model, pipeline_config):
    ssd_config = pipeline_config.model.ssd
    h = ssd_config.image_resizer.fixed_shape_resizer.height
    w = ssd_config.image_resizer.fixed_shape_resizer.width
    dummy = tf.zeros([1, h, w, 3], dtype=tf.float32)
    image, shapes = detection_model.preprocess(dummy)
    detection_model.predict(image, shapes)


# =========================================================
# Whole-model QAT: weight-preserving functional rebuild of the detection head.
#
# object_detection's feature_map_generator / FPN generator and box-predictor
# heads are *subclassed* Keras models. Swapping folded / quantize-wrapped layers
# into them in place breaks TFLite conversion (the swapped layers are not tracked
# sublayers, so the SavedModel trace prunes the graph to empty). Rebuilding them
# as FUNCTIONAL models -- reusing the converged layers, so weights are preserved
# exactly -- and wrapping them in tracked adapter Layers lets the same
# fold_model + quantize path used for the backbone quantize them, so a QAT model
# covers the whole graph up to the (float) TFLite_Detection_PostProcess.
# =========================================================


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


def _quantize_weights_only(model):
    """
    Annotate every conv weight-only (no output fake-quant) and quantize_apply.

    Used for the SSD box predictor: its 1x1 box/class convs feed reshape ->
    CONCAT (across feature maps) -> the float TFLite_Detection_PostProcess.
    Pinning each conv's output scale would give the feature maps different
    scales, forcing stray requant QUANTIZE ops before the concat that the NPU
    delegate cannot consume. Leaving the output scale FREE lets the converter
    align all concat inputs to one scale (exactly like PTQ). Weights are still
    QAT-trained; per-tensor vs per-channel emission is the converter's call.
    """
    cfg = FreeOutputConvQuantConfig()

    def clone_function(layer):
        if isinstance(layer, _CONV):
            return tfmot.quantization.keras.quantize_annotate_layer(
                layer, quantize_config=cfg
            )
        return layer

    with tfmot.quantization.keras.quantize_scope(_QUANT_SCOPE):
        return tfmot.quantization.keras.quantize_apply(
            tf.keras.models.clone_model(model, clone_function=clone_function)
        )


def quantize_detection_head(detection_model, image_size, *, per_channel=False):
    """
    Quantize the detection head in place via weight-preserving functional
    rebuilds, so QAT covers the whole graph up to the postprocess. Call AFTER the
    backbone has been folded + quantized.

    ``per_channel`` must match the backbone. Dispatches on the head architecture:

      * plain SSD MobileNetV2 (KerasMultiResolutionFeatureMaps +
        ConvolutionalBoxPredictor) -- below.
      * FPNLite (KerasFpnTopDownFeatureMaps + WeightSharedConvolutionalBox
        Predictor) -- see ``_quantize_fpn_detection_head``.

    In both, the feature generator uses the full scheme (same pin placement as
    the backbone) and the box predictor is always weight-only.
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
        fold_model(rebuild_feature_map_generator_functional(fmg, feature_specs)),
        per_channel=per_channel,
    )
    fe.feature_map_generator = FMGAdapter(qfmg, out_keys)

    qbp = _quantize_weights_only(
        rebuild_box_predictor_functional(detection_model._box_predictor, feature_shapes)
    )
    detection_model._box_predictor = BoxPredictorAdapter(qbp, len(feature_shapes))

    return detection_model


# ---------------------------------------------------------
# Plain SSD MobileNetV2 head (KerasMultiResolutionFeatureMaps +
# ConvolutionalBoxPredictor).
# ---------------------------------------------------------


def rebuild_feature_map_generator_functional(fmg, feature_specs):
    """
    Functionally reconstruct a KerasMultiResolutionFeatureMaps, reusing its
    converged layers (weights preserved). ``feature_specs`` is an OrderedDict
    {backbone_feature_key: TensorShape} for the inputs it consumes.

    ReLU6 ``Lambda`` layers are swapped for an explicit ``keras.layers.ReLU``
    (same name, no weights) so ``fold_model`` can pre-fold them into the conv
    (a conv + Lambda(relu6) does NOT fuse in TFLite -- it leaves a
    dequant->relu6->quant sandwich the NPU delegate cannot consume).
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
                if isinstance(layer, tf.keras.layers.Lambda) and is_relu6(layer):
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
        # NB: no `strict=` -- under @tf.function autograph rewrites `zip` into its
        # own `zip_`, which does not accept Python 3.10's strict keyword.
        return collections.OrderedDict(zip(self.output_keys, outs))  # noqa: B905


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
        for layer in box_predictor._shared_nets[i]:  # empty unless a tower is configured
            x = layer(x)
        bh = box_predictor._prediction_heads[BOX_ENCODINGS][i]
        b = x
        for layer in bh._box_encoder_layers:
            b = (
                _clone_conv_unique(layer, f"BoxEncodingPredictor_{i}")
                if isinstance(layer, _CONV)
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
                if isinstance(layer, _CONV)
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


# =========================================================
# FPNLite head QAT (SSD MobileNetV2 FPN).
#
# The FPN head is structurally different from the plain SSD head:
#   * feature generation is a top-down FPN (KerasFpnTopDownFeatureMaps:
#     projections + nearest-neighbour upsample + residual ADD + smoothing convs)
#     followed by extra "coarse" stride-2 layers;
#   * the box predictor is a WeightSharedConvolutionalBoxPredictor (one shared
#     tower + shared box/class heads applied to every feature map, per-level BN).
# Both use SeparableConv2D everywhere, which our fold/quant only handle once
# split into DepthwiseConv2D + Conv2D(1x1).
#
# The WHOLE post-backbone head (generator + coarse blocks + weight-shared box
# predictor) is rebuilt as ONE functional model and quantized in ONE pass. This
# is required for the per-TENSOR scheme: a feature-map ReLU6 that fans out to the
# internal RESIZE *and* to the box predictor must be interior to a single
# quantize_apply graph, otherwise (across a separate-model boundary) conv+ReLU6
# cannot fuse and TFLite emits a stray float-ReLU6 island (DEQUANTIZE -> ReLU6 ->
# QUANTIZE per consumer) the NPU delegate cannot consume. One combined graph lets
# it fuse: 0 stray quant/dequant AND (per-tensor) 0 per-channel weight tensors.
# Per-channel converts equally cleanly through the same graph.
#
# Installation: the meta-arch calls extract_features (-> feature_maps) and the
# box predictor as two separate steps, but the combined model must run ONCE. So
# the generator adapter runs it and caches all outputs; the coarse + box-predictor
# adapters return the cached tensors.
# =========================================================


def _split_separable_conv(sep, tag):
    """
    Split a SeparableConv2D into (DepthwiseConv2D no-bias, Conv2D 1x1 +bias),
    weights copied. Our fold/quant primitives are Conv2D/DepthwiseConv2D only
    (TFLite splits separables the same way). ``tag`` makes the names unique --
    the FPN / weight-shared graphs reuse one separable across feature maps, and
    the per-map BatchNorm folds into a distinct Conv2D each, so they cannot share.
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


def _apply(layer, x, counter, conv_sink=None):
    """
    Apply one reused head layer onto functional tensor(s), fold/quant/trace
    friendly: SeparableConv2D -> split into DepthwiseConv2D + Conv2D(1x1); ReLU6
    -> keras ReLU6 (so fold_model pre-folds it and TFLite fuses); other Lambda
    (nearest-neighbour upsample) -> freshly-named copy. When ``conv_sink`` is
    given, the Conv2D / DepthwiseConv2D names touched here are recorded (used to
    force the box-predictor convs weight-only).
    """
    if isinstance(layer, tf.keras.layers.SeparableConv2D):
        dw, pw = _split_separable_conv(layer, next(counter))
        if conv_sink is not None:
            conv_sink.update({dw.name, pw.name})
        return pw(dw(x))
    if is_relu6(layer):
        return tf.keras.layers.ReLU(max_value=6.0, name=f"relu6_{next(counter)}")(x)
    if isinstance(layer, tf.keras.layers.Lambda):
        return tf.keras.layers.Lambda(
            layer.function, name=f"{layer.name}_{next(counter)}"
        )(x)
    if isinstance(layer, _CONV) and conv_sink is not None:
        conv_sink.add(layer.name)
    return layer(x)


def _gen_body(fpn_gen, feature_items, counter):
    """Replay KerasFpnTopDownFeatureMaps onto ``feature_items`` (list of
    (key, tensor)); return (ordered_keys, ordered_tensors) in min..max level
    order."""
    top_down = feature_items[-1][1]
    for layer in fpn_gen.top_layers:
        top_down = _apply(layer, top_down, counter)
    outs = [top_down]
    keys = [f"top_down_{feature_items[-1][0]}"]
    num_levels = len(feature_items)
    for index, level in enumerate(reversed(list(range(num_levels - 1)))):
        residual = feature_items[level][1]
        top_down = outs[-1]
        for layer in fpn_gen.residual_blocks[index]:
            residual = _apply(layer, residual, counter)
        for layer in fpn_gen.top_down_blocks[index]:
            top_down = _apply(layer, top_down, counter)
        for layer in fpn_gen.reshape_blocks[index]:
            top_down = _apply(layer, [residual, top_down], counter)
        top_down = tf.keras.layers.Add(name=f"fpn_add_{next(counter)}")(
            [top_down, residual]
        )
        for layer in fpn_gen.conv_layers[index]:
            top_down = _apply(layer, top_down, counter)
        outs.append(top_down)
        keys.append(f"top_down_{feature_items[level][0]}")
    ordered = collections.OrderedDict(reversed(list(zip(keys, outs, strict=True))))
    return list(ordered.keys()), list(ordered.values())


def _coarse_body(coarse_layers, deepest, counter):
    """Replay the extractor coarse stride-2 blocks, fed by the deepest top-down
    map then each other; return the list of coarse feature tensors."""
    last = deepest
    extra = []
    for block in coarse_layers:
        x = last
        for layer in block:
            x = _apply(layer, x, counter)
        extra.append(x)
        last = x
    return extra


def _bp_body(box_predictor, feature_tensors, counter, conv_sink):
    """Replay a WeightSharedConvolutionalBoxPredictor over the feature tensors;
    return (box_out, cls_out). Convs touched are recorded in ``conv_sink`` and
    forced weight-only (except any relu6-fed tower conv, which keeps its [0,6]
    pin via ``_has_intrinsic_relu6`` precedence in ``_quantize_full``)."""
    from object_detection.core.box_predictor import (
        BOX_ENCODINGS,
        CLASS_PREDICTIONS_WITH_BACKGROUND,
    )

    code_size = box_predictor._box_prediction_head._box_code_size
    num_class_slots = box_predictor._prediction_heads[
        CLASS_PREDICTIONS_WITH_BACKGROUND
    ]._num_class_slots

    box_out, cls_out = [], []
    for i, x0 in enumerate(feature_tensors):
        x = x0
        for layer in box_predictor._additional_projection_layers[i]:
            x = _apply(layer, x, counter, conv_sink)
        for layer in box_predictor._base_tower_layers_for_heads[BOX_ENCODINGS][i]:
            x = _apply(layer, x, counter, conv_sink)
        tower = x  # shared between box and class heads (share_prediction_tower)

        b = tower
        for layer in box_predictor._box_prediction_head._box_encoder_layers:
            b = _apply(layer, b, counter, conv_sink)
        box_out.append(
            tf.keras.layers.Reshape((-1, code_size), name=f"ws_box_reshape_{i}")(b)
        )

        c = tower
        for layer in box_predictor._prediction_heads[
            CLASS_PREDICTIONS_WITH_BACKGROUND
        ]._class_predictor_layers:
            c = _apply(layer, c, counter, conv_sink)
        cls_out.append(
            tf.keras.layers.Reshape((-1, num_class_slots), name=f"ws_cls_reshape_{i}")(c)
        )
    return box_out, cls_out


def _build_combined_fpn_functional(detection_model, image_size):
    """
    Build ONE functional model: backbone FPN-input feature maps ->
    [feature_maps..., box_out..., cls_out...]. Returns
    (model, top_down_keys, num_coarse, num_maps, box_predictor_conv_names).
    """
    from object_detection.utils import ops as od_ops

    fe = detection_model.feature_extractor
    pp, _ = detection_model.preprocess(
        tf.zeros([1, image_size, image_size, 3], dtype=tf.float32)
    )
    backbone_feats = fe.classification_backbone(
        od_ops.pad_to_multiple(pp, fe._pad_to_multiple)
    )

    start = len(fe._feature_blocks) - fe._num_levels
    keys = [
        fe._feature_blocks[level - 2]
        for level in range(fe._fpn_min_level, fe._base_fpn_max_level + 1)
    ]
    feature_specs = collections.OrderedDict(
        (k, backbone_feats[start + i].shape) for i, k in enumerate(keys)
    )

    inp = collections.OrderedDict(
        (k, tf.keras.Input(shape=tuple(v.as_list()[1:]), name=k.replace("/", "__")))
        for k, v in feature_specs.items()
    )
    counter = itertools.count()
    feature_items = list(inp.items())

    td_keys, td_maps = _gen_body(fe._fpn_features_generator, feature_items, counter)
    coarse_maps = _coarse_body(fe._coarse_feature_layers, td_maps[-1], counter)
    feature_maps = td_maps + coarse_maps

    bp_convs = set()
    box_out, cls_out = _bp_body(
        detection_model._box_predictor, feature_maps, counter, bp_convs
    )

    model = tf.keras.Model(list(inp.values()), feature_maps + box_out + cls_out)
    return model, td_keys, len(coarse_maps), len(feature_maps), bp_convs


class _CombinedFpnHead:
    """Runs the combined model once and caches (feature_maps, box, cls) so the
    generator / coarse / box-predictor adapters can each return their slice from
    a single graph evaluation."""

    def __init__(self, qmodel, top_down_keys, num_coarse, num_maps):
        self.q = qmodel
        self.top_down_keys = list(top_down_keys)
        self.num_coarse = num_coarse
        self.num_maps = num_maps
        self._cache = None

    def run(self, feats_by_input_order):
        outs = self.q(feats_by_input_order)
        outs = list(outs) if isinstance(outs, (list, tuple)) else [outs]
        maps = outs[: self.num_maps]
        rest = outs[self.num_maps :]
        half = len(rest) // 2
        box, cls = rest[:half], rest[half:]
        self._cache = (maps, box, cls)
        return maps, box, cls


class _GenAdapter:
    """Drop-in for ``feature_extractor._fpn_features_generator``: runs the
    combined model (caching everything) and returns just the top-down maps."""

    def __init__(self, head):
        self.head = head

    def __call__(self, image_features):
        san = {k.replace("/", "__"): v for k, v in image_features}
        feats = [san[name] for name in self.head.q.input_names]
        maps, _box, _cls = self.head.run(feats)
        n_td = len(self.head.top_down_keys)
        return collections.OrderedDict(
            zip(self.head.top_down_keys, maps[:n_td])  # noqa: B905
        )


class _CoarseBlockAdapter:
    """Drop-in for one ``feature_extractor._coarse_feature_layers`` entry:
    returns the cached coarse map (the combined model already computed it)."""

    def __init__(self, head, index):
        self.head = head
        self.index = index

    def __call__(self, x):
        n_td = len(self.head.top_down_keys)
        return self.head._cache[0][n_td + self.index]


class _BoxPredictorAdapter(tf.keras.layers.Layer):
    """Drop-in for ``_box_predictor``: returns the cached box/class tensors as
    the {BOX_ENCODINGS, CLASS_PREDICTIONS_WITH_BACKGROUND} dict."""

    def __init__(self, head, **kw):
        super().__init__(**kw)
        self.head = head
        self.is_keras_model = True

    def call(self, image_features):
        from object_detection.core.box_predictor import (
            BOX_ENCODINGS,
            CLASS_PREDICTIONS_WITH_BACKGROUND,
        )

        _maps, box, cls = self.head._cache
        return {
            BOX_ENCODINGS: list(box),
            CLASS_PREDICTIONS_WITH_BACKGROUND: list(cls),
        }


def _quantize_fpn_detection_head(detection_model, image_size, *, per_channel=False):
    """
    FPNLite head QAT via ONE combined functional model + ONE quantize_apply.
    Call AFTER the backbone has been folded + quantized. See the section banner.
    """
    fe = detection_model.feature_extractor
    model, td_keys, num_coarse, num_maps, bp_convs = _build_combined_fpn_functional(
        detection_model, image_size
    )
    folded = fold_model(model)

    # Box-predictor tower convs fold their per-level BatchNorm into a "_folded"
    # conv, so the post-fold name may carry that suffix.
    weight_only = set(bp_convs) | {f"{n}_folded" for n in bp_convs}

    q = _quantize_full(folded, per_channel=per_channel, weight_only_names=weight_only)

    head = _CombinedFpnHead(q, td_keys, num_coarse, num_maps)
    fe._q_combined_head = q  # track variables for conversion
    fe._fpn_features_generator = _GenAdapter(head)
    fe._coarse_feature_layers = [
        [_CoarseBlockAdapter(head, i)] for i in range(num_coarse)
    ]
    detection_model._box_predictor = _BoxPredictorAdapter(head)
    return detection_model
