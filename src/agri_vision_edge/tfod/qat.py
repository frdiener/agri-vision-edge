"""
Quantization-aware training for folded TFOD SSD / SSD-FPNLite MobileNetV2.

The deployment targets require an int8 TFLite graph. Before QAT, every
functional subgraph is folded with `folding.fold_model`: BatchNorm parameters
are absorbed into their preceding convolution and a following ReLU6 is moved
into that convolution's intrinsic `activation`. Consequently, the folded
Conv2D / DepthwiseConv2D output is the post-ReLU6 tensor that TFLite may fuse
into the corresponding convolution operator.

`quantize_backbone` applies the common convolutional QAT scheme. Conv2D and
DepthwiseConv2D layers are annotated throughout; residual `Add` outputs are
also annotated for the per-channel target. Detection heads are rebuilt as
functional graphs before applying the same machinery, because TFOD's original
head components are subclassed models whose layers cannot safely be replaced
in place for SavedModel/TFLite export.

`per_channel` selects the intended *deployment weight granularity*, not the
granularity of the QAT fake-quant variables. In both modes,
`BaseQuantConfig` uses per-tensor, symmetric weight fake-quant. Per-channel
TFLite weights are deliberately left to the converter: weight-only QAT layers,
combined with `_experimental_disable_per_channel=False`, allow conversion to
choose per-channel weight quantization without inserting per-channel
fake-quant nodes into the training graph.

The two modes differ primarily in whether QAT explicitly constrains the output
of a folded ReLU6 convolution:

```
* ``per_channel=False`` (per-tensor deployment target, e.g. the i.MX8M
  Plus path): a convolution carrying intrinsic ``tf.nn.relu6`` receives
  ``ReLU6ConvQuantConfig``. Its output is fake-quantized to the fixed
  interval ``[0, 6]``. This explicitly establishes the post-ReLU6
  activation quantization used by the per-tensor QAT path and avoids
  relying on converter calibration for that tensor.

* ``per_channel=True`` (per-channel deployment target, e.g. Ethos-U):
  a convolution carrying intrinsic ``tf.nn.relu6`` receives
  ``FreeOutputConvQuantConfig``. Its weights are fake-quantized during
  QAT, but its output is not given a QAT output fake-quantizer. During
  TFLite conversion, calibration selects the activation quantization
  parameters and may emit per-channel weights. The intrinsic ReLU6 still
  constrains the real-valued activation to ``[0, 6]``, but it is not an
  explicit fixed-QAT pin and does not by itself guarantee
  ``scale == 6 / 255`` or ``zero_point == -128``.

  Leaving the output free (rather than pinning [0, 6] as the per-tensor
  scheme does) is required, not merely preferred, and was validated on the
  combined FPN graph (experiments/fpn_qat_probe,
  out/PIN_VS_FREE_FINDINGS.md): an explicit [0, 6] output pin here CRASHES
  the FPN export (native abort in flatbuffer_export) under BOTH the legacy
  and the new converter/quantizer, and -- even where it converts -- makes
  each relu6-fed conv a self-contained per-tensor op, which drops
  per-channel weight emission to zero. Per-channel weights + per-tensor
  int8 activations are otherwise the intended, working representation (the
  free path emits per-channel weights with no stray requant and aligned
  concat inputs); the incompatibility is specifically the fixed-pin /
  per-channel converter interaction, not per-channel weights themselves.
```

Linear box-predictor convolutions are also kept output-free. Their outputs feed
reshape/concat paths across feature-map levels, where independently fixed
output scales would require requantization to make concat inputs compatible.
Leaving these outputs free lets conversion choose compatible scales for the
prediction path while retaining QAT-trained weights.

For the same reason, graph boundaries matter. Tensors passed between separately
quantized functional submodels may acquire incompatible quantization domains or
extra Quantize/Dequantize/Requantize operations at export. Where a feature-map
tensor is consumed by multiple head components, the preferred representation is
one combined functional graph folded and passed through `quantize_apply` once.
This keeps the relevant producer, consumers, and concat paths in a single QAT
graph.

Per-channel fake-quant nodes are intentionally avoided. In this conversion
pipeline, baking `fake_quant_with_min_max_vars_per_channel` into the model
graph interferes with activation calibration and has previously produced
near-zero collected activation ranges and severe AP loss. Therefore,
`per_channel` changes output-pin placement and permits converter-side
per-channel weight emission; it does not request per-channel fake-quant during
training.

With a representative dataset normalized to `[-1, 1]`, plain PTQ can already
remain close to floating-point accuracy. QAT is used here chiefly to make the
exported int8 graph and its activation ranges more stable and deployment-safe,
rather than as a prerequisite for recovering baseline accuracy.
"""

import collections
import contextlib
import itertools

import numpy as np
import tensorflow as tf
import tensorflow_model_optimization as tfmot
from object_detection.core.freezable_batch_norm import FreezableBatchNorm
from tensorflow.keras.utils import register_keras_serializable

from agri_vision_edge.tfod.folding import fold_model, is_relu6

_CONV = (tf.keras.layers.Conv2D, tf.keras.layers.DepthwiseConv2D)


# ---------------------------------------------------------------------------
# Validation hook (default OFF -- does not affect production export).
#
# Normally the per-channel scheme leaves relu6-fed conv outputs FREE (calibrated
# by the converter) so TFLite may emit per-channel weights, and pins a fixed
# [0, 6] output only in the per-tensor scheme. This flag lets a probe force the
# explicit ReLU6ConvQuantConfig ([0, 6] pin) on intrinsic-relu6 convs *even in
# per_channel mode*, to empirically test whether an explicit activation pin is
# compatible with converter-emitted per-channel weights. Toggle only via the
# ``force_relu6_pin_in_per_channel`` context manager below; leave False for
# real exports.
# ---------------------------------------------------------------------------
_FORCE_RELU6_PIN_IN_PER_CHANNEL = False


@contextlib.contextmanager
def force_relu6_pin_in_per_channel(enabled: bool = True):
    """Temporarily pin relu6-fed conv outputs to a fixed [0, 6] even under the
    per-channel scheme (validation/probe use only). Restores the previous value
    on exit. See ``_FORCE_RELU6_PIN_IN_PER_CHANNEL``."""
    global _FORCE_RELU6_PIN_IN_PER_CHANNEL
    previous = _FORCE_RELU6_PIN_IN_PER_CHANNEL
    _FORCE_RELU6_PIN_IN_PER_CHANNEL = enabled
    try:
        yield
    finally:
        _FORCE_RELU6_PIN_IN_PER_CHANNEL = previous


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
          (<= 6). A fixed [0,6] output pin cannot be used here -- validated
          empirically on the combined FPN graph (see
          experiments/fpn_qat_probe/out/PIN_VS_FREE_FINDINGS.md):
            * TFMOT cannot quantize tf.nn.relu6 as an activation; the pin can
              only live on the conv OUTPUT.
            * That explicit output pin + per-channel weights CRASHES the FPN
              export (native SIGABRT in flatbuffer_export) under BOTH the legacy
              AND the new quantizer (``_experimental_new_quantizer`` False/True).
            * Even where it converts (backbone alone, no FPN fanout), the pin
              defeats per-channel weight emission entirely (104 -> 0 per-channel
              weight tensors): an explicit output quantizer makes the conv a
              self-contained per-tensor op, so the converter emits PER-TENSOR
              weights. Explicit output pin and per-channel weights are mutually
              exclusive on the same conv -- a QAT/converter interaction limit,
              not a limit of per-channel weight quantization.
          Hence per-channel keeps free outputs; the exact (6/255, -128) pin is
          reserved for the per-TENSOR scheme (ReLU6ConvQuantConfig), which wants
          per-tensor weights anyway.

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
    # Per-tensor always pins relu6 to [0, 6]. Per-channel normally leaves it free
    # (so TFLite can emit per-channel weights); the validation hook can force the
    # [0, 6] pin in per-channel mode to test converter compatibility.
    pin_relu6 = (not per_channel) or _FORCE_RELU6_PIN_IN_PER_CHANNEL
    relu6_conv_cfg = ReLU6ConvQuantConfig() if pin_relu6 else FreeOutputConvQuantConfig()
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


def quantize_detection_head(detection_model, image_size, *, per_channel=False):
    """
    Quantize the detection head in place via weight-preserving functional
    rebuilds, so QAT covers the whole graph up to the postprocess. Call AFTER the
    backbone has been folded + quantized.

    ``per_channel`` must match the backbone. Dispatches on the head architecture:

      * plain SSD MobileNetV2 (KerasMultiResolutionFeatureMaps +
        ConvolutionalBoxPredictor) -- see ``_quantize_ssd_detection_head``.
      * FPNLite (KerasFpnTopDownFeatureMaps + WeightSharedConvolutionalBox
        Predictor) -- see ``_quantize_fpn_detection_head``.

    In both, the whole post-backbone head (feature generator + box/class
    predictor) is rebuilt as ONE functional model and quantized in ONE
    quantize_apply pass (full scheme; box/class predictor convs weight-only),
    so no QAT/model boundary is left between the generator and the predictor.
    """
    fe = detection_model.feature_extractor
    if hasattr(fe, "_fpn_features_generator"):
        return _quantize_fpn_detection_head(
            detection_model, image_size, per_channel=per_channel
        )

    return _quantize_ssd_detection_head(
        detection_model, image_size, per_channel=per_channel
    )


# =========================================================
# Plain SSD MobileNetV2 head QAT (KerasMultiResolutionFeatureMaps +
# ConvolutionalBoxPredictor).
#
# Like the FPNLite head, the WHOLE post-backbone head (feature-map generator +
# box/class predictor) is rebuilt as ONE functional model and quantized in ONE
# quantize_apply pass. Previously the generator and predictor were quantized as
# two separate models, leaving a QAT/model boundary at every generated feature
# map. In the per-channel scheme the relu6-fed convs have free-calibrated
# outputs, so conversion could assign incompatible activation-quant domains on
# the two sides of such a boundary and insert stray QUANTIZE / DEQUANTIZE /
# requant nodes the NPU delegate cannot consume. One combined graph keeps the
# feature-map producer, its consumers, and the reshape/CONCAT prediction path
# interior to a single quantize_apply, so the converter aligns compatible scales
# through the concat while still emitting per-channel weights.
#
# Installation mirrors the FPN adapters: the meta-arch calls extract_features
# (-> feature_maps) and the box predictor as two separate steps, but the
# combined model must run ONCE. The generator adapter runs it and caches
# (feature_maps, box, cls); the box-predictor adapter returns the cached
# box/class tensors.
# =========================================================


def _ssd_bp_body(box_predictor, feature_tensors, conv_sink):
    """Replay a ConvolutionalBoxPredictor over the feature tensors; return
    (box_out, cls_out). Box/class heads are one 1x1 conv each (no BN) followed by
    a reshape; the convs share names across heads, so each is cloned to a unique
    name. Every conv touched (optional shared tower + box/class heads) is
    recorded in ``conv_sink`` and forced weight-only, so the reshape/CONCAT
    prediction path can share one scale (any relu6-fed tower conv keeps its
    intrinsic-relu6 handling via ``_has_intrinsic_relu6`` precedence in
    ``_quantize_full``)."""
    from object_detection.core.box_predictor import (
        BOX_ENCODINGS,
        CLASS_PREDICTIONS_WITH_BACKGROUND,
    )

    box_out, cls_out = [], []
    for i, x0 in enumerate(feature_tensors):
        x = x0
        for layer in box_predictor._shared_nets[i]:  # empty unless a tower is configured
            if isinstance(layer, _CONV):
                conv_sink.add(layer.name)
            x = layer(x)

        bh = box_predictor._prediction_heads[BOX_ENCODINGS][i]
        b = x
        for layer in bh._box_encoder_layers:
            if isinstance(layer, _CONV):
                layer = _clone_conv_unique(layer, f"BoxEncodingPredictor_{i}")
                conv_sink.add(layer.name)
            b = layer(b)
        box_out.append(
            tf.keras.layers.Reshape(
                (-1, 1, bh._box_code_size), name=f"box_reshape_{i}"
            )(b)
        )

        ch = box_predictor._prediction_heads[CLASS_PREDICTIONS_WITH_BACKGROUND][i]
        c = x
        for layer in ch._class_predictor_layers:
            if isinstance(layer, _CONV):
                layer = _clone_conv_unique(layer, f"ClassPredictor_{i}")
                conv_sink.add(layer.name)
            c = layer(c)
        cls_out.append(
            tf.keras.layers.Reshape((-1, ch._num_class_slots), name=f"cls_reshape_{i}")(
                c
            )
        )
    return box_out, cls_out


def _build_combined_ssd_functional(detection_model, image_size):
    """
    Build ONE functional model for the plain SSD head: backbone feature inputs ->
    feature-map generator -> box/class predictor -> [feature_maps..., box_out...,
    cls_out...]. Converged layers are reused (weights preserved); ReLU6 Lambdas
    are swapped for explicit ``keras.layers.ReLU`` so ``fold_model`` can pre-fold
    them into the conv (a conv + Lambda(relu6) does NOT fuse in TFLite -- it
    leaves a dequant->relu6->quant sandwich the NPU delegate cannot consume).

    Returns (model, feature_map_keys, num_feature_maps, box_predictor_conv_names).
    """
    from object_detection.utils import ops as od_ops

    fe = detection_model.feature_extractor
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
    # Feature-map output keys, in the order KerasMultiResolutionFeatureMaps emits.
    out_keys = list(fmg({k: tf.zeros(v) for k, v in feature_specs.items()}).keys())

    inp = collections.OrderedDict(
        (k, tf.keras.Input(shape=tuple(v.as_list()[1:]), name=k.replace("/", "__")))
        for k, v in feature_specs.items()
    )

    # Feature-map generator body (KerasMultiResolutionFeatureMaps).
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
    feature_maps = fmaps  # aligned with out_keys

    bp_convs = set()
    box_out, cls_out = _ssd_bp_body(
        detection_model._box_predictor, feature_maps, bp_convs
    )

    model = tf.keras.Model(list(inp.values()), feature_maps + box_out + cls_out)
    return model, out_keys, len(feature_maps), bp_convs


class _CombinedSsdHead:
    """Runs the combined SSD model once and caches (feature_maps, box, cls) so the
    generator and box-predictor adapters can each return their slice from a
    single graph evaluation."""

    def __init__(self, qmodel, feature_map_keys, num_maps):
        self.q = qmodel
        self.feature_map_keys = list(feature_map_keys)
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


class _SsdGenAdapter:
    """Drop-in for ``feature_extractor.feature_map_generator``: runs the combined
    model (caching everything) and returns just the feature maps as the
    meta-arch's dict-in / OrderedDict-out contract expects."""

    def __init__(self, head):
        self.head = head

    def __call__(self, image_features):
        san = {k.replace("/", "__"): v for k, v in image_features.items()}
        feats = [san[name] for name in self.head.q.input_names]
        maps, _box, _cls = self.head.run(feats)
        # NB: no `strict=` -- under @tf.function autograph rewrites `zip` into its
        # own `zip_`, which does not accept Python 3.10's strict keyword.
        return collections.OrderedDict(
            zip(self.head.feature_map_keys, maps)  # noqa: B905
        )


class _SsdBoxPredictorAdapter(tf.keras.layers.Layer):
    """Drop-in for ``_box_predictor``: returns the cached box/class tensors as
    the {BOX_ENCODINGS, CLASS_PREDICTIONS_WITH_BACKGROUND} dict (the combined
    model already computed them when the generator adapter ran)."""

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


def _quantize_ssd_detection_head(detection_model, image_size, *, per_channel=False):
    """
    Plain SSD head QAT via ONE combined functional model + ONE quantize_apply.
    Call AFTER the backbone has been folded + quantized. See the section banner.
    """
    fe = detection_model.feature_extractor
    model, out_keys, num_maps, bp_convs = _build_combined_ssd_functional(
        detection_model, image_size
    )
    folded = fold_model(model)

    # Box/class predictor convs are linear (no BN), so no "_folded" suffix; the
    # variant is included defensively in case a configured tower conv folds BN.
    weight_only = set(bp_convs) | {f"{n}_folded" for n in bp_convs}

    q = _quantize_full(folded, per_channel=per_channel, weight_only_names=weight_only)

    head = _CombinedSsdHead(q, out_keys, num_maps)
    fe._q_combined_head = q  # track variables for conversion
    fe.feature_map_generator = _SsdGenAdapter(head)
    detection_model._box_predictor = _SsdBoxPredictorAdapter(head)
    return detection_model


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
# Keeping the head in one quantized functional graph avoids quantization-domain
# boundaries between feature generation and prediction, which is particularly
# important for per-channel conversion where several outputs intentionally remain
# free-calibrated.
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
