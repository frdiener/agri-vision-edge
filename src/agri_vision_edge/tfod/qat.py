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
granularity of the QAT fake-quant variables: `BaseQuantConfig` always uses
per-tensor, symmetric weight fake-quant. Per-channel TFLite weights are left to
the converter, which emits them for a weight-only-annotated conv when
`_experimental_disable_per_channel=False`.

Crucially, the pin placement is therefore an EXPORT-TIME choice. Note the
consequence: ``per_channel`` no longer changes the TRAINING graph at all -- a
per-channel run and a per-tensor run currently train the identical graph, and
differ only in how their checkpoint is exported. (What would legitimately differ
is per-AXIS weight fake-quant, matching the grid per-channel actually deploys;
see ``DepthwisePerAxisQuantizer``.)

```
* TRAINING (``for_export=False``, both granularities): a convolution
  carrying an intrinsic ``tf.nn.relu6`` receives ``ReLU6ConvQuantConfig``,
  pinning its output to the fixed interval ``[0, 6]``. QAT therefore
  simulates the int8 activation the model actually deploys with, which is
  what makes the exported graph track the trained one (measured: per-layer
  correlation 0.999 through the backbone).

* EXPORT for ``per_channel=True``: the same graph is rebuilt WEIGHT-ONLY
  throughout -- every conv output quantizer dropped, including the relu6
  pins, the signed-conv ranges and the Add pins. One QAT checkpoint
  restores into either graph (the pins are stateless; the dropped range
  variables are simply left unused). The intrinsic ReLU6 still clamps the
  real activation to ``[0, 6]``, so the converter calibrates a range
  bounded by 6.

  Weight-only *throughout* is the point. A calibrated export re-derives
  EVERY activation range, so any QAT-trained range left in the graph is
  overridden anyway -- and the weights were tuned against it, so a partial
  set is worse than none. Handing all of them to calibration makes the
  export self-consistent, and it is also what frees the converter to emit
  per-channel weights.

  Dropping the pin at export is required, not merely preferred, and was
  validated on the combined FPN graph (experiments/fpn_qat_probe,
  out/PIN_VS_FREE_FINDINGS.md): an explicit [0, 6] output pin CRASHES the
  FPN export (native abort in flatbuffer_export) under BOTH the legacy and
  the new converter/quantizer, and -- even where it converts -- makes each
  relu6-fed conv a self-contained per-tensor op, which drops per-channel
  weight emission to zero. Per-channel weights + per-tensor int8
  activations are otherwise the intended, working representation.
```

Getting there took two corrections, both of which come down to keeping the
export self-consistent. Pinning during training used to be tied to the
per-tensor target, leaving the per-channel graph with no activation simulation
at all: it trained against float activations. And the export kept a SUBSET of
the QAT-trained ranges (signed convs, Add outputs) that calibration then
overrode, so the weights deployed against ranges they were never tuned for.
Measured on ssd-mn2-fpnlite_mc_phenobench-tiled_320, one pin-trained checkpoint:

    free-trained, partial ranges kept (old)   AP 0.3529
    pin-trained, partial ranges kept          AP 0.3970
    pin-trained, weight-only export           AP 0.4526   <- per-channel
    per-tensor export of the same checkpoint  AP 0.4499
    int8 PTQ per-channel                      AP 0.4406
    FP32                                      AP 0.4464

An alternative would be to never calibrate, so the trained ranges deploy as-is.
That needs the graph fully QDQ-specified: per-AXIS weight fake-quant (so the
per-channel grid comes from the graph rather than the converter), pinned relu6 /
Add outputs, and a pinned input range. It is implemented behind
``fully_quantized`` and is UNUSABLE on this converter (TF 2.11) -- but note that
it is the OUTPUT pins that it rejects, not per-axis weights. Probed on the
combined FPN graph with float weights and observers over 196 representative
images:

    with the fully-QDQ output pins
      per-axis weights, Conv2D + DepthwiseConv2D  -> exported graph is EMPTY
      per-axis weights, Conv2D only               -> SIGABRT in flatbuffer_export
      per-tensor weights                          -> SIGABRT in flatbuffer_export

    with the weight-only export (the working one)
      per-tensor weights (current)                -> converts, 214 per-channel
      per-axis weights, Conv2D only               -> converts, 214 per-channel
      per-axis weights, Conv2D + DepthwiseConv2D  -> exported graph is EMPTY

The SIGABRT rows reproduce the crash recorded in
experiments/fpn_qat_probe/out/PIN_VS_FREE_FINDINGS.md: explicit output
quantization and per-channel weights cannot be combined here, which is why the
weight-only export is not a workaround for a missing feature but the
representation this converter actually supports.

Per-axis WEIGHT fake-quant, on the other hand, converts fine in that export --
see ``DepthwisePerAxisQuantizer`` for the one case that does not, and why it is
worth fixing.

Linear box-predictor convolutions are also kept output-free. Their outputs feed
reshape/concat paths across feature-map levels, where independently fixed
output scales would require requantization to make concat inputs compatible.
Leaving these outputs free lets conversion choose compatible scales for the
prediction path while retaining QAT-trained weights.

For the same reason, graph boundaries matter. Tensors passed between separately
quantized functional submodels may acquire incompatible quantization domains or
extra Quantize/Dequantize/Requantize operations at export. Where a feature-map
tensor is consumed by multiple components, the preferred representation is one
combined functional graph folded and passed through `quantize_apply` once. This
keeps the relevant producer, consumers, and concat paths in a single QAT graph.

`quantize_detection_model` is the single self-contained entry point (callers do
not pre-fold / pre-quantize the backbone). It dispatches on architecture:

  * FPNLite folds + quantizes the backbone as its own graph, then rebuilds the
    whole post-backbone head (FPN generator + coarse blocks + weight-shared box
    predictor) as one combined functional graph. Its backbone taps are signed /
    Add outputs, not free-relu6 tensors, so the backbone/head boundary is clean.

  * plain SSD INLINES the backbone with the head into a single full-model
    functional graph (image -> backbone -> feature-map generator -> box/class
    predictor), folded and quantized in one pass. This is required because the
    SSD tap `layer_15/expansion_output` is `block_13_expand`'s ReLU6 output,
    which is DUAL-USE (it feeds the backbone's own `block_13_depthwise` and is
    exported as a feature map). In the per-channel scheme relu6 outputs are
    free-calibrated; a separately quantized backbone and head would calibrate
    that shared tensor independently, pick mismatched (scale, zero_point), and
    force the converter to dequantize + recompute the conv in float + requantize
    per consumer -- stray ops the NPU delegate cannot consume. One full-model
    graph makes the tap interior, so it is calibrated once, both consumers share
    the scale, no stray requant appears, and per-channel weights are retained.

Per-channel fake-quant nodes are currently avoided. Baking
`fake_quant_with_min_max_vars_per_channel` into the graph alongside converter
calibration has previously produced near-zero collected activation ranges and
severe AP loss -- note that this is an interaction with calibration, so it would
have to be re-evaluated as part of the fully-QDQ (never-calibrate) scheme
sketched above, where the two no longer meet. `per_channel` therefore changes
export-time pin placement and permits converter-side per-channel weight
emission; it does not request per-channel fake-quant during training.

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

#: Range of the image tensor entering the graph. ``SSDModule.inference_fn``
#: consumes an already-normalized image, and the representative dataset and the
#: runtime both produce [-1, 1], so this is a property of the pipeline rather
#: than an estimate.
INPUT_RANGE = (-1.0, 1.0)


# ---------------------------------------------------------------------------
# Validation hook (default OFF -- does not affect production export).
#
# Training pins relu6-fed conv outputs to [0, 6] for both granularities; only
# the per-channel EXPORT rewrite frees them (so TFLite may emit per-channel
# weights). This flag lets a probe keep the explicit ReLU6ConvQuantConfig
# ([0, 6] pin) even in that rewrite, to empirically re-test whether an explicit
# activation pin is compatible with converter-emitted per-channel weights.
# Toggle only via the ``force_relu6_pin_in_per_channel`` context manager below;
# leave False for real exports.
# ---------------------------------------------------------------------------
_FORCE_RELU6_PIN_IN_PER_CHANNEL = False


@contextlib.contextmanager
def force_relu6_pin_in_per_channel(enabled: bool = True):
    """Temporarily keep the fixed [0, 6] relu6 pin in the per-channel export
    rewrite (validation/probe use only). Restores the previous value on exit.
    See ``_FORCE_RELU6_PIN_IN_PER_CHANNEL``."""
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
    Shared configuration for convolutional layers: symmetric weight fake-quant.

    ``per_axis_weights=False`` (default) keeps the weight grid per-tensor and
    leaves per-channel emission to the CONVERTER (weight-only convs +
    ``_experimental_disable_per_channel=False``). That path needs calibration,
    and calibration discards the QAT-trained activation ranges.

    ``per_axis_weights=True`` puts the per-channel grid in the graph instead, so
    the export needs no calibration at all and the trained ranges survive. See
    the module docstring.
    """

    def __init__(self, per_axis_weights: bool = False):
        # Per-axis weight fake-quant puts the per-channel weight grid in the
        # GRAPH instead of leaving it to converter calibration, which is what
        # a fully-QDQ (never-calibrated) export needs.
        self.per_axis_weights = per_axis_weights

    def _weight_quantizer(self, layer):
        if not self.per_axis_weights:
            return tfmot.quantization.keras.quantizers.LastValueQuantizer(
                num_bits=8,
                per_axis=False,
                symmetric=True,
                narrow_range=True,
            )

        quantizer = (
            DepthwisePerAxisQuantizer
            if isinstance(layer, tf.keras.layers.DepthwiseConv2D)
            else tfmot.quantization.keras.quantizers.LastValueQuantizer
        )
        return quantizer(
            num_bits=8,
            per_axis=True,
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
        return [(self._kernel(layer), self._weight_quantizer(layer))]

    def set_quantize_weights(self, layer, quantize_weights):
        if isinstance(layer, tf.keras.layers.DepthwiseConv2D):
            layer.depthwise_kernel = quantize_weights[0]
        else:
            layer.kernel = quantize_weights[0]

    def get_config(self):
        return {"per_axis_weights": self.per_axis_weights}

    @classmethod
    def from_config(cls, config):
        return cls(**config)


@register_keras_serializable()
class FreeOutputConvQuantConfig(BaseQuantConfig):
    """
    Quantize convolution weights only; leave the OUTPUT scale FREE (calibrated
    by the converter). No output/activation fake-quant node is inserted.

    Two uses, both relying on the converter -- not a fixed pin -- to set the
    output scale:

      (a) relu6-fed convs in the per-channel EXPORT rewrite (never while
          training): a free output range lets TFLite emit PER-CHANNEL weights,
          while the intrinsic ``tf.nn.relu6`` still pins the fused output MIN at
          0 (zp -128) and the converter calibrates the MAX (<= 6). A fixed [0,6]
          output pin cannot be used here -- validated empirically on the
          combined FPN graph (see
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
class FixedRangeQuantizer(tfmot.quantization.keras.quantizers.Quantizer):
    """
    A fixed, stateless fake-quant over a known interval.

    Statelessness is load-bearing: a graph carrying these pins holds exactly the
    same variables as one without them, so a checkpoint restores into either and
    the pins can be added or dropped when rebuilding the graph.
    """

    def __init__(self, min_value: float, max_value: float, num_bits: int = 8):
        self.min_value = float(min_value)
        self.max_value = float(max_value)
        self.num_bits = num_bits

    def build(self, tensor_shape, name, layer):
        return {}

    def __call__(self, inputs, training, weights, **kwargs):
        return tf.quantization.fake_quant_with_min_max_args(
            inputs,
            min=self.min_value,
            max=self.max_value,
            num_bits=self.num_bits,
            narrow_range=False,
        )

    def get_config(self):
        return {
            "min_value": self.min_value,
            "max_value": self.max_value,
            "num_bits": self.num_bits,
        }


@register_keras_serializable()
class FixedRelu6Quantizer(FixedRangeQuantizer):
    """A fixed [0, 6] fake-quant: scale = 6/255, zero_point = -128."""

    def __init__(self, num_bits: int = 8):
        super().__init__(0.0, 6.0, num_bits)

    def get_config(self):
        return {"num_bits": self.num_bits}


@register_keras_serializable()
class DepthwisePerAxisQuantizer(
    tfmot.quantization.keras.quantizers.LastValueQuantizer
):
    """
    Per-axis weight fake-quant for a DepthwiseConv2D kernel.

    ``per_axis`` quantizers quantize along the LAST axis. For a Conv2D kernel
    ``[kh, kw, in, out]`` that is the output channel, which is what TFLite
    quantizes too. A DepthwiseConv2D kernel is ``[kh, kw, in, multiplier]``, and
    TFLite quantizes it along the flattened ``in * multiplier`` -- so pointing a
    stock per-axis quantizer at one silently produces a SINGLE scale (multiplier
    is 1 here), i.e. a per-tensor grid that then disagrees with the per-channel
    weights written into the flatbuffer.

    Flattening the two trailing axes before quantizing puts the channels last,
    which makes the stock implementation correct; the kernel is reshaped back
    afterwards. Reusing ``LastValueQuantizer`` this way keeps its range-update
    and symmetric/narrow-range handling rather than reimplementing it.

    BROKEN FOR EXPORT. The reshapes around the weight constant defeat the
    converter: a graph using this exports EMPTY (0 conv ops), while the same
    export with per-axis fake-quant on Conv2D only converts normally. Do not
    enable it for depthwise layers until it is reimplemented without reshaping
    the kernel.

    Worth fixing, because per-tensor weight fake-quant currently simulates a
    COARSER grid than the per-channel one deployment uses, and MobileNetV2's
    depthwise kernels -- whose per-channel ranges vary the most -- are exactly
    where that mismatch costs the most. Note the export does not need to consume
    this fake-quant at all: in the weight-only export the converter derives the
    per-channel weight scales from the float values itself (proven by the
    per-tensor control emitting 214 per-channel tensors). The quantizer only has
    to be numerically right during TRAINING, so a straight-through estimator
    with a broadcast per-channel scale -- no reshape, no FakeQuant op -- would
    do, and would fold away at export.
    """

    @staticmethod
    def _flat_shape(shape):
        return tf.TensorShape([shape[0], shape[1], int(shape[2]) * int(shape[3])])

    def build(self, tensor_shape, name, layer):
        return super().build(self._flat_shape(tensor_shape), name, layer)

    def __call__(self, inputs, training, weights, **kwargs):
        shape = inputs.shape
        flat = tf.reshape(inputs, self._flat_shape(shape))
        quantized = super().__call__(flat, training, weights, **kwargs)
        return tf.reshape(quantized, shape)


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

    NO LONGER APPLIED. It patched a coverage gap that only existed while the
    per-channel TRAINING graph left relu6 outputs free, making the ``Add`` the
    one un-fake-quantized tensor in an inverted-residual block. Training now
    pins relu6 for both granularities, so the gap is closed at its source, and
    the export graph is calibrated end to end anyway (verified: 0 of 123 conv
    outputs fall back to float/dynamic without it).

    Retained so graphs and checkpoints written by the previous scheme still
    deserialize.
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
    "FixedRangeQuantizer": FixedRangeQuantizer,
    "FixedRelu6Quantizer": FixedRelu6Quantizer,
    "DepthwisePerAxisQuantizer": DepthwisePerAxisQuantizer,
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


def _quantize_full(
    model,
    *,
    per_channel: bool,
    weight_only_names=frozenset(),
    for_export: bool = False,
    fully_quantized: bool = False,
    input_range: tuple[float, float] | None = None,
):
    """
    One ``quantize_apply`` over a folded functional model, applying the full
    int8 scheme. Each conv is classified by what its folded output is:

      * intrinsic relu6 (absorbed by ``fold_model``)
          -> ``ReLU6ConvQuantConfig`` -- weights + fixed [0, 6] OUTPUT pin, so
             training simulates the int8 activation the model deploys with.
      * name in ``weight_only_names`` (box-predictor convs feeding a concat)
          -> ``FreeOutputConvQuantConfig`` -- free output for shared concat scale.
      * otherwise (signed / linear) -> ``SignedConvQuantConfig``.

    The per-channel EXPORT rewrite overrides all of that with
    ``FreeOutputConvQuantConfig`` everywhere; see below.

    ``weight_only_names`` lets the FPN combined graph force box-predictor convs
    weight-only; intrinsic-relu6 detection takes precedence for tower convs that
    happen to feed a relu6.
    """
    # Fully-QDQ (per-channel): every tensor carries a fake-quant, so the export
    # needs no calibration and the trained ranges are what deploy. The
    # per-channel weight grid then has to come from the graph itself.
    per_axis_weights = fully_quantized
    weight_only_cfg = FreeOutputConvQuantConfig(per_axis_weights=per_axis_weights)

    # A CALIBRATED per-channel export is weight-only THROUGHOUT: no relu6 pins,
    # no signed-conv output quantizers, no Add pins. Whichever activation ranges
    # such a graph does carry, calibration overrides them all anyway -- so
    # keeping a subset of QAT-trained ranges is the worst of both, leaving the
    # weights tuned against ranges that never deploy. Handing every range to
    # calibration instead makes the export self-consistent, and it is also what
    # frees the converter to emit per-channel weights.
    #
    # Measured on ssd-mn2-fpnlite_mc_phenobench-tiled_320, one pin-trained
    # checkpoint exported three ways:
    #     trained ranges kept, relu6 pinned  -> per-tensor weights,   AP 0.4499
    #     relu6 freed, signed/Add kept       -> per-channel weights,  AP 0.3970
    #     weight-only throughout             -> per-channel weights,  AP 0.4526
    export_weight_only = (
        per_channel
        and for_export
        and not fully_quantized
        and not _FORCE_RELU6_PIN_IN_PER_CHANNEL
    )

    signed_cfg = (
        weight_only_cfg
        if export_weight_only
        else SignedConvQuantConfig(per_axis_weights=per_axis_weights)
    )
    # The relu6 pin is what makes QAT simulate the activation quantization, so
    # it is always present while training -- for BOTH deployment granularities.
    relu6_conv_cfg = (
        weight_only_cfg
        if export_weight_only
        else ReLU6ConvQuantConfig(per_axis_weights=per_axis_weights)
    )
    # Residual Add outputs are the one tensor in an inverted-residual block with
    # no conv of its own. The calibrated export lets the converter cover them; a
    # fully-QDQ graph has to pin them, since the FPN backbone taps are Add
    # outputs and leaving them open would force calibration back on.
    add_cfg = AddOutputConfig() if fully_quantized else None

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
        quantized = tfmot.quantization.keras.quantize_apply(annotated)

    if input_range is None:
        return quantized

    return _pin_model_input(quantized, input_range)


def _pin_model_input(model, input_range: tuple[float, float]):
    """
    Fake-quantize the model's INPUT to a known fixed range.

    The image reaching the graph is normalized to [-1, 1] by construction, but
    nothing in the graph says so, and the very first convolution cannot be
    quantized without a range for its input. With calibration that range comes
    from the representative data; a fully-QDQ export has no calibration, so the
    range has to be stated. It is stateless, like the relu6 pins.
    """
    from tensorflow_model_optimization.python.core.quantization.keras import (
        quantize_layer,
    )

    inputs = tf.keras.Input(
        batch_shape=model.input_shape,
        dtype=model.inputs[0].dtype,
        name="qat_input",
    )
    pinned = quantize_layer.QuantizeLayer(
        FixedRangeQuantizer(*input_range),
        name="quant_model_input",
    )(inputs)

    # Replayed onto the new input rather than called as a nested model: calling
    # it (`model(pinned)`) leaves an opaque sub-Model whose layers the
    # SavedModel trace does not follow, and the exported graph comes out empty.
    outputs = _replay_functional(model, pinned)

    return tf.keras.Model(
        inputs,
        outputs[0] if len(outputs) == 1 else outputs,
        name=model.name,
    )


def quantize_backbone(
    backbone,
    *,
    per_channel: bool,
    for_export: bool = False,
    fully_quantized: bool = False,
):
    """
    Apply the full int8 scheme to a FOLDED MobileNetV2 backbone (or any folded
    functional feature graph). See the module docstring for the training vs
    export pin placement. Input must already be folded
    (``folding.fold_model`` / ``fold_mobilenetv2_backbone``).

    The backbone's input is the (normalized) image, so under the fully-QDQ
    scheme it is also where the model's input range is pinned.
    """
    return _quantize_full(
        backbone,
        per_channel=per_channel,
        for_export=for_export,
        fully_quantized=fully_quantized,
        input_range=INPUT_RANGE if fully_quantized else None,
    )


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


def quantize_detection_model(
    detection_model,
    image_size,
    *,
    per_channel=False,
    for_export=False,
    fully_quantized=None,
):
    """
    Quantize the detection head in place via weight-preserving functional
    rebuilds, so QAT covers the whole graph up to the postprocess. Call AFTER the
    backbone has been folded + quantized.

    Self-contained: it folds + quantizes the backbone itself, so callers pass a
    freshly built (unfolded, unquantized) detection model -- they must NOT
    pre-fold / pre-quantize ``classification_backbone``.

    ``per_channel`` selects the deployment weight granularity, and with it the
    scheme (override with ``fully_quantized`` to convert a checkpoint trained
    under the other one):

      * ``per_channel=False`` -- per-tensor weights, calibrated export. Every
        relu6-fed conv output is pinned to [0, 6] so QAT simulates the int8
        activations, and the converter fills in the rest from the
        representative dataset.

      * ``per_channel=True`` -- the CONVERTER emits the per-channel weights,
        which it only does for a conv whose output range it is free to choose.
        Hence ``for_export=True`` rebuilds the graph weight-only. Those pins are
        stateless, so the rebuild does not disturb the restore.

    ``fully_quantized=True`` opts into the alternative scheme in which the graph
    specifies its own quantization completely (per-AXIS weight grid, pinned
    relu6 / Add / input ranges) so that no calibration is needed and the trained
    ranges survive. It is IMPLEMENTED BUT NOT USABLE on this converter; see the
    module docstring for what it does instead of converting.

    Dispatches on the head architecture:

      * FPNLite (KerasFpnTopDownFeatureMaps + WeightSharedConvolutionalBox
        Predictor): the backbone is folded + quantized as its OWN graph, then the
        whole post-backbone head (FPN generator + coarse blocks + weight-shared
        box predictor) is rebuilt as ONE combined functional graph and quantized
        in ONE pass -- see ``_quantize_fpn_detection_head``. The FPN taps are not
        free-relu6 dual-use tensors, so the backbone/head boundary stays clean.

      * plain SSD MobileNetV2 (KerasMultiResolutionFeatureMaps +
        ConvolutionalBoxPredictor): the backbone is INLINED with the head into a
        single full-model functional graph, folded + quantized in ONE pass -- see
        ``_quantize_ssd_detection_head``. This is required because the SSD tap
        ``layer_15/expansion_output`` is a free-relu6 tensor used both inside the
        backbone and by the head; a separate backbone graph would calibrate it
        inconsistently and leave stray requant nodes.
    """
    if fully_quantized is None:
        # Off by default: the fully-QDQ scheme is not usable on this converter
        # (see the module docstring -- it either aborts in flatbuffer_export or
        # exports an empty graph). Kept opt-in so it can be retried on a newer
        # TFLite converter without having to rebuild it.
        fully_quantized = False

    fe = detection_model.feature_extractor
    if hasattr(fe, "_fpn_features_generator"):
        fe.classification_backbone = fold_model(fe.classification_backbone)
        fe.classification_backbone = quantize_backbone(
            fe.classification_backbone,
            per_channel=per_channel,
            for_export=for_export,
            fully_quantized=fully_quantized,
        )
        return _quantize_fpn_detection_head(
            detection_model,
            image_size,
            per_channel=per_channel,
            for_export=for_export,
            fully_quantized=fully_quantized,
        )

    return _quantize_ssd_detection_model(
        detection_model,
        image_size,
        per_channel=per_channel,
        for_export=for_export,
        fully_quantized=fully_quantized,
    )


# =========================================================
# Plain SSD MobileNetV2 QAT -- ONE full-model combined functional graph.
#
# Unlike the FPNLite head (which starts from backbone feature inputs, keeping the
# backbone a separate quantized graph), the plain SSD path folds+quantizes the
# WHOLE model -- backbone + feature-map generator + box/class predictor -- as a
# single functional graph in ONE quantize_apply pass:
#
#     padded image
#       -> MobileNetV2 backbone (inlined)  -> layer_15/expansion & layer_19 taps
#       -> feature-map generator
#       -> box/class predictor convs -> reshape
#
# Why the backbone must be INSIDE this graph (not quantized separately): the SSD
# backbone tap ``layer_15/expansion_output`` is ``block_13_expand``'s ReLU6
# output, which is DUAL-USE -- it feeds the backbone's own ``block_13_depthwise``
# AND is exported as a feature map. In the per-channel scheme relu6 outputs are
# free-calibrated; a separately-quantized backbone and head calibrate that shared
# tensor INDEPENDENTLY, pick mismatched (scale, zero_point), and the converter
# then dequantizes + recomputes ``block_13_expand`` in float + requantizes per
# consumer -> stray QUANTIZE/DEQUANTIZE. Folding+quantizing the backbone together
# with the head makes the tap interior to ONE quantize_apply, so it is calibrated
# ONCE, both consumers share the same scale, and no stray requant appears -- while
# TFLite still emits per-channel weights (the relu6-fed convs stay weight-only).
#
# Installation: the meta-arch calls extract_features -- which calls
# classification_backbone then feature_map_generator -- and then the box
# predictor, as separate steps; but the combined model must run ONCE. The
# BACKBONE adapter runs it (it is the first call, and receives the padded image =
# the combined graph's input) and caches (feature_maps, box, cls); the generator
# and box-predictor adapters return their cached slice.
# =========================================================


def _replay_functional(model, input_tensor):
    """
    Re-apply a functional Keras model's layers onto ``input_tensor``, reusing the
    original layer objects (weights preserved), and return the model's output
    tensor(s). This flattens a nested model into the surrounding functional graph
    -- needed so ``fold_model`` (which folds conv/BN/ReLU6 across ``model.layers``)
    can see the backbone's layers instead of an opaque nested Model. Mirrors
    ``fold_model``'s single-input, single-inbound-node replay; the backbone is a
    single-input graph, exactly what ``fold_mobilenetv2_backbone`` already folds.
    """
    out: dict[str, tf.Tensor] = {}
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.InputLayer):
            out[layer.name] = input_tensor
            continue
        parents = list(tf.nest.flatten(layer.inbound_nodes[0].inbound_layers))
        inputs = [out[p.name] for p in parents]
        out[layer.name] = layer(inputs[0] if len(inputs) == 1 else inputs)
    return [out[t._keras_history.layer.name] for t in model.outputs]


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
    Build ONE functional model for the WHOLE plain SSD model: padded image ->
    MobileNetV2 backbone (inlined) -> feature-map generator -> box/class predictor
    -> [feature_maps..., box_out..., cls_out...]. Converged layers are reused
    (weights preserved, BN/ReLU6 still present so ``fold_model`` folds the whole
    graph at once); ReLU6 Lambdas in the generator are swapped for explicit
    ``keras.layers.ReLU`` so folding pre-folds them into the conv (a conv +
    Lambda(relu6) does NOT fuse in TFLite -- it leaves a dequant->relu6->quant
    sandwich the NPU delegate cannot consume).

    The backbone is INLINED (not a separate quantized graph) so its dual-use tap
    ``layer_15/expansion_output`` is interior to one quantize_apply -- see the
    section banner.

    Returns (model, feature_map_keys, num_feature_maps, num_taps,
    box_predictor_conv_names).
    """
    from object_detection.utils import ops as od_ops

    fe = detection_model.feature_extractor
    fmg = fe.feature_map_generator
    backbone = fe.classification_backbone

    tap_keys = [fl for fl in fmg.feature_map_layout["from_layer"] if fl]
    pp, _ = detection_model.preprocess(
        tf.zeros([1, image_size, image_size, 3], dtype=tf.float32)
    )
    padded = od_ops.pad_to_multiple(pp, fe._pad_to_multiple)
    # Concrete run to recover the fmg output key order.
    feats = backbone(padded)
    out_keys = list(
        fmg({k: feats[i] for i, k in enumerate(tap_keys)}).keys()
    )

    # Inline the raw backbone into the combined graph (single image input).
    image_input = tf.keras.Input(
        shape=tuple(padded.shape.as_list()[1:]), name="padded_image"
    )
    taps = _replay_functional(backbone, image_input)
    tap_map = {k: taps[i] for i, k in enumerate(tap_keys)}

    # Feature-map generator body (KerasMultiResolutionFeatureMaps).
    fmaps = []
    for index, from_layer in enumerate(fmg.feature_map_layout["from_layer"]):
        if from_layer:
            fm = tap_map[from_layer]
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

    model = tf.keras.Model(image_input, feature_maps + box_out + cls_out)
    return model, out_keys, len(feature_maps), len(tap_keys), bp_convs


class _CombinedSsdHead:
    """Runs the combined full SSD model (image -> everything) once and caches
    (feature_maps, box, cls) so the backbone / generator / box-predictor adapters
    can each return their slice from a single graph evaluation."""

    def __init__(self, qmodel, feature_map_keys, num_maps):
        self.q = qmodel
        self.feature_map_keys = list(feature_map_keys)
        self.num_maps = num_maps
        self._cache = None

    def run(self, padded_image):
        outs = self.q(padded_image)
        outs = list(outs) if isinstance(outs, (list, tuple)) else [outs]
        maps = outs[: self.num_maps]
        rest = outs[self.num_maps :]
        half = len(rest) // 2
        box, cls = rest[:half], rest[half:]
        self._cache = (maps, box, cls)
        return maps, box, cls


class _SsdBackboneAdapter:
    """Drop-in for ``feature_extractor.classification_backbone``. extract_features
    calls it FIRST, with the padded image (= the combined graph's input), so it
    RUNS the full combined model once and caches (feature_maps, box, cls). It
    returns ``num_taps`` placeholder tap tensors purely to satisfy
    ``image_features[0]``/``[1]`` indexing in extract_features -- the
    feature-map-generator adapter ignores them and returns the cached maps."""

    def __init__(self, head, num_taps):
        self.head = head
        self.num_taps = num_taps

    def __call__(self, padded_image):
        maps, _box, _cls = self.head.run(padded_image)
        # The returned taps are only handed to the (cache-returning) fmg adapter,
        # so any tensors of the right count suffice; reuse a real feature map.
        return [maps[0]] * self.num_taps


class _SsdGenAdapter:
    """Drop-in for ``feature_extractor.feature_map_generator``: returns the cached
    feature maps (the backbone adapter already ran the combined model) as the
    meta-arch's OrderedDict contract expects."""

    def __init__(self, head):
        self.head = head

    def __call__(self, image_features):
        maps, _box, _cls = self.head._cache
        # NB: no `strict=` -- under @tf.function autograph rewrites `zip` into its
        # own `zip_`, which does not accept Python 3.10's strict keyword.
        return collections.OrderedDict(
            zip(self.head.feature_map_keys, maps)  # noqa: B905
        )


class _SsdBoxPredictorAdapter(tf.keras.layers.Layer):
    """Drop-in for ``_box_predictor``: returns the cached box/class tensors as
    the {BOX_ENCODINGS, CLASS_PREDICTIONS_WITH_BACKGROUND} dict (the combined
    model already computed them when the backbone adapter ran)."""

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


def _quantize_ssd_detection_model(
    detection_model,
    image_size,
    *,
    per_channel=False,
    for_export=False,
    fully_quantized=False,
):
    """
    Plain SSD QAT via ONE full-model combined functional graph (backbone + head)
    + ONE quantize_apply. Call with the ORIGINAL (unfolded, unquantized) backbone
    -- this path folds+quantizes the backbone together with the head so the
    dual-use ``layer_15/expansion_output`` tap is interior. See the section banner.
    """
    fe = detection_model.feature_extractor
    model, out_keys, num_maps, num_taps, bp_convs = _build_combined_ssd_functional(
        detection_model, image_size
    )

    # Inlining the backbone REUSES its layers, so those layers now carry two
    # inbound nodes (their original backbone-graph node + the one just created).
    # ``fold_model`` reads ``inbound_nodes[0]`` (the original), which points at the
    # backbone's own InputLayer and is not part of this combined graph. Clone the
    # combined model to fresh single-node layers (copying the trained weights) so
    # folding sees only this graph's connectivity.
    clean = tf.keras.models.clone_model(model)
    clean.set_weights(model.get_weights())
    folded = fold_model(clean)

    # Box/class predictor convs are linear (no BN), so no "_folded" suffix; the
    # variant is included defensively in case a configured tower conv folds BN.
    weight_only = set(bp_convs) | {f"{n}_folded" for n in bp_convs}

    # The plain-SSD graph is the whole model, so its input is the image.
    q = _quantize_full(
        folded,
        per_channel=per_channel,
        weight_only_names=weight_only,
        for_export=for_export,
        fully_quantized=fully_quantized,
        input_range=INPUT_RANGE if fully_quantized else None,
    )

    head = _CombinedSsdHead(q, out_keys, num_maps)
    fe._q_combined_model = q  # track variables for conversion/training
    fe.classification_backbone = _SsdBackboneAdapter(head, num_taps)
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


def _quantize_fpn_detection_head(
    detection_model,
    image_size,
    *,
    per_channel=False,
    for_export=False,
    fully_quantized=False,
):
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

    # The head's inputs are backbone taps, which are interior tensors of the
    # exported graph (and pinned there by the backbone), so no input pin here.
    q = _quantize_full(
        folded,
        per_channel=per_channel,
        weight_only_names=weight_only,
        for_export=for_export,
        fully_quantized=fully_quantized,
    )

    head = _CombinedFpnHead(q, td_keys, num_coarse, num_maps)
    fe._q_combined_head = q  # track variables for conversion
    fe._fpn_features_generator = _GenAdapter(head)
    fe._coarse_feature_layers = [
        [_CoarseBlockAdapter(head, i)] for i in range(num_coarse)
    ]
    detection_model._box_predictor = _BoxPredictorAdapter(head)
    return detection_model
