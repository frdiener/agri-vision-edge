"""Quantization-aware training for folded TFOD SSD models.

Detection heads use functional graphs to preserve TFLite graph visibility.
Per-channel export removes output pins for converter calibration.
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

#: Normalized graph input range used by inference and representative data.
INPUT_RANGE = (-1.0, 1.0)


# Test hook for retaining the ReLU6 pin during per-channel export.
_FORCE_RELU6_PIN_IN_PER_CHANNEL = False


@contextlib.contextmanager
def force_relu6_pin_in_per_channel(enabled: bool = True):
    """Temporarily retain the [0, 6] ReLU6 pin in per-channel exports."""
    global _FORCE_RELU6_PIN_IN_PER_CHANNEL
    previous = _FORCE_RELU6_PIN_IN_PER_CHANNEL
    _FORCE_RELU6_PIN_IN_PER_CHANNEL = enabled
    try:
        yield
    finally:
        _FORCE_RELU6_PIN_IN_PER_CHANNEL = previous


# Quantization configs


@register_keras_serializable()
class BaseQuantConfig(tfmot.quantization.keras.QuantizeConfig):
    """Configure symmetric convolution weight fake quantization.

    Per-axis weights are embedded in fully QDQ graphs; per-tensor weights leave
    per-channel emission to converter calibration.
    """

    def __init__(self, per_axis_weights: bool = False):
        self.per_axis_weights = per_axis_weights

    def _weight_quantizer(self, layer):
        if not self.per_axis_weights:
            return tfmot.quantization.keras.quantizers.LastValueQuantizer(
                num_bits=8,
                per_axis=False,
                symmetric=True,
                narrow_range=True,
            )

        return PerChannelWeightQuantizer(
            num_bits=8,
            depthwise=isinstance(layer, tf.keras.layers.DepthwiseConv2D),
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
    """Quantize convolution weights and leave output ranges for calibration."""

    def get_activations_and_quantizers(self, layer):
        return []

    def set_quantize_activations(self, layer, quantize_activations):
        pass

    def get_output_quantizers(self, layer):
        return []


@register_keras_serializable()
class SignedConvQuantConfig(BaseQuantConfig):
    """Quantize weights and track signed linear-convolution output ranges."""

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
    """Apply stateless fake quantization over a fixed interval.

    Stateless pins may be added or removed without changing checkpoint variables.
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
class PerChannelWeightQuantizer(tfmot.quantization.keras.quantizers.Quantizer):
    """Apply stateless symmetric per-output-channel weight fake quantization.

    Conv2D uses ``[kh, kw, in, out]`` and reduces axes 0-2. DepthwiseConv2D uses
    ``[kh, kw, in, mult]`` and reduces axes 0-1 to retain ``in * mult`` scales.
    A straight-through implementation avoids the last-axis-only TensorFlow op.
    """

    def __init__(self, num_bits: int = 8, depthwise: bool = False):
        self.num_bits = num_bits
        self.depthwise = depthwise

    @property
    def _reduce_axes(self) -> list[int]:
        # Everything except the channel axis (or axes, for depthwise).
        return [0, 1] if self.depthwise else [0, 1, 2]

    def build(self, tensor_shape, name, layer):
        return {}

    def __call__(self, inputs, training, weights, **kwargs):
        # Narrow range, as TFLite uses for int8 weights: [-127, 127].
        limit = float(2 ** (self.num_bits - 1) - 1)

        abs_max = tf.reduce_max(tf.abs(inputs), axis=self._reduce_axes, keepdims=True)
        # An all-zero channel would divide by zero; its scale is arbitrary.
        scale = tf.maximum(abs_max, 1e-8) / limit

        quantized = tf.clip_by_value(tf.round(inputs / scale), -limit, limit) * scale

        # Straight-through: quantize forwards, pass gradients back unchanged.
        return inputs + tf.stop_gradient(quantized - inputs)

    def get_config(self):
        return {"num_bits": self.num_bits, "depthwise": self.depthwise}


@register_keras_serializable()
class ReLU6ConvQuantConfig(BaseQuantConfig):
    """Quantize weights and pin a folded intrinsic ReLU6 output to [0, 6].

    The output pin applies after folded ReLU6 and keeps the fused TFLite
    convolution's weights per-tensor.
    """

    def get_activations_and_quantizers(self, layer):
        return []

    def set_quantize_activations(self, layer, quantize_activations):
        pass

    def get_output_quantizers(self, layer):
        # TFMOT rejects tf.nn.relu6 in its activation-quantizer path.
        return [FixedRelu6Quantizer()]


@register_keras_serializable()
class AddOutputConfig(tfmot.quantization.keras.QuantizeConfig):
    """Quantize residual Add outputs with a signed range.

    Retained only to deserialize graphs and checkpoints from the previous scheme.
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
    "PerChannelWeightQuantizer": PerChannelWeightQuantizer,
}


def _has_intrinsic_relu6(layer: tf.keras.layers.Layer) -> bool:
    """Return whether a convolution has a folded intrinsic ``tf.nn.relu6``."""
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


# Full int8 scheme shared by the backbone and detection heads.


def _quantize_full(
    model,
    *,
    per_channel: bool,
    weight_only_names=frozenset(),
    for_export: bool = False,
    fully_quantized: bool = False,
    input_range: tuple[float, float] | None = None,
):
    """Apply one quantization pass to a folded functional model.

    Intrinsic ReLU6 outputs use [0, 6], named predictor outputs remain free for
    concat scaling, and other convolution outputs use signed ranges. Calibrated
    per-channel export leaves every output free; intrinsic ReLU6 takes precedence.
    """
    per_axis_weights = fully_quantized
    weight_only_cfg = FreeOutputConvQuantConfig(per_axis_weights=per_axis_weights)

    # Converter calibration requires all output ranges to remain free.
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
    relu6_conv_cfg = (
        weight_only_cfg
        if export_weight_only
        else ReLU6ConvQuantConfig(per_axis_weights=per_axis_weights)
    )
    # Fully QDQ graphs must pin residual Add outputs, including FPN taps.
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
    """Add a stateless fixed-range fake quantizer at the model input."""
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

    # Replay exposes nested layers to SavedModel tracing; direct calls export empty.
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
    """Quantize a folded functional backbone.

    Fully QDQ mode pins the normalized image input to ``INPUT_RANGE``.
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


# Functional head rebuilds preserve layer tracking during TFLite conversion.


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
    """Fold and quantize a fresh detection model in place.

    FPNLite uses separate backbone and combined-head graphs. Plain SSD uses one
    graph because ``layer_15/expansion_output`` has backbone and head consumers.
    Per-channel export frees output ranges; fully QDQ embeds all ranges.
    """
    if fully_quantized is None:
        # Current TFLite conversion aborts or exports an empty fully QDQ graph.
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


# Plain SSD uses one graph so dual-use backbone taps share a quantization domain.
# The backbone adapter runs the graph once; later adapters return cached outputs.


def _replay_functional(model, input_tensor):
    """Replay a single-input functional model while reusing its layers.

    Flattening exposes nested backbone layers to ``fold_model``.
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
    """Replay a convolutional predictor and return box and class tensors.

    Head convolutions receive unique names and enter ``conv_sink`` so concat
    inputs can share a converter-selected scale.
    """
    from object_detection.core.box_predictor import (
        BOX_ENCODINGS,
        CLASS_PREDICTIONS_WITH_BACKGROUND,
    )

    box_out, cls_out = [], []
    for i, x0 in enumerate(feature_tensors):
        x = x0
        for layer in box_predictor._shared_nets[
            i
        ]:  # empty unless a tower is configured
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
    """Build the plain SSD backbone and head as one functional model.

    ReLU6 Lambdas become Keras ReLU layers for convolution fusion. The return
    value includes output metadata and predictor convolution names.
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
    feats = backbone(padded)
    out_keys = list(fmg({k: feats[i] for i, k in enumerate(tap_keys)}).keys())

    image_input = tf.keras.Input(
        shape=tuple(padded.shape.as_list()[1:]), name="padded_image"
    )
    taps = _replay_functional(backbone, image_input)
    tap_map = {k: taps[i] for i, k in enumerate(tap_keys)}

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
    """Run the combined SSD graph once and cache map, box, and class outputs."""

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
    """Run the combined graph and return placeholder backbone taps."""

    def __init__(self, head, num_taps):
        self.head = head
        self.num_taps = num_taps

    def __call__(self, padded_image):
        maps, _box, _cls = self.head.run(padded_image)
        # The generator adapter ignores tap values but requires their count.
        return [maps[0]] * self.num_taps


class _SsdGenAdapter:
    """Return cached feature maps using the meta-architecture key order."""

    def __init__(self, head):
        self.head = head

    def __call__(self, image_features):
        maps, _box, _cls = self.head._cache
        # AutoGraph's zip replacement does not accept ``strict``.
        return collections.OrderedDict(
            zip(self.head.feature_map_keys, maps)  # noqa: B905
        )


class _SsdBoxPredictorAdapter(tf.keras.layers.Layer):
    """Return cached tensors using the TFOD box-predictor dictionary contract."""

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
    """Fold and quantize plain SSD as one graph.

    The input backbone must be original and unquantized so its dual-use tap
    remains inside the combined quantization domain.
    """
    fe = detection_model.feature_extractor
    model, out_keys, num_maps, num_taps, bp_convs = _build_combined_ssd_functional(
        detection_model, image_size
    )

    # Clone because ``fold_model`` reads the first inbound node of reused layers.
    clean = tf.keras.models.clone_model(model)
    clean.set_weights(model.get_weights())
    folded = fold_model(clean)

    # Include names produced when configured tower convolutions fold BatchNorm.
    weight_only = set(bp_convs) | {f"{n}_folded" for n in bp_convs}

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


# FPNLite uses one post-backbone graph to preserve ReLU6 fusion and scale domains.
# Separable convolutions are split for folding; adapters return cached outputs.


def _split_separable_conv(sep, tag):
    """Split a separable convolution into depthwise and pointwise layers.

    ``tag`` separates reused FPN layers whose per-map BatchNorm folds differ.
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
    """Apply a reused head layer in a foldable functional graph.

    Separable layers are split, ReLU6 is explicit, Lambdas receive unique names,
    and convolution names are optionally recorded in ``conv_sink``.
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
    """Replay FPN top-down layers and return keys and tensors by level."""
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
    """Replay chained coarse stride-2 blocks from the deepest FPN map."""
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
    """Replay a weight-shared predictor and return box and class tensors.

    Record its convolutions in ``conv_sink`` for weight-only output handling.
    """
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
            tf.keras.layers.Reshape((-1, num_class_slots), name=f"ws_cls_reshape_{i}")(
                c
            )
        )
    return box_out, cls_out


def _build_combined_fpn_functional(detection_model, image_size):
    """Build one functional FPN head from backbone maps to predictions.

    Return the model, output metadata, and predictor convolution names.
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
    """Run the combined FPN head once and cache all output groups."""

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
    """Run the combined FPN head and return its top-down maps."""

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
    """Return one cached coarse FPN map."""

    def __init__(self, head, index):
        self.head = head
        self.index = index

    def __call__(self, x):
        n_td = len(self.head.top_down_keys)
        return self.head._cache[0][n_td + self.index]


class _BoxPredictorAdapter(tf.keras.layers.Layer):
    """Return cached tensors using the TFOD box-predictor dictionary contract."""

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
    """Quantize a combined FPNLite head after quantizing its backbone."""
    fe = detection_model.feature_extractor
    model, td_keys, num_coarse, num_maps, bp_convs = _build_combined_fpn_functional(
        detection_model, image_size
    )
    folded = fold_model(model)

    # Include names produced by folding per-level BatchNorm.
    weight_only = set(bp_convs) | {f"{n}_folded" for n in bp_convs}

    # Backbone taps already define the head input quantization domain.
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
