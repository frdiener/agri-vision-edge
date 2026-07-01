"""
Generic BatchNorm folding + ReLU6 pre-folding for TFOD Keras graphs.

A single ``fold_model`` reconstructs any functional Keras graph with the
Batch-Normalization layers removed and every ReLU6 pre-folded into the
convolution that produces it:

    Conv2D       -> BatchNorm -> ReLU6   becomes   Conv2D(activation=relu6)
    DepthwiseConv2D -> BatchNorm -> ReLU6           DepthwiseConv2D(activation=relu6)
    Conv2D       -> BatchNorm            becomes   Conv2D(linear, bias folded)

while preserving residual connections, non-conv activations (e.g. the FPN
nearest-neighbour upsample ``Lambda``), feature-map outputs and the overall
graph topology.

Pre-folding the ReLU6 into the conv is what makes the downstream int8 pin land
on the right tensor: because the conv's output tensor now *is* the post-ReLU6
tensor, TFLite fuses conv+ReLU6 into one op whose output range is the known
[0, 6] (scale 6/255, zero_point -128). This is load-bearing for the QAT scheme
(see ``qat.quantize_backbone``): the same folded representation is used for the
MobileNetV2 backbone and for the functionally-rebuilt SSD / FPN detection heads,
so a single quantization path covers all of them.

``fold_model`` is topology-generic. It makes no MobileNetV2 assumptions, so it
works equally on the subclassed backbone's functional graph and on the rebuilt
feature-map generator / FPN head graphs (which mix split depthwise/pointwise
convs, residual adds and upsample lambdas).
"""

from __future__ import annotations

import collections

import numpy as np
import tensorflow as tf
from object_detection.core.freezable_batch_norm import FreezableBatchNorm

_CONV = (tf.keras.layers.Conv2D, tf.keras.layers.DepthwiseConv2D)
_BN = (tf.keras.layers.BatchNormalization, FreezableBatchNorm)


def is_relu6(layer: tf.keras.layers.Layer) -> bool:
    """
    True for a ReLU6 activation layer, whether it is a
    ``keras.layers.ReLU(max_value=6)`` (the folded / rebuilt graphs) or a
    ``Lambda(tf.nn.relu6)`` (the raw TFOD MobileNetV2 backbone / SSD head).

    A ``Lambda`` carries no metadata identifying it as ReLU6, so we probe it: a
    layer is ReLU6 iff it clips its input to [0, 6].
    """
    if isinstance(layer, tf.keras.layers.ReLU):
        return getattr(layer, "max_value", None) == 6

    if isinstance(layer, tf.keras.layers.Lambda):
        try:
            probe = tf.constant([-6.0, -1.0, 0.0, 3.0, 6.0, 9.0], dtype=tf.float32)
            out = np.asarray(layer(probe))
            return np.allclose(out, np.clip(probe.numpy(), 0.0, 6.0))
        except Exception:
            return False

    return False


def _fold_conv2d(
    conv: tf.keras.layers.Conv2D,
    bn: FreezableBatchNorm | None,
    activation,
) -> tf.keras.layers.Conv2D:
    """
    Rebuild a ``Conv2D`` with the given intrinsic activation and, when ``bn`` is
    provided, the BatchNorm parameters folded into a bias-enabled kernel.

    Built from the kernel shape (subclassed-layer convs, e.g. the split FPN
    pointwise convs, do not expose ``input_shape``).
    """
    kernel = conv.kernel.numpy()
    bias = (
        conv.bias.numpy()
        if conv.use_bias
        else np.zeros(kernel.shape[-1], dtype=np.float32)
    )

    if bn is not None:
        scale = bn.gamma.numpy() / np.sqrt(bn.moving_variance.numpy() + bn.epsilon)
        kernel = kernel * scale.reshape(1, 1, 1, -1)
        bias = bn.beta.numpy() + (bias - bn.moving_mean.numpy()) * scale

    folded = tf.keras.layers.Conv2D(
        filters=conv.filters,
        kernel_size=conv.kernel_size,
        strides=conv.strides,
        padding=conv.padding,
        dilation_rate=conv.dilation_rate,
        activation=activation,
        use_bias=True,
        name=f"{conv.name}_folded",
    )
    folded.build((None, None, None, int(kernel.shape[2])))
    folded.set_weights([kernel, bias])
    return folded


def _fold_depthwise(
    depthwise: tf.keras.layers.DepthwiseConv2D,
    bn: FreezableBatchNorm | None,
    activation,
) -> tf.keras.layers.DepthwiseConv2D:
    """
    Rebuild a ``DepthwiseConv2D`` with the given intrinsic activation and, when
    ``bn`` is provided, the BatchNorm parameters folded into a bias-enabled
    depthwise kernel. Built from the kernel shape (see ``_fold_conv2d``).
    """
    kernel = depthwise.depthwise_kernel.numpy()
    out_channels = kernel.shape[2] * kernel.shape[3]
    bias = (
        depthwise.bias.numpy()
        if depthwise.use_bias
        else np.zeros(out_channels, dtype=np.float32)
    )

    if bn is not None:
        scale = bn.gamma.numpy() / np.sqrt(bn.moving_variance.numpy() + bn.epsilon)
        kernel = kernel * scale.reshape(1, 1, -1, 1)
        bias = bn.beta.numpy() + (bias - bn.moving_mean.numpy()) * scale

    folded = tf.keras.layers.DepthwiseConv2D(
        kernel_size=depthwise.kernel_size,
        strides=depthwise.strides,
        padding=depthwise.padding,
        dilation_rate=depthwise.dilation_rate,
        depth_multiplier=depthwise.depth_multiplier,
        activation=activation,
        use_bias=True,
        name=f"{depthwise.name}_folded",
    )
    folded.build((None, None, None, int(kernel.shape[2])))
    folded.set_weights([kernel, bias])
    return folded


def _fold_conv(conv, bn, activation):
    if isinstance(conv, tf.keras.layers.DepthwiseConv2D):
        return _fold_depthwise(conv, bn, activation)
    return _fold_conv2d(conv, bn, activation)


def fold_model(model: tf.keras.Model) -> tf.keras.Model:
    """
    Fold every ``conv (-> BN) (-> ReLU6)`` chain in a functional Keras model,
    baking the BatchNorm into a bias-enabled conv and the ReLU6 into that conv's
    intrinsic ``activation``. Every other layer (residual ``Add``, upsample
    ``Lambda``, ``Reshape``, ...) is reused verbatim, so the graph topology and
    all weights are preserved exactly.

    Works on any functional graph -- the subclassed MobileNetV2 backbone and the
    rebuilt SSD / FPN head graphs alike. Both the per-tensor and per-channel
    quantization schemes use this fold: the per-channel scheme relies on TFLite's
    calibration of the fused conv+relu6 output, which is bounded by 6 (so
    zero_point -128, scale <= 6/255) and the per-tensor scheme pins it with a
    fixed [0, 6] output quantizer on the conv (``ReLU6ConvQuantConfig``).
    """
    consumers: dict[str, list] = collections.defaultdict(list)
    for layer in model.layers:
        for node in layer.inbound_nodes:
            for parent in tf.nest.flatten(node.inbound_layers):
                consumers[parent.name].append(layer)

    def single_consumer(name):
        cs = consumers.get(name, [])
        return cs[0] if len(cs) == 1 else None

    # Classify: which convs fold a BN (conv_bn) and which absorb a ReLU6
    # (conv_act); which layers disappear as a result (drop).
    conv_bn: dict[str, FreezableBatchNorm] = {}
    conv_act: dict[str, tf.keras.layers.Layer] = {}
    drop: set[str] = set()

    for layer in model.layers:
        if not isinstance(layer, _CONV):
            continue

        consumer = single_consumer(layer.name)

        bn = consumer if isinstance(consumer, _BN) else None
        if bn is not None:
            conv_bn[layer.name] = bn
            drop.add(bn.name)
            after = single_consumer(bn.name)
        else:
            after = consumer

        if after is not None and is_relu6(after):
            conv_act[layer.name] = after
            drop.add(after.name)

    out: dict[str, tf.Tensor] = {}
    new_inputs: list = []

    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.InputLayer):
            tensor = tf.keras.Input(shape=layer.output.shape[1:], name=layer.name)
            out[layer.name] = tensor
            new_inputs.append(tensor)
            continue

        parents = list(tf.nest.flatten(layer.inbound_nodes[0].inbound_layers))
        inputs = [out[p.name] for p in parents]
        inputs = inputs[0] if len(inputs) == 1 else inputs

        if layer.name in drop:
            # A folded-away BN or absorbed ReLU6: its output is its (single)
            # parent's already-folded output tensor.
            out[layer.name] = out[parents[0].name]
        elif layer.name in conv_bn or layer.name in conv_act:
            folded = _fold_conv(
                layer,
                conv_bn.get(layer.name),
                activation=tf.nn.relu6 if layer.name in conv_act else None,
            )
            out[layer.name] = folded(inputs)
        else:
            out[layer.name] = layer(inputs)  # reuse (Add / Lambda / Reshape / ...)

    outputs = [out[t._keras_history.layer.name] for t in model.outputs]

    folded = tf.keras.Model(
        inputs=new_inputs,
        outputs=outputs,
        name=f"{model.name}_folded",
    )

    remaining = sum(isinstance(layer, _BN) for layer in folded.layers)
    assert remaining == 0, f"Folded model still contains {remaining} BatchNorm layers."

    return folded


def fold_mobilenetv2_backbone(backbone: tf.keras.Model) -> tf.keras.Model:
    """
    Reconstruct a TFOD MobileNetV2 backbone without Batch-Normalization layers
    (BN folded into the convs, ReLU6 pre-folded into the conv activation).

    Thin backward-compatible wrapper around the topology-generic ``fold_model``;
    kept as the public entry point used by the trainer / converter / notebooks.
    """
    return fold_model(backbone)
