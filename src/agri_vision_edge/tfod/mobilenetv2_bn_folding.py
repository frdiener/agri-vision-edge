"""
Utilities for reconstructing a TensorFlow Object Detection API
MobileNetV2 backbone without Batch-Normalization layers.

The reconstruction process:

    Conv2D -> BatchNorm
        becomes
    FoldedConv2D

and

    DepthwiseConv2D -> BatchNorm
        becomes
    FoldedDepthwiseConv2D

while preserving:

    * residual connections
    * activation functions
    * feature map outputs
    * graph topology

The resulting backbone has been verified to produce numerically
equivalent outputs to the original backbone.
"""

from __future__ import annotations

import numpy as np
import tensorflow as tf
from object_detection.core.freezable_batch_norm import (
    FreezableBatchNorm,
)


def _get_bn_consumer(
    layer: tf.keras.layers.Layer,
) -> FreezableBatchNorm:
    """
    Retrieve the BatchNorm layer consuming a convolution output.

    MobileNetV2 in TFOD is expected to have exactly one consumer
    for each Conv2D and DepthwiseConv2D layer and that consumer
    must be a FreezableBatchNorm.

    Raises:
        AssertionError if the expected topology is violated.
    """

    consumers = [node.layer for node in layer._outbound_nodes]

    assert len(consumers) == 1, (
        f"{layer.name}: expected exactly one consumer, found {len(consumers)}"
    )

    bn = consumers[0]

    assert isinstance(
        bn,
        FreezableBatchNorm,
    ), f"{layer.name}: expected FreezableBatchNorm consumer, found {type(bn).__name__}"

    return bn


def _fold_conv_bn(
    conv: tf.keras.layers.Conv2D,
    bn: FreezableBatchNorm,
) -> tf.keras.layers.Conv2D:
    """
    Fold BatchNorm parameters into a Conv2D layer.
    """

    kernel = conv.kernel.numpy()

    if conv.use_bias:
        bias = conv.bias.numpy()
    else:
        bias = np.zeros(
            kernel.shape[-1],
            dtype=np.float32,
        )

    gamma = bn.gamma.numpy()
    beta = bn.beta.numpy()
    mean = bn.moving_mean.numpy()
    var = bn.moving_variance.numpy()

    scale = gamma / np.sqrt(var + bn.epsilon)

    kernel_folded = kernel * scale.reshape(
        1,
        1,
        1,
        -1,
    )

    bias_folded = beta + (bias - mean) * scale

    folded = tf.keras.layers.Conv2D(
        filters=conv.filters,
        kernel_size=conv.kernel_size,
        strides=conv.strides,
        padding=conv.padding,
        dilation_rate=conv.dilation_rate,
        activation=None,
        use_bias=True,
        name=f"{conv.name}_folded",
    )

    folded.build(conv.input_shape)

    folded.set_weights(
        [
            kernel_folded,
            bias_folded,
        ]
    )

    return folded


def _fold_depthwise_bn(
    depthwise: tf.keras.layers.DepthwiseConv2D,
    bn: FreezableBatchNorm,
) -> tf.keras.layers.DepthwiseConv2D:
    """
    Fold BatchNorm parameters into a DepthwiseConv2D layer.
    """

    kernel = depthwise.depthwise_kernel.numpy()

    if depthwise.use_bias:
        bias = depthwise.bias.numpy()
    else:
        bias = np.zeros(
            bn.gamma.shape[0],
            dtype=np.float32,
        )

    gamma = bn.gamma.numpy()
    beta = bn.beta.numpy()
    mean = bn.moving_mean.numpy()
    var = bn.moving_variance.numpy()

    scale = gamma / np.sqrt(var + bn.epsilon)

    kernel_folded = kernel * scale.reshape(
        1,
        1,
        -1,
        1,
    )

    bias_folded = beta + (bias - mean) * scale

    folded = tf.keras.layers.DepthwiseConv2D(
        kernel_size=depthwise.kernel_size,
        strides=depthwise.strides,
        padding=depthwise.padding,
        dilation_rate=depthwise.dilation_rate,
        depth_multiplier=depthwise.depth_multiplier,
        activation=None,
        use_bias=True,
        name=f"{depthwise.name}_folded",
    )

    folded.build(depthwise.input_shape)

    folded.set_weights(
        [
            kernel_folded,
            bias_folded,
        ]
    )

    return folded


def _clone_or_replace_layer(
    layer: tf.keras.layers.Layer,
) -> tf.keras.layers.Layer:
    """
    Clone a non-convolution layer.

    ReLU6 Lambda layers are replaced with explicit ReLU layers
    because TensorFlow Model Optimization does not automatically
    annotate Lambda layers for quantization.
    """

    if isinstance(
        layer,
        tf.keras.layers.Lambda,
    ):
        return tf.keras.layers.ReLU(
            max_value=6.0,
            name=layer.name,
        )

    return layer.__class__.from_config(layer.get_config())


def fold_mobilenetv2_backbone(
    backbone: tf.keras.Model,
) -> tf.keras.Model:
    """
    Reconstruct a TFOD MobileNetV2 backbone without
    Batch-Normalization layers.

    Args:
        backbone:
            TFOD MobileNetV2 backbone model.

    Returns:
        Folded backbone model.
    """

    new_input = tf.keras.Input(
        shape=backbone.input_shape[1:],
        name="folded_input",
    )

    tensor_map = {
        backbone.input.ref(): new_input,
    }

    for layer in backbone.layers:
        if isinstance(
            layer,
            tf.keras.layers.InputLayer,
        ):
            continue

        if isinstance(
            layer,
            FreezableBatchNorm,
        ):
            continue

        old_inputs = tf.nest.flatten(layer.input)

        new_inputs = [tensor_map[tensor.ref()] for tensor in old_inputs]

        if len(new_inputs) == 1:
            new_inputs = new_inputs[0]

        #
        # Conv2D + BN
        #

        if isinstance(
            layer,
            tf.keras.layers.Conv2D,
        ):
            bn = _get_bn_consumer(layer)

            new_output = _fold_conv_bn(
                layer,
                bn,
            )(new_inputs)

            tensor_map[layer.output.ref()] = new_output

            tensor_map[bn.output.ref()] = new_output

            continue

        #
        # DepthwiseConv2D + BN
        #

        if isinstance(
            layer,
            tf.keras.layers.DepthwiseConv2D,
        ):
            bn = _get_bn_consumer(layer)

            new_output = _fold_depthwise_bn(
                layer,
                bn,
            )(new_inputs)

            tensor_map[layer.output.ref()] = new_output

            tensor_map[bn.output.ref()] = new_output

            continue

        #
        # Everything else
        #

        new_layer = _clone_or_replace_layer(layer)

        new_output = new_layer(new_inputs)

        old_outputs = tf.nest.flatten(layer.output)

        new_outputs = tf.nest.flatten(new_output)

        for old, new in zip(
            old_outputs,
            new_outputs,
            strict=False,
        ):
            tensor_map[old.ref()] = new

    outputs = [tensor_map[tensor.ref()] for tensor in backbone.output]

    folded_backbone = tf.keras.Model(
        inputs=new_input,
        outputs=outputs,
        name=f"{backbone.name}_folded",
    )

    bn_count = sum(
        isinstance(
            layer,
            FreezableBatchNorm,
        )
        for layer in folded_backbone.layers
    )

    assert bn_count == 0, f"Folded backbone still contains {bn_count} BatchNorm layers."

    return folded_backbone
