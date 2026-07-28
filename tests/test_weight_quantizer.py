"""
Tests for the per-channel weight fake-quant.

The failure this guards against is silent: a quantizer that reduces over the
wrong axes still runs, still trains, and still exports -- it just quantizes a
depthwise kernel per-tensor while calling itself per-channel.
"""

from __future__ import annotations

import numpy as np
import pytest
import tensorflow as tf

# `agri_vision_edge.tfod` imports the vendored object_detection package, which
# has to be put on the path first.
from agri_vision_edge.third_party import setup_tensorflow_models

setup_tensorflow_models()

try:
    from agri_vision_edge.tfod.qat import PerChannelWeightQuantizer
except ImportError as exc:  # pragma: no cover - only without the vendored deps
    pytest.skip(f"object_detection unavailable: {exc}", allow_module_level=True)


def _quantize(kernel, depthwise):
    quantizer = PerChannelWeightQuantizer(depthwise=depthwise)
    return np.asarray(
        quantizer(tf.constant(kernel, dtype=tf.float32), False, {})
    )


def _step(original, quantized):
    """Smallest non-zero gap between quantized values -- i.e. the scale."""
    values = np.unique(np.abs(quantized.ravel()))
    values = values[values > 0]
    return values.min() if values.size else 0.0


def test_conv_kernel_gets_one_scale_per_output_channel():
    # Two output channels with ranges an order of magnitude apart: quantized
    # per channel, each keeps its own resolution.
    kernel = np.zeros((3, 3, 4, 2), dtype=np.float32)
    kernel[..., 0] = np.linspace(-1.0, 1.0, 36).reshape(3, 3, 4)
    kernel[..., 1] = np.linspace(-100.0, 100.0, 36).reshape(3, 3, 4)

    quantized = _quantize(kernel, depthwise=False)

    small = _step(kernel[..., 0], quantized[..., 0])
    large = _step(kernel[..., 1], quantized[..., 1])
    assert large > small * 50, (
        "output channels should be quantized with independent scales"
    )


def test_depthwise_kernel_gets_one_scale_per_input_channel():
    # The regression: a depthwise kernel is [kh, kw, in, mult] and mult is 1, so
    # a quantizer that keeps only the LAST axis produces a single scale and the
    # small channel is crushed by the large one.
    kernel = np.zeros((3, 3, 2, 1), dtype=np.float32)
    kernel[:, :, 0, 0] = np.linspace(-1.0, 1.0, 9).reshape(3, 3)
    kernel[:, :, 1, 0] = np.linspace(-100.0, 100.0, 9).reshape(3, 3)

    quantized = _quantize(kernel, depthwise=True)

    small = _step(kernel[:, :, 0, 0], quantized[:, :, 0, 0])
    large = _step(kernel[:, :, 1, 0], quantized[:, :, 1, 0])
    assert large > small * 50, (
        "depthwise channels must be quantized per input channel, not per tensor"
    )


def test_quantization_error_is_bounded_by_half_a_step():
    rng = np.random.default_rng(0)
    kernel = rng.normal(size=(3, 3, 8, 4)).astype(np.float32)

    quantized = _quantize(kernel, depthwise=False)

    scale = np.max(np.abs(kernel), axis=(0, 1, 2)) / 127.0
    assert np.all(np.abs(kernel - quantized) <= scale / 2 + 1e-6)


def test_values_land_on_the_grid_and_stay_in_range():
    rng = np.random.default_rng(1)
    kernel = rng.normal(size=(3, 3, 4, 3)).astype(np.float32)

    quantized = _quantize(kernel, depthwise=False)

    scale = np.max(np.abs(kernel), axis=(0, 1, 2)) / 127.0
    levels = quantized / scale
    assert np.allclose(levels, np.round(levels), atol=1e-4)
    assert np.all(np.abs(levels) <= 127 + 1e-4)


def test_all_zero_channel_does_not_produce_nans():
    kernel = np.zeros((3, 3, 2, 2), dtype=np.float32)
    kernel[..., 1] = 0.5

    quantized = _quantize(kernel, depthwise=False)

    assert np.all(np.isfinite(quantized))
    assert np.all(quantized[..., 0] == 0.0)


def test_gradients_pass_straight_through():
    kernel = tf.Variable(np.linspace(-1, 1, 36).reshape(3, 3, 4, 1).astype("float32"))
    quantizer = PerChannelWeightQuantizer()

    with tf.GradientTape() as tape:
        loss = tf.reduce_sum(quantizer(kernel, True, {}) * 2.0)

    grad = tape.gradient(loss, kernel)
    assert np.allclose(np.asarray(grad), 2.0)


def test_quantized_model_trains():
    """
    A train step through the real QAT wrapper, per-channel.

    The quantizer is stateless -- ``build`` returns no weights -- and sits in
    the gradient path of every kernel, so this covers the integration tfmot
    actually performs rather than the quantizer in isolation: both conv kinds
    wrapped, gradients arriving at the kernels, and a step moving them.
    """
    from agri_vision_edge.tfod.qat import _quantize_full

    inputs = tf.keras.Input(shape=(8, 8, 3), batch_size=2)
    x = tf.keras.layers.Conv2D(4, 3, padding="same", activation=tf.nn.relu6)(inputs)
    x = tf.keras.layers.DepthwiseConv2D(3, padding="same", activation=tf.nn.relu6)(x)
    x = tf.keras.layers.Conv2D(2, 1, padding="same")(x)
    model = tf.keras.Model(inputs, x)

    quantized = _quantize_full(model, per_channel=True)

    rng = np.random.default_rng(0)
    batch = tf.constant(rng.normal(size=(2, 8, 8, 3)).astype(np.float32))
    target = tf.constant(rng.normal(size=(2, 8, 8, 2)).astype(np.float32))

    kernels = [v for v in quantized.trainable_variables if "kernel" in v.name]
    assert len(kernels) == 3, [v.name for v in quantized.trainable_variables]
    before = [v.numpy().copy() for v in kernels]

    with tf.GradientTape() as tape:
        loss = tf.reduce_mean(tf.square(quantized(batch, training=True) - target))
    grads = tape.gradient(loss, quantized.trainable_variables)

    kernel_grads = [
        g
        for g, v in zip(grads, quantized.trainable_variables, strict=True)
        if g is not None and "kernel" in v.name
    ]
    assert len(kernel_grads) == 3
    assert float(tf.linalg.global_norm(kernel_grads)) > 0.0
    assert all(bool(tf.reduce_all(tf.math.is_finite(g))) for g in kernel_grads)

    tf.keras.optimizers.legacy.SGD(learning_rate=1.0).apply_gradients(
        [(g, v) for g, v in zip(grads, quantized.trainable_variables, strict=True)
         if g is not None]
    )
    assert all(
        not np.array_equal(old, v.numpy())
        for old, v in zip(before, kernels, strict=True)
    ), "a training step must move every conv kernel"
