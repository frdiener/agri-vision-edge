"""
Combined single-functional QAT for the FPNLite head.

The original ``_quantize_fpn_detection_head`` quantizes the FPN top-down
generator, each coarse block, and the box predictor as SEPARATE functional
models (each its own ``quantize_apply``). That is fine for the per-channel
scheme -- it pins the ReLU6 *layer* output, so a conv+ReLU6 fuses into one int8
op even when the ReLU6 is a model-boundary tensor. But the per-tensor scheme
deliberately leaves the ReLU6 layer float (pinning the conv instead, to keep
weights per-tensor), so a feature-map ReLU6 that sits on a SEPARATE-MODEL
boundary -- e.g. a top-down map that fans out to the internal RESIZE *and* to the
separately-quantized box predictor -- cannot fuse: TFLite emits

    conv(int8) -> DEQUANTIZE -> float ReLU6 -> QUANTIZE (per consumer)

a stray float island the NPU delegate cannot consume.

A micro-experiment (experiments/fpn_qat_probe/toy_boundary.py) confirmed that the
SAME topology, when the boundary ReLU6 and BOTH its consumers live in ONE
quantize_apply graph, fuses cleanly in per-tensor AND stays purely per-tensor
(0 per-channel weight tensors). So the fix is to rebuild the WHOLE post-backbone
head -- generator + coarse blocks + weight-shared box predictor -- as a single
functional model and quantize it in one pass. The box-predictor convs are still
forced weight-only (free output scale) so the six feature maps share one concat
scale (no requant QUANTIZE at the concat); everything else uses the same
per-tensor / per-channel "full" scheme as the backbone.

Installation: the meta-arch calls extract_features (-> feature_maps, for anchors)
and the box predictor as two separate steps, but we must run the combined model
ONCE. So the generator adapter runs the combined model and caches all outputs;
the coarse + box-predictor adapters return the cached tensors. One graph, one
quantize_apply, no inter-model boundary.
"""

from __future__ import annotations

import collections
import itertools

import tensorflow as tf
import tensorflow_model_optimization as tfmot
from object_detection.core.freezable_batch_norm import FreezableBatchNorm

from agri_vision_edge.tfod.qat import (
    AddOutputConfig,
    ReLU6ConvQuantConfig,
    ReLU6OutputConfig,
    SignedConvQuantConfig,
    WeightOnlyQuantConfig,
    _is_relu6,
    _relu6_fed_conv_names,
    _split_separable_conv,
    fold_functional,
)

_CONV = (tf.keras.layers.Conv2D, tf.keras.layers.DepthwiseConv2D)


def _apply(layer, x, counter, conv_sink=None, name_sink=None):
    """
    Apply one reused head layer onto functional tensor(s), fold/quant/trace
    friendly (mirrors ``qat._fpn_apply``): SeparableConv2D -> split into
    DepthwiseConv2D + Conv2D(1x1); ReLU6 -> keras ReLU6 (fuses); other Lambda
    (nearest-neighbour upsample) -> freshly-named copy. When ``conv_sink`` /
    ``name_sink`` are given, the Conv2D/DepthwiseConv2D names / ALL layer names
    touched here are recorded (used to force the box-predictor convs weight-only
    and to keep the rest of the box-predictor region float).
    """
    if isinstance(layer, tf.keras.layers.SeparableConv2D):
        dw, pw = _split_separable_conv(layer, next(counter))
        for nm in (dw.name, pw.name):
            if conv_sink is not None:
                conv_sink.add(nm)
            if name_sink is not None:
                name_sink.add(nm)
        return pw(dw(x))
    if _is_relu6(layer):
        relu = tf.keras.layers.ReLU(max_value=6.0, name=f"relu6_{next(counter)}")
        if name_sink is not None:
            name_sink.add(relu.name)
        return relu(x)
    if isinstance(layer, tf.keras.layers.Lambda):
        lam = tf.keras.layers.Lambda(layer.function, name=f"{layer.name}_{next(counter)}")
        if name_sink is not None:
            name_sink.add(lam.name)
        return lam(x)
    if isinstance(layer, _CONV) and conv_sink is not None:
        conv_sink.add(layer.name)
    if name_sink is not None:
        name_sink.add(layer.name)
    return layer(x)


def _gen_body(fpn_gen, feature_items, counter):
    """Replay KerasFpnTopDownFeatureMaps onto ``feature_items`` (list of
    (key, tensor)); return (ordered_keys, ordered_tensors) in min..max level
    order. Mirrors ``qat.rebuild_fpn_generator_functional`` but onto shared
    tensors (no own Inputs)."""
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


def _bp_body(box_predictor, feature_tensors, counter, conv_sink, name_sink):
    """Replay a WeightSharedConvolutionalBoxPredictor over the feature tensors;
    return (box_out, cls_out). Convs touched are recorded in ``conv_sink``
    (forced weight-only) and EVERY layer name in ``name_sink`` (so the whole
    box-predictor region is excluded from the generator's ReLU6/Add pinning --
    matching the proven weight-only box-predictor treatment). Mirrors
    ``qat.rebuild_weight_shared_box_predictor_functional``."""
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
            x = _apply(layer, x, counter, conv_sink, name_sink)
        for layer in box_predictor._base_tower_layers_for_heads[BOX_ENCODINGS][i]:
            x = _apply(layer, x, counter, conv_sink, name_sink)
        tower = x  # shared between box and class heads (share_prediction_tower)

        b = tower
        for layer in box_predictor._box_prediction_head._box_encoder_layers:
            b = _apply(layer, b, counter, conv_sink, name_sink)
        rb = tf.keras.layers.Reshape((-1, code_size), name=f"ws_box_reshape_{i}")
        name_sink.add(rb.name)
        box_out.append(rb(b))

        c = tower
        for layer in box_predictor._prediction_heads[
            CLASS_PREDICTIONS_WITH_BACKGROUND
        ]._class_predictor_layers:
            c = _apply(layer, c, counter, conv_sink, name_sink)
        rc = tf.keras.layers.Reshape((-1, num_class_slots), name=f"ws_cls_reshape_{i}")
        name_sink.add(rc.name)
        cls_out.append(rc(c))
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
    bp_names = set()
    box_out, cls_out = _bp_body(
        detection_model._box_predictor, feature_maps, counter, bp_convs, bp_names
    )

    model = tf.keras.Model(list(inp.values()), feature_maps + box_out + cls_out)
    return model, td_keys, len(coarse_maps), len(feature_maps), bp_convs, bp_names


def _quantize_combined(model, *, per_channel, weight_only_names, box_predictor_names):
    """
    One ``quantize_apply`` over the whole combined head. Same per-tensor /
    per-channel "full" scheme as the backbone (see qat._quantize_backbone_full),
    except every conv whose name is in ``weight_only_names`` (the box-predictor
    convs) is forced weight-only so the multi-scale feature maps align to one
    concat scale (no requant QUANTIZE at the box/class concat).
    """
    relu6_fed = _relu6_fed_conv_names(model)
    signed_cfg = SignedConvQuantConfig()
    weight_only_cfg = WeightOnlyQuantConfig()

    if per_channel:
        relu6_conv_cfg = WeightOnlyQuantConfig()
        relu6_layer_cfg = ReLU6OutputConfig()
        add_cfg = AddOutputConfig()
    else:
        relu6_conv_cfg = ReLU6ConvQuantConfig()
        relu6_layer_cfg = None
        add_cfg = None

    def clone_function(layer):
        name = layer.name
        is_box_predictor = name in box_predictor_names

        if isinstance(layer, _CONV):
            # Must take precedence over the box-predictor weight-only rule:
            # this Conv's output is the input to a ReLU6, so its output
            # quantizer must be fixed to the ReLU6 range.
            if name in relu6_fed:
                config = relu6_conv_cfg

            # Keep all other predictor convolutions weight-only, especially
            # the terminal box/class prediction convolutions before concat.
            elif is_box_predictor or name in weight_only_names:
                config = weight_only_cfg

            else:
                config = signed_cfg

            return tfmot.quantization.keras.quantize_annotate_layer(
                layer,
                quantize_config=config,
            )

        # In the per-channel path, preserve the existing explicit ReLU6/Add
        # annotations for the non-box-predictor FPN region.
        if not is_box_predictor:
            if relu6_layer_cfg is not None and _is_relu6(layer):
                return tfmot.quantization.keras.quantize_annotate_layer(
                    layer,
                    quantize_config=relu6_layer_cfg,
                )

            if add_cfg is not None and isinstance(layer, tf.keras.layers.Add):
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
        annotated = tf.keras.models.clone_model(model, clone_function=clone_function)
        return tfmot.quantization.keras.quantize_apply(annotated)


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
        # noqa note: see qat.FMGAdapter -- autograph rewrites zip, rejects strict=
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


def quantize_fpn_detection_head_combined(
    detection_model, image_size, *, per_channel=False
):
    """
    FPNLite head QAT via ONE combined functional model + ONE quantize_apply.

    Call AFTER the backbone has been folded + quantized. Replaces
    ``qat._quantize_fpn_detection_head``'s three separate quantize_apply models
    with a single graph so per-tensor feature-map ReLU6s fuse (no float island)
    while staying purely per-tensor; per-channel behaviour is unchanged.
    """
    fe = detection_model.feature_extractor
    model, td_keys, num_coarse, num_maps, bp_convs, bp_names = (
        _build_combined_fpn_functional(detection_model, image_size)
    )
    folded = fold_functional(model)

    # Box-predictor tower convs fold their per-level BatchNorm into a "_folded"
    # conv, so the post-fold name may carry that suffix.
    weight_only = set(bp_convs) | {f"{n}_folded" for n in bp_convs}
    bp_region = set(bp_names) | {f"{n}_folded" for n in bp_names}

    q = _quantize_combined(
        folded,
        per_channel=per_channel,
        weight_only_names=weight_only,
        box_predictor_names=bp_region,
    )

    head = _CombinedFpnHead(q, td_keys, num_coarse, num_maps)
    fe._q_combined_head = q  # track variables for conversion
    fe._fpn_features_generator = _GenAdapter(head)
    fe._coarse_feature_layers = [
        [_CoarseBlockAdapter(head, i)] for i in range(num_coarse)
    ]
    detection_model._box_predictor = _BoxPredictorAdapter(head)
    return detection_model
