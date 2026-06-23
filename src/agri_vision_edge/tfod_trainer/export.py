"""
Export the best checkpoint of a finetune / QAT run.

This is the export analog of the rest of ``tfod_trainer``: rather than calling
the *forked* ``object_detection.exporter_lib_v2.export_inference_graph`` (whose
``qat_backbone`` / ``fold_bn`` arguments do not exist upstream), it reimplements
the inference-graph export here, keeping the custom fold / quantize logic in
``agri_vision_edge`` and using only stock-upstream symbols from
``object_detection``:

  * ``model_builder.build`` to construct the inference model,
  * ``exporter_lib_v2.DETECTION_MODULE_MAP`` for the serving modules (these
    classes are unchanged from upstream; only ``export_inference_graph`` was
    forked), and
  * ``config_util.save_pipeline_config`` to write the pipeline.

So this module keeps working once the vendored ``object_detection`` tree is
swapped for the stock upstream package.

It produces the standard TF model-zoo layout::

    <export_dir>/
    ├── checkpoint/ckpt-0.*   # model-only, restorable as a "detection" checkpoint
    ├── pipeline.config
    └── saved_model/          # fp32 SavedModel for test inference

The graph modifications default to whatever the run used, so a QAT export
reproduces the trained fake-quantized graph while a plain finetune exports a
clean fp32 detection checkpoint.

Because the layout matches a zoo base model, the export directory is drop-in
usable as the ``model_path`` of a follow-up ``FinetuneRunConfig``. That makes
"resume QAT from a finetune" identical to "finetune from the COCO17 checkpoint":
both load a model-only checkpoint via ``fine_tune_checkpoint_type="detection"``
and then (for QAT) fold + quantize the freshly restored weights — avoiding the
``model_lib_v2`` pain point of resuming through a full train-dir checkpoint
(model + optimizer + step), which is why the old workaround was to copy the
finetune ``train`` directory.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .run import FinetuneRunConfig


@dataclass
class ExportResult:
    """Paths to the artifacts produced by :func:`export_run`."""

    export_dir: Path
    checkpoint: Path  # <export_dir>/checkpoint/ckpt-0
    saved_model_dir: Path  # <export_dir>/saved_model
    pipeline_config: Path  # <export_dir>/pipeline.config


def export_run(
    cfg,
    *,
    export_dir=None,
    input_type: str = "image_tensor",
    fold_bn: bool | None = None,
    qat_backbone: str | None = None,
    quantize_head: bool | None = None,
    qat_per_channel: bool | None = None,
) -> ExportResult:
    """
    Export the best checkpoint of ``cfg``'s run to a checkpoint + SavedModel.

    ``cfg`` may be a ``FinetuneRunConfig`` or a plain dict. The best checkpoint
    is the latest one in ``cfg.train_dir`` (the trainer only saves on metric
    improvement, so latest == best).

    By default the export mirrors how the run was trained:
      * ``fold_bn`` defaults to ``cfg.fold_bn``.
      * ``qat_backbone`` defaults to ``cfg.qat_scheme`` (empty for a plain
        finetune), so a plain finetune yields a clean fp32 detection checkpoint
        ready to seed a follow-up QAT run.
      * ``quantize_head`` defaults to ``cfg.quantize_head`` -- it MUST match how
        the run was trained, otherwise the head's quantized variables won't be
        present to restore and ``assert_existing_objects_matched`` fails.
      * ``qat_per_channel`` defaults to ``cfg.qat_per_channel`` (per-tensor vs
        per-channel weights) -- it changes the fake-quant variable shapes, so it
        too MUST match the trained checkpoint.

    Pass any of these explicitly to override.
    """
    from agri_vision_edge.third_party import setup_tensorflow_models

    setup_tensorflow_models()

    import tensorflow as tf
    from google.protobuf import text_format
    from object_detection.builders import model_builder
    from object_detection.exporter_lib_v2 import DETECTION_MODULE_MAP
    from object_detection.protos import pipeline_pb2
    from object_detection.utils import config_util

    if not isinstance(cfg, FinetuneRunConfig):
        cfg = FinetuneRunConfig.from_mapping(cfg)

    export_dir = cfg.output_dir / "export" if export_dir is None else Path(export_dir)

    if fold_bn is None:
        fold_bn = cfg.fold_bn
    if qat_backbone is None:
        qat_backbone = cfg.qat_scheme.value if cfg.qat_scheme else ""
    if quantize_head is None:
        quantize_head = cfg.quantize_head
    if qat_per_channel is None:
        qat_per_channel = cfg.qat_per_channel

    if input_type not in DETECTION_MODULE_MAP:
        raise ValueError(
            f"Unrecognized input_type {input_type!r}; "
            f"expected one of {sorted(DETECTION_MODULE_MAP)}"
        )

    # Load the as-run pipeline proto.
    pipeline_config = pipeline_pb2.TrainEvalPipelineConfig()
    text_format.Merge(
        Path(cfg.pipeline_config_path).read_text(),
        pipeline_config,
    )

    # Build the inference model and reproduce the trained graph modifications
    # (fold / quantize) *before* restoring, so the variable structure matches
    # the saved checkpoint.
    detection_model = model_builder.build(
        model_config=pipeline_config.model,
        is_training=False,
    )

    if fold_bn or qat_backbone:
        from agri_vision_edge.tfod import fold_mobilenetv2_backbone
        from agri_vision_edge.tfod.qat import (
            ensure_model_is_built_for_qat,
            quantize_backbone,
        )

        ensure_model_is_built_for_qat(detection_model, pipeline_config)
        backbone = detection_model.feature_extractor.classification_backbone

        if fold_bn:
            print("Folding batchnorms into the convolutions...")
            backbone = fold_mobilenetv2_backbone(backbone)

        if qat_backbone:
            print("Adding fake quantization nodes to the backbone (full int8)...")
            backbone = quantize_backbone(backbone, per_axis=qat_per_channel)

        detection_model.feature_extractor.classification_backbone = backbone

        # Reproduce the head quantization (must run after the backbone is
        # quantized; it reads the backbone's output shapes).
        if qat_backbone and quantize_head:
            from agri_vision_edge.tfod.qat import quantize_detection_head

            print("Quantizing the detection head (feature maps + box predictor)...")
            image_size = (
                pipeline_config.model.ssd.image_resizer.fixed_shape_resizer.height
            )
            quantize_detection_head(
                detection_model,
                image_size,
                per_axis=qat_per_channel,
            )

    # Restore the best checkpoint. The trainer only saves on metric improvement,
    # so the latest checkpoint in train_dir is the best one.
    ckpt = tf.train.Checkpoint(model=detection_model)
    manager = tf.train.CheckpointManager(ckpt, str(cfg.train_dir), max_to_keep=1)
    if not manager.latest_checkpoint:
        raise FileNotFoundError(f"No checkpoint to export in {cfg.train_dir}")
    status = ckpt.restore(manager.latest_checkpoint).expect_partial()

    # Build the serving module; tracing the concrete function forces all
    # variables to be created, so the restore can be asserted and saved.
    module = DETECTION_MODULE_MAP[input_type](detection_model)
    concrete_function = module.__call__.get_concrete_function()
    status.assert_existing_objects_matched()

    checkpoint_dir = export_dir / "checkpoint"
    saved_model_dir = export_dir / "saved_model"

    exported_manager = tf.train.CheckpointManager(
        ckpt, str(checkpoint_dir), max_to_keep=1
    )
    exported_manager.save(checkpoint_number=0)

    tf.saved_model.save(
        module,
        str(saved_model_dir),
        signatures=concrete_function,
    )

    config_util.save_pipeline_config(pipeline_config, str(export_dir))

    return ExportResult(
        export_dir=export_dir,
        checkpoint=checkpoint_dir / "ckpt-0",
        saved_model_dir=saved_model_dir,
        pipeline_config=export_dir / "pipeline.config",
    )
