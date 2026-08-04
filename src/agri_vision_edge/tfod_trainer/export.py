"""
Export the best checkpoint of a finetune / QAT run.

This is the export analog of the rest of ``tfod_trainer``: it reimplements
the inference-graph export here, keeping the custom fold / quantize logic in
``agri_vision_edge`` and using upstream symbols from ``object_detection``:

  * ``model_builder.build`` to construct the inference model,
  * ``exporter_lib_v2.DETECTION_MODULE_MAP`` for the serving modules and
  * ``config_util.save_pipeline_config`` to write the pipeline.

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

from agri_vision_edge.conversion.tflite import stage_graph_flags

from .run import FinetuneRunConfig


@dataclass
class ExportResult:
    """Paths to the artifacts produced by :func:`export_run`."""

    export_dir: Path
    checkpoint: Path  # <export_dir>/checkpoint/ckpt-0
    saved_model_dir: Path  # <export_dir>/saved_model
    pipeline_config: Path  # <export_dir>/pipeline.config


#: Sub-directory holding the scoring re-export (see
#: :func:`export_scoring_saved_model`). Sits beside the stage's own
#: ``saved_model/`` rather than replacing it -- the stock export is what the
#: TFLite conversion is traced from and must not move.
SCORING_EXPORT_NAME = "saved_model_nms0"


def export_scoring_saved_model(
    stage_dir,
    *,
    output_dir=None,
    score_threshold: float = 0.0,
    qat: bool | None = None,
    qat_per_channel: bool | None = None,
    input_type: str = "image_tensor",
) -> Path:
    """
    Re-export a stage's checkpoint with the NMS score threshold removed.

    The stage ``saved_model/`` bakes the pipeline's
    ``batch_non_max_suppression.score_threshold`` (0.05 for these runs) into the
    graph, and it cannot be overridden at inference time. That is fatal for a
    *reference* measurement: COCO AP integrates the whole precision/recall
    curve, so a floored detector loses the low-score tail and scores below the
    TFLite export it is supposed to be the ceiling for. ``ave benchmark`` pins
    the TFLite runtimes to ``score_threshold=0`` for exactly this reason; this
    is the equivalent for the SavedModel, where the only way through is to
    re-export.

    Everything else is held fixed -- same checkpoint, same graph
    modifications, same ``iou_threshold`` and ``max_total_detections`` -- so the
    result differs from the stock export in the score floor alone.

    Writes ``<stage_dir>/saved_model_nms0/`` plus the patched
    ``pipeline.config`` beside it, and returns the SavedModel directory.
    """
    from agri_vision_edge.third_party import setup_tensorflow_models

    setup_tensorflow_models()

    import tensorflow as tf
    from object_detection.builders import model_builder
    from object_detection.exporter_lib_v2 import DETECTION_MODULE_MAP
    from object_detection.utils import config_util

    from agri_vision_edge.tfod import load_pipeline_config

    stage_dir = Path(stage_dir)

    if input_type not in DETECTION_MODULE_MAP:
        raise ValueError(
            f"Unrecognized input_type {input_type!r}; "
            f"expected one of {sorted(DETECTION_MODULE_MAP)}"
        )

    inferred_qat, inferred_per_channel = stage_graph_flags(stage_dir.name)
    qat = inferred_qat if qat is None else qat
    qat_per_channel = (
        inferred_per_channel if qat_per_channel is None else qat_per_channel
    )

    export_dir = (
        stage_dir / SCORING_EXPORT_NAME if output_dir is None else Path(output_dir)
    )

    pipeline_config = load_pipeline_config(stage_dir / "pipeline.config")
    pipeline_config.model.ssd.post_processing.batch_non_max_suppression.score_threshold = (  # noqa: E501
        score_threshold
    )
    resolution = pipeline_config.model.ssd.image_resizer.fixed_shape_resizer.height

    detection_model = model_builder.build(
        model_config=pipeline_config.model,
        is_training=False,
    )

    if qat:
        from agri_vision_edge.tfod.qat import (
            ensure_model_is_built_for_qat,
            quantize_detection_model,
        )

        ensure_model_is_built_for_qat(detection_model, pipeline_config)

        # Reproduce the *trained* graph, not the conversion rewrite: this export
        # stands in for the model as trained, so it mirrors `export_run` and
        # deliberately does not pass `for_export`.
        quantize_detection_model(
            detection_model,
            resolution,
            per_channel=qat_per_channel,
        )

    checkpoint = tf.train.latest_checkpoint(str(stage_dir / "checkpoint"))
    if not checkpoint:
        raise FileNotFoundError(f"No checkpoint to export in {stage_dir / 'checkpoint'}")

    ckpt = tf.train.Checkpoint(model=detection_model)
    status = ckpt.restore(checkpoint).expect_partial()

    module = DETECTION_MODULE_MAP[input_type](detection_model)
    concrete_function = module.__call__.get_concrete_function()
    status.assert_existing_objects_matched()

    tf.saved_model.save(
        module,
        str(export_dir),
        signatures=concrete_function,
    )

    # Keep the patched pipeline beside the export so the threshold it was built
    # with is recoverable from the artifact alone.
    config_util.save_pipeline_config(pipeline_config, str(export_dir))

    return export_dir


def export_run(
    cfg,
    *,
    export_dir=None,
    input_type: str = "image_tensor",
    qat: bool | None = None,
    qat_per_channel: bool | None = None,
) -> ExportResult:
    """
    Export the best checkpoint of ``cfg``'s run to a checkpoint + SavedModel.

    ``cfg`` may be a ``FinetuneRunConfig`` or a plain dict. The best checkpoint
    is the latest one in ``cfg.train_dir`` (the trainer only saves on metric
    improvement, so latest == best).

    By default the export mirrors how the run was trained:
      * ``qat`` defaults to ``cfg.qat``: False yields a clean fp32 detection
        checkpoint (ready to seed a follow-up QAT run); True reproduces the full
        int8 graph (fold BN + fake-quant backbone + head).
      * ``qat_per_channel`` defaults to ``cfg.qat_per_channel``. It no longer
        changes the training graph -- pin placement is chosen when the export
        graph is rebuilt for conversion -- so both settings produce the same
        variables and restore either checkpoint.

    Pass either explicitly to override.
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

    if qat is None:
        qat = cfg.qat
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

    if qat:
        from agri_vision_edge.tfod.qat import (
            ensure_model_is_built_for_qat,
            quantize_detection_model,
        )

        ensure_model_is_built_for_qat(detection_model, pipeline_config)

        # quantize_detection_model is self-contained: it folds BatchNorms and
        # inserts the fake-quant nodes for the WHOLE model, reproducing the exact
        # trained QAT graph so the checkpoint restores. FPN folds+quantizes the
        # backbone as its own graph then the combined head; plain SSD inlines the
        # backbone with the head into ONE combined functional graph.
        print("Folding + quantizing the full model (backbone + detection head)...")
        image_size = pipeline_config.model.ssd.image_resizer.fixed_shape_resizer.height
        quantize_detection_model(
            detection_model,
            image_size,
            per_channel=qat_per_channel,
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
