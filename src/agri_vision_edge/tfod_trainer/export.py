"""Export a finetune or QAT run's best checkpoint in TF model-zoo layout.

The export contains a model-only checkpoint, pipeline config, and SavedModel.
QAT recreates the trained folded/fake-quantized graph; plain finetuning exports
clean fp32 weights. The model-only layout can seed a later run without restoring
an earlier optimizer or global step.
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
#: The scoring export sits beside ``saved_model/``. TFLite conversion traces the
#: stock export at its original path.
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
    """Re-export a stage with a configurable baked-in NMS score threshold.

    All other graph and post-processing settings remain fixed. Returns the
    SavedModel directory, defaulting to ``<stage_dir>/saved_model_nms0``.
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

        # Reproduce the trained graph used by `export_run`; omit the conversion
        # rewrite selected by `for_export`.
        quantize_detection_model(
            detection_model,
            resolution,
            per_channel=qat_per_channel,
        )

    checkpoint = tf.train.latest_checkpoint(str(stage_dir / "checkpoint"))
    if not checkpoint:
        raise FileNotFoundError(
            f"No checkpoint to export in {stage_dir / 'checkpoint'}"
        )

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
    """Export a run's latest best checkpoint and SavedModel.

    ``cfg`` may be a mapping. QAT graph settings default to the training
    configuration and may be overridden explicitly.
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
