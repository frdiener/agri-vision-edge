"""
TensorFlow Object Detection export utilities.

Provides helpers for exporting TF-OD checkpoints
to TensorFlow SavedModel and TFLite-compatible
SavedModel formats.
"""

from pathlib import Path

import tensorflow as tf

from google.protobuf import text_format

from object_detection import exporter_lib_v2
from object_detection import export_tflite_graph_lib_tf2
from object_detection.protos import pipeline_pb2


def _load_pipeline_config(
    pipeline_config_path,
    config_override="",
):
    """
    Load and optionally override pipeline config.
    """

    pipeline_config = (
        pipeline_pb2.TrainEvalPipelineConfig()
    )

    with tf.io.gfile.GFile(
        pipeline_config_path,
        "r",
    ) as f:
        text_format.Parse(
            f.read(),
            pipeline_config,
        )

    if config_override:

        override_config = (
            pipeline_pb2.TrainEvalPipelineConfig()
        )

        text_format.Parse(
            config_override,
            override_config,
        )

        pipeline_config.MergeFrom(
            override_config
        )

    return pipeline_config


def export_saved_model(
    pipeline_config_path,
    checkpoint_dir,
    output_dir,
    input_type="image_tensor",
    config_override="",
):
    """
    Export standard TF2 SavedModel.

    Produces:
        output_dir/
        ├── checkpoint/
        ├── pipeline.config
        └── saved_model/
    """

    output_dir = Path(output_dir)

    pipeline_config = _load_pipeline_config(
        pipeline_config_path,
        config_override=config_override,
    )

    exporter_lib_v2.export_inference_graph(
        input_type=input_type,
        pipeline_config=pipeline_config,
        trained_checkpoint_dir=checkpoint_dir,
        output_directory=str(output_dir),
        use_side_inputs=False,
        side_input_shapes=None,
        side_input_types=None,
        side_input_names=None,
    )

    return output_dir


def export_tflite_graph(
    pipeline_config_path,
    checkpoint_dir,
    output_dir,
    config_override="",
    max_detections=10,
    ssd_use_regular_nms=False,
    centernet_include_keypoints=False,
    keypoint_label_map_path=None,
):
    """
    Export TF2 TFLite-ready graph.

    This exports the intermediate SavedModel intended
    for later TFLite conversion.

    Produces:
        output_dir/
        ├── saved_model/
        └── pipeline.config
    """

    output_dir = Path(output_dir)

    pipeline_config = _load_pipeline_config(
        pipeline_config_path,
        config_override=config_override,
    )

    export_tflite_graph_lib_tf2.export_tflite_model(
        pipeline_config=pipeline_config,
        trained_checkpoint_dir=checkpoint_dir,
        output_directory=str(output_dir),
        max_detections=max_detections,
        use_regular_nms=ssd_use_regular_nms,
        include_keypoints=(
            centernet_include_keypoints
        ),
        label_map_path=(
            keypoint_label_map_path or ""
        ),
    )

    return output_dir


def export_all(
    pipeline_config_path,
    checkpoint_dir,
    export_dir,
    config_override="",
    input_type="image_tensor",

    # TFLite export options
    max_detections=10,
    ssd_use_regular_nms=False,
    centernet_include_keypoints=False,
    keypoint_label_map_path=None,
):
    """
    Export BOTH:

    1. Standard SavedModel
    2. TFLite-ready graph export

    Structure:

    exports/
    ├── exported_model/
    └── tflite_graph/
    """

    export_dir = Path(export_dir)

    saved_model_dir = (
        export_dir / "exported_model"
    )

    tflite_graph_dir = (
        export_dir / "tflite_graph"
    )

    # =========================================================
    # Standard inference SavedModel
    # =========================================================

    export_saved_model(
        pipeline_config_path=pipeline_config_path,
        checkpoint_dir=checkpoint_dir,
        output_dir=saved_model_dir,
        input_type=input_type,
        config_override=config_override,
    )

    # =========================================================
    # TFLite-ready graph export
    # =========================================================

    export_tflite_graph(
        pipeline_config_path=pipeline_config_path,
        checkpoint_dir=checkpoint_dir,
        output_dir=tflite_graph_dir,
        config_override=config_override,
        max_detections=max_detections,
        ssd_use_regular_nms=ssd_use_regular_nms,
        centernet_include_keypoints=(
            centernet_include_keypoints
        ),
        keypoint_label_map_path=(
            keypoint_label_map_path
        ),
    )

    return {
        "saved_model": saved_model_dir,
        "tflite_graph": tflite_graph_dir,
    }
