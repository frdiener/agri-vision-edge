import marimo

__generated_with = "0.23.9"
app = marimo.App(width="full", app_title="")


@app.cell(hide_code=True)
def configuration():
    import marimo as mo

    # one of [ptq, qat, qat_weigths, qat_folded]
    config_form = mo.md("""
    **Quantization scheme**

    - {quant_scheme}
    """).batch(
        quant_scheme=mo.ui.dropdown(
            options={
                "Post Training Quantization": {
                    "path": "finetune/",
                    "saved_model_path": "finetune/local_saved_model",
                    "qat_routine": None,
                    "tflite_path": "./tflite_models/ptq.tflite",
                    "precision": "int8"
                },
                "Quantization Aware Training": {
                    "path": "qat/",
                    "saved_model_path": "qat/local_saved_model",
                    "qat_routine": "full",
                    "tflite_path": "./tflite_models/qat.tflite",
                    "precision": "int8"
                },
                "QAT fp32": {
                    "path": "qat/",
                    "saved_model_path": "qat/local_saved_model",
                    "qat_routine": "full",
                    "tflite_path": "./tflite_models/qat_fp32.tflite",
                    "precision": "int32"
                },
                "Quantization Aware Training with quantized weights only": {
                    "path": "qat_weights_only/",
                    "saved_model_path": "qat_weights_only/local_saved_model",
                    "qat_routine": "weights",
                    "tflite_path": "./tflite_models/qat_weights.tflite",
                    "precision": "int8"
                },
                "Quantization Aware Training with prefolded Batchnorms": {
                    "path": "qat_folded/",
                    "saved_model_path": "qat_folded/local_saved_model",
                    "qat_routine": "fold",
                    "tflite_path": "./tflite_models/qat_fold.tflite",
                    "precision": "int8"
                },
            },
            value="Post Training Quantization",
            label="Quantization Scheme",
        ),
    ).form(bordered=False, submit_button_label="Convert & Evaluate")
    config_form
    return config_form, mo


@app.cell(hide_code=True)
def imports():
    import os
    import sys
    import json
    from pathlib import Path
    import numpy as np
    import tensorflow.compat.v2 as tf
    from google.protobuf import text_format

    from agri_vision_edge.third_party import setup_tensorflow_models
    setup_tensorflow_models()

    from agri_vision_edge.evaluation.artifacts import save_benchmark_artifacts, save_failure_artifact
    from agri_vision_edge.evaluation.benchmark import benchmark_runtime
    from agri_vision_edge.runtime.inference.tflite import TFLiteRuntime
    from agri_vision_edge.evaluation.dataset import load_coco_images

    from agri_vision_edge.third_party.phenobench import PhenoBench
    from agri_vision_edge.tfod.export import _load_pipeline_config
    from agri_vision_edge.experiment import ExperimentManifest
    from agri_vision_edge.experiment.environment import capture_environment
    from agri_vision_edge.tfod.qat import ensure_model_is_built_for_qat, quantize_backbone #, quantize_backbone_full
    from agri_vision_edge.data.coco import phenobench_bbox_to_xyxy
    from agri_vision_edge.data.preprocessing import resize_image_and_boxes
    from agri_vision_edge.tfod import fold_mobilenetv2_backbone as fold

    from object_detection.protos import pipeline_pb2
    from object_detection.builders import model_builder
    from object_detection.builders import post_processing_builder
    from object_detection.core import box_list
    from object_detection.core import standard_fields as fields
    from object_detection.export_tflite_graph_lib_tf2 import get_const_center_size_encoded_anchors, SSDModule
    import tensorflow_model_optimization as tfmot
    from tensorflow_model_optimization.quantization.keras import default_8bit

    dataset_dir = Path("./phenobench")
    dataset_raw_dir = Path("../datasets/PhenoBench")
    model_base_path = "qat-exports/"

    IMAGE_SIZE=320
    return (
        IMAGE_SIZE,
        Path,
        PhenoBench,
        SSDModule,
        TFLiteRuntime,
        benchmark_runtime,
        dataset_dir,
        dataset_raw_dir,
        ensure_model_is_built_for_qat,
        fold,
        json,
        load_coco_images,
        model_builder,
        np,
        os,
        phenobench_bbox_to_xyxy,
        quantize_backbone,
        resize_image_and_boxes,
        save_benchmark_artifacts,
        tf,
        tfmot,
    )


@app.cell(hide_code=True)
def dataset___conversion_helpers(
    IMAGE_SIZE,
    PhenoBench,
    dataset_dir,
    dataset_raw_dir,
    json,
    np,
    phenobench_bbox_to_xyxy,
    resize_image_and_boxes,
):
    with open(dataset_dir / "rep_dataset.json", "r") as f:
        rep_ds_indices = json.load(f)

    train_dataset = PhenoBench(
        root=dataset_raw_dir,
        split="train",
        target_types=["plant_bboxes"],
        ignore_partial=False,
    )

    def representative_dataset(
        dataset,
        indices=None,
        num_samples=100,
        size=320,
    ):
        """
        TFLite representative dataset generator.
        """

        if indices is None:
            indices = range(len(dataset))

        count = 0

        for i in indices:
            if count >= num_samples:
                break

            sample = dataset[i]

            image = np.array(
                sample["image"],
                dtype=np.uint8,
            )

            boxes = [
                phenobench_bbox_to_xyxy(bbox) for bbox in sample["plant_bboxes"]
            ]

            #
            # Skip empty samples
            #

            if not boxes:
                continue

            image_resized, _ = resize_image_and_boxes(
                image,
                boxes,
                size=size,
            )

            image_resized = image_resized.astype(np.float32)

            yield [
                np.expand_dims(
                    image_resized,
                    axis=0,
                )
            ]

            count += 1

    def normalized_representative_dataset():
        for sample in representative_dataset(
            dataset=train_dataset,
            indices=rep_ds_indices,
            num_samples=200,
            size=IMAGE_SIZE,
        ):
            x = sample[0]

            x = (
                2.0 / 255.0
            ) * x - 1.0

            yield [x]

    return rep_ds_indices, representative_dataset, train_dataset


@app.cell(hide_code=True)
def legacy_override(tf, tfmot):
    from object_detection.core.freezable_batch_norm import FreezableBatchNorm


    def _annotate_layer(layer, quantize_config=None):
        if isinstance(
            layer,
            (
                tf.keras.layers.Conv2D,
                tf.keras.layers.DepthwiseConv2D,
            ),
        ):
            return tfmot.quantization.keras.quantize_annotate_layer(
                layer, quantize_config=quantize_config
            )

        return layer


    def quantize_backbone_full(backbone):
        """
        Convert a TFOD MobileNetV2 backbone into a
        TF-MOT QAT backbone while preserving weights.
        """

        with tfmot.quantization.keras.quantize_scope(
            {
                "FreezableBatchNorm": FreezableBatchNorm,
            }
        ):
            annotated = tf.keras.models.clone_model(
                backbone,
                clone_function=_annotate_layer,
            )

            qat_backbone = tfmot.quantization.keras.quantize_apply(
                annotated,
            )

        return qat_backbone

    return (quantize_backbone_full,)


@app.cell(hide_code=True)
def _(
    Path,
    SSDModule,
    config_form,
    ensure_model_is_built_for_qat,
    fold,
    mo,
    model_builder,
    quantize_backbone,
    quantize_backbone_full,
    tf,
    tfmot,
):
    mo.stop(config_form.value is None, mo.md("_Configure parameters above and click **Convert & Evaluate** to start._"))
    conf = config_form.value['quant_scheme']
    if not Path(conf['path']).exists():
        raise RuntimeError(f"Path not found: {conf['path']}")

    pipeline_config = _load_pipeline_config(conf["path"] + "pipeline.config")

    detection_model = model_builder.build(pipeline_config.model, is_training=False)

    ensure_model_is_built_for_qat(detection_model, pipeline_config)

    if conf["qat_routine"]:
        backbone = detection_model.feature_extractor.classification_backbone

        if conf["qat_routine"] == "fold":
            folded_backbone = fold(backbone)
            annotated = tfmot.quantization.keras.quantize_annotate_model(
                folded_backbone
            )
            qat_backbone = tfmot.quantization.keras.quantize_apply(
                annotated,
                # scheme=default_8bit.Default8BitQuantizationScheme(
                #     disable_per_axis=True
                # )
            )
        elif conf["qat_routine"] == "weights":
            qat_backbone = quantize_backbone(backbone)

        elif conf["qat_routine"] == "full":
            qat_backbone = quantize_backbone_full(backbone)

        detection_model.feature_extractor.classification_backbone = qat_backbone


    # The module helps build a TF SavedModel appropriate for TFLite conversion.
    detection_module = SSDModule(pipeline_config, detection_model, max_detections=60, use_regular_nms=False)

    # restore model wheights
    ckpt = tf.train.Checkpoint(model=detection_model)
    ckpt.restore(
        tf.train.latest_checkpoint(conf['path'])
    ).expect_partial().assert_existing_objects_matched()

    concrete_function = (
        detection_module.inference_fn
        .get_concrete_function(
            tf.TensorSpec(
                shape=detection_module.input_shape(),
                dtype=tf.float32,
                name="input",
            )
        )
    )
    return concrete_function, conf, detection_module


@app.cell(hide_code=True)
def tflite_conversion(
    IMAGE_SIZE,
    concrete_function,
    conf,
    detection_module,
    rep_ds_indices,
    representative_dataset,
    tf,
    train_dataset,
):
    converter = tf.lite.TFLiteConverter.from_concrete_functions(
        [concrete_function],
        trackable_obj=detection_module,
    )

    if conf['precision'] == 'int8':
        converter.optimizations = [tf.lite.Optimize.DEFAULT]

    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,
        tf.lite.OpsSet.TFLITE_BUILTINS_INT8,
    ]

    if conf['precision'] == 'int8':
        converter.inference_input_type = tf.int8
    elif conf['precision'] == 'fp32':
        converter.inference_input_type = tf.fp32
    converter.inference_output_type = tf.float32

    converter.representative_dataset = lambda: representative_dataset(
        dataset=train_dataset,
        indices=rep_ds_indices,
        num_samples=200,
        size=IMAGE_SIZE
    )

    converter._experimental_new_quantizer = True
    # converter._experimental_disable_per_channel = True

    tflite_model = converter.convert()

    # interpreter = tf.lite.Interpreter(model_content=tflite_model)
    return (tflite_model,)


@app.cell
def _(FLAGS, Path, conf, flags, os):
    import flatbuffers

    from tflite_support import metadata_schema_py_generated as _metadata_fb
    from tflite_support import metadata as _metadata


    def define_flags():
      flags.DEFINE_string("model_file", None,
                          "Path and file name to the TFLite model file.")
      flags.DEFINE_string("label_file", None, "Path to the label file.")
      flags.DEFINE_string("export_directory", None,
                          "Path to save the TFLite model files with metadata.")
      flags.mark_flag_as_required("model_file")
      flags.mark_flag_as_required("label_file")
      flags.mark_flag_as_required("export_directory")


    class ModelSpecificInfo(object):
      """Holds information that is specificly tied to an image classifier."""

      def __init__(self, name, version, image_width, image_height, image_min,
                   image_max, mean, std, num_classes, author):
        self.name = name
        self.version = version
        self.image_width = image_width
        self.image_height = image_height
        self.image_min = image_min
        self.image_max = image_max
        self.mean = mean
        self.std = std
        self.num_classes = num_classes
        self.author = author


    _MODEL_INFO = {
        "mobilenet_v1_0.75_160_quantized.tflite":
            ModelSpecificInfo(
                name="MobileNetV1 image classifier",
                version="v1",
                image_width=160,
                image_height=160,
                image_min=0,
                image_max=255,
                mean=[127.5],
                std=[127.5],
                num_classes=1001,
                author="TensorFlow")
    }


    class MetadataPopulatorForImageClassifier(object):
      """Populates the metadata for an image classifier."""

      def __init__(self, model_file, model_info, label_file_path):
        self.model_file = model_file
        self.model_info = model_info
        self.label_file_path = label_file_path
        self.metadata_buf = None

      def populate(self):
        """Creates metadata and then populates it for an image classifier."""
        self._create_metadata()
        self._populate_metadata()

      def _create_metadata(self):
        """Creates the metadata for an image classifier."""

        # Creates model info.
        model_meta = _metadata_fb.ModelMetadataT()
        model_meta.name = self.model_info.name
        model_meta.description = ("Identify the most prominent object in the "
                                  "image from a set of %d categories." %
                                  self.model_info.num_classes)
        model_meta.version = self.model_info.version
        model_meta.author = self.model_info.author
        model_meta.license = ("Apache License. Version 2.0 "
                              "http://www.apache.org/licenses/LICENSE-2.0.")

        # Creates input info.
        input_meta = _metadata_fb.TensorMetadataT()
        input_meta.name = "image"
        input_meta.description = (
            "Input image to be classified. The expected image is {0} x {1}, with "
            "three channels (red, blue, and green) per pixel. Each value in the "
            "tensor is a single byte between {2} and {3}.".format(
                self.model_info.image_width, self.model_info.image_height,
                self.model_info.image_min, self.model_info.image_max))
        input_meta.content = _metadata_fb.ContentT()
        input_meta.content.contentProperties = _metadata_fb.ImagePropertiesT()
        input_meta.content.contentProperties.colorSpace = (
            _metadata_fb.ColorSpaceType.RGB)
        input_meta.content.contentPropertiesType = (
            _metadata_fb.ContentProperties.ImageProperties)
        input_normalization = _metadata_fb.ProcessUnitT()
        input_normalization.optionsType = (
            _metadata_fb.ProcessUnitOptions.NormalizationOptions)
        input_normalization.options = _metadata_fb.NormalizationOptionsT()
        input_normalization.options.mean = self.model_info.mean
        input_normalization.options.std = self.model_info.std
        input_meta.processUnits = [input_normalization]
        input_stats = _metadata_fb.StatsT()
        input_stats.max = [self.model_info.image_max]
        input_stats.min = [self.model_info.image_min]
        input_meta.stats = input_stats

        # Creates output info.
        output_meta = _metadata_fb.TensorMetadataT()
        output_meta.name = "probability"
        output_meta.description = "Probabilities of the %d labels respectively." % self.model_info.num_classes
        output_meta.content = _metadata_fb.ContentT()
        output_meta.content.content_properties = _metadata_fb.FeaturePropertiesT()
        output_meta.content.contentPropertiesType = (
            _metadata_fb.ContentProperties.FeatureProperties)
        output_stats = _metadata_fb.StatsT()
        output_stats.max = [1.0]
        output_stats.min = [0.0]
        output_meta.stats = output_stats
        label_file = _metadata_fb.AssociatedFileT()
        label_file.name = os.path.basename(self.label_file_path)
        label_file.description = "Labels for objects that the model can recognize."
        label_file.type = _metadata_fb.AssociatedFileType.TENSOR_AXIS_LABELS
        output_meta.associatedFiles = [label_file]

        # Creates subgraph info.
        subgraph = _metadata_fb.SubGraphMetadataT()
        subgraph.inputTensorMetadata = [input_meta]
        subgraph.outputTensorMetadata = [output_meta]
        model_meta.subgraphMetadata = [subgraph]

        b = flatbuffers.Builder(0)
        b.Finish(
            model_meta.Pack(b),
            _metadata.MetadataPopulator.METADATA_FILE_IDENTIFIER)
        self.metadata_buf = b.Output()

      def _populate_metadata(self):
        """Populates metadata and label file to the model file."""
        populator = _metadata.MetadataPopulator.with_model_file(self.model_file)
        populator.load_metadata_buffer(self.metadata_buf)
        populator.load_associated_files([self.label_file_path])
        populator.populate()



    model_basename = os.path.basename(conf["tflite_path"])
    if model_basename not in _MODEL_INFO:
        raise ValueError(
            "The model info for, {0}, is not defined yet.".format(model_basename))

    # Generate the metadata objects and put them in the model file
    populator = MetadataPopulatorForImageClassifier(
      conf["tflite_path"], _MODEL_INFO.get(model_basename), FLAGS.label_file)
    populator.populate()

    # Validate the output model file by reading the metadata and produce
    # a json file with the metadata under the export path
    displayer = _metadata.MetadataDisplayer.with_model_file(conf["tflite_path"])
    export_json_file = os.path.join(Path(conf["tflite_path"]).parent(),
                                  os.path.splitext(model_basename)[0] + ".json")
    json_file = displayer.get_metadata_json()
    with open(export_json_file, "w") as export_json_file:
        export_json_file.write(json_file)

    print("Finished populating metadata and associated file to the model:")
    print(conf["tflite_path"])
    print("The metadata json file has been saved to:")
    print(export_json_file)
    print("The associated file that has been been packed to the model is:")
    print(displayer.get_packed_associated_file_list())
    return


@app.cell(hide_code=True)
def _(Path, conf, tflite_model):
    with open(conf["tflite_path"], "wb") as tflite_file:
        tflite_file.write(tflite_model)

    if Path(conf["tflite_path"]).exists():
        print(f"tflite model written to {Path(conf['tflite_path'])}")
    return


@app.cell(disabled=True, hide_code=True)
def _(interpreter):
    input_details = (
        interpreter.get_input_details()
    )

    output_details = (
        interpreter.get_output_details()
    )

    # interpreter.resize_tensor_input(
    #     input_details[0]["index"],
    #     [1, 320, 320, 3],
    # )

    # input_details = (
    #     interpreter.get_input_details()
    # )

    interpreter.allocate_tensors()

    print(input_details)

    for d in output_details:
        print(
            d["name"],
            d["shape"],
            d["dtype"],
            d["quantization"],
        )
    return


@app.cell
def _(config_form):
    config_form.element.value 
    return


@app.cell(hide_code=True)
def _(
    Path,
    TFLiteRuntime,
    benchmark_runtime,
    conf,
    load_coco_images,
    save_benchmark_artifacts,
):
    annotations_path = Path("./phenobench/val_annotations.json")
    image_records = load_coco_images(
        Path("../datasets/PhenoBench/val/images/"),
        annotations_path,
    )
    model_path = Path(conf["tflite_path"])
    output_dir = Path("./benchmark_results/") / model_path.stem

    print(
        f"\n=== Benchmarking: "
        f"{model_path.name} ==="
    )

    runtime = TFLiteRuntime(
        model_path=model_path,
        delegate_path=None,
    )

    result = benchmark_runtime(
        runtime,
        image_records,
    )

    save_benchmark_artifacts(
        output_dir=output_dir,
        benchmark_result=result,
        runtime=runtime,
        model_name=model_path.name,
        delegate=None,
    )

    mean_latency = (
        sum(result.latencies_ms)
        / len(result.latencies_ms)
    )

    print(
        f"mean latency: "
        f"{mean_latency:.2f} ms"
    )

    print(
        f"exported "
        f"{len(result.predictions)} "
        f"prediction(s)"
    )
    return annotations_path, output_dir


@app.cell(hide_code=True)
def _(annotations_path, output_dir):
    from agri_vision_edge.evaluation.coco import (
        evaluate_predictions,
        save_metrics,
    )

    predictions_path = output_dir / "predictions.json"
    metrics_path = output_dir / "metrics.json"

    print(
        f"\n=== Evaluating: "
        f"{output_dir.name} ==="
    )

    metrics = (
        evaluate_predictions(
            annotations_path,
            predictions_path,
        )
    )

    save_metrics(
        metrics,
        metrics_path,
    )

    print()

    print(
        f"AP:   "
        f"{metrics['AP']:.4f}"
    )

    print(
        f"AP50: "
        f"{metrics['AP50']:.4f}"
    )

    print(
        f"AP75: "
        f"{metrics['AP75']:.4f}"
        )
    return


if __name__ == "__main__":
    app.run()
