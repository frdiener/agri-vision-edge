import marimo

__generated_with = "0.23.9"
app = marimo.App(width="full", app_title="")


@app.cell
def _(Path):
    import marimo as mo

    ARTIFACTS_DIR = Path("artifacts")
    models = list((ARTIFACTS_DIR / "tf").iterdir())

    model_root = mo.ui.dropdown(
        options={str(p.name): p for p in models},
        value=str(models[1].name),
        label="Model Root",
    )
    return ARTIFACTS_DIR, mo, model_root


@app.cell(hide_code=True)
def configuration(ARTIFACTS_DIR, Path, mo, model_root):
    _finetune_config = _load_pipeline_config(
        next(Path(model_root.value).iterdir()) / "pipeline.config"
    )

    _original_dataset = Path(
        str(_finetune_config.train_input_reader.tf_record_input_reader.input_path)
    ).parent.name

    TFLITE_MODELS_DIR = ARTIFACTS_DIR / "tflite"
    TFLITE_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    DATASETS_DIR = Path("datasets")

    config_form = mo.md("""
    **TFLite Conversion**

    - {model_root}
    - {model}
    - {dataset}
      {tiled}
      {classes}
    - {quantization}
      {precision} {per_channel}
    """).batch(
        model_root=model_root,
        model=mo.ui.dropdown(
            options={
                "SSD MobileNet v2": "mnv2",
                "SSD MobileNet v2 FPNlite": "mnv1_fpnlite",
            },
            value="SSD MobileNet v2",
            label="Model",
        ),
        dataset=mo.ui.dropdown(
            options={
                "Phenobench": "phenobench",
            },
            value="Phenobench",
            label="Dataset",
        ),
        tiled=mo.ui.switch(
            value=False,
            label="Tiled 2x2",
        ),
        classes=mo.ui.radio(
            options={
                "Single class (Weeds)": "sc",
                "Multiclass (Crops vs Weeds)": "mc",
            },
            value="Single class (Weeds)",
        ),
        quantization=mo.ui.dropdown(
            options={
                "Finetuned": "ptq",
                "Finetuned with full QAT": "qat0",
                "Finetuned with QAT (Quantized Weights only)": "qat1",
                "Finetuned with QAT (Prefolded Batchnorms)": "qat2",
            },
            value="Finetuned",
            label="Checkpoint",
        ),
        precision=mo.ui.dropdown(
            options={
                "int8": "int8",
                "fp32": "fp32",
            },
            value="int8",
            label="Precision",
        ),
        per_channel=mo.ui.switch(
            value=False,
            label="Per Channel Quantization",
        ),
    ).form(bordered=False, submit_button_label="Convert & Evaluate")

    summary = mo.md(f"""
    Dataset:

    | Setting | Value |
    |----------|----------|
    | Dataset | {_original_dataset} |
    | Tiled | {'tile' in _original_dataset} |
    | Labels | {'Single Class' if _finetune_config.model.ssd.num_classes == 1 else 'Multi Class'} |
    | Resolution | {_finetune_config.model.ssd.image_resizer.fixed_shape_resizer.width} |
    """)

    mo.hstack([
        config_form,
        summary,
    ])
    return DATASETS_DIR, TFLITE_MODELS_DIR, config_form


@app.cell(hide_code=True)
def imports():
    import os
    import json
    from pathlib import Path
    import numpy as np
    import tensorflow.compat.v2 as tf
    from phenobench import PhenoBench
    from google.protobuf import text_format

    from agri_vision_edge.third_party import setup_tensorflow_models

    setup_tensorflow_models()

    from agri_vision_edge.data.tiling import TiledPhenoBench, FilterConfig
    from agri_vision_edge.data.rep_dataset import (
        representative_dataset,
        normalized_representative_dataset,
    )
    from agri_vision_edge.evaluation.artifacts import save_benchmark_artifacts
    from agri_vision_edge.evaluation.benchmark import benchmark_runtime
    from agri_vision_edge.runtime.inference.tflite import TFLiteRuntime
    from agri_vision_edge.evaluation.dataset import load_coco_images

    from agri_vision_edge.tfod.export import _load_pipeline_config
    from agri_vision_edge.experiment import ExperimentManifest
    from agri_vision_edge.tfod.qat import (
        ensure_model_is_built_for_qat,
        quantize_backbone,
        # quantize_backbone_full,
    ) 
    from agri_vision_edge.data.coco import phenobench_bbox_to_xyxy
    from agri_vision_edge.data.preprocessing import resize_image_and_boxes
    from agri_vision_edge.tfod import fold_mobilenetv2_backbone as fold

    from object_detection.builders import model_builder
    from object_detection.export_tflite_graph_lib_tf2 import SSDModule
    import tensorflow_model_optimization as tfmot

    return (
        FilterConfig,
        Path,
        PhenoBench,
        SSDModule,
        TFLiteRuntime,
        TiledPhenoBench,
        benchmark_runtime,
        ensure_model_is_built_for_qat,
        fold,
        json,
        load_coco_images,
        model_builder,
        np,
        os,
        quantize_backbone,
        representative_dataset,
        save_benchmark_artifacts,
        tf,
    )


@app.cell
def dataset___conversion_helpers(
    DATASETS_DIR,
    FilterConfig,
    Path,
    PhenoBench,
    TFLITE_MODELS_DIR,
    TiledPhenoBench,
    config_form,
    json,
    mo,
):
    mo.stop(
        config_form.value is None,
        mo.md(
            "_Configure parameters above and click **Convert & Evaluate** to start._"
        ),
    )
    config = config_form.value

    MODEL_ROOT = config['model_root']
    CHECKPOINT = MODEL_ROOT / config["quantization"] / "checkpoint"
    PIPELINE_CONFIG = _load_pipeline_config(
        CHECKPOINT.parent / "pipeline.config"
    )
    config['original_dataset'] = Path(str(PIPELINE_CONFIG.train_input_reader.tf_record_input_reader.input_path)).parent.name
    config['num_classes'] = PIPELINE_CONFIG.model.ssd.num_classes
    config['resolution'] = PIPELINE_CONFIG.model.ssd.image_resizer.fixed_shape_resizer.width


    QUANT = config["quantization"]
    MODEL_FILE_PATH = TFLITE_MODELS_DIR / (
        f"{config['model_root'].name}_"
        + f"{'sc' if config['num_classes'] == 1 else 'mc'}_"
        + f"{config['dataset']}{'-tiled' if 'tiled' in config['original_dataset'] else ''}_"
        + f"{config['resolution']}_"
        + f"{config['precision']}_"
        + f"{config['quantization']}"
        + f"{'_per-channel' if config['per_channel'] else ''}"
        + ".tflite"
    )

    dataset_dir = DATASETS_DIR / f"{config['dataset']}_{config['classes']}{'_tiled' if config['tiled'] else ''}"
    dataset_raw_dir = (
        DATASETS_DIR
        / f"{config['dataset']}_raw_{'tiled' if config['tiled'] else 'full'}"
    )

    if not dataset_dir.exists():
        raise RuntimeError("Configured dataset not found.")



    IMAGE_SIZE = config["resolution"]

    rep_ds_indices = json.load((dataset_dir / "rep_dataset.json").open())

    if config["tiled"]:
        train_dataset = PhenoBench(
            root=dataset_raw_dir,
            split="train",
            target_types=[
                "semantics",
                "plant_instances",
            ],
            ignore_partial=True,
        )

        train_dataset = TiledPhenoBench(
            train_dataset,
            rows=2,
            cols=2,
            overlap=0.0,
            filter_config=FilterConfig(
                min_instance_pixels=32,
                min_bbox_width=4,
                min_bbox_height=4,
                min_bbox_area=32,
            ),
        )
    else:
        train_dataset = PhenoBench(
            root=dataset_raw_dir,
            split="train",
            target_types=["plant_bboxes"],
            ignore_partial=False,
        )
    return (
        CHECKPOINT,
        IMAGE_SIZE,
        MODEL_FILE_PATH,
        PIPELINE_CONFIG,
        config,
        dataset_dir,
        dataset_raw_dir,
        rep_ds_indices,
        train_dataset,
    )


@app.cell
def _():
    from tensorflow_model_optimization.python.core.quantization.keras.default_8bit.default_8bit_quantize_configs import (
        NoOpQuantizeConfig,
    )

    return


@app.cell
def _():
    return


@app.cell(hide_code=True)
def _(
    CHECKPOINT,
    PIPELINE_CONFIG,
    SSDModule,
    config,
    ensure_model_is_built_for_qat,
    fold,
    model_builder,
    quantize_backbone,
    quantize_backbone_full,
    tf,
):
    detection_model = model_builder.build(PIPELINE_CONFIG.model, is_training=False)

    ensure_model_is_built_for_qat(detection_model, PIPELINE_CONFIG)

    if config["quantization"] != "ptq":
        backbone = detection_model.feature_extractor.classification_backbone

        if config["quantization"] == "qat0":  # full quantization
            qat_backbone = quantize_backbone_full(backbone)

        elif config["quantization"] == "qat1":  # weights only quantization
            qat_backbone = quantize_backbone(backbone)

        elif config["quantization"] == "qat2":  # pre-folded quantization
            folded_backbone = fold(backbone)
            qat_backbone = quantize_backbone(
                folded_backbone,
                scheme="mixed"
            )

        detection_model.feature_extractor.classification_backbone = qat_backbone


    # The module helps build a TF SavedModel appropriate for TFLite conversion.
    detection_module = SSDModule(PIPELINE_CONFIG, detection_model, max_detections=60, use_regular_nms=False)

    # restore model wheights
    ckpt = tf.train.Checkpoint(model=detection_model)
    ckpt.restore(
        tf.train.latest_checkpoint(CHECKPOINT)
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

    # backbone = detection_model.feature_extractor.classification_backbone
    # folded_backbone = fold(backbone)
    # qat_backbone = quantize_backbone(
    #     folded_backbone,
    #     scheme="full"
    # )
    return concrete_function, detection_module, qat_backbone


@app.cell
def _(qat_backbone):
    for layer in qat_backbone.layers:
        print(layer.name)
        try:
            print(
                f"===>{type(layer.quantize_config).__name__}"
            )
        except Exception:
            pass

        for w in layer.weights:
            if "kernel" in w.name or "depthwise" in w.name:
                print("   ", w.name)

    for layer in qat_backbone.layers:
        if layer.name == "quant_Conv1_folded":
            print(type(layer.layer))
            print(layer.layer.activation)


    for layer in qat_backbone.layers:
        if layer.name == "quant_block_1_project_folded":
            print(type(layer.layer))
            print(layer.layer.activation)
    return


@app.cell(hide_code=True)
def tflite_conversion(
    IMAGE_SIZE,
    concrete_function,
    config,
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

    if config['precision'] == 'int8':
        converter.optimizations = [tf.lite.Optimize.DEFAULT]

    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,
        tf.lite.OpsSet.TFLITE_BUILTINS_INT8,
    ]

    if config['precision'] == 'int8':
        converter.inference_input_type = tf.int8
    elif config['precision'] == 'fp32':
        converter.inference_input_type = tf.float32
    converter.inference_output_type = tf.float32

    # SSDModule.inference_fn calls model.predict() directly, WITHOUT the SSD
    # preprocessing step, so the converted graph expects already-normalized
    # [-1, 1] input. representative_dataset yields raw [0, 255] images, so we
    # must normalize here. Feeding [0, 255] saturates the backbone during
    # calibration and mis-calibrates the (non-QAT) class head: the class-logit
    # tensor's max gets pinned to 0, capping every detection score at
    # sigmoid(0) = 0.5. (Independent of the QAT scheme.)
    def _nordmalized_rep_dataset():
        for sample in representative_dataset(
            dataset=train_dataset,
            indices=rep_ds_indices,
            num_samples=200,
            size=IMAGE_SIZE,
        ):
            yield [(2.0 / 255.0) * sample[0] - 1.0]

    converter.representative_dataset = _normalized_rep_dataset

    converter._experimental_new_quantizer = False
    converter._experimental_disable_per_channel = not config['per_channel']

    tflite_model = converter.convert()
    return (tflite_model,)


@app.cell(hide_code=True)
def _(
    FLAGS,
    MODEL_FILE_PATH,
    PIPELINE_CONFIG,
    conf,
    config,
    flags,
    os,
    tflite_model,
):
    tflite_model

    import flatbuffers

    from tflite_support import metadata_schema_py_generated as _metadata_fb
    from tflite_support import metadata as _metadata


    def define_flags():
        flags.DEFINE_string(
            "model_file", None, "Path and file name to the TFLite model file."
        )
        flags.DEFINE_string("label_file", None, "Path to the label file.")
        flags.DEFINE_string(
            "export_directory",
            None,
            "Path to save the TFLite model files with metadata.",
        )
        flags.mark_flag_as_required("model_file")
        flags.mark_flag_as_required("label_file")
        flags.mark_flag_as_required("export_directory")


    class ModelSpecificInfo(object):
        """Holds information that is specificly tied to an image classifier."""

        def __init__(
            self,
            name,
            version,
            image_width,
            image_height,
            image_min,
            image_max,
            mean,
            std,
            num_classes,
            author,
        ):
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
        "mobilenet_v1_0.75_160_quantized.tflite": ModelSpecificInfo(
            name="MobileNetV1 image classifier",
            version="v1",
            image_width=160,
            image_height=160,
            image_min=0,
            image_max=255,
            mean=[127.5],
            std=[127.5],
            num_classes=1001,
            author="TensorFlow",
        ),
        MODEL_FILE_PATH.name: ModelSpecificInfo(
            name="SSD MobileNetV2 Object Detector",
            version="v1",
            image_height=config['resolution'],
            image_width=config['resolution'],
            image_min=-128,
            image_max=127,
            mean=[0],
            std=[0],
            num_classes=PIPELINE_CONFIG.model.ssd.num_classes,
            author="fdi",
        )
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
            model_meta.description = (
                "Identify the most prominent object in the "
                "image from a set of %d categories." % self.model_info.num_classes
            )
            model_meta.version = self.model_info.version
            model_meta.author = self.model_info.author
            model_meta.license = (
                "Apache License. Version 2.0 "
                "http://www.apache.org/licenses/LICENSE-2.0."
            )

            # Creates input info.
            input_meta = _metadata_fb.TensorMetadataT()
            input_meta.name = "image"
            input_meta.description = (
                "Input image to be classified. The expected image is {0} x {1}, with "
                "three channels (red, blue, and green) per pixel. Each value in the "
                "tensor is a single byte between {2} and {3}.".format(
                    self.model_info.image_width,
                    self.model_info.image_height,
                    self.model_info.image_min,
                    self.model_info.image_max,
                )
            )
            input_meta.content = _metadata_fb.ContentT()
            input_meta.content.contentProperties = _metadata_fb.ImagePropertiesT()
            input_meta.content.contentProperties.colorSpace = (
                _metadata_fb.ColorSpaceType.RGB
            )
            input_meta.content.contentPropertiesType = (
                _metadata_fb.ContentProperties.ImageProperties
            )
            input_normalization = _metadata_fb.ProcessUnitT()
            input_normalization.optionsType = (
                _metadata_fb.ProcessUnitOptions.NormalizationOptions
            )
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
            output_meta.description = (
                "Probabilities of the %d labels respectively."
                % self.model_info.num_classes
            )
            output_meta.content = _metadata_fb.ContentT()
            output_meta.content.content_properties = (
                _metadata_fb.FeaturePropertiesT()
            )
            output_meta.content.contentPropertiesType = (
                _metadata_fb.ContentProperties.FeatureProperties
            )
            output_stats = _metadata_fb.StatsT()
            output_stats.max = [1.0]
            output_stats.min = [0.0]
            output_meta.stats = output_stats
            label_file = _metadata_fb.AssociatedFileT()
            label_file.name = os.path.basename(self.label_file_path)
            label_file.description = (
                "Labels for objects that the model can recognize."
            )
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
                _metadata.MetadataPopulator.METADATA_FILE_IDENTIFIER,
            )
            self.metadata_buf = b.Output()

        def _populate_metadata(self):
            """Populates metadata and label file to the model file."""
            populator = _metadata.MetadataPopulator.with_model_file(
                self.model_file
            )
            populator.load_metadata_buffer(self.metadata_buf)
            populator.load_associated_files([self.label_file_path])
            populator.populate()


    model_basename = MODEL_FILE_PATH.stem
    if model_basename not in _MODEL_INFO:
        print(
            "The model info for, {0}, is not defined yet.".format(model_basename)
        )
        # raise ValueError(
        #     "The model info for, {0}, is not defined yet.".format(model_basename))

    else:
        # Generate the metadata objects and put them in the model file
        populator = MetadataPopulatorForImageClassifier(
            MODEL_FILE_PATH, _MODEL_INFO.get(model_basename), FLAGS.label_file
        )
        populator.populate()

        # Validate the output model file by reading the metadata and produce
        # a json file with the metadata under the export path
        displayer = _metadata.MetadataDisplayer.with_model_file(
            MODEL_FILE_PATH
        )
        export_json_file = os.path.join(
            MODEL_FILE_PATH.parent(),
            os.path.splitext(model_basename)[0] + ".json",
        )
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
def _(MODEL_FILE_PATH, tflite_model):
    MODEL_FILE_PATH.write_bytes(tflite_model)
    if MODEL_FILE_PATH.exists():
        print(f"tflite model written to {MODEL_FILE_PATH}")
        written = True
    return (written,)


@app.cell(hide_code=True)
def _(tf, tflite_model):
    interpreter = tf.lite.Interpreter(model_content=tflite_model)

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
    return (interpreter,)


@app.cell
def _(interpreter):
    for t in interpreter.get_tensor_details():
        q = t["quantization"]
        if q != (0.0, 0):
            print(
                t["index"],
                t["name"],
                q,
            )
    return


@app.cell
def _(interpreter, np):
    for tensor in interpreter.get_tensor_details():
        if not 'relu6' in tensor['name'].lower():
            continue
        qparams = tensor["quantization_parameters"]

        scales = qparams["scales"]
        zero_points = qparams["zero_points"]

        if len(scales) == 0:
            continue

        dtype = tensor["dtype"]

        if dtype == np.int8:
            qmax = 127
        elif dtype == np.uint8:
            qmax = 255
        else:
            continue

        failed = False

        for i, scale in enumerate(scales):
            zp_idx = 0 if len(zero_points) == 1 else i
            zp = int(zero_points[zp_idx])

            mesa_range = (qmax - zp) * float(scale)

            if mesa_range > 6.0:
                failed = True

            print(
                f"{tensor['index']:4d} "
                f"{tensor['name']:<60}\n"
                f"     scale={scale:.9f} "
                f"zp={zp:4d} "
                f"range={mesa_range:.6f}"
            )

        if failed:
            print("     <-- FAILS MESA RELU6 CHECK")
    return


@app.cell
def _(dataset_dir):
    dataset_dir
    return


@app.cell(hide_code=True)
def _(
    MODEL_FILE_PATH,
    Path,
    TFLiteRuntime,
    benchmark_runtime,
    dataset_dir,
    dataset_raw_dir,
    load_coco_images,
    save_benchmark_artifacts,
    written,
):
    assert written and MODEL_FILE_PATH.exists()

    annotations_path = dataset_dir / "test_annotations.json"
    image_records = load_coco_images(
        dataset_raw_dir / "val/images/",
        annotations_path,
    )

    output_dir = Path("./benchmark_results/") / MODEL_FILE_PATH.stem

    print(
        f"\n=== Benchmarking: "
        f"{MODEL_FILE_PATH.name} ==="
    )

    runtime = TFLiteRuntime(
        model_path=MODEL_FILE_PATH,
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
        model_name=MODEL_FILE_PATH.name,
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
    return annotations_path, output_dir, result


@app.cell
def _(result):
    max(
        pred["score"]
        for pred in result.predictions
    )
    return


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
        f"mAP:  "
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
