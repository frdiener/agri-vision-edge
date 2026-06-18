import marimo

__generated_with = "0.23.9"
app = marimo.App(width="full", app_title="")


@app.cell(hide_code=True)
def _(Path):
    import marimo as mo

    ARTIFACTS_DIR = Path("artifacts")
    models = sorted(list((ARTIFACTS_DIR / "tf").iterdir()))

    model_root = mo.ui.dropdown(
        options={str(p.name): p for p in models},
        value=str(models[1].name),
        label="Model",
    )
    model_root
    return ARTIFACTS_DIR, mo, model_root


@app.cell(hide_code=True)
def configuration(ARTIFACTS_DIR, Path, json, mo, model_root):
    try:
        _finetune_config = _load_pipeline_config(
            Path(model_root.value) / "ptq" / "pipeline.config"
        )
    except Exception as e:
        _finetune_config = None

    _original_dataset = (
        Path(
            str(_finetune_config.train_input_reader.tf_record_input_reader.input_path)
        ).parent.name
        if _finetune_config
        else "not found"
    )

    fp32_map = {}
    for schema in ["ptq", "qat0", "qat1", "qat2", "qat3"]:
        try:
            fp32_map[schema] = json.loads(
                (model_root.value / schema / "best_metric.json").read_text()
            )["metric_value"]
        except Exception:
            fp32_map[schema] = -1

    TFLITE_MODELS_DIR = ARTIFACTS_DIR / "tflite"
    TFLITE_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    DATASETS_DIR = Path("datasets")

    config_form = (
        mo.md("""
    **TFLite Conversion**

    - {dataset}
      {tiled}
      {classes}
    - {quantization}
      {precision} {per_channel}
    - {eval_split}
      {regular_nms}
    """)
        .batch(
            dataset=mo.ui.dropdown(
                options={
                    "Phenobench": "phenobench",
                },
                value="Phenobench",
                label="Dataset",
            ),
            tiled=mo.ui.switch(
                value=True if "tiled" in model_root.value.name else False,
                label="Tiled 2x2",
            ),
            classes=mo.ui.radio(
                options={
                    "Single class (Weeds)": "sc",
                    "Multiclass (Crops vs Weeds)": "mc",
                },
                value="Single class (Weeds)"
                if "_sc_" in model_root.value.name
                else "Multiclass (Crops vs Weeds)",
            ),
            quantization=mo.ui.dropdown(
                options={
                    "Finetuned": "ptq",
                    "Finetuned with full QAT": "qat0",
                    "Finetuned with QAT (Quantized Weights only)": "qat1",
                    "Finetuned with QAT (Prefolded Batchnorms)": "qat2",
                    "Finetuned with full QAT (Backbone + Head)": "qat3",
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
            eval_split=mo.ui.radio(
                options={
                    "Eval (val split)": "val",
                    "Test": "test",
                },
                value="Eval (val split)",
            ),
            regular_nms=mo.ui.switch(
                value=False,
                label="Regular (per-class) NMS",
            ),
        )
        .form(bordered=False, submit_button_label="Convert & Evaluate")
    )

    summary = mo.md(f"""
    Dataset:

    | Setting | Value |
    |:----------|----------|
    | Dataset | {_original_dataset} |
    | Tiled | {"tile" in _original_dataset} |
    | Labels | {("Single Class" if _finetune_config.model.ssd.num_classes == 1 else "Multi Class") if _finetune_config else "not found"} |
    | Resolution | {(_finetune_config.model.ssd.image_resizer.fixed_shape_resizer.width) if _finetune_config else "not found"} |
    | FP32 PTQ mAP | {fp32_map["ptq"]:.4f} |
    | FP32 QAT0 mAP | {fp32_map["qat0"]:.4f} |
    | FP32 QAT1 mAP | {fp32_map["qat1"]:.4f} |
    | FP32 QAT2 mAP | {fp32_map["qat2"]:.4f} |
    | FP32 QAT3 mAP | {fp32_map["qat3"]:.4f} |
    """)

    mo.hstack(
        [
            config_form,
            summary,
        ]
    )
    return DATASETS_DIR, TFLITE_MODELS_DIR, config_form, fp32_map


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
        quantize_detection_head,
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
        quantize_backbone,
        quantize_detection_head,
        representative_dataset,
        save_benchmark_artifacts,
        tf,
    )


@app.cell(hide_code=True)
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
    model_root,
):
    mo.stop(
        config_form.value is None,
        mo.md(
            "_Configure parameters above and click **Convert & Evaluate** to start._"
        ),
    )
    config = config_form.value

    MODEL_ROOT = config["model_root"] = model_root.value
    CHECKPOINT = MODEL_ROOT / config["quantization"] / "checkpoint"
    PIPELINE_CONFIG = _load_pipeline_config(CHECKPOINT.parent / "pipeline.config")
    config["original_dataset"] = Path(
        str(PIPELINE_CONFIG.train_input_reader.tf_record_input_reader.input_path)
    ).parent.name
    config["num_classes"] = PIPELINE_CONFIG.model.ssd.num_classes
    config["resolution"] = (
        PIPELINE_CONFIG.model.ssd.image_resizer.fixed_shape_resizer.width
    )

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

    dataset_dir = (
        DATASETS_DIR
        / f"{config['dataset']}_{config['classes']}{'_tiled' if config['tiled'] else ''}"
    )
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
    quantize_detection_head,
    tf,
):
    detection_model = model_builder.build(PIPELINE_CONFIG.model, is_training=False)

    ensure_model_is_built_for_qat(detection_model, PIPELINE_CONFIG)

    # Rebuild the exact QAT graph the checkpoint was trained with so the weights
    # restore cleanly. Mirrors tfod_trainer.setup: fold BatchNorms into the convs
    # (fold_bn defaults on whenever QAT is enabled), then quantize_backbone with
    # the per-dir scheme, then optionally quantize_detection_head for qat3.
    # Scheme map is user-specified: qat0=annotate_all, qat1=weights, qat2=full;
    # qat3 = full on backbone + head.
    qat_backbone = None

    QAT_SCHEMES = {
        "qat0": "annotate_all",  # TFMOT default 8-bit (legacy / comparison)
        "qat1": "weights",  # weight-only int8
        "qat2": "full",  # full int8, backbone
        "qat3": "full",  # full int8, backbone + head
    }

    if config["quantization"] != "ptq":
        scheme = QAT_SCHEMES[config["quantization"]]

        folded_backbone = fold(
            detection_model.feature_extractor.classification_backbone
        )
        qat_backbone = quantize_backbone(folded_backbone, scheme=scheme)
        detection_model.feature_extractor.classification_backbone = qat_backbone

        if config["quantization"] == "qat3":
            # Quantize the SSD head (feature maps + box predictor) in place;
            # must run after the backbone is folded + quantized.
            quantize_detection_head(
                detection_model, config["resolution"], scheme="full"
            )

    # The module helps build a TF SavedModel appropriate for TFLite conversion.
    # max_detections=100 matches the pipeline's max_total_detections so the
    # TFLite mAP tracks the checkpoint metric (COCO also scores up to 100/image).
    detection_module = SSDModule(
        PIPELINE_CONFIG,
        detection_model,
        max_detections=100,
        use_regular_nms=config["regular_nms"],
    )

    # restore model wheights
    ckpt = tf.train.Checkpoint(model=detection_model)
    ckpt.restore(
        tf.train.latest_checkpoint(CHECKPOINT)
    ).expect_partial().assert_existing_objects_matched()

    concrete_function = detection_module.inference_fn.get_concrete_function(
        tf.TensorSpec(
            shape=detection_module.input_shape(),
            dtype=tf.float32,
            name="input",
        )
    )

    # backbone = detection_model.feature_extractor.classification_backbone
    # folded_backbone = fold(backbone)
    # qat_backbone = quantize_backbone(
    #     folded_backbone,
    #     scheme="full"
    # )
    return concrete_function, detection_module, qat_backbone


@app.cell(disabled=True)
def _(qat_backbone):
    for layer in qat_backbone.layers:
        print(layer.name)
        try:
            print(f"===>{type(layer.quantize_config).__name__}")
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

    converter.inference_output_type = tf.float32

    if config["precision"] == "int8":
        converter.optimizations = [tf.lite.Optimize.DEFAULT]

        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS,
            tf.lite.OpsSet.TFLITE_BUILTINS_INT8,
        ]

        converter.inference_input_type = tf.int8

        # SSDModule.inference_fn calls model.predict() directly, WITHOUT the SSD
        # preprocessing step, so the converted graph expects already-normalized
        # [-1, 1] input. representative_dataset yields raw [0, 255] images, so we
        # must normalize here. Feeding [0, 255] saturates the backbone during
        # calibration and mis-calibrates the (non-QAT) class head: the class-logit
        # tensor's max gets pinned to 0, capping every detection score at
        # sigmoid(0) = 0.5. (Independent of the QAT scheme.)
        def _normalized_rep_dataset():
            for sample in representative_dataset(
                dataset=train_dataset,
                indices=rep_ds_indices,
                num_samples=200,
                size=IMAGE_SIZE,
            ):
                yield [(2.0 / 255.0) * sample[0] - 1.0]

        converter.representative_dataset = _normalized_rep_dataset

        converter._experimental_new_quantizer = False
        converter._experimental_disable_per_channel = not config["per_channel"]

    elif config["precision"] == "fp32":
        # Plain float conversion: the finetuned ptq/ weights are kept in
        # float32 and no quantization happens. No optimizations, no
        # representative dataset, and only float builtin ops — the graph
        # already expects normalized [-1, 1] float input.
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
        converter.inference_input_type = tf.float32

    tflite_model = converter.convert()
    return (tflite_model,)


@app.cell(hide_code=True)
def _(MODEL_FILE_PATH, config, dataset_dir, written):
    # Write valid TFLite ObjectDetector metadata for the converted model.
    #
    # object_detector.MetadataWriter reads the input tensor type (int8 vs
    # float32) and the SSD output ordering straight from the model buffer, so the
    # same call produces correct metadata for every precision/quant combination
    # (fp32, ptq, qat0-qat3) and every sc/mc variant. The exported graph expects
    # normalized [-1, 1] input (normalized = (px - 127.5) / 127.5 for px in
    # [0, 255]); metadata normalization is always expressed in float terms, and
    # for the int8 model the converter's own quant params carry the float->int8
    # step, so mean/std are identical for both precisions.
    import re

    from tflite_support.metadata_writers import object_detector, writer_utils
    from tflite_support import metadata as _metadata

    assert written and MODEL_FILE_PATH.exists()

    # Class names in label-map id order (id 1 -> line 0). The detection head emits
    # 0-based class indices, so labels.txt must follow that order: ["crop",
    # "weed"] for mc, ["weed"] for sc.
    _label_map = (dataset_dir / "label_map.pbtxt").read_text()
    _items = re.findall(r'id:\s*(\d+)\s+name:\s*"([^"]+)"', _label_map)
    labels = [name for _id, name in sorted(_items, key=lambda kv: int(kv[0]))]
    assert len(labels) == config["num_classes"], (labels, config["num_classes"])

    label_file_path = MODEL_FILE_PATH.with_name(f"{MODEL_FILE_PATH.stem}_labels.txt")
    label_file_path.write_text("\n".join(labels) + "\n")

    writer = object_detector.MetadataWriter.create_for_inference(
        writer_utils.load_file(str(MODEL_FILE_PATH)),
        input_norm_mean=[127.5],
        input_norm_std=[127.5],
        label_file_paths=[str(label_file_path)],
    )
    writer_utils.save_file(writer.populate(), str(MODEL_FILE_PATH))

    metadata_json_path = MODEL_FILE_PATH.with_name(
        f"{MODEL_FILE_PATH.stem}.metadata.json"
    )
    metadata_json_path.write_text(
        _metadata.MetadataDisplayer.with_model_file(
            str(MODEL_FILE_PATH)
        ).get_metadata_json()
    )

    print(f"metadata written: {MODEL_FILE_PATH.name}  (labels={labels})")
    print(f"metadata json:    {metadata_json_path.name}")
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

    input_details = interpreter.get_input_details()

    output_details = interpreter.get_output_details()

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
        if not "relu6" in tensor["name"].lower():
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

            if mesa_range > 6.0000003:
                failed = True

            print(
                f"{tensor['index']:4d} "
                f"{tensor['name']:<60}\n"
                f"     scale={scale:.9f} "
                f"zp={zp:4d} "
                f"range={mesa_range:.6f}"
            )

        if failed:
            print("     ^^^ FAILS MESA RELU6 CHECK")
    return


@app.cell(hide_code=True)
def _(
    MODEL_FILE_PATH,
    Path,
    TFLiteRuntime,
    benchmark_runtime,
    config,
    dataset_dir,
    dataset_raw_dir,
    load_coco_images,
    save_benchmark_artifacts,
    written,
):
    assert written and MODEL_FILE_PATH.exists()

    # eval_split: "val" (the split used for the checkpoint metric) or "test".
    annotations_path = dataset_dir / f"{config['eval_split']}_annotations.json"
    image_records = load_coco_images(
        dataset_raw_dir / "val/images/",
        annotations_path,
    )

    output_dir = Path("./benchmark_results/") / MODEL_FILE_PATH.stem

    print(f"\n=== Benchmarking: {MODEL_FILE_PATH.name} ===")

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

    mean_latency = sum(result.latencies_ms) / len(result.latencies_ms)

    print(f"mean latency: {mean_latency:.2f} ms")

    print(f"exported {len(result.predictions)} prediction(s)")
    return annotations_path, output_dir, result


@app.cell
def _(result):
    max(pred["score"] for pred in result.predictions)
    return


@app.cell(hide_code=True)
def _(annotations_path, config, json, output_dir):
    import contextlib
    import io

    from agri_vision_edge.evaluation.coco import (
        METRIC_NAMES,
        evaluate_predictions,
        print_per_class,
        save_metrics,
    )

    # Maps our short COCO metric names to the keys used in metrics_history.json.
    _TF_HISTORY_KEY = {
        "AP": "DetectionBoxes_Precision/mAP",
        "AP50": "DetectionBoxes_Precision/mAP@.50IOU",
        "AP75": "DetectionBoxes_Precision/mAP@.75IOU",
        "APS": "DetectionBoxes_Precision/mAP (small)",
        "APM": "DetectionBoxes_Precision/mAP (medium)",
        "APL": "DetectionBoxes_Precision/mAP (large)",
        "AR1": "DetectionBoxes_Recall/AR@1",
        "AR10": "DetectionBoxes_Recall/AR@10",
        "AR100": "DetectionBoxes_Recall/AR@100",
        "ARS": "DetectionBoxes_Recall/AR@100 (small)",
        "ARM": "DetectionBoxes_Recall/AR@100 (medium)",
        "ARL": "DetectionBoxes_Recall/AR@100 (large)",
    }

    predictions_path = output_dir / "predictions.json"
    metrics_path = output_dir / "metrics.json"

    print(f"\n=== Evaluating: {output_dir.name} ===")

    # Swallow pycocotools' own index/summarize chatter; we print our own table.
    with contextlib.redirect_stdout(io.StringIO()):
        metrics = evaluate_predictions(
            annotations_path,
            predictions_path,
        )

    save_metrics(
        metrics,
        metrics_path,
    )

    # TF-side (pre-conversion) reference: the best eval row of the checkpoint
    # being converted (best-mAP row of metrics_history.json, same checkpoint as
    # best_metric.json). Missing -> NaN, rendered as x.xxxx.
    tf_best = {}
    try:
        _history = json.loads(
            (
                config["model_root"] / config["quantization"] / "metrics_history.json"
            ).read_text()
        )
        tf_best = max(_history, key=lambda r: r["DetectionBoxes_Precision/mAP"])
    except Exception:
        pass

    def _fmt(v):
        return f"{v:>8.4f}" if v == v else f"{'x.xxxx':>8}"  # nan != nan

    print()

    print(f"{'':<6}{'tf':>8}    {'tflite':>8}")

    for name in METRIC_NAMES:
        tf_v = tf_best.get(_TF_HISTORY_KEY[name], float("nan"))
        print(f"{name + ':':<6}{_fmt(tf_v)} => {_fmt(metrics[name])}")

    print_per_class(metrics)
    return


if __name__ == "__main__":
    app.run()
