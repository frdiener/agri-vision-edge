"""
Batch TFLite conversion of trained TF model variants.

Ports the conversion path of ``notebooks/tflite_conversion.py`` into a reusable
function for the ``ave convert`` CLI: for a model variant under ``artifacts/tf/``
it rebuilds the deployable TFLite models (default IoU threshold, fast NMS) and
embeds ObjectDetector metadata. Conversion + metadata only -- no evaluation.

For each variant the standard targets below are produced *as long as the backing
stage is present* (``ptq/``, ``qat/`` or ``qat_per-channel/``):

    fp32_ptq               plain float         (ptq stage)
    int8_ptq               per-tensor PTQ      (ptq stage)
    int8_ptq_per-channel   per-channel PTQ     (ptq stage)
    int8_qat               per-tensor QAT      (qat stage)
    int8_qat_per-channel   per-channel QAT     (qat_per-channel stage)

PTQ per-channel reuses the per-tensor ``ptq/`` checkpoint (granularity is a
converter flag); QAT per-channel is its own checkpoint trained with per-channel
fake-quant, hence the separate ``qat_per-channel/`` stage directory.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

# Calibration samples drawn from the representative dataset for int8 PTQ.
_REPRESENTATIVE_SAMPLES = 200


@dataclass(frozen=True)
class ConversionTarget:
    """One deployable model: a (precision, checkpoint, granularity) combination."""

    precision: str  # "int8" | "fp32"
    quantization: str  # "ptq" | "qat"
    per_channel: bool

    @property
    def stage_subdir(self) -> str:
        """Variant subdirectory holding the source checkpoint for this target."""
        # PTQ per-channel reuses the per-tensor ptq/ checkpoint; only QAT has a
        # distinct per-channel checkpoint directory.
        suffix = (
            "_per-channel" if self.per_channel and self.quantization != "ptq" else ""
        )
        return self.quantization + suffix

    @property
    def suffix(self) -> str:
        """Filename suffix appended to the variant stem (always fast NMS)."""
        per_channel = "_per-channel" if self.per_channel else ""
        return f"{self.precision}_{self.quantization}{per_channel}_fastnms"

    @property
    def label(self) -> str:
        return self.suffix.removesuffix("_fastnms")


# Emitted per variant, in dependency order (PTQ first). A target is skipped when
# its stage_subdir is absent.
STANDARD_TARGETS: tuple[ConversionTarget, ...] = (
    ConversionTarget("int8", "ptq", per_channel=False),
    ConversionTarget("int8", "ptq", per_channel=True),
    ConversionTarget("fp32", "ptq", per_channel=False),
    ConversionTarget("int8", "qat", per_channel=False),
    ConversionTarget("int8", "qat", per_channel=True),
)


def _parse_variant(name: str) -> tuple[str, bool]:
    """Return (classes, tiled) parsed from a variant directory name."""
    classes = "mc" if "_mc_" in name else "sc"
    tiled = "tiled" in name
    return classes, tiled


def _dataset_dir(variant_name: str, datasets_dir: Path) -> Path:
    classes, tiled = _parse_variant(variant_name)
    return datasets_dir / f"phenobench_{classes}{'_tiled' if tiled else ''}"


def _build_train_dataset(variant_name: str, datasets_dir: Path):
    """Build the PhenoBench train split used to draw representative samples."""
    from phenobench import PhenoBench

    from agri_vision_edge.data.tiling import TiledPhenoBench

    _, tiled = _parse_variant(variant_name)
    raw_dir = datasets_dir / f"phenobench_raw_{'tiled' if tiled else 'full'}"

    if not raw_dir.exists():
        raise FileNotFoundError(f"Raw PhenoBench dataset not found: {raw_dir}")

    if tiled:
        base = PhenoBench(
            root=str(raw_dir),
            split="train",
            target_types=["semantics", "plant_instances"],
            ignore_partial=True,
        )
        # Match the export notebooks (03/04): 3x3 tiles with 0.5 overlap so the
        # representative-dataset indices in rep_dataset.json line up with the
        # same 512px tiles the model was trained/exported on.
        return TiledPhenoBench(
            base,
            rows=3,
            cols=3,
            overlap=0.5,
        )

    return PhenoBench(
        root=str(raw_dir),
        split="train",
        target_types=["plant_bboxes"],
        ignore_partial=False,
    )


def _representative_dataset_fn(
    variant_name: str, datasets_dir: Path, resolution: int
) -> Callable[[], object]:
    """A converter representative_dataset callable yielding [-1, 1] inputs."""
    from agri_vision_edge.data.rep_dataset import normalized_representative_dataset

    dataset_dir = _dataset_dir(variant_name, datasets_dir)
    indices = json.loads((dataset_dir / "rep_dataset.json").read_text())
    train_dataset = _build_train_dataset(variant_name, datasets_dir)

    # SSDModule.inference_fn expects already-normalized [-1, 1] input, so the
    # raw [0, 255] samples must be normalized for calibration -- feeding [0, 255]
    # mis-calibrates the class head and caps scores at sigmoid(0) = 0.5.
    def representative_dataset():
        return normalized_representative_dataset(
            dataset=train_dataset,
            indices=indices,
            num_samples=_REPRESENTATIVE_SAMPLES,
            size=resolution,
        )

    return representative_dataset


def _convert_one(
    variant_dir: Path,
    target: ConversionTarget,
    datasets_dir: Path,
    out_path: Path,
    iou_threshold: float,
    native_resize: bool = True,
) -> None:
    """Rebuild + convert a single target and embed its metadata.

    ``native_resize`` (default True) builds FPN models with the NPU-delegatable
    ``RESIZE_NEAREST_NEIGHBOR`` upsample instead of the ``PACK``-based reshape
    trick; it is a no-op for non-FPN models. See
    ``agri_vision_edge.tfod.fpn_native_resize_upsampling``.
    """
    import tensorflow as tf

    from agri_vision_edge.third_party import setup_tensorflow_models

    setup_tensorflow_models()

    from object_detection.builders import model_builder
    from object_detection.export_tflite_graph_lib_tf2 import SSDModule

    from agri_vision_edge.conversion.metadata import write_object_detector_metadata
    from agri_vision_edge.tfod import (
        fpn_native_resize_upsampling,
        load_pipeline_config,
    )
    from agri_vision_edge.tfod.qat import (
        ensure_model_is_built_for_qat,
        quantize_detection_model,
    )

    stage_dir = variant_dir / target.stage_subdir
    pipeline_config = load_pipeline_config(stage_dir / "pipeline.config")

    nms = pipeline_config.model.ssd.post_processing.batch_non_max_suppression
    nms.iou_threshold = iou_threshold
    num_classes = pipeline_config.model.ssd.num_classes
    max_detections = nms.max_total_detections
    score_threshold = nms.score_threshold
    resolution = pipeline_config.model.ssd.image_resizer.fixed_shape_resizer.width

    with fpn_native_resize_upsampling(native_resize):
        detection_model = model_builder.build(pipeline_config.model, is_training=False)
    ensure_model_is_built_for_qat(detection_model, pipeline_config)

    # Rebuild the exact QAT graph the checkpoint was trained with so the weights
    # restore cleanly. quantize_detection_model is self-contained: it folds +
    # quantizes the backbone (FPN: as its own graph; plain SSD: inlined into one
    # full-model combined graph) and the head in one path. per_channel must match
    # the trained checkpoint.
    if target.quantization != "ptq":
        quantize_detection_model(
            detection_model, resolution, per_channel=target.per_channel
        )

    detection_module = SSDModule(
        pipeline_config,
        detection_model,
        max_detections=max_detections,
        use_regular_nms=False,
    )

    ckpt = tf.train.Checkpoint(model=detection_model)
    ckpt.restore(
        tf.train.latest_checkpoint(str(stage_dir / "checkpoint"))
    ).expect_partial().assert_existing_objects_matched()

    concrete_function = detection_module.inference_fn.get_concrete_function(
        tf.TensorSpec(
            shape=detection_module.input_shape(),
            dtype=tf.float32,
            name="input",
        )
    )

    converter = tf.lite.TFLiteConverter.from_concrete_functions(
        [concrete_function],
        trackable_obj=detection_module,
    )
    converter.inference_output_type = tf.float32

    if target.precision == "int8":
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS,
            tf.lite.OpsSet.TFLITE_BUILTINS_INT8,
        ]
        converter.inference_input_type = tf.int8
        converter.representative_dataset = _representative_dataset_fn(
            variant_dir.name, datasets_dir, resolution
        )
        converter._experimental_new_quantizer = True # this may be set to false if problems arise.
        converter._experimental_disable_per_channel = not target.per_channel
    else:  # fp32: keep float weights, no calibration, float builtins only.
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
        converter.inference_input_type = tf.float32

    tflite_model = converter.convert()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(tflite_model)

    write_object_detector_metadata(
        model_path=out_path,
        label_map_path=_dataset_dir(variant_dir.name, datasets_dir) / "label_map.pbtxt",
        num_classes=num_classes,
        extra_metadata={
            "iou_threshold": iou_threshold,
            "nms": "fast",
            "max_detections": max_detections,
            "score_threshold": score_threshold,
        },
    )


def convert_variant(
    variant_dir: Path,
    *,
    datasets_dir: Path,
    out_dir: Path,
    targets: tuple[ConversionTarget, ...] = STANDARD_TARGETS,
    iou_threshold: float = 0.5,
    native_resize: bool = True,
    overwrite: bool = False,
    log: Callable[[str], None] = print,
) -> list[Path]:
    """
    Convert every applicable target of ``variant_dir`` to a TFLite model.

    A target is converted only when its backing stage directory is present;
    existing outputs are skipped unless ``overwrite`` is set. Returns the list of
    written ``.tflite`` paths.

    ``native_resize`` (default True) builds FPN models with the NPU-delegatable
    ``RESIZE_NEAREST_NEIGHBOR`` upsample instead of the ``PACK`` reshape trick, so
    the full FPN graph delegates to the Teflon/etnaviv NPU; no-op for non-FPN
    models.
    """
    written: list[Path] = []

    for target in targets:
        stage_dir = variant_dir / target.stage_subdir

        if not (stage_dir / "checkpoint").is_dir():
            log(f"  skip {target.label}: no '{target.stage_subdir}/' stage")
            continue

        out_path = out_dir / f"{variant_dir.name}_{target.suffix}.tflite"

        if out_path.exists() and not overwrite:
            log(f"  skip {target.label}: exists ({out_path.name})")
            continue

        log(f"  convert {target.label} -> {out_path.name}")
        _convert_one(
            variant_dir,
            target,
            datasets_dir,
            out_path,
            iou_threshold,
            native_resize=native_resize,
        )
        written.append(out_path)

    return written
