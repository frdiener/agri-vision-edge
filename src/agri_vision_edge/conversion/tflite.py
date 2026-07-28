"""
Batch TFLite conversion of trained TF model variants.

Ports the conversion path of ``notebooks/tflite_conversion.py`` into a reusable
function for the ``ave convert`` CLI: for a model variant under ``artifacts/tf/``
it rebuilds the deployable TFLite models (default IoU threshold, fast NMS) and
embeds ObjectDetector metadata. Conversion + metadata only -- no evaluation.

For each variant the standard targets below are produced *as long as the backing
stage is present* (``ptq/``, ``qat_per-tensor/`` or ``qat_per-channel/``):

    fp32_ptq               plain float         (ptq stage)
    int8_ptq_per-tensor    per-tensor PTQ      (ptq stage)
    int8_ptq_per-channel   per-channel PTQ     (ptq stage)
    int8_qat_per-tensor    per-tensor QAT      (qat_per-tensor stage)
    int8_qat_per-channel   per-channel QAT     (qat_per-channel stage)

PTQ per-channel reuses the per-tensor ``ptq/`` checkpoint (granularity is a
converter flag); QAT trains a distinct checkpoint per granularity, so each has
its own stage directory (``qat_per-tensor/`` and ``qat_per-channel/``).
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
    def stage_candidates(self) -> tuple[str, ...]:
        """
        Variant subdirectories that can source this target, most preferred first.

        PTQ shares one checkpoint regardless of granularity -- there it is just
        a converter flag.

        QAT does too: the training graph does not depend on granularity (relu6
        outputs are pinned either way), which is chosen when the export graph is
        rebuilt. Training a run per granularity was tried and measured to gain
        nothing, so ``qat_per-tensor/`` is the canonical run and
        ``qat_per-channel/`` is used only when it is the sole QAT stage present.
        """
        if self.quantization == "ptq":
            return ("ptq",)

        granularity = "per-channel" if self.per_channel else "per-tensor"
        return (
            f"{self.quantization}_per-tensor",
            f"{self.quantization}_{granularity}",
        )

    @property
    def suffix(self) -> str:
        """Filename suffix appended to the variant stem (always fast NMS)."""
        # int8 models carry their weight granularity explicitly (per-tensor vs
        # per-channel); fp32 has no quantization granularity, so it stays bare.
        if self.precision == "int8":
            granularity = "_per-channel" if self.per_channel else "_per-tensor"
        else:
            granularity = ""
        return f"{self.precision}_{self.quantization}{granularity}_fastnms"

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
    """
    Locate the exported dataset bundle a variant was trained from.

    The models are trained on the ``_no-partials`` bundles (partials dropped in
    train, do-not-care in eval); the unsuffixed directories are earlier exports
    kept around. They are not interchangeable -- their ``rep_dataset.json``
    indices were drawn against different sample counts -- so prefer the
    no-partials bundle and only fall back to the legacy name when it is absent.
    """
    classes, tiled = _parse_variant(variant_name)
    stem = f"phenobench_{classes}{'_tiled' if tiled else ''}"

    for name in (f"{stem}_no-partials", stem):
        candidate = datasets_dir / name
        if candidate.is_dir():
            return candidate

    raise FileNotFoundError(
        f"No exported dataset bundle for {variant_name}: looked for "
        f"{stem}_no-partials and {stem} under {datasets_dir}"
    )


def _export_metadata(dataset_dir: Path) -> dict:
    """The exported bundle's ``dataset_metadata.json`` (empty when absent)."""
    path = dataset_dir / "dataset_metadata.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def _masks_dataset(raw_dir: Path):
    """PhenoBench split with the masks the tile boxes are derived from."""
    from phenobench import PhenoBench

    return PhenoBench(
        root=str(raw_dir),
        split="train",
        target_types=["semantics", "plant_instances"],
        ignore_partial=True,
    )


def _build_train_dataset(variant_name: str, datasets_dir: Path):
    """
    Rebuild the train split the representative-dataset indices were drawn from.

    ``rep_dataset.json`` stores *positions*, so this has to reproduce the
    exported bundle's ordering exactly -- a dataset built with different tiling
    still indexes fine and still yields plausible field images, it just
    calibrates on the wrong ones.

    Tiled bundles are cut from the FULL frames, so the tiling is applied to
    ``phenobench_raw_full`` with the geometry the export recorded. (Applying it
    to ``phenobench_raw_tiled`` instead re-cuts tiles that are already tiles:
    512px training tiles became 256px sub-tiles and every index shifted.)
    Bundles exported before the geometry was recorded are matched by the
    materialized ``phenobench_raw_tiled`` as-is; either way
    :func:`_check_calibration_dataset` verifies the result against the export's
    own sample count.
    """
    from phenobench import PhenoBench

    from agri_vision_edge.data.tiling import TiledPhenoBench

    _, tiled = _parse_variant(variant_name)

    if not tiled:
        raw_dir = datasets_dir / "phenobench_raw_full"
        if not raw_dir.exists():
            raise FileNotFoundError(f"Raw PhenoBench dataset not found: {raw_dir}")
        return PhenoBench(
            root=str(raw_dir),
            split="train",
            target_types=["plant_bboxes"],
            ignore_partial=False,
        )

    tiling = _export_metadata(_dataset_dir(variant_name, datasets_dir)).get("tiling")

    if tiling:
        raw_dir = datasets_dir / "phenobench_raw_full"
        if not raw_dir.exists():
            raise FileNotFoundError(f"Raw PhenoBench dataset not found: {raw_dir}")
        return TiledPhenoBench(
            _masks_dataset(raw_dir),
            rows=int(tiling["rows"]),
            cols=int(tiling["cols"]),
            overlap=float(tiling.get("overlap", 0.0)),
        )

    # Legacy bundle: no recorded geometry, but the materialized tiled dataset is
    # that geometry. Wrapped in a 1x1 grid purely to derive boxes from the
    # instance masks -- unlike the full dataset it ships no `plant_bboxes/`.
    raw_dir = datasets_dir / "phenobench_raw_tiled"
    if not raw_dir.exists():
        raise FileNotFoundError(f"Raw PhenoBench dataset not found: {raw_dir}")
    return TiledPhenoBench(
        _masks_dataset(raw_dir),
        rows=1,
        cols=1,
        overlap=0.0,
    )


def _check_calibration_dataset(dataset, dataset_dir: Path, indices: list) -> None:
    """
    Verify the calibration dataset is the one ``rep_dataset.json`` indexes.

    The indices are positions, not identifiers: a calibration dataset built with
    different geometry than the exported training set still indexes fine, still
    yields plausible field images, and silently calibrates on the wrong ones.
    The export records its own sample count, so compare against that -- it is
    the only cheap check that catches a renumbering.
    """
    expected = _export_metadata(dataset_dir).get("train_samples")
    if expected is None:
        return

    if len(dataset) != expected:
        raise ValueError(
            f"Calibration dataset has {len(dataset)} samples but "
            f"{dataset_dir.name} was exported with {expected}. The "
            "rep_dataset.json indices address the exported ordering, so "
            "calibration would silently run on the wrong images. Re-export the "
            "dataset bundle, or check the tiling recorded in its "
            "dataset_metadata.json."
        )

    if indices and max(indices) >= len(dataset):
        raise ValueError(
            f"rep_dataset.json indexes up to {max(indices)} but the "
            f"calibration dataset has only {len(dataset)} samples."
        )


def _representative_dataset_fn(
    variant_name: str, datasets_dir: Path, resolution: int
) -> Callable[[], object]:
    """A converter representative_dataset callable yielding [-1, 1] inputs."""
    from agri_vision_edge.data.rep_dataset import normalized_representative_dataset

    dataset_dir = _dataset_dir(variant_name, datasets_dir)
    indices = json.loads((dataset_dir / "rep_dataset.json").read_text())
    train_dataset = _build_train_dataset(variant_name, datasets_dir)
    _check_calibration_dataset(train_dataset, dataset_dir, indices)

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
    stage_dir: Path | None = None,
) -> None:
    """Rebuild + convert a single target and embed its metadata.

    ``stage_dir`` is the resolved source checkpoint directory (see
    ``ConversionTarget.stage_candidates``), defaulting to the preferred one.

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

    if stage_dir is None:
        stage_dir = variant_dir / target.stage_candidates[0]
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

    # Rebuild the QAT graph the checkpoint was trained with so the weights
    # restore cleanly. quantize_detection_model is self-contained: it folds +
    # quantizes the backbone (FPN: as its own graph; plain SSD: inlined into one
    # full-model combined graph) and the head in one path.
    #
    # `for_export` asks for the export rewrite of that graph: for a per-channel
    # target the stateless [0, 6] relu6 pins are dropped, which is what lets the
    # converter emit per-channel weights. Same variables either way, so the
    # checkpoint restores unchanged.
    if target.quantization != "ptq":
        quantize_detection_model(
            detection_model,
            resolution,
            per_channel=target.per_channel,
            for_export=True,
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
        stage_dir = next(
            (
                variant_dir / name
                for name in target.stage_candidates
                if (variant_dir / name / "checkpoint").is_dir()
            ),
            None,
        )

        if stage_dir is None:
            log(
                f"  skip {target.label}: none of "
                f"{'/'.join(target.stage_candidates)} present"
            )
            continue

        out_path = out_dir / f"{variant_dir.name}_{target.suffix}.tflite"

        if out_path.exists() and not overwrite:
            log(f"  skip {target.label}: exists ({out_path.name})")
            continue

        log(f"  convert {target.label} ({stage_dir.name}) -> {out_path.name}")
        _convert_one(
            variant_dir,
            target,
            datasets_dir,
            out_path,
            iou_threshold,
            native_resize=native_resize,
            stage_dir=stage_dir,
        )
        written.append(out_path)

    return written
