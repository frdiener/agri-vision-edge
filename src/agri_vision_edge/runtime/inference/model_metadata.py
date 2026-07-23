"""
Read deployment configuration embedded in a converted TFLite model.

The converter (:mod:`agri_vision_edge.conversion.metadata`) embeds standard
TFLite ObjectDetector metadata into every exported ``.tflite``:

- ``labels.txt`` — the category names, stored as an *associated file*. TFLite's
  metadata populator appends associated files as a ZIP archive to the end of the
  flatbuffer, so they are readable with the standard-library :mod:`zipfile`
  module (a ZIP reader locates its central directory from the end of the file
  and ignores the flatbuffer prefix).
- ``NormalizationOptions`` — the input ``mean``/``std``, and
- a ``DETECTOR_POSTPROCESSING`` custom blob — ``score_threshold``,
  ``iou_threshold``, ``max_detections`` and the ``nms`` type. Both of these are
  also mirrored into the ``<model>.metadata.json`` sidecar the converter writes
  next to the model.

This module reads that configuration back at runtime using **only the standard
library** (``json`` + ``zipfile``), so it works on-device where ``tflite_support``
is not installed — the ``device`` extra ships ``tflite-runtime`` only.
``tflite_support``, when available (the prep/eval environment), is used as an
additional fallback for the embedded normalization if the sidecar is absent.

This replaces the earlier ``<model>.runtime.json`` sidecar, which nothing in the
pipeline ever produced.
"""

from __future__ import annotations

import json
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

# Name of the associated label file embedded by the converter.
_LABELS_ASSOCIATED_FILE = "labels.txt"

# CustomMetadata key holding the post-processing parameters
# (mirrors conversion.metadata.DETECTOR_PARAMS_KEY).
_DETECTOR_PARAMS_KEY = "DETECTOR_POSTPROCESSING"

# SSD MobileNetV2 default: [0, 255] -> [-1, 1] via (px - 127.5) / 127.5.
_DEFAULT_NORM = 127.5


@dataclass
class ModelMetadata:
    """
    Deployment configuration read back from a converted ``.tflite`` model.

    All fields degrade gracefully: missing metadata leaves the numeric
    thresholds as ``None`` (so the caller can apply its own default) and the
    normalization at the SSD ``127.5`` default.
    """

    # 1-based category_id -> human-readable name (matches the COCO/label-map
    # convention: labels.txt line 0 -> category_id 1).
    labels: dict[int, str] = field(default_factory=dict)

    norm_mean: list[float] = field(default_factory=lambda: [_DEFAULT_NORM])
    norm_std: list[float] = field(default_factory=lambda: [_DEFAULT_NORM])

    score_threshold: float | None = None
    iou_threshold: float | None = None
    max_detections: int | None = None
    nms: str | None = None

    # Where each piece was resolved from, for diagnostics.
    sources: dict[str, str] = field(default_factory=dict)

    @classmethod
    def load(
        cls,
        model_path: str | Path,
        *,
        label_offset: int = 1,
        verbose: bool = True,
    ) -> ModelMetadata:
        """
        Read the metadata embedded in ``model_path``.

        Labels come from the embedded ``labels.txt`` (stdlib :mod:`zipfile`);
        normalization and post-processing come from the ``<model>.metadata.json``
        sidecar (stdlib :mod:`json`), falling back to an embedded read via
        ``tflite_support`` when the sidecar is absent but that package is
        importable.
        """

        model_path = Path(model_path)
        meta = cls()

        meta._load_labels(model_path, label_offset)
        has_norm = meta._load_from_sidecar(model_path)

        if not has_norm:
            meta._load_embedded_via_tflite_support(model_path)

        if verbose:
            meta._log(model_path)

        return meta

    #
    # Labels — embedded associated file, read as a stdlib zip.
    #

    def _load_labels(self, model_path: Path, label_offset: int) -> None:
        try:
            with zipfile.ZipFile(model_path) as archive:
                if _LABELS_ASSOCIATED_FILE not in archive.namelist():
                    return
                raw = archive.read(_LABELS_ASSOCIATED_FILE).decode("utf-8")
        except (zipfile.BadZipFile, OSError, KeyError):
            # Not a zip-bearing model, or unreadable — leave labels empty.
            return

        names = [line.strip() for line in raw.splitlines() if line.strip()]

        if not names:
            return

        self.labels = {index + label_offset: name for index, name in enumerate(names)}
        self.sources["labels"] = "embedded"

    #
    # Normalization + post-processing — the sidecar the converter writes.
    #

    def _load_from_sidecar(self, model_path: Path) -> bool:
        """Return ``True`` when normalization was resolved from the sidecar."""

        sidecar = model_path.with_suffix(".metadata.json")

        if not sidecar.is_file():
            return False

        try:
            meta = json.loads(sidecar.read_text(encoding="utf-8"))
            subgraph = meta["subgraph_metadata"][0]
        except (json.JSONDecodeError, KeyError, IndexError, OSError):
            return False

        has_norm = self._extract_normalization(subgraph, "sidecar")
        self._extract_postprocessing(subgraph, "sidecar")

        return has_norm

    def _load_embedded_via_tflite_support(self, model_path: Path) -> None:
        """
        Last-resort read of the embedded metadata via ``tflite_support``.

        Only usable where the package is installed (the prep/eval environment);
        on-device (``tflite-runtime`` only) the import fails and the defaults are
        kept.
        """

        try:
            from tflite_support import metadata as _metadata

            displayer = _metadata.MetadataDisplayer.with_model_file(str(model_path))
            meta = json.loads(displayer.get_metadata_json())
            subgraph = meta["subgraph_metadata"][0]
        except Exception:
            return

        self._extract_normalization(subgraph, "embedded")

        if "postprocessing" not in self.sources:
            self._extract_postprocessing(subgraph, "embedded")

    #
    # Shared extractors (operate on a parsed subgraph_metadata entry).
    #

    def _extract_normalization(self, subgraph: dict, source: str) -> bool:
        try:
            units = subgraph["input_tensor_metadata"][0]["process_units"]
        except (KeyError, IndexError, TypeError):
            return False

        for unit in units or []:
            if unit.get("options_type") != "NormalizationOptions":
                continue

            options = unit.get("options", {})

            if "mean" in options and "std" in options:
                self.norm_mean = [float(v) for v in options["mean"]]
                self.norm_std = [float(v) for v in options["std"]]
                self.sources["normalization"] = source
                return True

        return False

    def _extract_postprocessing(self, subgraph: dict, source: str) -> None:
        for entry in subgraph.get("custom_metadata", []) or []:
            if entry.get("name") != _DETECTOR_PARAMS_KEY:
                continue

            try:
                params = json.loads(bytes(entry["data"]).decode("utf-8"))
            except (KeyError, ValueError, TypeError):
                return

            if "score_threshold" in params:
                self.score_threshold = float(params["score_threshold"])
            if "iou_threshold" in params:
                self.iou_threshold = float(params["iou_threshold"])
            if "max_detections" in params:
                self.max_detections = int(params["max_detections"])
            if "nms" in params:
                self.nms = str(params["nms"])

            self.sources["postprocessing"] = source
            return

    #
    # Diagnostics
    #

    def _log(self, model_path: Path) -> None:
        label_names = list(self.labels.values()) or "-"
        origins = ", ".join(f"{k}:{v}" for k, v in self.sources.items()) or "defaults"

        print(
            f"[metadata] {model_path.name}: labels={label_names} "
            f"norm(mean={self.norm_mean}, std={self.norm_std}) "
            f"score_threshold={self.score_threshold} iou={self.iou_threshold} "
            f"max_detections={self.max_detections} nms={self.nms} [{origins}]"
        )
