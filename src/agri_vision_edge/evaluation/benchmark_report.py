"""
Benchmark result aggregation and publication-level reporting.

This is the evaluation-time counterpart to :mod:`agri_vision_edge.evaluation.curves`:
where ``curves`` charts *training* scalars over steps, this module aggregates the
*evaluation* artifacts written by ``bin/benchmark_tflite.py`` (``latency.json`` /
``runtime.json``) and ``bin/evaluate_coco.py`` (``metrics.json``) into one tidy
frame, then renders the figures and LaTeX tables used in the thesis.

Layout consumed::

    benchmark_results/<platform>/<run>/{metrics,latency,runtime}.json
    benchmark_results/<platform>/resize.json

The second is not a run. It is the host's measured ``cv2.resize`` cost, written
by ``scripts/benchmark_resize.py``, and it exists because every latency above
contains one: ``predict()`` resizes the source frame to the model's input
inside the timed region, by an amount that depends on both the evaluation
regime and the input size. :func:`add_resize_cost` subtracts it, which is what
makes the resolution ladder and the tiling study comparisons of the detector
rather than of its preprocessing.

The run name encodes the configuration and is decoded by :func:`parse_run_name`::

    <tiling>_<arch>_<classes>_<dataset>_<size>_<precision>_<quant>[_<granularity>]_<nms>_<split>
    untiled_ssd-mn2-fpnlite_mc_phenobench-tiled_320_int8_ptq_per-tensor_fastnms_val

``<tiling>`` (``tiled`` / ``untiled``) is the prefix ``benchmark_all.sh`` puts on
each result directory and names the *input regime the model was evaluated on* --
every model is swept over both. It is orthogonal to the ``<dataset>`` token
(``phenobench`` / ``phenobench-tiled``), which names the data the model was
*trained* on; the cross of the two is the interesting axis. ``<granularity>``
(``per-tensor`` / ``per-channel``) is present on int8 runs and omitted on fp32.
Everything else is discovered dynamically: new platforms (i.MX 8M Plus, i.MX 93)
and quantization schemes (``qatN``) appear automatically once their artifacts
exist.
Plot helpers return ``None`` when a comparison has too few groups to be
meaningful, so callers can skip them without special-casing.
"""

from __future__ import annotations

import json
import re
import warnings
from collections.abc import Iterable
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter

# Reuse the shared axis styling so benchmark figures match the training-curve
# figures (despined, grid-below, consistent spines).
from .curves import _prepare_axis

# =========================================================
# Run-name parsing
# =========================================================

_PRECISIONS = {"fp32", "fp16", "int8", "int16", "dynamic"}
_SPLITS = {"val", "test", "train", "eval"}
_CLASSES = {"sc", "mc"}
#: int8 weight granularities. Carried as a first-class field so it never leaks
#: into the ``dataset`` token (both per-tensor and per-channel exports name it
#: explicitly now, mirroring the conversion artifact filenames).
_GRANULARITIES = {"per-tensor", "per-channel"}
#: Evaluation-input regimes, carried as the leading token of a benchmark result
#: directory. Stripped before the architecture is read, otherwise it would be
#: swallowed by the ``arch`` prefix and split every model in two.
_EVAL_TILINGS = {"tiled", "untiled"}

ARCH_LABELS = {
    "ssd-mn2": "SSD MobileNetV2",
    "ssd-mn2-fpnlite": "SSD MobileNetV2 FPNLite",
    # Auxiliary reference detectors. They are not part of the SSD comparison
    # the thesis is built on -- they exist to say how far the operating point
    # sits from a modern anchor-free single-stage detector -- but they share
    # the deployment chain, so they belong in the same frame.
    "yolov7-tiny": "YOLOv7-tiny",
    "yolox-nano": "YOLOX-Nano",
}

#: Architectures the thesis' controlled comparison is built on. Everything else
#: is auxiliary and has to be opted into, otherwise a detector that exists in
#: one configuration only silently joins means taken over the full matrix.
PRIMARY_ARCHS = ("ssd-mn2", "ssd-mn2-fpnlite")
CLASS_LABELS = {
    "sc": "Single-class (weed)",
    "mc": "Multi-class (crop+weed)",
}
PRECISION_LABELS = {"fp32": "FP32", "fp16": "FP16", "int8": "INT8"}
EVAL_TILING_LABELS = {
    "tiled": "Tiled input",
    "untiled": "Full-frame input",
}

#: The post-processing the toolchain emits by default, and what "shipping"
#: means unless told otherwise. ``ave convert`` builds a matched pair for every
#: deployable, keyed by the filename's NMS token: ``_fastnms`` is
#: ``TFLite_Detection_PostProcess`` with ``use_regular_nms=False`` -- one
#: class-agnostic pass over each anchor's argmax class -- and ``_regnms`` is the
#: per-class pass the training checkpoint runs.
DEFAULT_NMS = "fastnms"

#: The per-class alternative. **Also a deployable, and deployed.** It was
#: converted, flashed and benchmarked on both boards under the delegate, and it
#: carries the same metrics as any other variant, official evaluator included.
#: Do not describe it as a diagnostic that "does not ship": the fused
#: post-processing operator supports both modes, and which one an export uses is
#: a deployment decision with a measurable accuracy/latency trade-off, not a
#: property of the toolchain.
#:
#: It nevertheless plays a second role as the algorithm-matched **control**,
#: because it is the mode the training checkpoint runs -- which is what lets the
#: SavedModel rung be compared like for like and the substitution's cost be
#: isolated from the format change.
REGULAR_NMS = "regnms"

NMS_LABELS = {
    "fastnms": "Fast NMS (default export)",
    "regnms": "Per-class NMS",
}

#: Top-level entries under ``benchmark_results`` that are not measurement platforms.
NON_PLATFORM_DIRS = frozenset({"attic"})

_METRIC_KEYS = (
    "AP",
    "AP50",
    "AP75",
    "APS",
    "APM",
    "APL",
    "AR1",
    "AR10",
    "AR100",
    "ARS",
    "ARM",
    "ARL",
)


def _classify_token(token: str) -> str | None:
    """
    Classify one run-name token into a configuration field.

    Order-independent so renamed datasets and new ``qatN`` schemes parse
    without code changes.
    """
    if token in _PRECISIONS:
        return "precision"
    if re.fullmatch(r"qat\d*|ptq", token):
        return "quant"
    if token in _GRANULARITIES:
        return "granularity"
    if token in _SPLITS:
        return "split"
    if "nms" in token:
        return "nms"
    if token.isdigit():
        return "size"
    return None


def parse_run_name(name: str) -> dict | None:
    """
    Decompose a run directory name into configuration fields.

    Also accepts bare variant names (``artifacts/tf`` folders), which carry no
    evaluation-tiling prefix; ``eval_tiling`` is then ``None``.

    Args:
        name:
            Run directory name, e.g.
            ``untiled_ssd-mn2-fpnlite_mc_phenobench-tiled_320_int8_ptq_fastnms_val``.

    Returns:
        Dict of configuration fields plus display labels, or ``None`` if the
        name does not carry a recognizable ``sc``/``mc`` class token.
    """
    tokens = name.split("_")

    # The evaluation-input regime prefixes the run directory, i.e. it sits left
    # of the architecture; strip it first so `arch` stays the plain model name
    # and keeps matching ARCH_LABELS / the coverage variant keys.
    eval_tiling = None
    if tokens and tokens[0] in _EVAL_TILINGS:
        eval_tiling = tokens[0]
        tokens = tokens[1:]

    cls_idx = next(
        (i for i, t in enumerate(tokens) if t in _CLASSES),
        None,
    )

    if cls_idx is None:
        return None

    info: dict = {
        "run": name,
        "eval_tiling": eval_tiling,
        "arch": "_".join(tokens[:cls_idx]) or "unknown",
        "classes": tokens[cls_idx],
    }

    dataset_parts: list[str] = []

    for tok in tokens[cls_idx + 1 :]:
        kind = _classify_token(tok)

        if kind is None:
            dataset_parts.append(tok)
        else:
            info.setdefault(kind, tok)

    info["dataset"] = "_".join(dataset_parts) or "unknown"

    info["arch_label"] = ARCH_LABELS.get(info["arch"], info["arch"])
    info["class_label"] = CLASS_LABELS.get(info["classes"], info["classes"])
    info["precision_label"] = PRECISION_LABELS.get(
        info.get("precision", ""),
        info.get("precision", "?").upper(),
    )
    info["quant_label"] = info.get("quant", "?").upper()
    info["granularity_label"] = {
        "per-tensor": "Per-tensor",
        "per-channel": "Per-channel",
    }.get(info.get("granularity"), "")
    info["eval_tiling_label"] = EVAL_TILING_LABELS.get(eval_tiling, "")

    # Compact label used on chart axes. It has to name the training dataset and
    # the evaluation regime: the same architecture is trained on both the tiled
    # and the untiled dataset and each model is then benchmarked on both inputs,
    # so a label of arch + classes alone would collapse four distinct runs into
    # one group and silently average them.
    info["config"] = (
        f"{info['arch_label'].replace('SSD MobileNetV2', 'MNv2')}"
        f" | {info['classes'].upper()}"
        f" | {info['dataset']}"
    )
    if eval_tiling:
        info["config"] += f" | eval:{eval_tiling}"

    return info


# =========================================================
# Loading
# =========================================================


def _read_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def _latency_fields(latency: dict) -> dict:
    """
    Latency summary for one run, preferring order statistics over the mean.

    The benchmark already discards warm-up iterations, but a run still picks up
    the odd scheduling outlier (seen: a single 367 ms sample against a 15 ms
    median) which drags the mean and blows up the min/max range. The median and
    p95 describe the achievable rate far better, so throughput is derived from
    the median; the mean is kept for continuity.
    """

    samples = sorted(latency.get("latencies_ms") or [])

    fields = {
        key: latency.get(key)
        for key in (
            "mean_latency_ms",
            "median_latency_ms",
            "min_latency_ms",
            "max_latency_ms",
        )
    }
    fields["n_latency"] = len(samples)

    if samples:
        fields["p95_latency_ms"] = float(np.percentile(samples, 95))
        # How far the mean is inflated by outliers -- a cheap tell that a run's
        # timing was disturbed (see the sanity checks).
        median = fields.get("median_latency_ms") or float(np.median(samples))
        if median:
            fields["latency_outlier_ratio"] = float(max(samples)) / float(median)
            fields["fps"] = 1000.0 / float(median)

    return fields


def _runtime_fields(runtime: dict) -> dict:
    """
    Execution backend for one run.

    ``delegate`` is only the delegate that was *requested*; a missing or
    unloadable one falls back to CPU. Newer artifacts record what was really
    used, older ones do not, so an unknown backend is reported as such instead
    of being optimistically read as accelerated.
    """

    requested = runtime.get("delegate_requested", runtime.get("delegate"))

    if "backend" in runtime:
        backend = runtime["backend"]
    elif requested is None:
        backend = "cpu"
    else:
        backend = "unknown"

    return {
        "delegate": requested,
        "delegate_active": runtime.get("delegate_active"),
        "backend": backend,
        # Which pipeline stage produced the predictions. Artifacts written
        # before this existed are all TFLite.
        "format": runtime.get("format", "tflite"),
        "input_dtype": _input_dtype(runtime),
    }


def _input_dtype(runtime: dict) -> str | None:
    """
    Input dtype of the first input, across both artifact flavours.

    TFLite records a repr (``<class 'numpy.float32'>``); the SavedModel runtime
    records a bare name (``uint8``). Splitting on quotes unconditionally raises
    on the latter, so fall back to the value as written.
    """
    details = runtime.get("input_details") or []

    if not details:
        return None

    raw = details[0].get("dtype") or ""
    parts = raw.split("'")

    return parts[-2] if len(parts) >= 2 else (raw or None)


def _faithful_fields(faithful: dict | None, *, classes: str | None) -> dict:
    """
    Official PhenoBench (``metrics_faithful.json``) metrics for one run.

    Upstream reports percentages; they are rescaled to the 0-1 range so they sit
    in the same units as the pycocotools columns. ``mAP_cls`` is positional and
    only interpretable with the class order, which newer artifacts carry as
    ``class_names`` -- for older ones the upstream ``[crop, weed]`` order is
    assumed.

    ``faithful_mAP`` is upstream's own aggregate and is **not** comparable
    across runs: it is the unweighted mean over whichever classes upstream
    happened to score, which includes classes the model cannot predict (``crop``
    for a weed-only model) and PhenoBench's partial semantic ids when upstream's
    partial filter did not run. Prefer ``faithful_mAP_plants``, which averages
    only the predictable classes and lines up with the pycocotools ``AP`` to
    within ~0.6 AP -- see
    :func:`agri_vision_edge.evaluation.faithful.annotate_class_metrics`.

    Single-class runs need two further caveats, both flagged rather than
    silently averaged away:

    * ``faithful_mAP`` averages over crop and weed, so a weed-only model is
      penalised for a class it cannot predict -- ``faithful_weed_AP`` is the
      comparable number.
    * results produced before the label remap scored weed predictions against
      crop ground truth and are simply invalid (``faithful_stale``).
    """

    if not faithful:
        return {"faithful": False}

    names = faithful.get("class_names") or ["crop", "weed"]
    per_class = faithful.get("mAP_cls") or []

    fields: dict = {
        "faithful": True,
        "faithful_mAP": _pct(faithful.get("mAP")),
        "faithful_mAP50": _pct(faithful.get("mAP_50")),
        "faithful_mAP75": _pct(faithful.get("mAP_75")),
        # The comparable aggregate (newer artifacts only).
        "faithful_mAP_plants": _pct(faithful.get("mAP_plants")),
        "faithful_upstream_classes": faithful.get("upstream_class_count"),
        "faithful_images_without_predictions": faithful.get(
            "images_without_predictions"
        ),
    }

    for name, value in zip(names, per_class, strict=False):
        fields[f"faithful_{name}_AP"] = _pct(value)

    predicted = faithful.get("predicted_classes")

    # Artifacts without `class_names` predate the label remap; for single-class
    # runs that remap is exactly what was broken, so those numbers are unusable.
    fields["faithful_stale"] = "class_names" not in faithful and classes == "sc"
    fields["faithful_partial_classes"] = bool(
        predicted is not None and len(predicted) < len(names)
    ) or (predicted is None and classes == "sc")

    return fields


def _pct(value) -> float | None:
    """Upstream percentage -> 0-1 fraction, matching the pycocotools columns."""
    return None if value is None else float(value) / 100.0


def load_benchmark_results(
    root: str | Path = "benchmark_results",
    exclude_dirs: Iterable[str] = NON_PLATFORM_DIRS,
) -> tuple[pd.DataFrame, list[str]]:
    """
    Scan every platform / run under ``root`` into one tidy frame.

    Args:
        root:
            ``benchmark_results`` directory.
        exclude_dirs:
            Top-level directory names to skip (non-platform, e.g. ``attic``).

    Returns:
        ``(df, skipped)`` where ``df`` has one row per successful run with
        configuration fields, pycocotools metrics (overall + per-class), the
        official PhenoBench metrics when ``metrics_faithful.json`` is present,
        latency order statistics and the execution backend; ``skipped`` lists
        ``platform/run`` entries that were not loadable.
    """
    root = Path(root)
    exclude = set(exclude_dirs)

    rows: list[dict] = []
    skipped: list[str] = []

    platforms = sorted(
        p for p in root.iterdir() if p.is_dir() and p.name not in exclude
    )

    for platform_dir in platforms:
        for run_dir in sorted(p for p in platform_dir.iterdir() if p.is_dir()):
            tag = f"{platform_dir.name}/{run_dir.name}"

            if (run_dir / "error.json").exists():
                skipped.append(f"{tag} (failed)")
                continue

            metrics = _read_json(run_dir / "metrics.json")
            if metrics is None:
                # A run whose predictions failed the integrity check has its
                # metrics.json removed and a metrics_invalid.json left behind.
                # Say so, otherwise it is indistinguishable from a run that was
                # simply never evaluated.
                if (run_dir / "metrics_invalid.json").exists():
                    skipped.append(f"{tag} (corrupt predictions)")
                else:
                    skipped.append(f"{tag} (no metrics)")
                continue

            info = parse_run_name(run_dir.name)
            if info is None:
                skipped.append(f"{tag} (unparsed)")
                continue

            row = dict(info)
            row["platform"] = platform_dir.name
            # A `<board>_cpu` results tree is the same board with the delegate
            # switched off, not a separate target. Carry the board separately
            # from the backend so figures can group by hardware instead of
            # treating the CPU reference run as a peer platform.
            row["device"] = platform_dir.name.removesuffix("_cpu")

            for key in _METRIC_KEYS:
                row[key] = metrics.get(key)

            for cls, cls_metrics in (metrics.get("per_class") or {}).items():
                row[f"{cls}_AP"] = cls_metrics.get("AP")
                row[f"{cls}_AP50"] = cls_metrics.get("AP50")
                row[f"{cls}_AP75"] = cls_metrics.get("AP75")

            row.update(_latency_fields(_read_json(run_dir / "latency.json") or {}))
            row.update(_runtime_fields(_read_json(run_dir / "runtime.json") or {}))
            row.update(
                _faithful_fields(
                    _read_json(run_dir / "metrics_faithful.json"),
                    classes=row.get("classes"),
                )
            )

            rows.append(row)

    df = pd.DataFrame(rows)

    if not df.empty:
        df = df.sort_values(
            ["platform", "arch", "classes", "precision", "quant"]
        ).reset_index(drop=True)

    return df, skipped


def select_nms(df: pd.DataFrame, nms: str | None = DEFAULT_NMS) -> pd.DataFrame:
    """
    Narrow a run frame to one post-processing variant.

    **Call this before any aggregation.** Every deployable is converted twice,
    once with each NMS implementation, so a frame straight out of
    :func:`load_benchmark_results` carries two rows per configuration that are
    identical in every grouping key a table or figure uses. Without this filter
    a "per scheme" row silently becomes a mean over two post-processing
    algorithms, and a per-run table grows a duplicate for every deployable --
    measured on the current sweep: 224 of 590 configuration groups.

    Rows with no NMS token are kept regardless: the SavedModel reference tree
    and the YOLO exports have no such pair, and dropping them would remove the
    very rung the ladder needs. Pass ``nms=None`` to keep everything, which is
    only correct when the caller groups on ``nms`` itself.
    """
    if df.empty or nms is None or "nms" not in df.columns:
        return df
    return df[df["nms"].isna() | (df["nms"] == nms)]


# =========================================================
# Preprocessing cost
# =========================================================

#: Written by ``scripts/benchmark_resize.py``, one per measurement host::
#:
#:     benchmark_results/<platform>/resize.json
#:
#: A file rather than a directory, deliberately: :func:`load_benchmark_results`
#: walks directories only, so the artifact can sit beside the runs it corrects
#: without being scanned as a run that has no metrics.
RESIZE_ARTIFACT = "resize.json"

#: Latency columns the correction is applied to, and the corrected name each
#: gets. ``min``/``max`` are excluded on purpose: a run's fastest inference and
#: its slowest did not both pay the median resize, and subtracting a central
#: estimate from an extreme produces a number with no defensible meaning.
_CORRECTED_LATENCY_COLUMNS = {
    "median_latency_ms": "median_latency_ms_net",
    "mean_latency_ms": "mean_latency_ms_net",
    "p95_latency_ms": "p95_latency_ms_net",
}


def load_resize_costs(
    root: str | Path = "benchmark_results",
) -> pd.DataFrame:
    """
    Every host's measured ``cv2.resize`` cost, as one tidy frame.

    One row per ``(platform, eval_tiling, size)``: the preprocessing every run
    in that cell paid inside its timed region, measured in isolation on the
    same host by ``scripts/benchmark_resize.py`` (which documents what the
    number is and is not).

    ``size`` is a string, matching :func:`parse_run_name`'s token rather than
    the integer the artifact stores, so the frame joins straight onto a run
    frame without a cast at every call site.
    """
    root = Path(root)

    if not root.is_dir():
        return pd.DataFrame()

    rows: list[dict] = []

    for platform_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        artifact = _read_json(platform_dir / RESIZE_ARTIFACT)

        if not artifact:
            continue

        opencv = artifact.get("opencv") or {}

        for entry in artifact.get("resizes") or []:
            latency = entry.get("latency") or {}

            rows.append(
                {
                    "platform": platform_dir.name,
                    "eval_tiling": entry.get("eval_tiling"),
                    "source_px": entry.get("source_px"),
                    "target_px": entry.get("target_px"),
                    "size": str(entry.get("target_px")),
                    "identity": entry.get("identity"),
                    "resize_median_ms": latency.get("median_latency_ms"),
                    "resize_mean_ms": latency.get("mean_latency_ms"),
                    "resize_p95_ms": latency.get("p95_latency_ms"),
                    "resize_n": latency.get("count"),
                    "opencv_version": opencv.get("version"),
                    "opencv_threads": opencv.get("threads"),
                    "measured": artifact.get("created"),
                }
            )

    return pd.DataFrame(rows)


def _resize_platform_candidates(platform: str) -> list[str]:
    """
    The trees a run's platform may find its resize measurement filed under.

    Resize runs on the CPU whatever the delegate is doing and whatever delegate
    *build* is installed, so ``frdm-imx93``, ``frdm-imx93_cpu`` and
    ``frdm-imx93_unpatched`` all pay the same preprocessing -- one measurement
    per board, not per results tree. Suffixes are joined with ``_`` and board
    hostnames use ``-``, so the leading token is the board.

    Exact match first regardless, so a tree that does have its own artifact is
    never overridden by the board's.
    """
    candidates = [platform, platform.removesuffix("_cpu"), platform.split("_")[0]]

    seen: list[str] = []

    for candidate in candidates:
        if candidate and candidate not in seen:
            seen.append(candidate)

    return seen


def add_resize_cost(
    df: pd.DataFrame,
    costs: pd.DataFrame | str | Path = "benchmark_results",
    *,
    statistic: str = "resize_median_ms",
) -> pd.DataFrame:
    """
    Attach the preprocessing cost of each run, and the latency net of it.

    Every latency in this frame is the wall time of ``runtime.predict()``, and
    that call begins by resizing the source frame to the model's input. The
    resize is genuine deployment work, so the uncorrected figure is the right
    one to quote for "what does this cost to run" -- but it is *not* model
    compute, and its size is set by the pair (source resolution, input size),
    which the sweep varies on two axes at once::

        untiled runs feed a 1024x1024 frame; tiled runs feed a 512x512 tile,
        and the input size ranges over 320 / 512 / 1024.

    So a 320 model evaluated untiled is charged a 1024->320 downscale that the
    1024 model evaluated on the same frames does not pay, and the tiling
    comparison moves the source resolution underneath everything in it. The
    ``*_net`` columns take that term out, which is what makes the resolution
    ladder and the tiling study comparisons of the detector rather than of the
    preprocessing.

    Runs with no measurement for their host get ``NaN``, never the uncorrected
    value: a silently uncorrected row in a corrected column is a wrong number,
    whereas a missing one is a missing measurement and says so.

    The ``tf-savedmodel*`` trees stay ``NaN`` for a second, better reason, and
    it is not an omission to be fixed: those graphs carry a
    ``fixed_shape_resizer`` and are fed at native resolution, so they contain
    no ``cv2.resize`` to subtract at all (see
    :mod:`agri_vision_edge.runtime.inference.saved_model`). Filing a resize
    artifact under them would subtract work that never happened.

    Args:
        df: run frame from :func:`load_benchmark_results`.
        costs: frame from :func:`load_resize_costs`, or the results root to
            load it from.
        statistic: which order statistic to subtract. The median matches the
            column the report quotes latency from; ``resize_mean_ms`` pairs
            with ``mean_latency_ms`` if a caller wants like for like.
    """
    if df.empty:
        return df

    if not isinstance(costs, pd.DataFrame):
        costs = load_resize_costs(costs)

    out = df.copy()

    for column in ("resize_ms", "resize_source_px", "resize_platform"):
        out[column] = np.nan

    for column in _CORRECTED_LATENCY_COLUMNS.values():
        out[column] = np.nan

    if costs.empty or "platform" not in out.columns:
        return out

    indexed = costs[
        ~costs.duplicated(subset=["platform", "eval_tiling", "size"], keep="first")
    ].set_index(["platform", "eval_tiling", "size"])

    def _lookup(row):
        for candidate in _resize_platform_candidates(str(row.get("platform", ""))):
            key = (candidate, row.get("eval_tiling"), str(row.get("size")))

            if key in indexed.index:
                entry = indexed.loc[key]
                return candidate, entry.get("source_px"), entry.get(statistic)

        return None, np.nan, np.nan

    resolved = out.apply(_lookup, axis=1, result_type="expand")
    out["resize_platform"] = resolved[0]
    out["resize_source_px"] = resolved[1]
    out["resize_ms"] = pd.to_numeric(resolved[2], errors="coerce")

    for column, corrected in _CORRECTED_LATENCY_COLUMNS.items():
        if column not in out.columns:
            continue

        # Clipped at zero rather than allowed negative. A negative "compute
        # time" is nonsense to print, and the only way to reach one is a resize
        # measured under different conditions than the run -- which the
        # magnitude of the clip makes obvious in the frame.
        out[corrected] = (
            pd.to_numeric(out[column], errors="coerce") - out["resize_ms"]
        ).clip(lower=0.0)

    net = out["median_latency_ms_net"]
    out["fps_net"] = np.where(net > 0, 1000.0 / net, np.nan)

    # What fraction of the measured latency was preprocessing. This is the
    # number that decides whether the correction is worth quoting at all: a few
    # percent on a 1024 model, far more on a 320 model fed full frames.
    if "median_latency_ms" in out.columns:
        measured = pd.to_numeric(out["median_latency_ms"], errors="coerce")
        out["resize_share"] = out["resize_ms"] / measured.where(measured > 0)

    return out


def resize_cost_table(
    costs: pd.DataFrame | str | Path = "benchmark_results",
) -> pd.DataFrame:
    """
    The measured preprocessing cost per host, as a table for the appendix.

    One row per host and source resolution, one column per model input size --
    the shape the correction is applied in, so a reader can check any ``*_net``
    figure against it by hand.
    """
    if not isinstance(costs, pd.DataFrame):
        costs = load_resize_costs(costs)

    if costs.empty:
        return pd.DataFrame()

    table = costs.copy()
    table["Host"] = table["platform"]
    table["Source"] = table["source_px"].map(lambda px: f"{px}x{px}")
    table["Input"] = table["target_px"]
    table["ms"] = table["resize_median_ms"].round(3)

    wide = table.pivot_table(
        index=["Host", "Source"],
        columns="Input",
        values="ms",
        aggfunc="first",
    )

    wide.columns = [f"-> {int(column)} (ms)" for column in wide.columns]

    return wide.reset_index()


# =========================================================
# Styling
# =========================================================

#: Colour-blind-friendly encodings reused across every figure.
PRECISION_COLORS = {"fp32": "#4C72B0", "int8": "#DD8452", "fp16": "#55A868"}
CLASSES_COLORS = {"sc": "#8172B3", "mc": "#C44E52"}
PALETTE = ("#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3", "#937860")

_PUBLICATION_RCPARAMS = {
    "figure.dpi": 130,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "font.size": 10,
    "font.family": "serif",
    "axes.titlesize": 11,
    "axes.titleweight": "bold",
    "axes.labelsize": 10,
    "legend.frameon": False,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
}


def apply_publication_style() -> None:
    """Apply the serif, despined publication theme (extends ``curves``)."""
    plt.rcParams.update(_PUBLICATION_RCPARAMS)


def _grouped_bars(
    ax,
    categories,
    group_labels,
    values,
    colors,
    ylabel,
    title,
    *,
    annotate=True,
    fmt=None,
    rotation=0,
    horizontal=False,
    percent=False,
):
    """
    Draw side-by-side bar groups.

    Args:
        values:
            Mapping ``{group_label: [value per category]}``.
        percent:
            Scale AP fractions to upstream's percentage-point convention
            (``39.4``, not ``0.3940``) and mark the axis with ``(%)``. The bar
            labels are the reason: at three decimals a fraction spends five
            glyphs to carry two digits of information, and on a thesis page
            that is the difference between a readable figure and a grey smear.
            Tables reach the same convention through ``percent=`` arguments of
            their own, so figure and table agree.
        fmt:
            Label format. ``None`` picks one to match ``percent``.
        horizontal:
            Put the categories on the **y** axis and the bars along x. Worth it
            once the category labels stop fitting side by side: a vertical
            chart has to share one figure width between every category, while a
            horizontal one gives each label a full line and grows downward
            instead (see :func:`plot_quantization_effect`).
    """

    def _missing(v):
        return v is None or (isinstance(v, float) and np.isnan(v))

    if percent:
        values = {
            label: [v if _missing(v) else 100.0 * v for v in vals]
            for label, vals in values.items()
        }
        if ylabel and "%" not in ylabel:
            ylabel = f"{ylabel} (%)"

    if fmt is None:
        fmt = "{:.1f}" if percent else "{:.3f}"

    n_groups = max(len(group_labels), 1)
    x = np.arange(len(categories))
    width = 0.8 / n_groups

    # Vertical bars split one figure width between *every* bar, so what decides
    # whether a label fits is the total, not the bars per group: five schemes
    # across two variants is roomy, the same five across eight is where
    # "46.546.3" comes from. Upright text needs only the bar's own width.
    # Horizontal bars each own a full line and never collide.
    label_rotation = 90 if (not horizontal and n_groups * len(categories) > 12) else 0

    for i, glabel in enumerate(group_labels):
        offset = (i - (n_groups - 1) / 2) * width
        vals = values[glabel]

        draw = ax.barh if horizontal else ax.bar
        size_kw = "height" if horizontal else "width"

        bars = draw(
            x + offset,
            vals,
            **{size_kw: width},
            label=glabel,
            color=colors[i % len(colors)],
            edgecolor="white",
            linewidth=0.5,
        )

        if annotate:
            for bar, v in zip(bars, vals, strict=False):
                if v is None or (isinstance(v, float) and np.isnan(v)):
                    continue

                # The label sits on the far side of the bar's tip, so it
                # follows the bar's direction: a negative bar grows away from
                # the axis and its label has to go with it, or it is written
                # back over the bar it belongs to.
                negative = v < 0

                if horizontal:
                    xy = (v, bar.get_y() + bar.get_height() / 2)
                    align = {"ha": "right" if negative else "left", "va": "center"}
                    offset_pts = (-2 if negative else 2, 0)
                else:
                    xy = (bar.get_x() + bar.get_width() / 2, v)
                    align = {"ha": "center", "va": "top" if negative else "bottom"}
                    offset_pts = (0, -1 if negative else 1)

                ax.annotate(
                    fmt.format(v),
                    xy,
                    fontsize=7,
                    xytext=offset_pts,
                    textcoords="offset points",
                    rotation=label_rotation,
                    **align,
                )

    if horizontal:
        ax.set_yticks(x)
        ax.set_yticklabels(categories)
        # Read the categories top-down, the order they were passed in.
        ax.invert_yaxis()
        ax.set_xlabel(ylabel)
        ax.margins(x=0.15)
    else:
        ax.set_xticks(x)
        # Anchor rotated labels by their upper (right) end at the tick, so the
        # text reads from the category position outward instead of being centred
        # under it (much easier to read with long labels).
        ax.set_xticklabels(
            categories,
            rotation=rotation,
            ha="right" if rotation else "center",
            rotation_mode="anchor" if rotation else "default",
        )
        ax.set_ylabel(ylabel)
        ax.margins(y=0.15)

    ax.set_title(title)
    _prepare_axis(ax)
    return ax


def _legend_outside(fig, ax, **kwargs):
    """
    Put the legend right of the axes and actually reserve room for it.

    ``tight_layout`` only measures artists *inside* the axes, and neither the
    notebook's inline render nor :func:`save_figure` passes
    ``bbox_inches="tight"``. An outside legend is therefore drawn over whatever
    happens to be beside it unless the space is taken out of the axes first --
    which is what the ``subplots_adjust`` here does, using the legend's own
    measured width rather than a guessed fraction.

    Call it *after* ``tight_layout``; that would otherwise undo the reservation.
    """
    legend = ax.legend(
        loc="upper left", bbox_to_anchor=(1.01, 1.0), borderaxespad=0.0, **kwargs
    )
    reserve_legend_space(fig)

    return legend


def reserve_legend_space(fig, pad: float = 0.04, passes: int = 3) -> None:
    """
    Take the width of the widest outside legend out of the axes.

    Measures *every* legend on the figure, not just one: a plot with a second
    legend below the first reserves too little if only the first is measured,
    and the wider one is then clipped at the figure edge.

    Iterates because the reservation is a fixed point, not a calculation. The
    legend is anchored in *axes* coordinates, so narrowing the axes moves it and
    changes the extent that decided how far to narrow them. A single pass
    undershoots and still clips; three settle it.
    """
    right = 1.0

    for _ in range(passes):
        fig.canvas.draw()

        widest = 0.0
        for ax in fig.axes:
            for artist in (ax.get_legend(), *getattr(ax, "artists", [])):
                if artist is None or not hasattr(artist, "get_window_extent"):
                    continue
                try:
                    widest = max(widest, artist.get_window_extent().width)
                except Exception:  # noqa: BLE001 - an unrenderable artist skips
                    continue

        if not widest:
            return

        share = widest / fig.get_window_extent().width
        target = max(0.5, 1.0 - share - pad)

        if abs(target - right) < 0.005:
            return

        right = target
        fig.subplots_adjust(right=right)


def _variant_label(
    frame: pd.DataFrame, columns: Iterable[str] = ("classes", "dataset", "size")
) -> pd.Series:
    """
    Architecture, joined with whichever of ``columns`` actually varies.

    Two reasons, one cosmetic and one not. A label repeating ``MC | phenobench``
    on every row spends axis width to say nothing, and that width comes out of
    the bars. And ``size`` belongs in the list because a resolution that is not
    part of the label is not part of the grouping either -- the rows are then
    averaged into one bar with nothing on the axis admitting it.
    """
    label = _short(frame["arch_label"])

    for col in columns:
        if col not in frame.columns or frame[col].nunique(dropna=False) <= 1:
            continue
        part = frame[col].astype(str)
        label = label + " | " + (part.str.upper() if col == "classes" else part)

    return label


def _short(series: pd.Series) -> pd.Series:
    return series.str.replace("SSD MobileNetV2", "MNv2")


# =========================================================
# Figures
# =========================================================


def plot_quantization_effect(
    df: pd.DataFrame,
    *,
    eval_tiling: str | None = None,
    platform: str | None = None,
    nms: str | None = DEFAULT_NMS,
):
    """FP32 vs INT8 on overall AP and weed AP (PTQ runs).

    INT8 is represented by the per-tensor export (the default deployment
    granularity) so each config contributes one FP32/INT8 pair; per-channel PTQ
    is compared separately elsewhere.

    Bars are **horizontal**, and the figure height grows with the number of
    categories. Vertical bars do not survive a multi-platform sweep: one
    category is ``platform | arch | classes | trained-on | eval-input``, so the
    count is the product of all five and every added platform multiplies it.
    At four platforms that is 67 categories sharing a 10-inch axis -- 0.15 in
    each, for labels averaging 56 characters -- which no rotation can rescue.
    Horizontal bars give each label its own line, and the two metric panels sit
    side by side sharing one set of labels instead of repeating them.

    ``eval_tiling`` / ``platform`` narrow the figure the same way
    :func:`plot_scheme_effect` does; without them every regime and board is
    shown at once.
    """
    df = select_nms(df, nms)
    df = df[df.get("quant") == "ptq"].copy() if "quant" in df else df.copy()
    if "granularity" in df.columns:
        df = df[
            (df["precision"] == "fp32") | (df["granularity"] == "per-tensor")
        ].copy()
    if eval_tiling is not None and "eval_tiling" in df.columns:
        df = df[df["eval_tiling"] == eval_tiling].copy()
    if platform is not None and "platform" in df.columns:
        df = df[df["platform"] == platform].copy()
    if df.empty or df["precision"].nunique() < 2:
        return None

    group_cols = ["platform", "config"] if platform is None else ["config"]
    df["group"] = df[group_cols].agg(" | ".join, axis=1)
    precisions = [p for p in ("fp32", "int8") if p in df["precision"].values]
    groups = sorted(df["group"].unique())

    # One line per category, floored so a short figure still looks deliberate.
    height = max(3.0, 0.28 * len(groups)) + 1.1

    fig, axes = plt.subplots(1, 2, figsize=(13, height), sharey=True)
    for ax, metric, title in (
        (axes[0], "AP", "Overall AP (COCO)"),
        (axes[1], "weed_AP", "Weed AP"),
    ):
        values = {
            p: [
                df[(df["group"] == g) & (df["precision"] == p)][metric].mean()
                for g in groups
            ]
            for p in precisions
        }
        _grouped_bars(
            ax,
            groups,
            precisions,
            values,
            [PRECISION_COLORS[p] for p in precisions],
            ylabel=metric,
            percent=True,
            title=title,
            horizontal=True,
        )
        ax.legend(title="precision", loc="lower right")

    suffix = []
    if platform:
        suffix.append(platform)
    if eval_tiling:
        suffix.append(f"{eval_tiling} input")
    title = "FP32 -> INT8 quantization effect"
    if suffix:
        title += f" ({', '.join(suffix)})"

    fig.suptitle(title)
    fig.tight_layout()
    return fig


#: Tree whose CPU runs stand in for "unaccelerated reference" everywhere: a
#: conventional x86-64 machine, named for the ISA rather than a hostname so the
#: contrast against the ARM boards is what the name carries. Nine functions
#: below take it as a default, so a value that matches no directory empties
#: them all silently -- `sanity_checks` asserts it resolves.
CPU_REFERENCE_PLATFORM = "x86_cpu"

#: Metrics compared when checking that every CPU tree agrees with the reference.
CPU_REFERENCE_METRICS = ("AP", "AP50", "AP75", "weed_AP", "crop_AP")

#: Largest absolute metric difference still counted as "same result". CPU INT8
#: predictions are bit-identical across x86 and ARM; only fp32 kernels differ,
#: and only by pycocotools accumulation noise (measured max 2.3e-07 AP). Four
#: orders of magnitude below the spread between INT8 *backends* (0.002-0.008).
CPU_REFERENCE_TOLERANCE = 1e-5


def cpu_reference_divergence(
    df: pd.DataFrame,
    *,
    reference: str = CPU_REFERENCE_PLATFORM,
    metrics: Iterable[str] = CPU_REFERENCE_METRICS,
) -> pd.DataFrame:
    """
    How far each CPU-backend results tree sits from the reference host's.

    Reporting one CPU curve instead of one per board is only honest if the
    boards actually agree, so this is the check that licenses it -- run it
    before any figure that collapses them. One row per
    ``(platform, metric)`` with the worst and mean absolute difference over the
    configs the two have in common.

    Empty when there is nothing to compare (no reference tree, or no other CPU
    tree), which callers should treat as "unverified", not "passed".
    """
    if df.empty or "backend" not in df.columns:
        return pd.DataFrame()

    cpu = df[df["backend"] == "cpu"]
    if reference not in set(cpu.get("platform", [])):
        return pd.DataFrame()

    keys = [
        k
        for k in (
            "arch_label",
            "class_label",
            "dataset",
            "size",
            "eval_tiling",
            "precision",
            "quant",
            "granularity",
            "nms",
        )
        if k in cpu.columns
    ]

    def _indexed(frame):
        f = frame.copy()
        # fp32 rows carry no granularity; NaN never equals NaN, so a NaN key
        # would silently drop every float config from the comparison.
        for k in keys:
            f[k] = f[k].astype("object").where(f[k].notna(), "-")
        f = f.set_index(keys)
        return f[~f.index.duplicated(keep="first")]

    ref = _indexed(cpu[cpu["platform"] == reference])

    rows = []
    for platform, group in cpu[cpu["platform"] != reference].groupby("platform"):
        other = _indexed(group)
        common = other.index.intersection(ref.index)

        for metric in metrics:
            if metric not in other.columns or metric not in ref.columns:
                continue

            diff = (other.loc[common, metric] - ref.loc[common, metric]).abs().dropna()
            if diff.empty:
                continue

            rows.append(
                {
                    "platform": platform,
                    "reference": reference,
                    "metric": metric,
                    "configs": int(len(diff)),
                    "max_abs_diff": float(diff.max()),
                    "mean_abs_diff": float(diff.mean()),
                    "bit_identical": int((diff < 1e-9).sum()),
                }
            )

    return pd.DataFrame(rows)


def cpu_reference_holds(
    divergence: pd.DataFrame,
    *,
    tolerance: float = CPU_REFERENCE_TOLERANCE,
) -> bool:
    """Whether every compared CPU tree matched the reference within tolerance."""
    if divergence.empty:
        return False
    return bool((divergence["max_abs_diff"] < tolerance).all())


#: Colours for the five deployable exports, float first then INT8 coarse->fine.
SCHEME_COLORS = {
    "fp32_ptq": "#4C72B0",
    "int8_ptq_per-tensor": "#DD8452",
    "int8_ptq_per-channel": "#E8B778",
    "int8_qat_per-tensor": "#55A868",
    "int8_qat_per-channel": "#8FC7A0",
}


def plot_scheme_effect(
    df: pd.DataFrame,
    *,
    eval_tiling: str | None = "untiled",
    metric: str = "AP",
    platform: str | None = CPU_REFERENCE_PLATFORM,
    nms: str | None = DEFAULT_NMS,
):
    """
    All five quantization schemes side by side, one bar group per variant.

    This is the figure for the PTQ-vs-QAT question, and unlike an aggregated
    FP32-vs-INT8 view it shows each export on its own: a single broken scheme
    stays visible instead of being averaged into an "INT8" bar.

    ``nms`` is pinned for the same reason ``platform`` is: a bar is a mean over
    the matching rows, and the default export and its algorithm-matched
    control both match, so leaving it open silently halves the substitution's
    cost into every INT8 bar.

    ``platform`` defaults to the CPU reference, which makes this the *export*
    comparison -- quantization cost with the accelerator held out. It has to be
    pinned to one platform: a bar is a mean over the matching rows, so leaving
    it open averages every board into each bar. With the i.MX8MP's collapsed
    per-channel runs in the frame that turned a 0.313 bar into 0.235, a number
    describing no configuration that exists. Pass ``platform=None`` deliberately
    to get the old cross-platform mean; use :func:`plot_backend_effect` for the
    CPU-vs-NPU question.
    """
    if df.empty or metric not in df.columns:
        return None

    sel = select_nms(df.copy(), nms)
    if eval_tiling is not None and "eval_tiling" in sel.columns:
        sel = sel[sel["eval_tiling"] == eval_tiling]
    if platform is not None:
        sel = sel[sel["platform"] == platform]
    if sel.empty:
        return None

    sel = add_scheme(sel)
    schemes = [s for s in SCHEME_ORDER if s in set(sel["scheme"])]
    if len(schemes) < 2:
        return None

    sel["group"] = _variant_label(sel)
    groups = sorted(sel["group"].unique())

    values = {
        s: [
            sel[(sel["group"] == g) & (sel["scheme"] == s)][metric].mean()
            for g in groups
        ]
        for s in schemes
    }

    fig, ax = plt.subplots(figsize=(max(9, 1.5 * len(groups)), 4.6))
    _grouped_bars(
        ax,
        groups,
        schemes,
        values,
        [SCHEME_COLORS.get(s, "#999999") for s in schemes],
        ylabel=metric,
        percent=True,
        title=(
            f"{metric} by quantization scheme"
            + (f" ({eval_tiling} input)" if eval_tiling else "")
            + (f" — {platform}" if platform else " — mean over platforms")
        ),
        rotation=20,
    )
    # Outside the axes: the bars reach the top of the plotting area and their
    # value labels sit above them, so an in-axes legend lands on the numbers.
    fig.tight_layout()
    _legend_outside(fig, ax, title="scheme")
    return fig


#: Bar colours for the backend comparison: the CPU reference first, then the
#: accelerated boards.
BACKEND_COLORS = ("#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3")


def plot_backend_effect(
    df: pd.DataFrame,
    *,
    eval_tiling: str | None = "untiled",
    metric: str = "AP",
    reference: str = CPU_REFERENCE_PLATFORM,
    nms: str | None = DEFAULT_NMS,
):
    """
    INT8 on the CPU reference vs each NPU delegate, per variant and scheme.

    The companion to :func:`plot_scheme_effect`: that one holds the hardware
    fixed and varies the export, this one holds the export fixed and varies the
    hardware. Splitting them is what keeps either readable -- a single figure
    crossing both axes is the product of every variant, scheme and board.

    Only INT8 exports appear, because that is the only precision an NPU
    delegate actually accelerates; the float baseline lives in the scheme
    figure. A delegate that reproduces its CPU reference should draw bars of
    equal height, so any visible gap is the accelerator changing the result.
    """
    if df.empty or metric not in df.columns or "backend" not in df.columns:
        return None

    sel = select_nms(df[df["precision"] == "int8"].copy(), nms)
    if eval_tiling is not None and "eval_tiling" in sel.columns:
        sel = sel[sel["eval_tiling"] == eval_tiling]
    if sel.empty:
        return None

    sel = add_scheme(sel)

    # One series per hardware backend: the CPU reference, then each board that
    # actually ran through a delegate.
    is_reference = (sel["platform"] == reference) & (sel["backend"] == "cpu")
    sel = sel[is_reference | (sel["backend"] == "delegate")].copy()
    sel["series"] = np.where(
        sel["platform"] == reference,
        f"{reference} (CPU ref)",
        sel.get("device", sel["platform"]).astype(str) + " (NPU)",
    )

    series = [f"{reference} (CPU ref)"] + sorted(
        s for s in set(sel["series"]) if not s.startswith(reference)
    )
    if len(series) < 2:
        return None

    sel["group"] = _variant_label(sel) + " | " + sel["scheme"].map(scheme_label)
    groups = sorted(sel["group"].unique())

    values = {
        s: [
            sel[(sel["group"] == g) & (sel["series"] == s)][metric].mean()
            for g in groups
        ]
        for s in series
    }

    height = max(3.0, 0.28 * len(groups)) + 1.1
    fig, ax = plt.subplots(figsize=(9, height))
    _grouped_bars(
        ax,
        groups,
        series,
        values,
        BACKEND_COLORS,
        ylabel=metric,
        percent=True,
        title=(
            f"INT8 {metric}: CPU reference vs NPU"
            + (f" ({eval_tiling} input)" if eval_tiling else "")
        ),
        horizontal=True,
    )
    fig.tight_layout()
    _legend_outside(fig, ax, title="backend")
    return fig


def plot_single_vs_multiclass(df: pd.DataFrame, *, nms: str | None = DEFAULT_NMS):
    """Weed AP under single-class vs multi-class regimes."""
    if df.empty or df["classes"].nunique() < 2:
        return None

    df = select_nms(df, nms).copy()
    df["group"] = (
        df["platform"]
        + " | "
        + _short(df["arch_label"])
        + " | "
        + df["precision"].str.upper()
    )
    groups = sorted(df["group"].unique())
    class_order = [c for c in ("sc", "mc") if c in df["classes"].values]

    fig, ax = plt.subplots(figsize=(9, 4.2))
    values = {
        c: [
            df[(df["group"] == g) & (df["classes"] == c)]["weed_AP"].mean()
            for g in groups
        ]
        for c in class_order
    }
    _grouped_bars(
        ax,
        groups,
        class_order,
        values,
        [CLASSES_COLORS[c] for c in class_order],
        ylabel="Weed AP",
        percent=True,
        title="Weed AP: single-class vs. multi-class",
        rotation=20,
    )
    ax.legend(
        title="regime",
        labels=["single-class", "multi-class"][: len(class_order)],
    )
    fig.tight_layout()
    return fig


def plot_architecture_effect(df: pd.DataFrame, *, nms: str | None = DEFAULT_NMS):
    """Overall AP per architecture (SSD MobileNetV2 vs FPNLite)."""
    if df.empty or df["arch"].nunique() < 2:
        return None

    df = select_nms(df, nms).copy()
    df["group"] = (
        df["platform"]
        + " | "
        + df["classes"].str.upper()
        + " | "
        + df["precision"].str.upper()
    )
    groups = sorted(df["group"].unique())
    archs = sorted(df["arch_label"].unique())

    fig, ax = plt.subplots(figsize=(9, 4.2))
    values = {
        a: [
            df[(df["group"] == g) & (df["arch_label"] == a)]["AP"].mean()
            for g in groups
        ]
        for a in archs
    }
    _grouped_bars(
        ax,
        groups,
        archs,
        values,
        PALETTE,
        ylabel="AP",
        percent=True,
        title="Architecture comparison (overall AP)",
        rotation=15,
    )
    ax.legend(title="architecture")
    fig.tight_layout()
    return fig


def plot_per_class_ap(df: pd.DataFrame, *, nms: str | None = DEFAULT_NMS):
    """Crop vs weed AP for multi-class runs."""
    if "crop_AP" not in df:
        return None
    df = select_nms(df, nms)
    df = df[(df["classes"] == "mc") & df["crop_AP"].notna()].copy()
    if df.empty:
        return None

    df["group"] = (
        df["platform"]
        + " | "
        + _short(df["arch_label"])
        + " | "
        + df["precision"].str.upper()
    )
    groups = sorted(df["group"].unique())

    fig, ax = plt.subplots(figsize=(9, 4.2))
    values = {
        "crop": [df[df["group"] == g]["crop_AP"].mean() for g in groups],
        "weed": [df[df["group"] == g]["weed_AP"].mean() for g in groups],
    }
    _grouped_bars(
        ax,
        groups,
        ["crop", "weed"],
        values,
        PALETTE,
        ylabel="AP",
        percent=True,
        title="Per-class AP (multi-class runs)",
        rotation=15,
    )
    ax.legend(title="class")
    fig.tight_layout()
    return fig


def plot_ap_by_area(df: pd.DataFrame, *, nms: str | None = DEFAULT_NMS):
    """AP broken down by COCO object area (small / medium / large)."""
    if df.empty:
        return None

    df = select_nms(df, nms).copy()
    df["label"] = (
        df["platform"] + " | " + df["config"] + " | " + df["precision"].str.upper()
    )
    labels = sorted(df["label"].unique())
    areas = ["APS", "APM", "APL"]

    fig, ax = plt.subplots(figsize=(max(8, 1.1 * len(labels)), 4.2))
    values = {a: [df[df["label"] == lbl][a].mean() for lbl in labels] for a in areas}
    _grouped_bars(
        ax,
        labels,
        areas,
        values,
        PALETTE,
        ylabel="AP",
        percent=True,
        title="AP by object area",
        rotation=25,
    )
    ax.legend(title="object size", labels=["small", "medium", "large"])
    fig.tight_layout()
    return fig


def plot_latency(df: pd.DataFrame, *, nms: str | None = DEFAULT_NMS):
    """Horizontal mean-latency bars with min/max whiskers, per run."""
    if "mean_latency_ms" not in df:
        return None
    df = select_nms(df, nms)
    df = df[df["mean_latency_ms"].notna()].copy()
    if df.empty:
        return None

    df["label"] = (
        df["platform"] + " | " + df["config"] + " | " + df["precision"].str.upper()
    )
    df = df.sort_values("mean_latency_ms")
    colors = [PRECISION_COLORS.get(p, "#999999") for p in df["precision"]]

    fig, ax = plt.subplots(figsize=(8, 0.5 * len(df) + 1.5))
    bars = ax.barh(
        df["label"],
        df["mean_latency_ms"],
        color=colors,
        edgecolor="white",
        linewidth=0.5,
    )
    err_low = df["mean_latency_ms"] - df["min_latency_ms"]
    err_high = df["max_latency_ms"] - df["mean_latency_ms"]
    ax.errorbar(
        df["mean_latency_ms"],
        df["label"],
        xerr=[err_low, err_high],
        fmt="none",
        ecolor="#444444",
        elinewidth=0.8,
        capsize=2,
    )
    for bar, v in zip(bars, df["mean_latency_ms"], strict=False):
        ax.annotate(
            f"{v:.1f} ms",
            (v, bar.get_y() + bar.get_height() / 2),
            va="center",
            ha="left",
            fontsize=7,
            xytext=(3, 0),
            textcoords="offset points",
        )
    ax.set_xlabel("mean latency (ms)  —  min/max whiskers")
    ax.set_title("Inference latency per run")
    _prepare_axis(ax)
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    return fig


#: Trees that are references rather than deployment targets. Their latency
#: answers no question the trade-off asks: the SavedModel rung is a scoring
#: reference, and the x86 CPU exists to establish that a device run computed
#: the right answer -- comparing a desktop's milliseconds against an embedded
#: board's is a statement about two machine classes, not about a deployment.
NON_TARGET_PLATFORMS = ("tf-savedmodel", "tf-savedmodel-nms0", CPU_REFERENCE_PLATFORM)


def platform_label(platform: str) -> str:
    """
    Short figure label for a results tree: ``frdm-imx8mp_cpu`` -> ``i.MX8MP CPU``.

    The directory names carry a vendor prefix and encode the backend as a
    suffix, which makes them long and makes the CPU/NPU distinction -- the one
    a reader is actually comparing -- the least visible part of the string.
    """
    name = platform
    backend = "CPU" if name.endswith("_cpu") else "NPU"
    name = name.removesuffix("_cpu").removesuffix("_unpatched")
    pretty = {
        "frdm-imx8mp": "i.MX8MP",
        "frdm-imx93": "i.MX93",
        CPU_REFERENCE_PLATFORM: "x86",
    }.get(name, name)
    suffix = " (unpatched)" if platform.endswith("_unpatched") else ""

    return f"{pretty} {backend}{suffix}"


#: Marker per export scheme -- the *whole* scheme, not just its granularity.
#: A scheme is a quantization method **and** a granularity, so keying the shape
#: on granularity alone does not identify a point: `ptq_per-tensor` and
#: `qat_per-tensor` drew the same circle in the same colour, and the same
#: token appeared twice in a panel meaning two different exports. Colour is
#: already platform, so the shape has to carry both halves.
SCHEME_MARKERS = {
    "fp32_ptq": "^",
    "int8_ptq_per-tensor": "o",
    "int8_ptq_per-channel": "s",
    "int8_qat_per-tensor": "D",
    "int8_qat_per-channel": "P",
}


def plot_accuracy_latency(
    df: pd.DataFrame,
    *,
    nms: str | None = DEFAULT_NMS,
    exclude_platforms: Iterable[str] = NON_TARGET_PLATFORMS,
    facet: str = "arch_label",
):
    """
    Accuracy / latency trade-off, one panel per architecture.

    Faceted rather than pooled because accuracy here is set by architecture and
    export scheme -- the delegates reproduce their CPU reference to within
    0.002 AP -- while latency is set by platform and granularity. Pooled, the
    two detectors form two AP bands that share no range, and most of the axis
    is empty space between them.

    Colour is platform and shape is the export scheme, which together identify
    a point uniquely, so none needs a text label. An earlier version annotated
    every point with its full configuration -- after scoping to one cell, the
    same string on all of them.
    """
    if "mean_latency_ms" not in df:
        return None

    from matplotlib.lines import Line2D

    sel = add_scheme(select_nms(df, nms))
    sel = sel[~sel["platform"].isin(set(exclude_platforms))]
    # `_unpatched` is the same board under an older delegate build: a build
    # comparison (§4.3), not a deployment option.
    sel = sel[~sel["platform"].str.endswith("_unpatched")]
    sel = sel[sel["mean_latency_ms"].notna() & sel["AP"].notna()].copy()
    if sel.empty:
        return None

    panels = sorted(sel[facet].dropna().unique())
    platforms = sorted(sel["platform"].unique())
    colors = {p: PALETTE[i % len(PALETTE)] for i, p in enumerate(platforms)}

    fig, axes = plt.subplots(
        len(panels), 1, figsize=(8, 3.2 * len(panels)), sharex=True
    )
    axes = np.atleast_1d(axes)

    for index, (ax, panel) in enumerate(zip(axes, panels, strict=False)):
        rows = sel[sel[facet] == panel]
        for _, r in rows.iterrows():
            ax.scatter(
                r["mean_latency_ms"],
                100.0 * r["AP"],
                color=colors[r["platform"]],
                marker=SCHEME_MARKERS.get(str(r.get("scheme")), "X"),
                s=70,
                edgecolor="black",
                linewidth=0.5,
                zorder=3,
            )
        ax.set_ylabel("AP (%)")
        ax.set_title(_short(pd.Series([panel])).iloc[0])
        if index == len(panels) - 1:
            ax.set_xlabel("mean latency (ms)")
        _prepare_axis(ax)

    platform_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label=platform_label(p),
            markerfacecolor=colors[p],
            markersize=9,
            markeredgecolor="black",
        )
        for p in platforms
    ]
    scheme_handles = [
        Line2D(
            [0],
            [0],
            marker=marker,
            color="w",
            label=scheme_label(scheme),
            markerfacecolor="#cccccc",
            markersize=9,
            markeredgecolor="black",
        )
        for scheme, marker in SCHEME_MARKERS.items()
        if sel["scheme"].eq(scheme).any()
    ]

    fig.suptitle("Accuracy / latency trade-off")
    fig.tight_layout()
    # No `add_artist` here. That idiom keeps a first legend alive when a second
    # is added to the *same* axes, but these sit on different ones -- and
    # re-registering the object matplotlib already holds as `axes[0].legend_`
    # draws it twice, text over itself, which reads as a doubled label.
    _legend_outside(fig, axes[0], handles=platform_handles, title="platform")
    axes[-1].legend(
        handles=scheme_handles,
        title="export scheme",
        loc="lower left",
        bbox_to_anchor=(1.01, 0.0),
        borderaxespad=0.0,
    )
    # Again, now that the second legend exists: the reservation is sized to the
    # widest one, and this is the first moment both can be measured.
    reserve_legend_space(fig)
    return fig


# =========================================================
# Thesis tables
# =========================================================

#: Published PhenoBench plant-detection baselines (test split), as printed in
#: the dataset paper. Percentages, so they share the units of the *faithful*
#: (upstream torchmetrics) columns -- never mix them with the pycocotools ones.
PHENOBENCH_BASELINES = (
    {
        "Approach": "Faster R-CNN",
        "mAP": 40.43,
        "mAP50": 65.07,
        "mAP75": 40.19,
        "Crop AP": 63.23,
        "Weed AP": 17.62,
        "Source": "upstream",
    },
    {
        "Approach": "Mask R-CNN",
        "mAP": 38.68,
        "mAP50": 63.72,
        "mAP75": 38.07,
        "Crop AP": 60.32,
        "Weed AP": 17.05,
        "Source": "upstream",
    },
    {
        "Approach": "YOLOv7",
        "mAP": 60.48,
        "mAP50": 82.47,
        "mAP75": 62.30,
        "Crop AP": 83.06,
        "Weed AP": 37.91,
        "Source": "upstream",
    },
)


def upstream_comparison_table(
    df: pd.DataFrame,
    *,
    platform: str | None = None,
    include_baselines: bool = True,
    nms: str | None = DEFAULT_NMS,
) -> pd.DataFrame:
    """
    Our detectors next to the published PhenoBench baselines.

    Only runs that are actually comparable to the upstream leaderboard are
    eligible, which is a narrow slice:

    * **multi-class** -- upstream averages mAP over crop and weed, so a
      weed-only model is scored against a class it cannot predict;
    * **untiled training data and untiled evaluation** -- the upstream number is
      a full-frame 1024x1024 one, and our tile-wise faithful evaluation is
      explicitly not that;
    * **official metrics only** -- the ``faithful_*`` columns, in upstream
      percentage units. Runs whose faithful metrics predate the label remap are
      dropped rather than shown as-is.

    Note that the upstream figures are on the *test* split while ours are on
    val, so this is an orientation, not a like-for-like ranking.
    """
    if df.empty or "faithful_mAP" not in df.columns:
        return pd.DataFrame()

    # The default export only. Both variants are deployable, so this is a
    # presentation choice rather than a correctness one: listing a model twice
    # under two post-processing settings on a leaderboard-style table invites
    # the reader to compare our best configuration against upstream's single
    # published one.
    df = select_nms(df, nms)

    sel = df[
        (df["classes"] == "mc")
        & (df["dataset"] == "phenobench")
        & (df.get("eval_tiling") == "untiled")
        & df["faithful_mAP"].notna()
        & ~df.get("faithful_stale", False).fillna(False)
    ].copy()

    if platform is not None:
        sel = sel[sel["platform"] == platform]

    if sel.empty:
        return pd.DataFrame()

    sel = add_scheme(sel)
    rows = [
        {
            "Approach": f"{r['arch_label']} ({r['scheme']})",
            "mAP": round(100 * r["faithful_mAP"], 2),
            "mAP50": round(100 * r["faithful_mAP50"], 2),
            "mAP75": round(100 * r["faithful_mAP75"], 2),
            "Crop AP": _round_pct(r.get("faithful_crop_AP")),
            "Weed AP": _round_pct(r.get("faithful_weed_AP")),
            "Source": r["platform"],
        }
        for _, r in sel.sort_values(["arch", "scheme"]).iterrows()
    ]

    if include_baselines:
        rows = [*PHENOBENCH_BASELINES, *rows]

    return pd.DataFrame(rows)


def _round_pct(value, digits: int = 2):
    return None if value is None or pd.isna(value) else round(100 * value, digits)


#: Metric columns shared by the per-scheme thesis tables.
_SCHEME_METRICS = (
    ("AP", "mAP"),
    ("AP50", "mAP50"),
    ("AP75", "mAP75"),
    ("crop_AP", "Crop AP"),
    ("weed_AP", "Weed AP"),
)

#: Order the quantization schemes appear in, coarse to fine.
SCHEME_ORDER = (
    "fp32_ptq",
    "int8_ptq_per-tensor",
    "int8_ptq_per-channel",
    "int8_qat_per-tensor",
    "int8_qat_per-channel",
)


def scheme_comparison_table(
    df: pd.DataFrame,
    *,
    platform: str | None = None,
    eval_tiling: str | None = None,
    nms: str | None = DEFAULT_NMS,
    metrics: Iterable[tuple[str, str]] = _SCHEME_METRICS,
    drop_constant_keys: bool = True,
) -> pd.DataFrame:
    """
    PTQ vs. QAT per model variant: one row per (variant, scheme).

    This is the table the quantization chapter is built on, so it also carries
    ``dAP vs fp32`` -- the whole point is how far each INT8 export falls behind
    (or ahead of) its own float baseline, which raw AP columns make the reader
    compute by hand.

    Rows are pycocotools metrics (our internally consistent numbers), not the
    upstream ones; mixing the two in a single table would be meaningless.

    Scoped to the default post-processing. The scheme axis and the NMS axis are
    independent, so leaving both open would put two rows per scheme in the
    table with nothing to tell them apart -- and quantization cost is not the
    place to also price a post-processing swap. Both variants are deployable,
    so ``nms=REGULAR_NMS`` reads the same table for the per-class export;
    ``nms=None`` only if you are grouping on it yourself.
    """
    if df.empty:
        return pd.DataFrame()

    sel = select_nms(df.copy(), nms)
    if platform is not None:
        sel = sel[sel["platform"] == platform]
    if eval_tiling is not None:
        sel = sel[sel["eval_tiling"] == eval_tiling]
    if sel.empty:
        return pd.DataFrame()

    sel = add_scheme(sel)

    variant_keys = [
        "platform",
        "arch_label",
        "classes",
        "dataset",
        "size",
        "eval_tiling",
    ]
    variant_keys = [k for k in variant_keys if k in sel.columns]

    rows = []
    for _, group in sel.groupby(variant_keys, dropna=False):
        baseline = group[group["precision"] == "fp32"]
        baseline_ap = (
            float(baseline["AP"].iloc[0])
            if not baseline.empty and pd.notna(baseline["AP"].iloc[0])
            else None
        )

        ordered = group.sort_values(
            "scheme",
            key=lambda s: s.map(
                {name: i for i, name in enumerate(SCHEME_ORDER)}
            ).fillna(len(SCHEME_ORDER)),
        )

        for _, r in ordered.iterrows():
            row = {k: r[k] for k in variant_keys}
            row["Scheme"] = r["scheme"]
            for col, label in metrics:
                row[label] = (
                    None if col not in r or pd.isna(r[col]) else round(float(r[col]), 4)
                )
            row["dAP vs fp32"] = (
                None
                if baseline_ap in (None, 0) or pd.isna(r.get("AP"))
                else round(100 * (float(r["AP"]) - baseline_ap) / baseline_ap, 1)
            )
            rows.append(row)

    table = pd.DataFrame(rows)

    if drop_constant_keys and not table.empty:
        # Columns the caller pinned carry no information per row -- `platform`
        # and `eval_tiling` are arguments of this function, so they repeat one
        # value down the table and push the metrics off the page. `arch_label`
        # and `Scheme` stay unconditionally: they are what a row *is*.
        constant = [
            k
            for k in variant_keys
            if k not in ("arch_label",) and table[k].nunique(dropna=False) <= 1
        ]
        table = table.drop(columns=constant)

    return table


def platform_metrics_table(
    df: pd.DataFrame,
    platform: str,
    *,
    eval_tiling: str | None = None,
) -> pd.DataFrame:
    """
    Full COCO metrics for every run on one device (one table per device).
    """
    sel = df[df["platform"] == platform] if not df.empty else df
    if eval_tiling is not None and not sel.empty:
        sel = sel[sel["eval_tiling"] == eval_tiling]
    if sel.empty:
        return pd.DataFrame()

    sel = add_scheme(sel)
    cols = [
        ("arch_label", "Architecture"),
        ("classes", "Classes"),
        ("dataset", "Trained on"),
        ("size", "Input"),
        ("eval_tiling", "Eval input"),
        ("scheme", "Scheme"),
        ("nms", "NMS"),
        ("backend", "Backend"),
        ("AP", "mAP"),
        ("AP50", "mAP50"),
        ("AP75", "mAP75"),
        ("crop_AP", "Crop AP"),
        ("weed_AP", "Weed AP"),
        ("APS", "APS"),
        ("median_latency_ms", "Lat med (ms)"),
        ("p95_latency_ms", "Lat p95 (ms)"),
        ("fps", "FPS"),
    ]
    cols = [(c, n) for c, n in cols if c in sel.columns]
    out = sel[[c for c, _ in cols]].copy()
    out.columns = [n for _, n in cols]

    round_map = {
        n: 4
        for c, n in cols
        if c in ("AP", "AP50", "AP75", "crop_AP", "weed_AP", "APS")
    }
    round_map.update({"Lat med (ms)": 2, "Lat p95 (ms)": 2, "FPS": 1})
    return out.round(round_map).reset_index(drop=True)


def latency_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Median / p95 latency and throughput per platform, backend and scheme.

    Reported on the median rather than the mean: the sweeps pick up occasional
    scheduling outliers an order of magnitude above the typical sample, which
    the mean happily absorbs (see ``latency_outlier_ratio``).
    """
    if df.empty or "median_latency_ms" not in df.columns:
        return pd.DataFrame()

    sel = add_scheme(df[df["median_latency_ms"].notna()].copy())
    if sel.empty:
        return pd.DataFrame()

    # `nms` is a grouping key, not a nuisance column: the two variants are
    # separate runs of separate graphs, so folding them into one median mixes
    # two post-processing implementations into a number describing neither.
    keys = [
        k
        for k in ("platform", "backend", "arch_label", "size", "scheme", "nms")
        if k in sel.columns
    ]

    grouped = sel.groupby(keys, dropna=False).agg(
        runs=("median_latency_ms", "size"),
        lat_median_ms=("median_latency_ms", "median"),
        lat_p95_ms=("p95_latency_ms", "median"),
        fps=("fps", "median"),
    )

    return (
        grouped.round({"lat_median_ms": 2, "lat_p95_ms": 2, "fps": 1})
        .reset_index()
        .rename(
            columns={
                "platform": "Platform",
                "backend": "Backend",
                "arch_label": "Architecture",
                "size": "Input",
                "scheme": "Scheme",
                "nms": "NMS",
                "runs": "Runs",
                "lat_median_ms": "Lat med (ms)",
                "lat_p95_ms": "Lat p95 (ms)",
                "fps": "FPS",
            }
        )
    )


# =========================================================
# Sanity checks
# =========================================================

#: A quantized export scoring this much below its own FP32 baseline is treated
#: as a broken export rather than a quantization cost.
QUANT_COLLAPSE_FRACTION = 0.5

#: Largest single-class fast-vs-per-class NMS difference still counted as zero.
#: It should be *exactly* zero -- the two algorithms coincide at one class --
#: so this only absorbs pycocotools accumulation noise.
NMS_CONTROL_TOLERANCE = 1e-6


def sanity_checks(df: pd.DataFrame) -> pd.DataFrame:
    """
    Flag results that are more likely bugs than findings.

    Averages and bar charts hide broken runs very effectively, so every table in
    the notebook is preceded by this: it returns one row per detected issue with
    a severity, the run it concerns and what was observed. An empty frame means
    nothing suspicious was found -- not that everything is correct.

    Checks:

    * ``quant-collapse`` -- an INT8 export scoring far below its own FP32
      baseline.
    * ``backend-unknown`` / ``delegate-fallback`` -- a run whose effective
      execution backend is not recorded, or which asked for a delegate and
      silently ran on the CPU. Both invalidate latency comparisons.
    * ``fp32-on-delegate`` -- a float graph pushed through an INT8 accelerator.
    * ``latency-outliers`` -- max sample far above the median, i.e. a disturbed
      timing run (harmless for the median, fatal for the mean).
    * ``faithful-stale`` -- official metrics produced before the upstream label
      remap.
    * ``faithful-divergence`` -- official and pycocotools metrics disagreeing
      beyond what the two implementations explain.
    * ``nms-control-broken`` -- a single-class fast-vs-per-class NMS pair that
      is not identically zero, which is impossible if the pair is what it says
      it is.
    """
    if df.empty:
        return pd.DataFrame()

    sel = add_scheme(df)
    issues: list[dict] = []

    def add(severity, check, run, platform, detail):
        issues.append(
            {
                "severity": severity,
                "check": check,
                "platform": platform,
                "run": run,
                "detail": detail,
            }
        )

    # `nms` belongs here: without it an INT8 run is compared against the FP32
    # baseline of the *other* post-processing, so the substitution's own loss
    # is charged to quantization.
    variant_keys = [
        k
        for k in (
            "platform",
            "arch",
            "classes",
            "dataset",
            "size",
            "eval_tiling",
            "nms",
        )
        if k in sel.columns
    ]

    for _, group in sel.groupby(variant_keys, dropna=False):
        fp32 = group[group["precision"] == "fp32"]
        base = (
            float(fp32["AP"].iloc[0])
            if not fp32.empty and pd.notna(fp32["AP"].iloc[0])
            else None
        )

        if base:
            for _, r in group[group["precision"] == "int8"].iterrows():
                if pd.isna(r["AP"]):
                    continue
                if float(r["AP"]) < QUANT_COLLAPSE_FRACTION * base:
                    add(
                        "error",
                        "quant-collapse",
                        r["run"],
                        r["platform"],
                        f"AP {r['AP']:.3f} vs fp32 {base:.3f} "
                        f"({100 * (r['AP'] - base) / base:+.0f}%)",
                    )

    for _, r in sel.iterrows():
        backend = r.get("backend")

        if backend == "unknown":
            add(
                "warning",
                "backend-unknown",
                r["run"],
                r["platform"],
                f"requested {r.get('delegate')!r}; artifact predates effective-"
                "delegate recording, so CPU vs NPU is unknown",
            )
        elif backend == "cpu" and r.get("delegate"):
            add(
                "error",
                "delegate-fallback",
                r["run"],
                r["platform"],
                f"requested {r.get('delegate')!r} but ran on CPU",
            )

        if r.get("precision") == "fp32" and backend == "delegate":
            add(
                "warning",
                "fp32-on-delegate",
                r["run"],
                r["platform"],
                "float graph routed through an INT8 accelerator",
            )

        ratio = r.get("latency_outlier_ratio")
        if pd.notna(ratio) and ratio and float(ratio) > 5:
            add(
                "info",
                "latency-outliers",
                r["run"],
                r["platform"],
                f"max sample {float(ratio):.0f}x the median; use median/p95",
            )

        # A run without official metrics has NaN here, and NaN is truthy --
        # only an explicit True means the metrics exist and are stale.
        if r.get("faithful_stale") is True:
            add(
                "error",
                "faithful-stale",
                r["run"],
                r["platform"],
                "official metrics predate the crop/weed label remap; re-run "
                "`ave evaluate --faithful`",
            )
        elif (
            pd.notna(r.get("faithful_mAP"))
            and pd.notna(r.get("AP"))
            and not r.get("faithful_partial_classes")
            and abs(float(r["faithful_mAP"]) - float(r["AP"])) > 0.05
        ):
            add(
                "warning",
                "faithful-divergence",
                r["run"],
                r["platform"],
                f"official mAP {r['faithful_mAP']:.3f} vs pycocotools AP {r['AP']:.3f}",
            )

    # The post-processing substitution ships with its own null control: at one
    # class the fast and per-class passes are the same algorithm, so a
    # single-class pair that differs at all is a mispaired run, a stale
    # metrics.json or a mislabelled export -- never a result.
    for _, r in nms_substitution_table(sel).iterrows():
        if r.get("classes") != "sc" or pd.isna(r.get("dAP")):
            continue
        if abs(float(r["dAP"])) > NMS_CONTROL_TOLERANCE:
            add(
                "error",
                "nms-control-broken",
                f"{r.get('arch_label')} | sc | {r.get('dataset')} | "
                f"{r.get('scheme')} | eval:{r.get('eval_tiling')}",
                r.get("platform"),
                f"single-class fast-vs-per-class NMS differs by "
                f"{float(r['dAP']):+.4f} AP; at one class the two are the same "
                "algorithm, so the pair does not describe what it claims to",
            )

    if not issues:
        return pd.DataFrame(columns=["severity", "check", "platform", "run", "detail"])

    order = {"error": 0, "warning": 1, "info": 2}
    return (
        pd.DataFrame(issues)
        .sort_values(
            ["severity", "check", "run"],
            key=lambda s: s.map(order) if s.name == "severity" else s,
        )
        .reset_index(drop=True)
    )


def sanity_summary(issues: pd.DataFrame) -> pd.DataFrame:
    """Issue counts per check and severity."""
    if issues.empty:
        return pd.DataFrame()
    return (
        issues.groupby(["severity", "check"])
        .size()
        .rename("runs")
        .reset_index()
        .sort_values("runs", ascending=False)
        .reset_index(drop=True)
    )


# =========================================================
# Tables
# =========================================================


def quantization_delta_table(
    df: pd.DataFrame, *, nms: str | None = DEFAULT_NMS
) -> pd.DataFrame:
    """FP32 vs INT8 AP with relative change, one row per config (PTQ runs).

    INT8 is the per-tensor export (the default deployment granularity), so each
    config yields a single FP32/INT8 delta row.
    """
    df = select_nms(df, nms)
    df = df[df.get("quant") == "ptq"] if "quant" in df else df
    if "granularity" in df.columns:
        df = df[(df["precision"] == "fp32") | (df["granularity"] == "per-tensor")]
    if df.empty:
        return pd.DataFrame()

    # Group on the full run identity. Anything left out of the keys collapses
    # several runs into one group, and since the row is built from `iloc[0]`
    # those extra runs would be dropped rather than reported.
    keys = [
        k
        for k in (
            "platform",
            "arch_label",
            "class_label",
            "dataset",
            "size",
            "eval_tiling",
        )
        if k in df.columns and df[k].notna().any()
    ]
    rows = []
    for _, group in df.groupby(keys):
        fp = group[group["precision"] == "fp32"]
        q8 = group[group["precision"] == "int8"]
        if fp.empty or q8.empty:
            continue

        rec = {k: group.iloc[0][k] for k in keys}
        for col, name in (
            ("AP", "AP"),
            ("weed_AP", "Weed AP"),
            ("crop_AP", "Crop AP"),
            ("AP50", "AP50"),
        ):
            if col in group and pd.notna(fp[col].iloc[0]) and pd.notna(q8[col].iloc[0]):
                f, q = fp[col].iloc[0], q8[col].iloc[0]
                rec[f"{name} FP32"] = round(f, 4)
                rec[f"{name} INT8"] = round(q, 4)
                rec[f"{name} chg%"] = round(100 * (q - f) / f, 1) if f else None
        rows.append(rec)

    return pd.DataFrame(rows)


# =========================================================
# The reference configuration, and collapsing onto it
# =========================================================

#: Verdicts a (config, platform) cell can carry, worst first. Used both by
#: :func:`deployability_matrix` and, as a filter, by everything that averages.
DEPLOYABILITY_VERDICTS = ("failed", "unscoreable", "collapsed", "degraded", "ok", "-")

#: Below this share of the CPU reference's AP a delegated run is not a
#: quantization cost, it is a broken deployment. Same threshold the
#: ``quant-collapse`` check uses, applied across hardware instead of precision.
DEPLOYABILITY_COLLAPSE_FRACTION = 0.5

#: Milder disagreement with the CPU reference. A faithful delegate reproduces
#: it; a few tenths of a point is requantization noise, several points is the
#: accelerator changing the answer.
DEPLOYABILITY_DEGRADED_FRACTION = 0.9

#: The configuration every analysis that is *not* a tiling or resolution study
#: is reported at. Untiled multi-class, trained on full frames: it is the cell
#: the published PhenoBench numbers are comparable to, and the one the resource
#: sweep already collapses onto.
REFERENCE_CONFIG = {
    "classes": "mc",
    "dataset": "phenobench",
    "eval_tiling": "untiled",
    # Pinned: once the resolution ladder lands, the other three fields no
    # longer identify a single cell.
    "size": "320",
}

#: Dimensions folded into the reference for cost-style metrics, and the reason
#: each is expected to be free. These are mechanisms, not curve fits:
#:
#: * ``classes`` -- only the class predictor changes width
#:   (``num_anchors x (num_classes + 1)``, 2 vs 3 channels per anchor), worth
#:   +0.34 % / +0.03 % in file size;
#: * ``dataset`` -- the untiled- and tiled-*trained* exports are the **same
#:   graph** with different weight values, so there is no mechanism by which
#:   their compute could differ (measured: -0.02 % / -0.05 % in file size);
#: * ``eval_tiling`` -- this one is *not* free and is collapsed only because it
#:   is out of scope here. Feeding 1024 px versus 512 px sources changes
#:   ``cv2.resize`` preprocessing cost, which is real work; it is simply not
#:   model compute. It carries the largest residual of the three and must be
#:   quoted whenever the collapsed figure is.
COLLAPSIBLE_DIMENSIONS = ("classes", "dataset", "eval_tiling")

#: Dimensions that stay: architecture and export scheme are what the thesis is
#: about, and they move cost by an order of magnitude more.
KEPT_DIMENSIONS = ("arch", "scheme")

#: Display names for the fields a collapse audit is resolved by. Architecture
#: is always one of them: a cost figure without the detector it belongs to is
#: not interpretable, and the two detectors here differ by ~20 % in inference
#: cost.
_GROUP_LABELS = {"platform": "Platform", "arch": "Architecture"}

DIMENSION_LABELS = {
    "classes": "single- vs multi-class",
    "dataset": "trained untiled vs tiled",
    "eval_tiling": "evaluated untiled vs tiled",
    "arch": "ssd-mn2 vs fpnlite",
    "scheme": "export scheme",
}


def reference_config_slice(df: pd.DataFrame, **overrides) -> pd.DataFrame:
    """
    Narrow to the reference configuration (see :data:`REFERENCE_CONFIG`).

    Pass ``dimension=None`` to leave one axis open -- e.g.
    ``reference_config_slice(df, eval_tiling=None)`` for the tiling study.
    """
    wanted = {**REFERENCE_CONFIG, **overrides}
    sel = df
    for column, value in wanted.items():
        if value is None or column not in sel.columns:
            continue
        sel = sel[sel[column] == value]
    return sel


def failed_deployment_mask(
    df: pd.DataFrame,
    *,
    reference: str = CPU_REFERENCE_PLATFORM,
    fraction: float = DEPLOYABILITY_COLLAPSE_FRACTION,
) -> pd.Series:
    """
    Rows whose accuracy collapsed against the same file on the CPU reference.

    The companion to :func:`deployability_matrix`, as a mask rather than a
    table, for the places that need to *exclude* those runs rather than report
    them.
    """
    if df.empty:
        return pd.Series(dtype=bool)

    sel = add_scheme(df)
    keys = ["arch_label", "classes", "dataset", "size", "eval_tiling", "scheme", "nms"]
    keys = [k for k in keys if k in sel.columns]

    ref = sel[sel["platform"] == reference]
    ref = ref[~ref.duplicated(subset=keys, keep="first")].set_index(keys)["AP"]

    def _failed(row) -> bool:
        if pd.isna(row.get("AP")):
            return False
        try:
            baseline = float(ref.loc[tuple(row[k] for k in keys)])
        except (KeyError, TypeError, ValueError):
            return False
        return bool(baseline) and float(row["AP"]) < fraction * baseline

    return sel.apply(_failed, axis=1)


def drop_failed_deployments(df: pd.DataFrame, **kwargs) -> pd.DataFrame:
    """
    Remove runs whose output is real but describes a computation that did not
    happen.

    A ``collapsed`` run still has entirely valid latency numbers, so it will
    happily enter a latency mean and pull it toward whatever the accelerator
    does when it is not really working. Averaging is exactly where that
    matters, so the collapse guard and the collapsed tables filter first.
    """
    if df.empty:
        return df
    return df[~failed_deployment_mask(df, **kwargs).reindex(df.index, fill_value=False)]


def collapse_divergence(
    df: pd.DataFrame,
    *,
    metric: str = "median_latency_ms",
    collapsible: Iterable[str] = COLLAPSIBLE_DIMENSIONS,
    kept: Iterable[str] = KEPT_DIMENSIONS,
) -> pd.DataFrame:
    """
    How much each dimension moves ``metric`` with everything else held fixed.

    This is the measurement that licenses -- or refuses -- reporting one
    collapsed number instead of the full matrix, and it has to be re-run per
    metric because the answer is not the same for cost and for accuracy.

    For every dimension, runs are grouped on *all other* configuration fields
    and the relative spread ``(max - min) / median`` within each group is
    recorded. A collapsed dimension is only harmless if its spread is small
    against the dimensions deliberately kept: collapsing something that moves
    the metric as much as the effect under study destroys the effect.

    Returns:
        One row per dimension with its role and the median / p90 / max relative
        spread, in percent. Empty if nothing varies.
    """
    if df.empty or metric not in df.columns:
        return pd.DataFrame()

    sel = add_scheme(df).dropna(subset=[metric])
    if sel.empty:
        return pd.DataFrame()

    fields = [
        f
        for f in (
            "platform",
            "backend",
            "arch",
            "classes",
            "dataset",
            "size",
            "eval_tiling",
            "scheme",
            "nms",
        )
        if f in sel.columns
    ]

    rows = []
    for role, dimensions in (("collapsed", collapsible), ("kept", kept)):
        for dimension in dimensions:
            if dimension not in fields:
                continue
            others = [f for f in fields if f != dimension]
            stats = sel.groupby(others, dropna=False)[metric].agg(
                ["size", "min", "max", "median"]
            )
            stats = stats[(stats["size"] > 1) & (stats["median"] != 0)]
            if stats.empty:
                continue

            rel = 100 * (stats["max"] - stats["min"]) / stats["median"].abs()
            rows.append(
                {
                    "dimension": dimension,
                    "varies": DIMENSION_LABELS.get(dimension, dimension),
                    "role": role,
                    "groups": int(len(rel)),
                    "median_spread_pct": round(float(rel.median()), 1),
                    "p90_spread_pct": round(float(rel.quantile(0.9)), 1),
                    "max_spread_pct": round(float(rel.max()), 1),
                }
            )

    if not rows:
        return pd.DataFrame()

    return (
        pd.DataFrame(rows)
        .sort_values(["role", "median_spread_pct"], ascending=[True, False])
        .reset_index(drop=True)
    )


def collapse_holds(divergence: pd.DataFrame) -> bool:
    """
    Whether collapsing is defensible for the metric ``divergence`` describes.

    The test is comparative, not a fixed threshold: every collapsed dimension
    has to move the metric less than the *smallest* effect being kept.
    Otherwise the reported number averages over something larger than what it
    is meant to show.
    """
    if divergence.empty:
        return False

    collapsed = divergence[divergence["role"] == "collapsed"]["median_spread_pct"]
    kept = divergence[divergence["role"] == "kept"]["median_spread_pct"]
    if collapsed.empty or kept.empty:
        return False

    return bool(collapsed.max() < kept.min())


def collapse_bias_table(
    df: pd.DataFrame,
    *,
    metric: str = "median_latency_ms",
    dimensions: Iterable[str] = COLLAPSIBLE_DIMENSIONS,
    group: Iterable[str] = ("platform", "arch"),
) -> pd.DataFrame:
    """
    Which direction each collapsed dimension pulls the metric, and by how much.

    :func:`collapse_divergence` answers "is the residual small enough to
    ignore"; this answers "is it noise or is it a bias", which is the question
    that decides whether a collapsed mean can be quoted as an operating point.

    Both are needed because a dimension can be individually small and
    collectively directional. Here all three are: the reference configuration
    is evaluated on full frames, so it pays the largest ``cv2.resize`` cost of
    any cell folded into it, and every off-reference cell is therefore cheaper.
    Small residuals that all point the same way do not cancel -- they add.

    Returns:
        One row per (group, dimension) with the median and mean relative change
        of the off-reference value against the reference value, in percent.
        Negative means the off-reference runs are cheaper, i.e. the collapsed
        mean is optimistic.
    """
    if df.empty or metric not in df.columns:
        return pd.DataFrame()

    sel = add_scheme(df).dropna(subset=[metric])
    group = [g for g in group if g in sel.columns]
    if sel.empty or not group:
        return pd.DataFrame()

    fields = [
        f
        for f in (
            "platform",
            "backend",
            "arch",
            "classes",
            "dataset",
            "size",
            "eval_tiling",
            "scheme",
            "nms",
        )
        if f in sel.columns
    ]

    rows = []
    for dimension in dimensions:
        baseline = REFERENCE_CONFIG.get(dimension)
        if dimension not in fields or baseline is None:
            continue

        others = [f for f in fields if f != dimension]
        pivot = sel.pivot_table(index=others, columns=dimension, values=metric)
        if baseline not in pivot.columns:
            continue

        for other in (c for c in pivot.columns if c != baseline):
            change = (100 * (pivot[other] - pivot[baseline]) / pivot[baseline]).dropna()
            if change.empty:
                continue
            for group_value, values in change.groupby(level=list(group)):
                key = group_value if isinstance(group_value, tuple) else (group_value,)
                rows.append(
                    {
                        **{
                            _GROUP_LABELS.get(name, name): (
                                ARCH_LABELS.get(value, value)
                                if name == "arch"
                                else value
                            )
                            for name, value in zip(group, key, strict=False)
                        },
                        "dimension": dimension,
                        "off-reference": f"{other} (vs {baseline})",
                        "pairs": int(len(values)),
                        "median change %": round(float(values.median()), 1),
                        "mean change %": round(float(values.mean()), 1),
                    }
                )

    if not rows:
        return pd.DataFrame()

    sort_keys = [_GROUP_LABELS.get(name, name) for name in group]
    return (
        pd.DataFrame(rows).sort_values([*sort_keys, "dimension"]).reset_index(drop=True)
    )


def collapsed_latency_table(
    df: pd.DataFrame,
    *,
    keys: Iterable[str] = ("platform", "backend", "arch_label", "size", "scheme"),
    collapse: Iterable[str] = COLLAPSIBLE_DIMENSIONS,
) -> pd.DataFrame:
    """
    Runtime cost at the reference configuration, with the collapsed dimensions
    averaged in and the spread they contribute carried alongside.

    The point of the extra columns is that the collapse never becomes
    invisible. ``Ref cell (ms)`` is the reference configuration measured on its
    own, ``Lat med (ms)`` is the mean over everything collapsed into it,
    ``spread %`` is what was averaged away, and ``bias %`` is the *direction*
    of the disagreement.

    ``bias %`` is the column to read, because the residual here is not
    symmetric noise. The reference configuration is evaluated on full frames,
    i.e. it is the one fed 1024 px sources and therefore the one paying the
    largest ``cv2.resize`` cost, so folding the tiled-input runs in pulls the
    figure **down**. On the fastest board that is worth around a tenth of the
    total, which is not a rounding error. Quote the reference cell when the
    claim is about a deployed operating point, and the collapsed mean only when
    the claim is about the model.

    ``collapse`` selects which dimensions are folded in; anything left out is
    pinned to its reference value instead. Dropping ``eval_tiling`` from it
    removes most of the bias, at the cost of a smaller sample per row.
    """
    if df.empty or "median_latency_ms" not in df.columns:
        return pd.DataFrame()

    sel = add_scheme(drop_failed_deployments(df)).dropna(subset=["median_latency_ms"])

    # Pin every dimension that is not being collapsed to its reference value,
    # so a row is never a mean over an axis nobody asked to average.
    sel = reference_config_slice(sel, **dict.fromkeys(collapse, None))

    if sel.empty:
        return pd.DataFrame()

    keys = [k for k in keys if k in sel.columns]
    reference = reference_config_slice(sel)

    grouped = sel.groupby(keys, dropna=False)
    out = grouped.agg(
        runs=("median_latency_ms", "size"),
        lat_med=("median_latency_ms", "mean"),
        lat_p95=("p95_latency_ms", "mean"),
        _min=("median_latency_ms", "min"),
        _max=("median_latency_ms", "max"),
    )
    out["fps"] = 1000.0 / out["lat_med"]
    out["spread_pct"] = 100 * (out["_max"] - out["_min"]) / out["lat_med"]

    if not reference.empty:
        out["ref_ms"] = reference.groupby(keys, dropna=False)[
            "median_latency_ms"
        ].mean()
    else:
        out["ref_ms"] = np.nan

    out["bias_pct"] = 100 * (out["lat_med"] - out["ref_ms"]) / out["ref_ms"]

    return (
        out.drop(columns=["_min", "_max"])
        .round(
            {
                "lat_med": 2,
                "lat_p95": 2,
                "fps": 1,
                "spread_pct": 1,
                "ref_ms": 2,
                "bias_pct": 1,
            }
        )
        .reset_index()
        .rename(
            columns={
                "platform": "Platform",
                "backend": "Backend",
                "arch_label": "Architecture",
                "size": "Input",
                "scheme": "Scheme",
                "runs": "Runs",
                "lat_med": "Lat med (ms)",
                "lat_p95": "Lat p95 (ms)",
                "fps": "FPS",
                "ref_ms": "Ref cell (ms)",
                "spread_pct": "spread %",
                "bias_pct": "bias %",
            }
        )
    )


# =========================================================
# Deployability
# =========================================================


def _skipped_frame(skipped: Iterable[str]) -> pd.DataFrame:
    """Parse ``load_benchmark_results``' skip list into ``platform/run/reason``."""
    rows = []
    for entry in skipped or ():
        platform, _, rest = str(entry).partition("/")
        run, _, reason = rest.partition(" (")
        rows.append(
            {
                "platform": platform,
                "run": run,
                "reason": reason.rstrip(")") or "unknown",
            }
        )
    return pd.DataFrame(rows, columns=["platform", "run", "reason"])


def deployability_matrix(
    runs_df: pd.DataFrame,
    skipped: Iterable[str] = (),
    *,
    reference: str = CPU_REFERENCE_PLATFORM,
    eval_tiling: str | None = "untiled",
    nms: str | None = DEFAULT_NMS,
    platforms: Iterable[str] | None = None,
    drop_constant_keys: bool = True,
) -> pd.DataFrame:
    """
    Which exports actually survive each target, one row per (variant, scheme).

    This is the question that has to be answered *before* any accuracy or
    latency table is read, because the three ways a deployment fails here do
    not look alike and only one of them raises anything:

    ``unscoreable``
        The run produced output, and the output is not a detection. The Teflon
        delegate claims float convolutions and returns ``NaN`` boxes with
        out-of-range scores; nothing errors. Worse, ``pycocotools`` *rewards*
        this -- its match test is ``if iou < threshold: continue``, which a
        ``NaN`` fails, so every detection is accepted at every threshold and the
        run reports a high AP with ``AP == AP50``. The integrity gate is what
        turns that into a missing ``metrics.json``, so an absent cell here is a
        finding, not a gap in the sweep.

    ``collapsed`` / ``degraded``
        The run loads, runs, and produces plausible-looking boxes that score far
        below the *same file* on the same board's CPU. Measured for INT8
        per-channel weights under Teflon on the i.MX8M Plus.

    ``ok``
        Reproduces the CPU reference.

    Comparing each cell against the CPU reference rather than against a fixed
    AP is what separates "this accelerator broke it" from "this export was
    always weak": a per-tensor export that scores poorly on every backend is a
    quantization result and belongs in the scheme table, not here.

    Args:
        runs_df: Frame from :func:`load_benchmark_results`.
        skipped: Its companion skip list, which is where the unscoreable runs
            went. Omit it and those cells read as never-run.
        platforms: Columns to show. The reference is always used for scoring
            even when it is not shown, so narrowing to the boards does not turn
            every cell into an unjudged ``ok``. Note this holds for *this*
            argument only -- filtering ``runs_df`` by platform upstream does
            remove the reference from scoring, and warns.
    """
    if runs_df.empty:
        return pd.DataFrame()

    sel = add_scheme(select_nms(runs_df, nms))
    if eval_tiling is not None and "eval_tiling" in sel.columns:
        sel = sel[sel["eval_tiling"] == eval_tiling]

    keys = ["arch_label", "classes", "dataset", "size", "scheme"]
    if any(k not in sel.columns for k in keys):
        return pd.DataFrame()

    # A configuration with no baseline is scored `ok` (see `_verdict`), which
    # is the right default for a genuinely unjudgeable cell but catastrophic
    # when the *whole* reference tree is missing: every verdict then reads `ok`
    # and the table reports that nothing ever broke. Narrowing `runs_df` by
    # platform is the usual way to arrive here -- `platforms=` selects columns
    # without removing the reference, and is what callers want instead.
    if reference not in set(sel["platform"].unique()):
        warnings.warn(
            f"deployability_matrix: reference platform {reference!r} is absent "
            "from the frame, so no cell can be scored against it and every "
            "verdict falls back to 'ok'. Keep the reference tree in scope "
            "(it need not be a column -- use `platforms=` to choose those).",
            RuntimeWarning,
            stacklevel=2,
        )

    ref = sel[sel["platform"] == reference].set_index(keys)["AP"]
    ref = ref[~ref.index.duplicated(keep="first")]

    def _verdict(row) -> str:
        ap = row["AP"]
        if pd.isna(ap):
            return "failed"
        try:
            baseline = float(ref.loc[tuple(row[k] for k in keys)])
        except (KeyError, TypeError):
            return "ok"
        if not baseline or pd.isna(baseline):
            return "ok"
        share = float(ap) / baseline
        if share < DEPLOYABILITY_COLLAPSE_FRACTION:
            return "collapsed"
        if share < DEPLOYABILITY_DEGRADED_FRACTION:
            return "degraded"
        return "ok"

    cells = {}
    for _, r in sel.iterrows():
        cells[(*(r[k] for k in keys), r["platform"])] = _verdict(r)

    # The frame is the caller's statement of scope, but the skip list is not
    # filtered by it -- so a caller who narrowed to, say, multi-class would get
    # the single-class rows resurrected here as `unscoreable`. Recover only
    # configurations the frame still contains; a genuinely unscoreable one is
    # present on at least the CPU reference, which is what it is judged against.
    in_scope = set(sel[keys].itertuples(index=False, name=None))

    # Runs that were benchmarked but could not be scored. They are the whole
    # point of the table, and they exist only in the skip list.
    for _, s in _skipped_frame(skipped).iterrows():
        info = parse_run_name(s["run"])
        if info is None:
            continue
        if eval_tiling is not None and info.get("eval_tiling") != eval_tiling:
            continue
        if nms is not None and info.get("nms") not in (None, nms):
            continue
        # Built from `keys` rather than spelled out, so adding a field to the
        # configuration identity cannot silently desynchronise the two paths.
        info["scheme"] = scheme_name(
            info.get("precision"), info.get("quant"), info.get("granularity")
        )
        config = tuple(info.get(k) for k in keys)
        if config not in in_scope:
            continue
        cells.setdefault((*config, s["platform"]), "unscoreable")

    if platforms is not None:
        keep = set(platforms)
        cells = {k: v for k, v in cells.items() if k[-1] in keep}

    if not cells:
        return pd.DataFrame()

    long = pd.DataFrame(
        [
            {**dict(zip([*keys, "platform"], k, strict=True)), "verdict": v}
            for k, v in cells.items()
        ]
    )

    matrix = long.pivot_table(
        index=keys, columns="platform", values="verdict", aggfunc="first"
    ).fillna("-")

    order = {name: i for i, name in enumerate(SCHEME_ORDER)}
    table = (
        matrix.reset_index()
        .sort_values(
            ["arch_label", "classes", "dataset", "size", "scheme"],
            key=lambda s: s.map(order).fillna(len(order)) if s.name == "scheme" else s,
        )
        .reset_index(drop=True)
    )

    if drop_constant_keys:
        # A caller scoped to one class regime and one training set gets three
        # columns repeating one value beside the verdicts they came for.
        # `arch_label` and `scheme` stay: they are what a row is.
        constant = [
            k
            for k in ("classes", "dataset", "size")
            if k in table.columns and table[k].nunique(dropna=False) <= 1
        ]
        table = table.drop(columns=constant)

    return table


#: Config identity within one evaluation regime -- what makes a row of the
#: deployability matrix joinable against the runs frame.
_DEPLOY_KEYS = ("arch_label", "classes", "dataset", "size", "scheme")


def delegate_build_table(
    runs_df: pd.DataFrame,
    skipped: Iterable[str] = (),
    *,
    eval_tiling: str | None = "untiled",
    nms: str | None = DEFAULT_NMS,
    suffix: str = "_unpatched",
    reference: str = CPU_REFERENCE_PLATFORM,
    by_arch: bool = True,
    latency_scope: str = "ok-both",
    show_board_column=True,
) -> pd.DataFrame:
    """
    What rebuilding the delegate changed -- in verdicts, and in latency.

    A ``<board>{suffix}`` tree is the same board carrying an older delegate
    build, so the pair prices the operator-support work in the two currencies
    that matter: configurations that changed verdict, and what the ones that
    already worked now cost.

    Rows are resolved **per architecture** by default (``by_arch``). The plain
    head and FPNLite do not respond to a delegate rebuild alike -- measured
    here, the same rebuild moved per-tensor 320 by -30.6 % on ``ssd-mn2`` and
    -60.6 % on FPNLite -- so pooling them reports a median belonging to neither.

    ``latency_scope`` picks what the latency columns cover:

    ``"ok-both"`` (default)
        Only configurations both builds executed correctly. A collapsed run
        still produces perfectly real timings, and fast ones -- the work it
        skipped is exactly the work it got wrong -- so differencing against it
        reports a speedup for breaking the model.

    ``"paired"``
        Every configuration both trees *timed*, whatever the verdict. A
        collapsed run's wall clock is still a real measurement of dispatch and
        CPU<->NPU tensor movement, which is the cost a delegate rebuild is
        meant to shift, so this is the scope to read when the question is about
        sync overhead rather than about compute. Read it against ``Before`` /
        ``After``: the two medians describe runs of different correctness.

    Either way the pairing is per configuration and the width is reported as
    ``timed``. Empty latency columns mean the two trees share no timed
    configuration at all -- which, note, includes the case where the older tree
    never ran them, not only the case where the gate rejected them. Check
    ``timed`` against ``Configs`` before reading a difference.
    """
    if runs_df.empty:
        return pd.DataFrame()

    boards = sorted(runs_df["platform"].unique())
    pairs = [(b, f"{b}{suffix}") for b in boards if f"{b}{suffix}" in boards]
    if not pairs:
        return pd.DataFrame()

    matrix = deployability_matrix(
        runs_df,
        skipped,
        reference=reference,
        eval_tiling=eval_tiling,
        nms=nms,
        platforms=[p for pair in pairs for p in pair],
    )
    if matrix.empty:
        return pd.DataFrame()

    keys = [k for k in _DEPLOY_KEYS if k in matrix.columns]
    lat = add_scheme(select_nms(runs_df, nms))
    if eval_tiling is not None and "eval_tiling" in lat.columns:
        lat = lat[lat["eval_tiling"] == eval_tiling]

    def _timed(platform, subset):
        """Median latency per config, indexed by ``keys``, timed runs only."""
        if subset.empty:
            return pd.Series(dtype=float)
        rows = lat[lat["platform"] == platform].merge(subset[keys], on=keys)
        rows = rows.dropna(subset=["median_latency_ms"])
        if rows.empty:
            return pd.Series(dtype=float)
        return rows.groupby(keys)["median_latency_ms"].median()

    def _paired(older, board, subset):
        """
        The two medians over configs *both* trees actually timed.

        Returned as a triple with the pairing width, because a latency
        difference whose support is one configuration is a different claim
        from one over six, and the table has to be able to say which.
        """
        was_all, now_all = _timed(older, subset), _timed(board, subset)
        common = was_all.index.intersection(now_all.index)
        if len(common) == 0:
            return None, None, 0
        return (
            round(float(was_all[common].median()), 1),
            round(float(now_all[common].median()), 1),
            len(common),
        )

    group_keys = ["scheme"]
    if by_arch and "arch_label" in matrix.columns:
        group_keys.insert(0, "arch_label")

    rows = []
    for board, older in pairs:
        if board not in matrix.columns or older not in matrix.columns:
            continue

        for label, group in matrix.groupby(group_keys, dropna=False):
            label = label if isinstance(label, tuple) else (label,)
            before, after = group[older], group[board]
            both_ok = group[(before == "ok") & (after == "ok")]
            scope = both_ok if latency_scope == "ok-both" else group
            was, now, timed = _paired(older, board, scope)

            row = {"Board": board} if show_board_column else {}
            row.update(dict(zip(group_keys, label, strict=True)))
            row.update(
                {
                    "Configs": len(group),
                    "Before": before.value_counts().idxmax()
                    if before.notna().any()
                    else "-",
                    "After": after.value_counts().idxmax()
                    if after.notna().any()
                    else "-",
                    "Fixed": int(((before != "ok") & (after == "ok")).sum()),
                    "Broken": int(((before == "ok") & (after != "ok")).sum()),
                    "ok both": len(both_ok),
                    "timed": timed,
                    "ms before": was,
                    "ms after": now,
                    "d ms %": (
                        None
                        if not was or now is None
                        else round(100.0 * (now - was) / was, 1)
                    ),
                }
            )
            rows.append(row)

    out = pd.DataFrame(rows)
    return out.rename(columns={"arch_label": "Arch", "scheme": "Scheme"})


def deployability_summary(matrix: pd.DataFrame) -> pd.DataFrame:
    """Verdict counts per platform, worst first."""
    if matrix.empty:
        return pd.DataFrame()

    keys = {"arch_label", "classes", "dataset", "size", "scheme"}
    platforms = [c for c in matrix.columns if c not in keys]

    counts = (
        matrix[platforms].apply(lambda col: col.value_counts()).fillna(0).astype(int).T
    )
    ordered = [v for v in DEPLOYABILITY_VERDICTS if v in counts.columns]
    return counts[ordered].reset_index().rename(columns={"index": "platform"})


# =========================================================
# Post-processing substitution (fast vs per-class NMS)
# =========================================================

#: Keys that identify one deployable across the two NMS variants. Everything
#: except ``nms`` itself, so the two rows differ in exactly one thing.
NMS_PAIR_KEYS = (
    "platform",
    "eval_tiling",
    "arch_label",
    "classes",
    "dataset",
    "size",
    "scheme",
)


def nms_substitution_table(
    df: pd.DataFrame,
    *,
    deployed: str = DEFAULT_NMS,
    control: str = REGULAR_NMS,
) -> pd.DataFrame:
    """
    Price the post-processing substitution, one row per matched pair.

    The exported graph does not run the NMS the checkpoint runs.
    ``TFLite_Detection_PostProcess`` with ``use_regular_nms=False`` makes a
    single class-agnostic pass over each anchor's argmax class, while the
    checkpoint suppresses per class. SSD shares one box regression across
    classes, so an anchor's crop and weed hypotheses are the *same box* and only
    the higher-scoring one survives the class-agnostic pass.

    That mechanism predicts the shape of the result before it is measured, and
    the prediction is testable in this very table:

    * **single-class runs must show exactly zero.** With one class the two
      algorithms are the same algorithm. Any non-zero ``dAP`` on an ``sc`` row
      is a broken pairing, not a finding -- :func:`sanity_checks` flags it.
    * **multi-class loss must land on the suppressed class**, i.e. on
      ``crop``, with ``weed`` left alone.

    Because the pair shares one checkpoint, one graph and one calibration set,
    the delta is the substitution and nothing else, at whichever rung the row
    happens to sit -- float, INT8, or INT8 on a delegate.

    Returns:
        One row per configuration with both variants' metrics and their
        differences, or an empty frame if the sweep has no pairs.
    """
    if df.empty or "nms" not in df.columns:
        return pd.DataFrame()

    sel = add_scheme(df[df["nms"].notna()].copy())
    keys = [k for k in NMS_PAIR_KEYS if k in sel.columns]
    if not keys:
        return pd.DataFrame()

    def _side(variant: str) -> pd.DataFrame:
        side = sel[sel["nms"] == variant]
        return side[~side.duplicated(subset=keys, keep="first")].set_index(keys)

    fast, reg = _side(deployed), _side(control)
    common = fast.index.intersection(reg.index)
    if common.empty:
        return pd.DataFrame()

    fast, reg = fast.loc[common], reg.loc[common]

    out = pd.DataFrame(index=common)
    for col, name in (
        ("AP", "AP"),
        ("crop_AP", "Crop AP"),
        ("weed_AP", "Weed AP"),
        ("AR100", "AR100"),
    ):
        if col not in fast.columns:
            continue
        out[f"{name} ({control})"] = reg[col]
        out[f"{name} ({deployed})"] = fast[col]
        out[f"d{name}"] = fast[col] - reg[col]

    if "median_latency_ms" in fast.columns:
        out["dLatency (ms)"] = fast["median_latency_ms"] - reg["median_latency_ms"]
        out["dLatency (%)"] = 100 * out["dLatency (ms)"] / reg["median_latency_ms"]

    # `+ 0.0` collapses the negative zero left by rounding a tiny negative. The
    # single-class rows are the null control and have to read as a clean 0.
    return (out.round(4) + 0.0).reset_index()


def nms_pair_coverage(df: pd.DataFrame) -> pd.DataFrame:
    """
    Where both post-processing variants exist, per platform.

    Both are deployable and both were swept on the boards, but the CPU-only and
    unpatched control trees carry the default export alone. That is deliberate,
    so it does not belong in :func:`build_coverage` as a gap -- but it does
    need saying somewhere, otherwise a reader wonders why the substitution
    analysis covers three platforms and the accuracy tables cover nine.
    """
    if df.empty or "nms" not in df.columns:
        return pd.DataFrame()

    sel = add_scheme(df[df["nms"].notna()])
    if sel.empty:
        return pd.DataFrame()

    keys = [
        k
        for k in ("arch_label", "classes", "dataset", "size", "eval_tiling", "scheme")
        if k in sel.columns
    ]

    rows = []
    for platform, group in sel.groupby("platform"):
        variants = group.groupby(keys, dropna=False)["nms"].nunique()
        rows.append(
            {
                "Platform": platform,
                "configurations": int(len(variants)),
                "both variants": int((variants > 1).sum()),
                "default only": int((variants == 1).sum()),
            }
        )

    return pd.DataFrame(rows)


def nms_substitution_summary(
    df: pd.DataFrame, *, drop_constant_keys: bool = True, **kwargs
) -> pd.DataFrame:
    """
    The substitution's accuracy cost aggregated over the cells it depends on.

    Grouped by architecture, class regime and evaluation input. The last two
    are the axes the mechanism predicts -- single-class is the null control and
    must be identically zero, and the tiled/full-frame split changes how many
    crop and weed hypotheses compete for the same anchor. **Architecture is a
    grouping key rather than something to average over**: the two detectors
    have different anchor layouts and different numbers of candidates entering
    suppression, so a mean over both describes neither.

    ``drop_constant_keys`` removes grouping columns the caller has already
    scoped to a single value. They carry no information in a summary and only
    widen the table -- a scoped frame otherwise prints a ``classes`` column
    reading ``mc`` on every row.
    """
    pairs = nms_substitution_table(df, **kwargs)
    if pairs.empty:
        return pd.DataFrame()

    keys = [
        k
        for k in ("arch_label", "classes", "eval_tiling", "dataset", "size")
        if k in pairs.columns
    ]

    if drop_constant_keys:
        # `arch_label` stays regardless: it is the axis the docstring above
        # refuses to average over, so its absence would be misread as a mean.
        varying = [
            k for k in keys if k == "arch_label" or pairs[k].nunique(dropna=False) > 1
        ]
        keys = varying or keys
    metrics = [c for c in ("dAP", "dCrop AP", "dWeed AP", "dAR100") if c in pairs]

    summary = pairs.groupby(keys, dropna=False)[metrics].agg(["mean", "min", "max"])
    summary.insert(0, ("pairs", ""), pairs.groupby(keys, dropna=False).size())
    return (summary.round(4) + 0.0).reset_index()


def nms_latency_tradeoff_table(
    df: pd.DataFrame, *, precision: str | None = "int8", **kwargs
) -> pd.DataFrame:
    """
    What the substitution *buys*, estimated against its own null control.

    The naive answer -- median latency of the fast runs minus the regular ones
    -- is not trustworthy here, because the two variants were benchmarked as
    separate runs and any drift between them lands in that difference. The
    single-class pairs bound that drift: with one class the two graphs run the
    *same* algorithm, so their latency difference is measurement error by
    construction and nothing else.

    So the estimate is a difference in differences::

        saving = mean(dLatency | mc) - mean(dLatency | sc)
                 \\_______________/   \\_______________/
                   algorithm + drift        drift

    ``sc drift`` is reported alongside it: when it is large the platform's
    paired timings are simply not reliable at this resolution, and the row
    should be read as an upper bound rather than a measurement.

    The estimate carries its own uncertainty, because a difference of two noisy
    means is itself noisy. ``SE`` combines the standard errors of both arms,
    ``sigma`` is the estimate in units of that, and ``95% CI`` spans it. A row
    whose interval contains zero has not resolved a saving from drift, however
    large its point estimate looks -- which is the difference between a small
    effect and no measured effect, and the two are not interchangeable in a
    trade-off argument.

    Restricted to ``precision`` -- INT8 by default, the only precision that
    ships. The quantity being estimated is a roughly fixed millisecond cost, so
    pooling a 340 ms fp32 regime with a 33 ms INT8 one adds variance without
    adding signal: measured here, the fp32 pairs carry some forty times the
    absolute timing noise (single-class drift sd 4.7 ms against 0.02-0.12 ms),
    and four such pairs in twenty set the width of every interval. Pass
    ``precision=None`` to pool them again.

    Resolved **per architecture as well as per platform**. A latency figure
    without the detector it belongs to is not interpretable -- the two
    architectures differ by roughly 20 % in inference cost here -- and the
    quantity being estimated is a fixed post-processing cost, so expressing it
    as a share of the wrong total would misstate it.
    """
    if precision is not None and "precision" in df.columns:
        df = df[df["precision"] == precision]

    pairs = nms_substitution_table(df, **kwargs)
    if pairs.empty or "dLatency (ms)" not in pairs.columns:
        return pd.DataFrame()

    pairs = pairs.dropna(subset=["dLatency (ms)"])
    if pairs.empty:
        return pd.DataFrame()

    keys = [k for k in ("platform", "arch_label") if k in pairs.columns]

    rows = []
    for key, group in pairs.groupby(keys):
        key = key if isinstance(key, tuple) else (key,)
        mc = group[group["classes"] == "mc"]["dLatency (ms)"]
        sc = group[group["classes"] == "sc"]["dLatency (ms)"]
        if mc.empty or sc.empty:
            continue

        drift = float(sc.mean())
        saving = float(mc.mean()) - drift

        # Standard error of a difference of independent means. Needs two
        # samples per arm; with fewer, the interval is undefined rather than
        # zero-width, so it is left empty instead of implying certainty.
        if len(mc) > 1 and len(sc) > 1:
            standard_error = float(
                np.sqrt(mc.var(ddof=1) / len(mc) + sc.var(ddof=1) / len(sc))
            )
        else:
            standard_error = float("nan")

        half_width = 1.96 * standard_error

        rows.append(
            {
                **dict(zip(("Platform", "Architecture"), key, strict=False)),
                "mc pairs": int(len(mc)),
                "sc pairs": int(len(sc)),
                "dLatency mc (ms)": round(float(mc.mean()), 3),
                "sc drift (ms)": round(drift, 3),
                "NMS saving (ms)": round(saving, 3),
                "SE (ms)": None
                if np.isnan(standard_error)
                else round(standard_error, 3),
                "sigma": (
                    None
                    if np.isnan(standard_error) or not standard_error
                    else round(abs(saving) / standard_error, 1)
                ),
                "95% CI (ms)": (
                    None
                    if np.isnan(half_width)
                    else f"[{saving - half_width:+.2f}, {saving + half_width:+.2f}]"
                ),
                "resolved": (
                    None if np.isnan(half_width) else bool(abs(saving) > half_width)
                ),
                "sc |drift| worst (ms)": round(float(sc.abs().max()), 3),
            }
        )

    return pd.DataFrame(rows)


def plot_nms_substitution(
    df: pd.DataFrame,
    *,
    eval_tiling: str | None = "untiled",
    platform: str | None = None,
):
    """
    Per-class AP change from the post-processing substitution.

    One panel per architecture, class regime along the y axis, export scheme as
    the colour. Architecture is a **panel** rather than a colour because the two
    detectors have different anchor layouts and different numbers of candidates
    entering suppression: putting them in one group invites a comparison
    between bars that do not describe the same competition.

    Crop and weed are drawn separately on purpose -- the overall-AP bar averages
    a class that loses heavily with one that barely moves, which is exactly the
    structure the figure exists to show.
    """
    pairs = nms_substitution_table(df)
    if pairs.empty or "dCrop AP" not in pairs.columns:
        return None

    if eval_tiling is not None and "eval_tiling" in pairs.columns:
        pairs = pairs[pairs["eval_tiling"] == eval_tiling]
    if platform is not None:
        pairs = pairs[pairs["platform"] == platform]
    if pairs.empty:
        return None

    series = [("dAP", "overall"), ("dCrop AP", "crop"), ("dWeed AP", "weed")]
    categories = [label for _, label in series]

    archs = sorted(pairs["arch_label"].unique())
    schemes = [s for s in SCHEME_ORDER if s in set(pairs["scheme"])]
    if not archs or not schemes:
        return None

    labels = [scheme_label(s) for s in schemes]
    colors = [SCHEME_COLORS[s] for s in schemes]

    height = max(1.9, 0.22 * len(categories) * len(schemes) + 0.9)
    fig, axes = plt.subplots(
        len(archs), 1, figsize=(9, height * len(archs)), sharex=True
    )
    axes = np.atleast_1d(axes)

    for index, (ax, arch) in enumerate(zip(axes, archs, strict=False)):
        rows = pairs[pairs["arch_label"] == arch]
        # The panels share one x axis, so only the bottom one carries its label.
        last = index == len(archs) - 1
        values = {
            label: [
                rows[rows["scheme"] == scheme][col].mean() if not rows.empty else None
                for col, _ in series
            ]
            for scheme, label in zip(schemes, labels, strict=False)
        }
        _grouped_bars(
            ax,
            categories,
            labels,
            values,
            colors,
            ylabel="AP change, fast NMS - per-class NMS (%)" if last else "",
            title=_short(pd.Series([arch])).iloc[0],
            fmt="{:+.2f}",
            percent=True,
            horizontal=True,
        )
        ax.axvline(0.0, color="#444444", linewidth=0.8, zorder=2)

    fig.suptitle(
        "Cost of the exported post-processing substitution"
        + (f" ({eval_tiling} input)" if eval_tiling else "")
        + (f" - {platform}" if platform else "")
    )
    fig.tight_layout()
    # Every bar is negative, so the plotting area is full from the zero line
    # leftwards and any in-axes corner would sit on data.
    _legend_outside(fig, axes[0], title="export scheme")
    return fig


def resolution_ladder_table(
    df: pd.DataFrame,
    *,
    platform: str | None = CPU_REFERENCE_PLATFORM,
    eval_tiling: str | None = "untiled",
    nms: str | None = DEFAULT_NMS,
    classes: str | None = "mc",
    dataset: str | None = "phenobench",
    latency_platforms: Iterable[str] | None = None,
) -> pd.DataFrame:
    """
    Accuracy and latency against input resolution, one row per (arch, size,
    scheme).

    The rung a model sits on is defined entirely by which finetune output was
    attached as a Kaggle input -- the quantization notebooks rebuild
    ``FineTuneConfig`` from the attached manifest, so ``image_size`` is
    inherited rather than set. That makes a mislabelled rung a plausible
    failure, so the table reports the parsed ``size`` from the artifact name
    next to the metrics instead of assuming the grouping is right.

    Scoped to one class regime, training set, platform and input regime,
    because resolution is the axis under study and everything else has to be
    held fixed for the comparison to mean anything.

    ``latency_platforms`` adds one median-latency column per target, joined on
    the same rows. Accuracy is measured once on the CPU reference, but the
    question a higher rung has to survive is what it costs **on the device**,
    and a rung can only be disqualified by a number from the board it would
    ship on. A target with no run at some resolution leaves the cell empty
    rather than falling back to the reference host, whose milliseconds answer a
    different question.

    Empty until the 512/1024 rungs are benchmarked; the shape is fixed now so
    the section lights up on its own when they land.
    """
    if df.empty or "size" not in df.columns:
        return pd.DataFrame()

    sel = add_scheme(select_nms(df, nms))
    for column, value in (
        ("platform", platform),
        ("eval_tiling", eval_tiling),
        ("classes", classes),
        ("dataset", dataset),
    ):
        if value is not None and column in sel.columns:
            sel = sel[sel[column] == value]

    sel = sel[sel["size"].notna()]
    if sel.empty or sel["size"].nunique() < 2:
        return pd.DataFrame()

    cols = [
        ("arch_label", "Architecture"),
        ("size", "Input"),
        ("scheme", "Scheme"),
        ("AP", "mAP"),
        ("AP50", "mAP50"),
        ("APS", "APS"),
        ("crop_AP", "Crop AP"),
        ("weed_AP", "Weed AP"),
        ("median_latency_ms", "x86 (ms)"),
    ]
    cols = [(c, n) for c, n in cols if c in sel.columns]

    out = sel[[c for c, _ in cols]].copy()
    out.columns = [n for _, n in cols]

    if latency_platforms:
        # Latency comes from the *unscoped* frame: `sel` is pinned to the
        # accuracy platform, so the boards were filtered out several lines ago.
        wide = add_scheme(select_nms(df, nms))
        for column, value in (
            ("eval_tiling", eval_tiling),
            ("classes", classes),
            ("dataset", dataset),
        ):
            if value is not None and column in wide.columns:
                wide = wide[wide[column] == value]

        for target in latency_platforms:
            median = (
                wide[wide["platform"] == target]
                .groupby(["arch_label", "size", "scheme"])["median_latency_ms"]
                .median()
            )
            latency = [
                median.get((a, s, c))
                for a, s, c in zip(
                    sel["arch_label"], sel["size"], sel["scheme"], strict=False
                )
            ]
            label = platform_label(target)
            out[f"{label} (ms)"] = [
                None if v is None or pd.isna(v) else round(float(v), 1) for v in latency
            ]
            # Throughput on the target, not on the reference host: the whole
            # point of the column is what the board sustains.
            out[f"{label} FPS"] = [
                None if v is None or pd.isna(v) or not v else round(1000.0 / v, 1)
                for v in latency
            ]

    out["Input"] = pd.to_numeric(out["Input"], errors="coerce")

    order = {name: i for i, name in enumerate(SCHEME_ORDER)}
    return (
        out.sort_values(
            ["Architecture", "Input", "Scheme"],
            key=lambda s: s.map(order).fillna(len(order)) if s.name == "Scheme" else s,
        )
        .round(
            {
                "mAP": 4,
                "mAP50": 4,
                "APS": 4,
                "Crop AP": 4,
                "Weed AP": 4,
                "x86 (ms)": 2,
            }
        )
        .reset_index(drop=True)
    )


#: Results tree holding the pre-conversion SavedModel reference. The *floored*
#: one: its NMS score threshold matches the TFLite graphs', which is what keeps
#: the first rung like-for-like. (`tf-savedmodel-nms0` is the floor-free control
#: and is worth +0.002 AP on average.)
REFERENCE_PLATFORM = "tf-savedmodel"


def degradation_ladder_table(
    df: pd.DataFrame,
    *,
    npu_platform: str = "frdm-imx93",
    eval_tiling: str | None = "untiled",
    quant: str = "ptq",
    reference: str = REFERENCE_PLATFORM,
    cpu_platform: str = CPU_REFERENCE_PLATFORM,
    metric: str = "AP",
) -> pd.DataFrame:
    """
    Decompose the deployment chain into its rungs, one loss per transformation.

    ::

        SavedModel
          --conversion-->    TFLite fp32 (per-class NMS)
          --nms-swap-->      TFLite fp32 (fast NMS, deployed)
          --quantization-->  int8 CPU
          --delegation-->    int8 NPU

    Each delta isolates one transformation by holding the others fixed. Two
    rungs carry the interesting content:

    * **The SavedModel rung.** Without it the conversion loss is invisible and
      gets folded into "quantization"; for at least one config it is *larger*
      than that config's quantization loss.
    * **The NMS rung.** "Conversion" used to be three changes at once, and the
      post-processing substitution dominated it. Splitting it out is what turns
      a format cost that could not be explained into an algorithm cost that
      can: feed identical pixels through both graphs with the same NMS and the
      float formats agree. What is left in ``conversion`` is the resampling
      difference (the SavedModel resizes in-graph with ``fixed_shape_resizer``,
      the TFLite path externally with ``cv2``) and the missing clip window --
      small, sign-random, and not a precision loss.

      ``nms-swap`` is the cost of the **default** export choice, not an
      unavoidable one: the per-class variant is equally deployable and was
      benchmarked on both boards. :func:`nms_substitution_table` breaks the
      cost down per class and :func:`nms_latency_tradeoff_table` prices what it
      buys; it is exactly zero for single-class models by construction.

    Rows are dropped only when the deployed chain itself is missing; an absent
    reference or control leaves that column empty rather than removing the
    config, so an incomplete sweep stays visible instead of silently narrowing
    the table.
    """
    if df.empty or metric not in df.columns:
        return pd.DataFrame()

    sel = add_scheme(df.copy())

    if eval_tiling is not None and "eval_tiling" in sel.columns:
        sel = sel[sel["eval_tiling"] == eval_tiling]

    keys = [k for k in ("arch_label", "classes", "dataset", "size") if k in sel.columns]
    if not keys:
        return pd.DataFrame()

    def _rung(
        platform, scheme=None, quantization=None, nms=DEFAULT_NMS, *, strict=False
    ):
        """
        One rung's metric per config.

        ``strict`` demands an explicit NMS token instead of accepting rows that
        carry none. The control rung needs it: a result tree that predates the
        paired export has untagged float runs, and letting one stand in for the
        control would report a measured ``nms-swap`` of exactly zero for a
        comparison that was never made.
        """
        rows = sel[sel["platform"] == platform]
        if scheme is not None:
            rows = rows[rows["scheme"] == scheme]
        if quantization is not None:
            rows = rows[rows["quant"] == quantization]
        if strict:
            if "nms" not in rows.columns:
                return pd.Series(dtype="float64")
            rows = rows[rows["nms"] == nms]
        else:
            rows = select_nms(rows, nms)
        rows = rows.dropna(subset=[metric])
        if rows.empty:
            return pd.Series(dtype="float64")
        # One row per config by construction now that the NMS variant is
        # pinned; `first` rather than `mean` so a leftover duplicate shows up
        # as a wrong number in one cell instead of a plausible average.
        return rows.groupby(keys)[metric].first()

    fp32_scheme = scheme_name("fp32", "ptq")
    int8_scheme = scheme_name("int8", quant, "per-tensor")

    control_label = f"TFLite fp32 ({REGULAR_NMS})"
    deployed_label = f"TFLite fp32 ({DEFAULT_NMS})"
    npu_label = f"int8 NPU ({npu_platform})"

    table = pd.DataFrame(
        {
            "SavedModel": _rung(reference, quantization=quant, nms=None),
            control_label: _rung(
                cpu_platform, scheme=fp32_scheme, nms=REGULAR_NMS, strict=True
            ),
            deployed_label: _rung(cpu_platform, scheme=fp32_scheme),
            "int8 CPU": _rung(cpu_platform, scheme=int8_scheme),
            npu_label: _rung(npu_platform, scheme=int8_scheme),
        }
    )

    # A config needs at least the deployed chain to be worth a row.
    table = table.dropna(subset=[deployed_label, "int8 CPU"], how="any")
    if table.empty:
        return pd.DataFrame()

    table["conversion"] = table[control_label] - table["SavedModel"]
    table["nms-swap"] = table[deployed_label] - table[control_label]
    table["quantization"] = table["int8 CPU"] - table[deployed_label]
    table["delegation"] = table[npu_label] - table["int8 CPU"]

    # `+ 0.0` collapses the negative zero a rounded -1e-7 leaves behind. It is
    # cosmetic in a notebook and not in a thesis table, where "-0" next to a
    # column of real losses reads as a measured effect.
    return table.round(4) + 0.0


# =========================================================
# The thesis story, in six steps
# =========================================================
#
# Chapter 6 is one argument, and these functions are its steps in order. Each
# takes the whole sweep and narrows it itself, so a caller cannot accidentally
# feed one step the scope of another:
#
#   1  baseline_table            float accuracy of the trained detectors
#   2  preparation_ladder_table  what conversion and PTQ cost
#      nms_substitution_*        ... and how much of that is the NMS swap
#   3  qat_reclaim_table         what QAT gets back
#   4  deployability_matrix      which exports the accelerators execute correctly
#   5  device_latency_table      what the survivors cost on CPU vs NPU
#   6  story_ablation_table      how single-class and tiling move each step above
#
# Steps 1-5 are reported at REFERENCE_CONFIG; step 6 is the only one that
# leaves it, one axis at a time.

#: Short architecture names for table columns, where the full label does not fit.
ARCH_SHORT = {
    "SSD MobileNetV2": "MNv2",
    "SSD MobileNetV2 FPNLite": "FPNLite",
}


def _arch_short(label: str) -> str:
    return ARCH_SHORT.get(label, label)


def baseline_table(
    df: pd.DataFrame,
    *,
    platform: str = REFERENCE_PLATFORM,
    quant: str = "ptq",
    include_baselines: bool = True,
    sizes: str | Iterable[str] | None = None,
) -> pd.DataFrame:
    """
    **Step 1** -- float accuracy of the fine-tuned detectors.

    The float SavedModel at the reference configuration -- multi-class, trained
    and evaluated full-frame -- across every input resolution present.  This is
    the number every later step is a cost against, so it is measured *before*
    TensorFlow Lite is involved at all: the checkpoint's own graph and its own
    post-processing.

    ``sizes`` selects the resolutions; ``None`` takes whatever the frame holds,
    so the ladder is as ragged as the artifacts are -- currently 320/512/1024
    for the plain detector and 320/512 for FPNLite. A missing cell means that
    rung is absent from ``artifacts/tf``, not that it was measured and lost.

    ``quant`` picks which float export stands for the checkpoint. The
    SavedModel tree carries one per quantization path (``ptq`` and ``qat``),
    which are separate fine-tunes and differ slightly; ``ptq`` is the float
    graph before the paths diverge.

    Reported with the **official PhenoBench evaluator** (``faithful_*``), not
    with pycocotools. This is the one table in the notebook that puts our
    numbers beside someone else's, so it has to use the metric stack those
    numbers were produced with; the two families are not interchangeable and
    mixing them in one column would be meaningless. Every *internal*
    comparison downstream stays on pycocotools.

    The published PhenoBench baselines are appended for orientation. They are
    not a like-for-like ranking: upstream reports the withheld test split at
    full resolution, this work the internal test split derived from the
    validation partition. Say so wherever the table is used.

    Runs whose official metrics predate the crop/weed label remap are dropped
    rather than shown, matching :func:`upstream_comparison_table` -- a stale
    ``faithful_*`` value is wrong in a way that looks plausible.
    """
    if df.empty or "faithful_mAP" not in df.columns:
        return pd.DataFrame()

    # `size` is the axis under study here, so it is relaxed rather than pinned;
    # every other axis of the reference configuration still holds.
    sel = reference_config_slice(df[df["platform"] == platform], size=None)
    if sizes is not None:
        wanted = {sizes} if isinstance(sizes, str) else set(sizes)
        sel = sel[sel["size"].isin(wanted)]
    sel = sel[sel.get("quant", quant) == quant]
    sel = sel[sel["arch"].isin(PRIMARY_ARCHS)] if "arch" in sel.columns else sel
    sel = sel[~sel.get("faithful_stale", False).fillna(False)]
    if sel.empty:
        return pd.DataFrame()

    rows = []
    if include_baselines:
        for base in PHENOBENCH_BASELINES:
            rows.append(
                {
                    "Detector": base["Approach"],
                    "Input": "full",
                    "AP": base["mAP"],
                    "AP50": base["mAP50"],
                    "AP75": base["mAP75"],
                    "Crop AP": base["Crop AP"],
                    "Weed AP": base["Weed AP"],
                    "Source": "published (test split)",
                }
            )

    # Detector, then resolution ascending -- `size` is a string token, so sort
    # it numerically or 1024 lands between 320 and 512.
    order = sel.assign(_px=pd.to_numeric(sel["size"], errors="coerce")).sort_values(
        ["arch_label", "_px"]
    )

    for _, r in order.iterrows():
        size = r.get("size") or "?"
        rows.append(
            {
                "Detector": r["arch_label"],
                "Input": "full" if int(size) == 1024 else f"{size}x{size}",
                "AP": _round_pct(r.get("faithful_mAP")),
                "AP50": _round_pct(r.get("faithful_mAP50")),
                "AP75": _round_pct(r.get("faithful_mAP75")),
                "Crop AP": _round_pct(r.get("faithful_crop_AP")),
                "Weed AP": _round_pct(r.get("faithful_weed_AP")),
                "Source": "this work (internal test split)",
            }
        )

    return pd.DataFrame(rows)


def _preparation_stages(nms: str = DEFAULT_NMS) -> tuple[tuple, ...]:
    """
    The rungs of model preparation, in the order they are applied.

    Each entry is ``(label, scheme, nms, delta_base)``; ``delta_base`` names the
    row the delta is quoted against -- ``"prev"`` for the float rung, a
    successive transformation, and ``"float"`` for the INT8 rows, which are
    alternative exports of the *same* deployed float model and are therefore
    quoted against it rather than against each other.

    The whole chain runs at **one** post-processing variant, so conversion and
    quantization are priced at matched suppression and neither absorbs the
    other's cost. Choosing the variant is a separate axis with its own
    accuracy/latency trade-off, priced by :func:`nms_substitution_summary` and
    :func:`nms_latency_tradeoff_table`.
    """
    float_stage = f"Float TFLite, {NMS_LABELS.get(nms, nms)}"

    return (
        ("Float SavedModel (reference)", None, None, None),
        (float_stage, "fp32_ptq", nms, "prev"),
        ("INT8 PTQ, per-channel", "int8_ptq_per-channel", nms, "float"),
        ("INT8 PTQ, per-tensor", "int8_ptq_per-tensor", nms, "float"),
        ("INT8 QAT, per-channel", "int8_qat_per-channel", nms, "float"),
        ("INT8 QAT, per-tensor", "int8_qat_per-tensor", nms, "float"),
    )


def preparation_ladder_table(
    df: pd.DataFrame,
    *,
    include_qat: bool = True,
    quant: str = "ptq",
    reference: str = REFERENCE_PLATFORM,
    cpu_platform: str = CPU_REFERENCE_PLATFORM,
    metric: str = "AP",
    percent: bool = True,
    nms: str = DEFAULT_NMS,
) -> pd.DataFrame:
    """
    **Steps 2 and 3** -- what preparing the model for deployment costs.

    Stage rows, architecture columns, which is the shape the thesis table uses
    and the transpose of :func:`degradation_ladder_table` (that one spans
    *configurations* and continues onto the accelerator; this one stays at the
    reference configuration and stops at the exported file).

    Every rung runs the post-processing given by ``nms``, so the float step
    prices the format conversion alone and the INT8 steps price precision
    alone. Substituting one suppressor for the other is a different axis and is
    priced separately -- see :func:`_preparation_stages`.

    ``include_qat=False`` gives the step-2 (PTQ) table; the default includes
    the step-3 rows so the whole preparation chain can be shown at once.

    Args:
        nms: Post-processing variant the whole ladder is built at. Must be
            present in the frame; the SavedModel rung carries no NMS token and
            is unaffected.
        percent: Report AP in upstream percentage units (the thesis
            convention) rather than 0-1 fractions.
    """
    if df.empty or metric not in df.columns:
        return pd.DataFrame()

    sel = add_scheme(reference_config_slice(df))
    if "arch" in sel.columns:
        sel = sel[sel["arch"].isin(PRIMARY_ARCHS)]
    if sel.empty:
        return pd.DataFrame()

    scale = 100.0 if percent else 1.0

    def _stage_values(scheme, nms) -> dict:
        if scheme is None:
            rows = sel[(sel["platform"] == reference) & (sel["quant"] == quant)]
        else:
            rows = sel[(sel["platform"] == cpu_platform) & (sel["scheme"] == scheme)]
            rows = rows[rows["nms"] == nms] if "nms" in rows.columns else rows
        rows = rows.dropna(subset=[metric])
        if rows.empty:
            return {}
        return {
            arch: scale * float(group[metric].iloc[0])
            for arch, group in rows.groupby("arch_label")
        }

    all_stages = _preparation_stages(nms)
    deployed_float_stage = all_stages[1][0]
    stages = [
        stage
        for stage in all_stages
        if include_qat or stage[1] is None or "qat" not in str(stage[1])
    ]

    values = {label: _stage_values(scheme, nms) for label, scheme, nms, _ in stages}
    archs = sorted({a for v in values.values() for a in v})
    if not archs:
        return pd.DataFrame()

    rows = []
    previous: dict = {}
    for label, _scheme, _nms, delta_base in stages:
        current = values[label]
        row = {"Stage": label}

        for arch in archs:
            short = _arch_short(arch)
            value = current.get(arch)
            row[f"{short} AP"] = None if value is None else round(value, 2)

            if delta_base is None:
                base = None
            elif delta_base == "prev":
                base = previous.get(arch)
            else:
                base = values.get(deployed_float_stage, {}).get(arch)

            row[f"{short} d"] = (
                None if value is None or base is None else round(value - base, 2)
            )

        rows.append(row)
        # Only the successive float rungs advance the running reference; the
        # INT8 rows are siblings, not a chain.
        if delta_base is None or delta_base == "prev":
            previous = current

    return pd.DataFrame(rows)


#: Smallest PTQ deficit (AP points) worth expressing a reclaim as a percentage
#: of. Below it the denominator is noise and the ratio explodes -- FPNLite
#: per-channel loses 0.0997 and reclaims 0.45, which would report 451 %. The
#: cell is left empty instead, and ``Reclaimed`` still carries the absolute
#: change. Applied to the unrounded cost, so a row displaying -0.10 can be
#: suppressed; ``Reclaimed %`` empty next to a small ``PTQ cost`` means the
#: deficit was under this bar, not that the measurement is missing.
QAT_RECLAIM_MIN_DEFICIT = 0.1


def qat_reclaim_table(
    df: pd.DataFrame,
    *,
    cpu_platform: str = CPU_REFERENCE_PLATFORM,
    metric: str = "AP",
    percent: bool = True,
    nms: str | None = DEFAULT_NMS,
) -> pd.DataFrame:
    """
    **Step 3** -- how much of the quantization cost QAT gets back.

    One row per (architecture, weight granularity), because that is the axis
    the answer depends on. QAT is not a general accuracy technique here: it is
    a targeted repair for the shared-scale constraint of per-tensor
    quantization. Where that constraint does not bind -- per-channel weights,
    which already lose almost nothing -- there is nothing to reclaim and QAT
    can come out *behind* the post-training export.

    ``Reclaimed`` is the QAT-minus-PTQ difference; ``Reclaimed %`` expresses it
    as a share of the PTQ cost, so a value near 100 means the deficit was
    repaired and a negative value means QAT made it worse. It is left empty
    where PTQ had no deficit to repair, since a percentage of roughly zero is
    noise amplified, not a result.
    """
    if df.empty or metric not in df.columns:
        return pd.DataFrame()

    sel = add_scheme(select_nms(reference_config_slice(df), nms))
    sel = sel[sel["platform"] == cpu_platform]
    if "arch" in sel.columns:
        sel = sel[sel["arch"].isin(PRIMARY_ARCHS)]
    if sel.empty:
        return pd.DataFrame()

    scale = 100.0 if percent else 1.0

    def _value(arch, scheme):
        rows = sel[(sel["arch_label"] == arch) & (sel["scheme"] == scheme)]
        rows = rows.dropna(subset=[metric])
        return None if rows.empty else scale * float(rows[metric].iloc[0])

    rows = []
    for arch in sorted(sel["arch_label"].unique()):
        float_ap = _value(arch, scheme_name("fp32", "ptq"))
        for granularity in ("per-channel", "per-tensor"):
            ptq = _value(arch, scheme_name("int8", "ptq", granularity))
            qat = _value(arch, scheme_name("int8", "qat", granularity))
            if ptq is None and qat is None:
                continue

            ptq_cost = None if (ptq is None or float_ap is None) else ptq - float_ap
            qat_cost = None if (qat is None or float_ap is None) else qat - float_ap
            reclaimed = None if (ptq is None or qat is None) else qat - ptq

            rows.append(
                {
                    "Architecture": arch,
                    "Granularity": granularity,
                    "Float AP": None if float_ap is None else round(float_ap, 2),
                    "PTQ AP": None if ptq is None else round(ptq, 2),
                    "QAT AP": None if qat is None else round(qat, 2),
                    "PTQ cost": None if ptq_cost is None else round(ptq_cost, 2),
                    "QAT cost": None if qat_cost is None else round(qat_cost, 2),
                    "Reclaimed": None if reclaimed is None else round(reclaimed, 2),
                    # A share of a deficit that never existed is meaningless.
                    # Tested on the unrounded cost, so a value displayed as
                    # -0.10 can still be suppressed: FPNLite per-channel is
                    # -0.0997, and its +0.45 reclaim over that would print
                    # 451 %.
                    "Reclaimed %": (
                        None
                        if reclaimed is None
                        or ptq_cost is None
                        or ptq_cost > -QAT_RECLAIM_MIN_DEFICIT
                        else round(100 * reclaimed / -ptq_cost, 0)
                    ),
                }
            )

    return pd.DataFrame(rows)


def discover_board_pairs(df: pd.DataFrame) -> list[tuple[str, str]]:
    """
    ``(delegated_tree, cpu_tree)`` for every board that has both.

    A ``<board>_cpu`` tree is the same board with the delegate switched off, so
    the pair is a controlled CPU-vs-NPU comparison on identical hardware --
    which is the only honest way to state an acceleration factor. Control
    trees such as ``_unpatched`` are excluded: they are a different delegate
    build, not a different backend.

    The naming convention alone is not enough to identify a pair. The dev host
    may carry a delegated/CPU split of its own (``x86`` / ``x86_cpu``) while
    having no accelerator at all
    -- the difference there is whether the run *asked* for a delegate, not
    whether it got one. Pairing those would report an "acceleration factor"
    for a machine with nothing to accelerate on, so the delegated side is
    additionally required to contain runs that actually executed on a delegate.
    """
    if df.empty or "platform" not in df.columns:
        return []

    platforms = set(df["platform"].unique())
    delegated = (
        set(df.loc[df["backend"] == "delegate", "platform"].unique())
        if "backend" in df.columns
        else platforms
    )

    return sorted(
        (platform, f"{platform}_cpu")
        for platform in platforms
        if f"{platform}_cpu" in platforms
        and platform in delegated
        and not platform.endswith("_unpatched")
    )


def device_latency_table(
    df: pd.DataFrame,
    *,
    metric: str = "AP",
    deployable_only: bool = True,
    nms: str | None = DEFAULT_NMS,
) -> pd.DataFrame:
    """
    **Step 5** -- what the surviving exports cost on each board, CPU vs NPU.

    Restricted to configurations step 4 found the accelerator executes
    *correctly*. That restriction is the point rather than a convenience: a
    delegate that returns ``NaN`` boxes or collapses to near-zero AP still
    produces perfectly real timings, and it is usually **fast**, so an
    unfiltered latency table ranks the broken configurations at the top. Pass
    ``deployable_only=False`` only to demonstrate that.

    The CPU column is the same board with the delegate switched off, not a
    host, so ``Speedup`` is a property of the accelerator and not of the two
    machines. Note that the CPU baseline is XNNPACK-accelerated, which makes
    the reported speedup conservative.

    ``dAP`` is carried alongside because acceleration is only worth having if
    the answer survives it: a large speedup next to a large accuracy drop is a
    finding, not a win.
    """
    if df.empty or "median_latency_ms" not in df.columns:
        return pd.DataFrame()

    sel = add_scheme(select_nms(reference_config_slice(df), nms))
    if "arch" in sel.columns:
        sel = sel[sel["arch"].isin(PRIMARY_ARCHS)]
    if deployable_only:
        sel = drop_failed_deployments(sel)
    if sel.empty:
        return pd.DataFrame()

    rows = []
    for npu_platform, cpu_platform in discover_board_pairs(sel):
        npu = sel[(sel["platform"] == npu_platform) & (sel["backend"] == "delegate")]
        cpu = sel[sel["platform"] == cpu_platform]
        if npu.empty or cpu.empty:
            continue

        keys = ["arch_label", "size", "scheme"]
        cpu_indexed = cpu[~cpu.duplicated(subset=keys, keep="first")].set_index(keys)

        for _, r in npu.sort_values(keys).iterrows():
            key = tuple(r[k] for k in keys)
            reference = cpu_indexed.loc[key] if key in cpu_indexed.index else None

            cpu_ms = (
                None
                if reference is None or pd.isna(reference["median_latency_ms"])
                else float(reference["median_latency_ms"])
            )
            npu_ms = (
                None
                if pd.isna(r["median_latency_ms"])
                else float(r["median_latency_ms"])
            )

            rows.append(
                {
                    "Board": npu_platform,
                    "Architecture": r["arch_label"],
                    "Input": r.get("size"),
                    "Scheme": r["scheme"],
                    "CPU (ms)": None if cpu_ms is None else round(cpu_ms, 1),
                    "NPU (ms)": None if npu_ms is None else round(npu_ms, 1),
                    "Speedup": (
                        None if not cpu_ms or not npu_ms else round(cpu_ms / npu_ms, 1)
                    ),
                    "NPU FPS": None if not npu_ms else round(1000.0 / npu_ms, 1),
                    "AP CPU": (
                        None
                        if reference is None or pd.isna(reference[metric])
                        else round(float(reference[metric]), 4)
                    ),
                    "AP NPU": (
                        None if pd.isna(r[metric]) else round(float(r[metric]), 4)
                    ),
                    "dAP": (
                        None
                        if reference is None
                        or pd.isna(r[metric])
                        or pd.isna(reference[metric])
                        else round(float(r[metric]) - float(reference[metric]), 4)
                    ),
                }
            )

    return pd.DataFrame(rows)


def plot_device_latency(df: pd.DataFrame, **kwargs):
    """CPU vs NPU median latency per board, architecture and export scheme."""
    table = device_latency_table(df, **kwargs)
    if table.empty:
        return None

    table = table.dropna(subset=["CPU (ms)", "NPU (ms)"]).copy()
    if table.empty:
        return None

    table["group"] = (
        table["Board"]
        + " | "
        + table["Architecture"].map(_arch_short)
        + " | "
        + table["Scheme"].map(scheme_label)
    )
    groups = list(table["group"])

    fig, ax = plt.subplots(figsize=(9, max(3.0, 0.32 * len(groups)) + 1.2))
    _grouped_bars(
        ax,
        groups,
        ["CPU (XNNPACK)", "NPU (delegate)"],
        {
            "CPU (XNNPACK)": list(table["CPU (ms)"]),
            "NPU (delegate)": list(table["NPU (ms)"]),
        },
        PALETTE,
        ylabel="median latency (ms)",
        title="Inference latency, CPU vs NPU (reference configuration)",
        fmt="{:.0f}",
        horizontal=True,
    )
    ax.legend(title="backend", loc="lower right")
    fig.tight_layout()
    return fig


#: The single-axis deviations from the reference configuration that step 6
#: contrasts against it. One axis at a time, because crossing them would make
#: an effect impossible to attribute.
ABLATION_AXES = (
    ("Reference (mc, trained full, eval full)", {}),
    ("Single-class", {"classes": "sc"}),
    ("Trained tiled", {"dataset": "phenobench-tiled"}),
    ("Evaluated tiled", {"eval_tiling": "tiled"}),
    ("Tiled end to end", {"dataset": "phenobench-tiled", "eval_tiling": "tiled"}),
)

#: The tiling square. Training tiling and evaluation tiling are *not* two
#: independent one-axis deviations, and reporting them that way is misleading:
#: each alone costs about 4 AP, while doing both gains 8.6 (MNv2, fp32, CPU
#: reference). The matched cell is a different pipeline, not the sum of two
#: perturbations, so :func:`tiling_cross_table` reports all four.
TILING_CELLS = (
    ("trained full / eval full", "phenobench", "untiled"),
    ("trained full / eval tiled", "phenobench", "tiled"),
    ("trained tiled / eval full", "phenobench-tiled", "untiled"),
    ("trained tiled / eval tiled", "phenobench-tiled", "tiled"),
)


def tiling_cross_table(
    df: pd.DataFrame,
    *,
    platform: str = CPU_REFERENCE_PLATFORM,
    classes: str | None = "mc",
    size: str | None = "320",
    precision: str = "fp32",
    quant: str = "ptq",
    nms: str | None = DEFAULT_NMS,
    metrics: Iterable[tuple[str, str]] = (
        ("AP", "AP"),
        ("crop_AP", "Crop AP"),
        ("weed_AP", "Weed AP"),
    ),
    percent: bool = True,
) -> pd.DataFrame:
    """
    All four training x evaluation tiling combinations, per architecture.

    The two axes interact, so the one-at-a-time ablation cannot describe them.
    Each cell answers a different question:

    ==========================  =========================================
    cell                        question
    ==========================  =========================================
    trained full / eval full    the reference
    trained full / eval tiled   does tiling only at inference help?
    trained tiled / eval full   does tile-training transfer to full frames?
    trained tiled / eval tiled  the matched tiled pipeline
    ==========================  =========================================

    ``d`` is against the reference cell of the same architecture, so a
    mismatched pair and the matched pipeline can be read off directly rather
    than inferred from two separate deltas that do not add up.
    """
    if df.empty or "dataset" not in df.columns:
        return pd.DataFrame()

    sel = select_nms(df, nms)
    sel = sel[sel["platform"] == platform] if platform else sel
    sel = sel[sel["precision"] == precision]
    if quant and "quant" in sel.columns:
        sel = sel[sel["quant"] == quant]
    if classes:
        sel = sel[sel["classes"] == classes]
    if size:
        sel = sel[sel["size"] == size]
    sel = sel[sel["arch"].isin(PRIMARY_ARCHS)] if "arch" in sel.columns else sel
    if sel.empty:
        return pd.DataFrame()

    scale = 100.0 if percent else 1.0
    rows = []

    for arch, group in sel.groupby("arch_label"):
        reference = {}
        for label, dataset, tiling in TILING_CELLS:
            cell = group[
                (group["dataset"] == dataset) & (group["eval_tiling"] == tiling)
            ]
            row = {"Architecture": _arch_short(arch), "Tiling": label}

            for col, name in metrics:
                value = (
                    None
                    if cell.empty or cell[col].isna().all()
                    else scale * float(cell[col].mean())
                )
                row[name] = None if value is None else round(value, 2)

                if label == TILING_CELLS[0][0]:
                    reference[name] = value
                base = reference.get(name)
                row[f"d {name}"] = (
                    None if value is None or base is None else round(value - base, 2)
                )

            rows.append(row)

    return pd.DataFrame(rows)


def story_ablation_table(
    df: pd.DataFrame,
    *,
    npu_platform: str = "frdm-imx93",
    cpu_platform: str = CPU_REFERENCE_PLATFORM,
    reference: str = REFERENCE_PLATFORM,
    metric: str = "AP",
    percent: bool = True,
) -> pd.DataFrame:
    """
    **Step 6** -- how single-class and tiling move each of the earlier steps.

    One row per (deviation, architecture), one column per step, so the question
    "does this dimension change the conclusion, or only the number" is
    answerable by reading across. Deviations are single-axis: crossing them
    would make any effect impossible to attribute.

    Reading guide, since the columns are not all the same kind of quantity:

    * ``Float AP`` is a level -- how good the detector is at all;
    * ``Conversion`` / ``NMS swap`` / ``PTQ`` are costs against the preceding
      rung, so they answer "does preparation behave differently here";
    * ``QAT reclaim`` is QAT minus PTQ at per-tensor weights;
    * ``NPU (ms)`` is cost on the accelerator.

    The one prediction worth checking against this table is that ``NMS swap``
    must be **exactly zero** on the single-class row: at one class the fast and
    per-class suppressors are the same algorithm.

    Cells whose rung was never benchmarked are left empty rather than dropping
    the row, so an incomplete sweep is visible instead of silently narrowing
    the comparison.

    .. warning::

       As with :func:`preparation_ladder_table`, pass a frame that still has
       both post-processing variants; :func:`select_nms` output empties the
       ``Conversion`` and ``NMS swap`` columns without complaining.
    """
    if df.empty or metric not in df.columns:
        return pd.DataFrame()

    scale = 100.0 if percent else 1.0
    base = add_scheme(df)
    if "arch" in base.columns:
        base = base[base["arch"].isin(PRIMARY_ARCHS)]
    if base.empty:
        return pd.DataFrame()

    def _ap(frame, platform, scheme=None, nms=DEFAULT_NMS, quant=None):
        rows = frame[frame["platform"] == platform]
        if scheme is not None:
            rows = rows[rows["scheme"] == scheme]
        if quant is not None:
            rows = rows[rows["quant"] == quant]
        if nms is not None and "nms" in rows.columns:
            rows = rows[rows["nms"].isna() | (rows["nms"] == nms)]
        rows = rows.dropna(subset=[metric])
        return None if rows.empty else scale * float(rows[metric].iloc[0])

    def _sub(a, b):
        return None if a is None or b is None else round(a - b, 2)

    rows = []
    for label, overrides in ABLATION_AXES:
        slice_ = reference_config_slice(base, **overrides)
        if slice_.empty:
            continue

        for arch in sorted(slice_["arch_label"].unique()):
            frame = slice_[slice_["arch_label"] == arch]

            saved = _ap(frame, reference, nms=None, quant="ptq")
            control = _ap(frame, cpu_platform, scheme_name("fp32", "ptq"), REGULAR_NMS)
            deployed = _ap(frame, cpu_platform, scheme_name("fp32", "ptq"))
            ptq = _ap(frame, cpu_platform, scheme_name("int8", "ptq", "per-tensor"))
            qat = _ap(frame, cpu_platform, scheme_name("int8", "qat", "per-tensor"))

            npu_rows = frame[
                (frame["platform"] == npu_platform)
                & (frame["scheme"] == scheme_name("int8", "ptq", "per-tensor"))
            ].dropna(subset=["median_latency_ms"])

            rows.append(
                {
                    "Variant": label,
                    "Architecture": _arch_short(arch),
                    "Float AP": None if saved is None else round(saved, 2),
                    "Conversion": _sub(control, saved),
                    "NMS swap": _sub(deployed, control),
                    "PTQ": _sub(ptq, deployed),
                    "QAT reclaim": _sub(qat, ptq),
                    "Deployed AP": None if ptq is None else round(ptq, 2),
                    "NPU (ms)": (
                        None
                        if npu_rows.empty
                        else round(float(npu_rows["median_latency_ms"].iloc[0]), 1)
                    ),
                }
            )

    return pd.DataFrame(rows)


def master_table(df: pd.DataFrame) -> pd.DataFrame:
    """Full benchmark matrix with renamed, rounded thesis-ready columns."""
    cols = [
        ("platform", "Platform"),
        ("arch_label", "Architecture"),
        ("classes", "Classes"),
        # Trained-on dataset and evaluated-on input regime are independent now
        # (every model is swept over both regimes), so both have to be shown or
        # otherwise identical-looking rows differ by an invisible field.
        ("dataset", "Trained on"),
        ("size", "Input"),
        ("eval_tiling", "Eval input"),
        ("precision", "Precision"),
        ("quant", "Quant"),
        ("granularity", "Granularity"),
        # Without this column the default export and its algorithm-matched
        # control are two rows that differ in nothing the reader can see.
        ("nms", "NMS"),
        ("AP", "AP"),
        ("AP50", "AP50"),
        ("AP75", "AP75"),
        ("weed_AP", "Weed AP"),
        ("crop_AP", "Crop AP"),
        ("APS", "APS"),
        ("mean_latency_ms", "Lat (ms)"),
        ("fps", "FPS"),
    ]
    cols = [(c, n) for c, n in cols if c in df.columns]
    out = df[[c for c, _ in cols]].copy()
    out.columns = [n for _, n in cols]

    round_map = {
        n: 4
        for c, n in cols
        if c in ("AP", "AP50", "AP75", "weed_AP", "crop_AP", "APS")
    }
    round_map.update({"Lat (ms)": 2, "FPS": 1})
    return out.round(round_map)


# =========================================================
# Coverage / completeness
# =========================================================

#: The deployable exports that make up a *full run* for one model variant, as
#: ``(precision, quant, granularity)`` -- the FP32 baseline plus one INT8 export
#: per quantization scheme and weight granularity, mirroring the conversion
#: targets in :mod:`agri_vision_edge.conversion.tflite`. FP32 has no
#: granularity; QAT is INT8-only by construction.
DEFAULT_SCHEMES = (
    ("fp32", "ptq", None),
    ("int8", "ptq", "per-tensor"),
    ("int8", "ptq", "per-channel"),
    ("int8", "qat", "per-tensor"),
    ("int8", "qat", "per-channel"),
)

#: Every model is benchmarked on both input regimes, so both belong in the
#: expected matrix.
DEFAULT_EVAL_TILINGS = ("untiled", "tiled")

#: Platforms a full run targets: the dev host plus the two embedded NPU boards.
DEFAULT_EXPECTED_PLATFORMS = (CPU_REFERENCE_PLATFORM, "frdm-imx8mp", "frdm-imx93")


def scheme_label(scheme: str) -> str:
    """
    Short display name for an export scheme.

    Two changes to the raw token. ``int8_`` is dropped, since every scheme but
    the float one is INT8 and the prefix only costs width. And ``fp32_ptq``
    becomes plain ``fp32``: the quant field records *which path exported the
    float graph*, not a quantization that was applied, so the suffix reads as
    though the float model were post-training quantized. Where both float
    exports are present the QAT one keeps a marker, because then the path is
    the thing that distinguishes them.
    """
    if scheme == "fp32_ptq":
        return "fp32"
    if scheme == "fp32_qat":
        return "fp32 (QAT path)"

    return scheme.replace("int8_", "")


def scheme_name(precision, quant, granularity=None) -> str:
    """
    Canonical short name of one export, e.g. ``int8_qat_per-channel``.

    Matches the artifact filename suffix so a scheme in a table can be traced
    back to the exact ``.tflite`` it came from.
    """
    parts = [str(precision), str(quant)]
    if granularity and str(granularity) != "nan":
        parts.append(str(granularity))
    return "_".join(parts)


def add_scheme(df: pd.DataFrame) -> pd.DataFrame:
    """Return ``df`` with a ``scheme`` column (see :func:`scheme_name`)."""
    if df.empty:
        return df
    out = df.copy()
    out["scheme"] = [
        scheme_name(p, q, g)
        for p, q, g in zip(
            out.get("precision", ""),
            out.get("quant", ""),
            out.get("granularity", pd.Series([None] * len(out), index=out.index)),
            strict=False,
        )
    ]
    return out


#: Fields that identify a model variant independent of precision/quant/platform.
_VARIANT_KEYS = ("arch", "classes", "dataset", "size")


def discover_model_variants(
    artifacts_tf_dir: str | Path = "artifacts/tf",
) -> pd.DataFrame:
    """
    Enumerate trained model variants from the ``artifacts/tf`` folders.

    Each subdirectory is a finetuned model named ``<arch>_<classes>_<dataset>_<size>``;
    these define the rows of the expected benchmark matrix.

    Returns:
        One row per variant with its parsed config fields, or an empty frame
        if the directory is absent.
    """
    artifacts_tf_dir = Path(artifacts_tf_dir)
    if not artifacts_tf_dir.is_dir():
        return pd.DataFrame()

    rows = []
    for variant_dir in sorted(p for p in artifacts_tf_dir.iterdir() if p.is_dir()):
        info = parse_run_name(variant_dir.name)
        if info is None:
            continue
        rows.append(
            {
                "variant": variant_dir.name,
                **{k: info.get(k) for k in (*_VARIANT_KEYS, "arch_label", "config")},
            }
        )
    return pd.DataFrame(rows)


#: Trees that are not part of the conversion matrix at all. The SavedModel rung
#: is the checkpoint's own TensorFlow graph: every export scheme in
#: `DEFAULT_SCHEMES` is produced by `ave convert` *downstream* of it, so none of
#: them is a thing that tree can be missing -- not the INT8 ones, and not
#: `fp32_ptq` either, whose `ptq` token records which training path emitted the
#: float graph rather than any quantization applied to it. Crossed with the
#: schemes anyway, the tree reported 22 of 110 cells done (20 %) when it was
#: complete for everything it can hold. It is produced by
#: `scripts/benchmark_reference_models.py`, not by the sweep this matrix counts.
NON_MATRIX_PLATFORMS = (REFERENCE_PLATFORM, "tf-savedmodel-nms0")


def build_coverage(
    runs_df: pd.DataFrame,
    variants_df: pd.DataFrame,
    schemes: Iterable[tuple[str, str, str | None]] = DEFAULT_SCHEMES,
    platforms: Iterable[str] = DEFAULT_EXPECTED_PLATFORMS,
    eval_tilings: Iterable[str] = DEFAULT_EVAL_TILINGS,
    nms: str | None = DEFAULT_NMS,
    exclude_platforms: Iterable[str] = NON_MATRIX_PLATFORMS,
) -> pd.DataFrame:
    """
    Cross variants × schemes × eval regimes × platforms into a long coverage
    frame, flagging which expected runs are already benchmarked.

    A cell is one benchmarked artifact, so the key has to carry everything that
    distinguishes one: the variant, the export scheme *including its weight
    granularity*, and the input regime it was evaluated on. Leaving the
    granularity out (as an earlier version did) makes the per-tensor and
    per-channel exports indistinguishable and reports a matrix that is both
    incomplete and satisfied by half the runs.

    Coverage is counted over the **default export** -- the ``fastnms``
    variants. The per-class variant is equally deployable and was swept on the
    boards, but only there: it is not converted for the CPU-only and unpatched
    control trees, so treating it as an expected cell would invent gaps that
    are deliberate. Treating it as *satisfying* a cell would be worse, letting
    a per-class run stand in for the default export. The NMS axis therefore
    gets its own completeness view rather than a column here.

    ``exclude_platforms`` drops trees the conversion matrix does not describe.
    No export scheme is a thing the SavedModel rung can be missing -- it holds
    the pre-conversion graph, and every scheme is produced downstream of it --
    so crossing it with them reported phantom gaps and understated its coverage
    fivefold. It still appears in §8's per-platform tables, where it belongs.

    Other platforms present in ``runs_df`` are always included, so unexpected
    ones still surface. Matching ignores the split token.
    """
    if variants_df.empty:
        return pd.DataFrame()

    runs_df = select_nms(runs_df, nms)

    def _granularity(value):
        # fp32 has no granularity; normalise NaN/"" to None so the expected
        # matrix and the loaded runs use the same key.
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return None
        return value or None

    present_keys = set()
    if not runs_df.empty:
        for _, r in runs_df.iterrows():
            present_keys.add(
                (
                    r["platform"],
                    *(r.get(k) for k in _VARIANT_KEYS),
                    r.get("eval_tiling"),
                    r.get("precision"),
                    r.get("quant"),
                    _granularity(r.get("granularity")),
                )
            )

    skip = set(exclude_platforms)
    platform_list = [
        p
        for p in dict.fromkeys(
            [*platforms, *(runs_df["platform"].unique() if not runs_df.empty else [])]
        )
        if p not in skip
    ]

    rows = []
    for platform in platform_list:
        for _, v in variants_df.iterrows():
            for eval_tiling in eval_tilings:
                for precision, quant, granularity in schemes:
                    key = (
                        platform,
                        *(v[k] for k in _VARIANT_KEYS),
                        eval_tiling,
                        precision,
                        quant,
                        _granularity(granularity),
                    )
                    rows.append(
                        {
                            "platform": platform,
                            "variant": v["variant"],
                            "config": v["config"],
                            "eval_tiling": eval_tiling,
                            "precision": precision,
                            "quant": quant,
                            "granularity": granularity,
                            "scheme": scheme_name(precision, quant, granularity),
                            "present": key in present_keys,
                        }
                    )
    return pd.DataFrame(rows)


def coverage_matrix(coverage_long: pd.DataFrame) -> pd.DataFrame:
    """
    Pivot the long coverage frame to a variant×(platform/scheme) ASCII grid
    ("x" = done, "-" = missing) suitable for a table / LaTeX export.
    """
    if coverage_long.empty:
        return pd.DataFrame()
    grid = coverage_long.assign(
        mark=lambda d: d["present"].map({True: "x", False: "-"})
    ).pivot_table(
        index=["variant", "eval_tiling"],
        columns=["platform", "scheme"],
        values="mark",
        aggfunc="first",
    )
    grid.columns = [f"{p} {s}" for p, s in grid.columns]
    return grid.reset_index()


def coverage_summary(coverage_long: pd.DataFrame) -> pd.DataFrame:
    """Per-platform done / total / percent counts."""
    if coverage_long.empty:
        return pd.DataFrame()
    g = coverage_long.groupby("platform")["present"]
    out = pd.DataFrame({"done": g.sum(), "total": g.count()})
    out["percent"] = (100 * out["done"] / out["total"]).round(1)
    return out.reset_index()


# =========================================================
# Resource / power sweeps (a separate measurement path)
# =========================================================

#: Metrics a resource sweep reports, as ``(column, label)``. Energy leads: it
#: is the quantity latency alone cannot answer, and the reason the i.MX93 can
#: lose on speed and still win a deployment.
RESOURCE_METRICS = (
    ("net mJ/inf", "Net energy per inference (mJ)", True),
    ("net P (W)", "Net power (W)", True),
    ("lat mean (ms)", "Mean latency (ms)", False),
    ("RSS (MiB)", "Peak resident memory (MiB)", False),
)


def annotate_resource_runs(power_df: pd.DataFrame) -> pd.DataFrame:
    """
    Add the configuration columns parsed out of each resource run's name.

    A resource sweep records the *model stem*, not a benchmark run directory,
    so it carries no ``<tiling>`` prefix -- the sweep feeds one image set and
    input regime is not an axis it varies. Everything else in the name follows
    the same grammar, which is what lets these rows be scoped with the same
    vocabulary as the rest of the notebook.
    """
    if power_df.empty or "run" not in power_df.columns:
        return power_df

    parsed = [parse_run_name(str(name)) or {} for name in power_df["run"]]
    fields = (
        "arch",
        "arch_label",
        "classes",
        "dataset",
        "size",
        "precision",
        "quant",
        "granularity",
        "nms",
    )

    out = power_df.copy()
    for field in fields:
        out[field] = [info.get(field) for info in parsed]

    return add_scheme(out)


def plot_resource_summary(
    power_df: pd.DataFrame,
    *,
    metrics: Iterable[tuple[str, str, bool]] = RESOURCE_METRICS,
    verified_only: bool = True,
):
    """
    One panel per resource metric, grouped by export scheme, coloured by device.

    ``verified_only`` is applied **per metric**, not to the frame. Only power
    and energy are joined to the meter trace; latency, CPU and memory are
    recorded by the board itself and are unaffected by whether the join could
    be checked. Gating them too punched holes in a matrix for a reason that
    does not apply to them, so each panel drops only the rows its own metric
    depends on -- and the panel title says when that happened.
    """
    if power_df.empty:
        return None

    sel = power_df
    if sel.empty:
        return None

    def _rows_for(joined: bool) -> pd.DataFrame:
        """Rows a metric may use: joined ones need a checked alignment."""
        if not (joined and verified_only) or "state" not in sel.columns:
            return sel
        return sel[sel["state"] == "verified"]

    usable = [
        (col, label, joined)
        for col, label, joined in metrics
        if col in sel.columns and _rows_for(joined)[col].notna().any()
    ]
    if not usable:
        return None

    schemes = [s for s in SCHEME_ORDER if s in set(sel.get("scheme", []))]
    if not schemes:
        return None

    # Series are device *and* architecture. Folding the detectors together
    # averages a ~40% latency difference into one bar and hides the thing a
    # reader compares: FPNLite's extra neck costs energy on one board and
    # almost nothing on the other.
    series = sorted(
        {
            (str(d), str(a))
            for d, a in zip(sel["device"], sel["arch_label"], strict=False)
        }
    )
    if not series:
        return None

    labels = [scheme_label(s) for s in schemes]
    series_labels = [
        f"{platform_label(device)} · {_short(pd.Series([arch])).iloc[0]}"
        for device, arch in series
    ]
    colors = [PALETTE[i % len(PALETTE)] for i in range(len(series))]

    fig, axes = plt.subplots(
        len(usable), 1, figsize=(9, 2.9 * len(usable)), sharex=True
    )
    axes = np.atleast_1d(axes)

    for index, (ax, (col, label, joined)) in enumerate(zip(axes, usable, strict=False)):
        rows = _rows_for(joined)
        dropped = len(sel) - len(rows)
        values = {
            series_label: [
                rows[
                    (rows["device"] == device)
                    & (rows["arch_label"] == arch)
                    & (rows["scheme"] == scheme)
                ][col].mean()
                for scheme in schemes
            ]
            for (device, arch), series_label in zip(series, series_labels, strict=False)
        }
        _grouped_bars(
            ax,
            labels,
            series_labels,
            values,
            colors,
            ylabel=label,
            title="",
            fmt="{:.1f}",
            rotation=20 if index == len(usable) - 1 else 0,
        )
        ax.set_title(
            label + (f"  (excludes {dropped} unaligned)" if dropped else ""),
            fontsize=10,
        )
        ax.set_ylabel("")

    fig.suptitle("Steady-state cost per export scheme")
    fig.tight_layout()
    _legend_outside(fig, axes[0], title="device · detector")
    return fig


def plot_class_regime_quantization(
    df: pd.DataFrame,
    *,
    platform: str | None = CPU_REFERENCE_PLATFORM,
    nms: str | None = DEFAULT_NMS,
    percent: bool = True,
):
    """
    Per-class AP across export schemes, multi-class beside single-class.

    Three series per scheme: the two classes a multi-class model reports, and
    the weed AP of the single-class model trained on the same data. Read across
    a panel, the question is whether quantization-aware training helps the same
    amount in both class regimes -- and it does not. The single-class weed bar
    is *not* a slice of the multi-class one; it is a different model, which is
    why it needs its own series rather than a facet.

    One panel per architecture, because the two detectors answer this
    differently and pooling them would average an effect that reverses sign.
    """
    if df.empty or "weed_AP" not in df.columns:
        return None

    sel = add_scheme(select_nms(df, nms))
    if platform:
        sel = sel[sel["platform"] == platform]
    sel = sel[sel["arch"].isin(PRIMARY_ARCHS)]
    if sel.empty:
        return None

    schemes = [s for s in SCHEME_ORDER if s in set(sel["scheme"])]
    archs = sorted(sel["arch_label"].unique())
    if not schemes or not archs:
        return None

    series = (
        ("crop AP (multi-class)", "mc", "crop_AP"),
        ("weed AP (multi-class)", "mc", "weed_AP"),
        ("weed AP (single-class)", "sc", "weed_AP"),
    )

    fig, axes = plt.subplots(len(archs), 1, figsize=(9, 3.4 * len(archs)), sharex=True)
    axes = np.atleast_1d(axes)
    labels = [scheme_label(s) for s in schemes]

    for index, (ax, arch) in enumerate(zip(axes, archs, strict=False)):
        rows = sel[sel["arch_label"] == arch]
        values = {}
        for label, classes, column in series:
            cells = rows[rows["classes"] == classes]
            values[label] = [
                (
                    None
                    if cells[cells["scheme"] == s][column].isna().all()
                    else cells[cells["scheme"] == s][column].mean()
                )
                for s in schemes
            ]

        _grouped_bars(
            ax,
            labels,
            [label for label, _, _ in series],
            values,
            PALETTE,
            ylabel="AP" if index == len(archs) - 1 else "",
            title="",
            fmt="{:.1f}",
            percent=percent,
            rotation=15 if index == len(archs) - 1 else 0,
        )
        ax.set_title(_short(pd.Series([arch])).iloc[0], fontsize=10)

    fig.suptitle("Quantization by class regime")
    fig.tight_layout()
    _legend_outside(fig, axes[0], title="metric")
    return fig


def plot_resolution_tradeoff(
    df: pd.DataFrame,
    *,
    platforms: Iterable[str] = ("frdm-imx8mp", "frdm-imx93"),
    scheme: str = "int8_qat_per-tensor",
    nms: str | None = DEFAULT_NMS,
    metric: str = "AP",
    percent: bool = True,
    latency_ticks: Iterable[float] = (30, 50, 100, 200, 500),
):
    """
    What a resolution rung buys against what it costs, on the target.

    Accuracy against **device** latency, with the resolution rungs of one
    detector on one board joined into a line. A rung is worth taking only if
    its step up the y axis justifies its step along x, and that is a question
    about the board -- the CPU reference answers a different one, so it is not
    drawn here.

    Latency is log-scaled: the ladder spans 30 ms to several seconds, and on a
    linear axis every rung below 1024 collapses onto the origin.

    Pinned to one ``scheme``. Granularity alone moves latency eightfold on the
    Vivante NPU, so a line mixing schemes would trace that rather than
    resolution. Points are labelled with their input size; a rung not yet
    benchmarked on a board simply does not appear.
    """
    if df.empty or metric not in df.columns:
        return None

    sel = add_scheme(select_nms(df, nms))
    sel = sel[
        sel["platform"].isin(set(platforms))
        & (sel["scheme"] == scheme)
        & sel["arch"].isin(PRIMARY_ARCHS)
        & (sel["classes"] == "mc")
        & (sel["dataset"] == "phenobench")
        & (sel["eval_tiling"] == "untiled")
    ]
    sel = sel[sel[metric].notna() & sel["median_latency_ms"].notna()].copy()
    if sel.empty:
        return None

    sel["px"] = pd.to_numeric(sel["size"], errors="coerce")
    sel = sel[sel["px"].notna()]
    if sel.empty:
        return None

    scale = 100.0 if percent else 1.0
    targets = sorted(sel["platform"].unique())
    archs = sorted(sel["arch_label"].unique())
    colors = {p: PALETTE[i % len(PALETTE)] for i, p in enumerate(targets)}
    markers = dict(zip(archs, ("o", "s", "^"), strict=False))

    fig, ax = plt.subplots(figsize=(8, 5.5))

    for target in targets:
        for arch in archs:
            rows = sel[
                (sel["platform"] == target) & (sel["arch_label"] == arch)
            ].sort_values("px")
            if rows.empty:
                continue
            grouped = rows.groupby("px").agg(
                lat=("median_latency_ms", "median"), ap=(metric, "mean")
            )
            ax.plot(
                grouped["lat"],
                grouped["ap"] * scale,
                marker=markers.get(arch, "o"),
                markersize=9,
                linewidth=1.5,
                color=colors[target],
                markeredgecolor="black",
                markeredgewidth=0.4,
                label=f"{platform_label(target)} · {_short(pd.Series([arch])).iloc[0]}",
                zorder=3,
            )
            for px, row in grouped.iterrows():
                ax.annotate(
                    f"  {int(px)}",
                    (row["lat"], row["ap"] * scale),
                    fontsize=7,
                    va="center",
                )

    ax.set_xscale("log")
    ax.set_xticks(list(latency_ticks))
    ax.xaxis.set_major_formatter(
        FuncFormatter(lambda x, _: f"{x:g}" if x < 1000 else f"{x / 1000:g}k")
    )

    ax.set_xlabel("median latency on device (ms, log scale)")
    ax.set_ylabel(f"{metric} (%)" if percent else metric)
    ax.set_title(f"Resolution trade-off on device — {scheme_label(scheme)}")
    _prepare_axis(ax)

    fig.tight_layout()
    _legend_outside(fig, ax, title="device · detector")
    return fig


def plot_resolution_ap(
    df: pd.DataFrame,
    *,
    platform: str | None = CPU_REFERENCE_PLATFORM,
    nms: str | None = DEFAULT_NMS,
    metric: str = "AP",
    schemes: Iterable[str] = (
        "fp32_ptq",
        "int8_ptq_per-channel",
        "int8_ptq_per-tensor",
    ),
    percent: bool = True,
):
    """
    AP against input resolution, one line per detector.

    Colour is the architecture and the line is its ladder, so the question the
    figure answers -- does more input resolution buy accuracy, and does it buy
    the same for both detectors -- is read off the slopes.

    ``schemes`` are drawn as separate line styles, and are listed rather than
    summarised by precision on purpose. An earlier version drew one "INT8" line
    per detector, which averaged per-tensor with per-channel and so hid up to
    2.4 AP (FPNLite at 320) of the granularity effect §2 exists to report --
    the same silent collapse this figure is meant to rule out for resolution.
    QAT is omitted by default to keep the line count readable; the table beside
    the figure carries every scheme.

    A detector with no run at some resolution simply ends early; the ladder is
    ragged while rungs are still training.
    """
    if df.empty or metric not in df.columns:
        return None

    sel = add_scheme(select_nms(df, nms))
    if platform:
        sel = sel[sel["platform"] == platform]
    sel = sel[sel["arch"].isin(PRIMARY_ARCHS)]
    sel = sel[
        (sel["classes"] == "mc")
        & (sel["dataset"] == "phenobench")
        & (sel["eval_tiling"] == "untiled")
    ]
    wanted = list(schemes)
    sel = sel[sel["scheme"].isin(wanted)]
    sel = sel[sel[metric].notna() & sel["size"].notna()].copy()
    if sel.empty:
        return None

    sel["px"] = pd.to_numeric(sel["size"], errors="coerce")
    sel = sel[sel["px"].notna()]
    if sel.empty:
        return None

    scale = 100.0 if percent else 1.0
    archs = sorted(sel["arch_label"].unique())
    colors = {a: PALETTE[i % len(PALETTE)] for i, a in enumerate(archs)}
    dashes = ("-", "--", ":", "-.", (0, (3, 1, 1, 1)))
    styles = {s: dashes[i % len(dashes)] for i, s in enumerate(wanted)}

    fig, ax = plt.subplots(figsize=(7.5, 5))

    for arch in archs:
        for scheme in wanted:
            rows = sel[(sel["arch_label"] == arch) & (sel["scheme"] == scheme)]
            if rows.empty:
                continue
            ladder = rows.groupby("px")[metric].mean().mul(scale).sort_index()
            ax.plot(
                ladder.index,
                ladder.to_numpy(),
                marker="*",
                markersize=14,
                linewidth=1.6,
                linestyle=styles[scheme],
                color=colors[arch],
                markeredgecolor="black",
                markeredgewidth=0.4,
                label=f"{_short(pd.Series([arch])).iloc[0]} · {scheme_label(scheme)}",
                zorder=3,
            )

    ax.set_xlabel("input resolution (px)")
    ax.set_ylabel(f"{metric} (%)" if percent else metric)
    ax.set_title("Accuracy against input resolution")
    ax.set_xticks(sorted(sel["px"].unique()))
    ax.set_xticklabels([f"{int(p)}" for p in sorted(sel["px"].unique())])
    _prepare_axis(ax)

    fig.tight_layout()
    _legend_outside(fig, ax, title="detector · scheme")
    return fig


def plot_coverage(coverage_long: pd.DataFrame):
    """Heatmap of benchmark coverage: rows = variant, cols = platform/scheme."""
    if coverage_long.empty:
        return None
    from matplotlib.colors import ListedColormap

    grid = coverage_long.pivot_table(
        index=["variant", "eval_tiling"],
        columns=["platform", "scheme"],
        values="present",
        aggfunc="first",
    )
    col_labels = [f"{p}\n{s}" for p, s in grid.columns]
    data = grid.to_numpy(dtype=float)

    fig, ax = plt.subplots(
        figsize=(max(7, 0.5 * data.shape[1]), max(3, 0.5 * data.shape[0]) + 1)
    )
    ax.imshow(
        data, cmap=ListedColormap(["#e8e8e8", "#55A868"]), vmin=0, vmax=1, aspect="auto"
    )
    ax.set_xticks(range(data.shape[1]))
    ax.set_xticklabels(col_labels, fontsize=7, rotation=90)
    ax.set_yticks(range(data.shape[0]))
    ax.set_yticklabels(grid.index, fontsize=8)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            ax.text(
                j,
                i,
                "x" if data[i, j] else "",
                ha="center",
                va="center",
                fontsize=8,
                color="white",
            )
    ax.set_title("Benchmark coverage (x = done)")
    ax.grid(False)
    fig.tight_layout()
    return fig


# =========================================================
# Export helpers
# =========================================================


def save_figure(
    fig,
    stem: str,
    fig_dir: str | Path,
    formats: Iterable[str] = ("pdf", "png"),
) -> None:
    """Export ``fig`` as ``<stem>.<ext>`` for each format into ``fig_dir``."""
    if fig is None:
        return
    fig_dir = Path(fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)
    for ext in formats:
        # `bbox_inches="tight"` grows the canvas to include artists placed
        # outside the axes. Legends are put there deliberately (they would sit
        # on the data otherwise), and without this they are cropped at the
        # figure edge in the exported file.
        fig.savefig(fig_dir / f"{stem}.{ext}", bbox_inches="tight")


# Unicode -> ASCII fallbacks so exported .tex compiles under plain pdflatex.
_ASCII_REPLACEMENTS = {
    "·": "-",  # middle dot
    "→": "->",  # right arrow
    "≤": "<=",  # less-or-equal
    "≥": ">=",  # greater-or-equal
    "Δ": "d",  # capital delta
    "²": "^2",  # superscript two
    "–": "-",  # en dash
    "—": "--",  # em dash
}


def _ascii(value):
    """Coerce a string to ASCII, applying known replacements first."""
    if not isinstance(value, str):
        return value
    for uni, repl in _ASCII_REPLACEMENTS.items():
        value = value.replace(uni, repl)
    return value.encode("ascii", "replace").decode("ascii")


def save_latex_table(
    df: pd.DataFrame,
    path: str | Path,
    *,
    caption: str = "",
    label: str | None = None,
    **to_latex_kwargs,
) -> None:
    """
    Write ``df`` as a (optionally wrapped) booktabs LaTeX table (ASCII-only).

    The index is dropped only when it carries no information. A default
    ``RangeIndex`` is row numbering and belongs nowhere near a thesis table,
    but a frame indexed by its configuration keys -- the degradation ladder is
    one -- loses its row labels entirely if the index is discarded, and a table
    of unlabelled numbers is worse than no table. Pass ``index=`` explicitly to
    override.
    """
    if df.empty:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    keep_index = not isinstance(df.index, pd.RangeIndex)
    if keep_index:
        # Flatten to plain columns rather than relying on to_latex's multirow
        # handling, which needs \multirow in the preamble to compile.
        df = df.reset_index()
        keep_index = False

    # Sanitize headers and string cells to ASCII before LaTeX escaping.
    df = df.rename(columns=_ascii)
    obj_cols = df.select_dtypes(include="object").columns
    if len(obj_cols):
        df[obj_cols] = df[obj_cols].apply(lambda c: c.map(_ascii))

    kwargs = {
        "index": keep_index,
        "escape": True,
        "na_rep": "--",
        "float_format": "%.4g",
    }
    kwargs.update(to_latex_kwargs)
    body = _ascii(df.to_latex(**kwargs))

    if caption:
        label = label or path.stem
        # `max width` scales the tabular only when it would overrun the text
        # block, so narrow tables keep the body font while the wide sweeps
        # (device_latency at 11 columns, nms_substitution_summary at 15) fit
        # instead of running into the margin. Unconditional \resizebox would
        # shrink every table, including the ones that already fit.
        body = (
            # [htbp] rather than [t]: top-only placement cannot satisfy a
            # chapter carrying ~19 floats over ~18 pages, because a table that
            # misses the top of the current page defers to the next one and the
            # backlog cascades for the rest of the chapter. Allowing `h` lets a
            # table set in the prose that discusses it, and `p` gives the wide
            # sweeps somewhere to go instead of displacing body text.
            "\\begin{table}[htbp]\n\\centering\n"
            f"\\caption{{{_ascii(caption)}}}\n\\label{{tab:{label}}}\n"
            "\\begin{adjustbox}{max width=\\linewidth}\n"
            f"{body}"
            "\\end{adjustbox}\n"
            "\\end{table}\n"
        )
    path.write_text(body)
