"""
Benchmark result aggregation and publication-level reporting.

This is the evaluation-time counterpart to :mod:`agri_vision_edge.evaluation.curves`:
where ``curves`` charts *training* scalars over steps, this module aggregates the
*evaluation* artifacts written by ``bin/benchmark_tflite.py`` (``latency.json`` /
``runtime.json``) and ``bin/evaluate_coco.py`` (``metrics.json``) into one tidy
frame, then renders the figures and LaTeX tables used in the thesis.

Layout consumed::

    benchmark_results/<platform>/<run>/{metrics,latency,runtime}.json

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
from collections.abc import Iterable
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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
}
CLASS_LABELS = {
    "sc": "Single-class (weed)",
    "mc": "Multi-class (crop+weed)",
}
PRECISION_LABELS = {"fp32": "FP32", "fp16": "FP16", "int8": "INT8"}
EVAL_TILING_LABELS = {
    "tiled": "Tiled input",
    "untiled": "Full-frame input",
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
    fmt="{:.3f}",
    rotation=0,
    horizontal=False,
):
    """
    Draw side-by-side bar groups.

    Args:
        values:
            Mapping ``{group_label: [value per category]}``.
        horizontal:
            Put the categories on the **y** axis and the bars along x. Worth it
            once the category labels stop fitting side by side: a vertical
            chart has to share one figure width between every category, while a
            horizontal one gives each label a full line and grows downward
            instead (see :func:`plot_quantization_effect`).
    """
    n_groups = max(len(group_labels), 1)
    x = np.arange(len(categories))
    width = 0.8 / n_groups

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

                if horizontal:
                    xy = (v, bar.get_y() + bar.get_height() / 2)
                    align = {"ha": "left", "va": "center"}
                    offset_pts = (2, 0)
                else:
                    xy = (bar.get_x() + bar.get_width() / 2, v)
                    align = {"ha": "center", "va": "bottom"}
                    offset_pts = (0, 1)

                ax.annotate(
                    fmt.format(v),
                    xy,
                    fontsize=7,
                    xytext=offset_pts,
                    textcoords="offset points",
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


#: Host whose CPU runs stand in for "unaccelerated reference" everywhere.
CPU_REFERENCE_PLATFORM = "gaia"

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
            "eval_tiling",
            "precision",
            "quant",
            "granularity",
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
):
    """
    All five quantization schemes side by side, one bar group per variant.

    This is the figure for the PTQ-vs-QAT question, and unlike an aggregated
    FP32-vs-INT8 view it shows each export on its own: a single broken scheme
    stays visible instead of being averaged into an "INT8" bar.

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

    sel = df.copy()
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

    sel["group"] = (
        _short(sel["arch_label"])
        + " | "
        + sel["classes"].str.upper()
        + " | "
        + sel["dataset"]
    )
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
        title=(
            f"{metric} by quantization scheme"
            + (f" ({eval_tiling} input)" if eval_tiling else "")
            + (f" — {platform}" if platform else " — mean over platforms")
        ),
        rotation=20,
    )
    ax.legend(title="scheme", ncol=2)
    fig.tight_layout()
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

    sel = df[df["precision"] == "int8"].copy()
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

    sel["group"] = (
        _short(sel["arch_label"])
        + " | "
        + sel["classes"].str.upper()
        + " | "
        + sel["dataset"]
        + " | "
        + sel["scheme"].str.replace("int8_", "", regex=False)
    )
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
        title=(
            f"INT8 {metric}: CPU reference vs NPU"
            + (f" ({eval_tiling} input)" if eval_tiling else "")
        ),
        horizontal=True,
    )
    ax.legend(title="backend", loc="lower right")
    fig.tight_layout()
    return fig


def plot_single_vs_multiclass(df: pd.DataFrame):
    """Weed AP under single-class vs multi-class regimes."""
    if df.empty or df["classes"].nunique() < 2:
        return None

    df = df.copy()
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
        title="Weed AP: single-class vs. multi-class",
        rotation=20,
    )
    ax.legend(
        title="regime",
        labels=["single-class", "multi-class"][: len(class_order)],
    )
    fig.tight_layout()
    return fig


def plot_architecture_effect(df: pd.DataFrame):
    """Overall AP per architecture (SSD MobileNetV2 vs FPNLite)."""
    if df.empty or df["arch"].nunique() < 2:
        return None

    df = df.copy()
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
        title="Architecture comparison (overall AP)",
        rotation=15,
    )
    ax.legend(title="architecture")
    fig.tight_layout()
    return fig


def plot_per_class_ap(df: pd.DataFrame):
    """Crop vs weed AP for multi-class runs."""
    if "crop_AP" not in df:
        return None
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
        title="Per-class AP (multi-class runs)",
        rotation=15,
    )
    ax.legend(title="class")
    fig.tight_layout()
    return fig


def plot_ap_by_area(df: pd.DataFrame):
    """AP broken down by COCO object area (small / medium / large)."""
    if df.empty:
        return None

    df = df.copy()
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
        title="AP by object area",
        rotation=25,
    )
    ax.legend(title="object size", labels=["small", "medium", "large"])
    fig.tight_layout()
    return fig


def plot_latency(df: pd.DataFrame):
    """Horizontal mean-latency bars with min/max whiskers, per run."""
    if "mean_latency_ms" not in df:
        return None
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


def plot_accuracy_latency(df: pd.DataFrame):
    """Accuracy / latency scatter; colour = precision, marker = architecture."""
    if "mean_latency_ms" not in df:
        return None
    df = df[df["mean_latency_ms"].notna() & df["AP"].notna()].copy()
    if df.empty:
        return None

    from matplotlib.lines import Line2D

    markers = {"ssd-mn2": "o", "ssd-mn2-fpnlite": "s"}
    fig, ax = plt.subplots(figsize=(7, 5))
    for _, r in df.iterrows():
        ax.scatter(
            r["mean_latency_ms"],
            r["AP"],
            color=PRECISION_COLORS.get(r["precision"], "#999999"),
            marker=markers.get(r["arch"], "^"),
            s=70,
            edgecolor="black",
            linewidth=0.5,
            zorder=3,
        )
        ax.annotate(
            f"  {r['config']}",
            (r["mean_latency_ms"], r["AP"]),
            fontsize=7,
            va="center",
        )
    ax.set_xlabel("mean latency (ms)")
    ax.set_ylabel("AP")
    ax.set_title("Accuracy / latency trade-off")
    _prepare_axis(ax)

    prec_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label=p.upper(),
            markerfacecolor=c,
            markersize=9,
            markeredgecolor="black",
        )
        for p, c in PRECISION_COLORS.items()
        if p in df["precision"].values
    ]
    arch_handles = [
        Line2D(
            [0],
            [0],
            marker=m,
            color="w",
            label=ARCH_LABELS.get(a, a),
            markerfacecolor="#cccccc",
            markersize=9,
            markeredgecolor="black",
        )
        for a, m in markers.items()
        if a in df["arch"].values
    ]
    leg1 = ax.legend(handles=prec_handles, title="precision", loc="lower right")
    ax.add_artist(leg1)
    ax.legend(handles=arch_handles, title="architecture", loc="upper right")
    fig.tight_layout()
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
    metrics: Iterable[tuple[str, str]] = _SCHEME_METRICS,
) -> pd.DataFrame:
    """
    PTQ vs. QAT per model variant: one row per (variant, scheme).

    This is the table the quantization chapter is built on, so it also carries
    ``dAP vs fp32`` -- the whole point is how far each INT8 export falls behind
    (or ahead of) its own float baseline, which raw AP columns make the reader
    compute by hand.

    Rows are pycocotools metrics (our internally consistent numbers), not the
    upstream ones; mixing the two in a single table would be meaningless.
    """
    if df.empty:
        return pd.DataFrame()

    sel = df.copy()
    if platform is not None:
        sel = sel[sel["platform"] == platform]
    if eval_tiling is not None:
        sel = sel[sel["eval_tiling"] == eval_tiling]
    if sel.empty:
        return pd.DataFrame()

    sel = add_scheme(sel)

    variant_keys = ["platform", "arch_label", "classes", "dataset", "eval_tiling"]
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
                    None
                    if col not in r or pd.isna(r[col])
                    else round(float(r[col]), 4)
                )
            row["dAP vs fp32"] = (
                None
                if baseline_ap in (None, 0) or pd.isna(r.get("AP"))
                else round(100 * (float(r["AP"]) - baseline_ap) / baseline_ap, 1)
            )
            rows.append(row)

    return pd.DataFrame(rows)


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
        ("eval_tiling", "Eval input"),
        ("scheme", "Scheme"),
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
        n: 4 for c, n in cols if c in ("AP", "AP50", "AP75", "crop_AP", "weed_AP", "APS")
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

    keys = [
        k
        for k in ("platform", "backend", "arch_label", "scheme")
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
                "scheme": "Scheme",
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

#: Relative AP gap (per-channel vs per-tensor) worth reporting: per-channel is
#: the finer scheme and should never be materially *worse*.
GRANULARITY_INVERSION_FRACTION = 0.1


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
    * ``granularity-inversion`` -- per-channel scoring materially below
      per-tensor for the same quantization scheme, which is backwards: the
      finer granularity is strictly more expressive.
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

    variant_keys = [
        k
        for k in ("platform", "arch", "classes", "dataset", "eval_tiling")
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

        # per-channel should never be materially worse than per-tensor
        for quant, quant_group in group.groupby("quant", dropna=False):
            pt = quant_group[quant_group.get("granularity") == "per-tensor"]
            pc = quant_group[quant_group.get("granularity") == "per-channel"]
            if pt.empty or pc.empty:
                continue
            pt_ap, pc_ap = pt["AP"].iloc[0], pc["AP"].iloc[0]
            if pd.isna(pt_ap) or pd.isna(pc_ap) or not pt_ap:
                continue
            if pc_ap < (1 - GRANULARITY_INVERSION_FRACTION) * pt_ap:
                add(
                    "error",
                    "granularity-inversion",
                    pc["run"].iloc[0],
                    pc["platform"].iloc[0],
                    f"{quant} per-channel AP {pc_ap:.3f} < per-tensor "
                    f"{pt_ap:.3f} ({100 * (pc_ap - pt_ap) / pt_ap:+.0f}%)",
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
                f"official mAP {r['faithful_mAP']:.3f} vs pycocotools AP "
                f"{r['AP']:.3f}",
            )

    if not issues:
        return pd.DataFrame(
            columns=["severity", "check", "platform", "run", "detail"]
        )

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


def quantization_delta_table(df: pd.DataFrame) -> pd.DataFrame:
    """FP32 vs INT8 AP with relative change, one row per config (PTQ runs).

    INT8 is the per-tensor export (the default deployment granularity), so each
    config yields a single FP32/INT8 delta row.
    """
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
        for k in ("platform", "arch_label", "class_label", "dataset", "eval_tiling")
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


#: Results tree holding the pre-conversion SavedModel reference. The *floored*
#: one: its NMS score threshold matches the TFLite graphs', which is what keeps
#: the first rung like-for-like. (`tf-savedmodel-nms0` is the floor-free control
#: and is worth +0.002 AP on average -- see CLAUDE.md.)
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
    Decompose the deployment chain into its four rungs and three losses.

    ::

        SavedModel  --conversion-->  TFLite fp32  --quantization-->  int8 CPU
                    --delegation-->  int8 NPU

    Each delta isolates one transformation by holding the others fixed, which is
    the point: without the SavedModel rung the conversion loss is invisible and
    gets folded into "quantization". It is not small -- averaged over the
    variants it is about a third of the quantization loss, and for at least one
    config it is *larger*.

    Two things to keep in mind when reading the first column:

    * **Conversion is not only numerical.** The TFLite export swaps TFOD's
      post-processing for ``TFLite_Detection_PostProcess``, a different NMS
      implementation, so this rung measures an algorithm substitution as much as
      a precision change.
    * **Resampling is folded in.** The SavedModel resizes inside the graph
      (``fixed_shape_resizer``) while the TFLite path resizes externally with
      ``cv2``, so part of the gap is interpolation. Splitting the two needs a
      pre-resized control run.

    Rows are dropped only when a rung is missing entirely; a missing reference
    leaves the conversion column empty rather than removing the config, so an
    incomplete reference sweep is visible instead of silently narrowing the
    table.
    """
    if df.empty or metric not in df.columns:
        return pd.DataFrame()

    sel = add_scheme(df.copy())

    if eval_tiling is not None and "eval_tiling" in sel.columns:
        sel = sel[sel["eval_tiling"] == eval_tiling]

    keys = [k for k in ("arch_label", "classes", "dataset") if k in sel.columns]
    if not keys:
        return pd.DataFrame()

    def _rung(platform, scheme=None, quantization=None):
        rows = sel[sel["platform"] == platform]
        if scheme is not None:
            rows = rows[rows["scheme"] == scheme]
        if quantization is not None:
            rows = rows[rows["quant"] == quantization]
        rows = rows.dropna(subset=[metric])
        if rows.empty:
            return pd.Series(dtype="float64")
        return rows.groupby(keys)[metric].mean()

    int8_scheme = scheme_name("int8", quant, "per-tensor")

    rungs = {
        "SavedModel": _rung(reference, quantization=quant),
        "TFLite fp32": _rung(cpu_platform, scheme=scheme_name("fp32", "ptq")),
        "int8 CPU": _rung(cpu_platform, scheme=int8_scheme),
        f"int8 NPU ({npu_platform})": _rung(npu_platform, scheme=int8_scheme),
    }

    table = pd.DataFrame(rungs)

    # A config needs at least the deployed chain to be worth a row.
    table = table.dropna(subset=["TFLite fp32", "int8 CPU"], how="any")
    if table.empty:
        return pd.DataFrame()

    columns = list(table.columns)
    table["conversion"] = table["TFLite fp32"] - table["SavedModel"]
    table["quantization"] = table["int8 CPU"] - table["TFLite fp32"]
    table["delegation"] = table[columns[3]] - table["int8 CPU"]

    return table.round(4)


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
        ("eval_tiling", "Eval input"),
        ("precision", "Precision"),
        ("quant", "Quant"),
        ("granularity", "Granularity"),
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
DEFAULT_EXPECTED_PLATFORMS = ("gaia", "imx8mp", "imx93")


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


def build_coverage(
    runs_df: pd.DataFrame,
    variants_df: pd.DataFrame,
    schemes: Iterable[tuple[str, str, str | None]] = DEFAULT_SCHEMES,
    platforms: Iterable[str] = DEFAULT_EXPECTED_PLATFORMS,
    eval_tilings: Iterable[str] = DEFAULT_EVAL_TILINGS,
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

    Platforms already present in ``runs_df`` are always included, so unexpected
    platforms still surface. Matching ignores NMS / split.
    """
    if variants_df.empty:
        return pd.DataFrame()

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

    platform_list = list(
        dict.fromkeys(
            [*platforms, *(runs_df["platform"].unique() if not runs_df.empty else [])]
        )
    )

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
        fig.savefig(fig_dir / f"{stem}.{ext}")


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
    """Write ``df`` as a (optionally wrapped) booktabs LaTeX table (ASCII-only)."""
    if df.empty:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Sanitize headers and string cells to ASCII before LaTeX escaping.
    df = df.rename(columns=_ascii)
    obj_cols = df.select_dtypes(include="object").columns
    if len(obj_cols):
        df[obj_cols] = df[obj_cols].apply(lambda c: c.map(_ascii))

    kwargs = {"index": False, "escape": True, "na_rep": "--", "float_format": "%.4g"}
    kwargs.update(to_latex_kwargs)
    body = _ascii(df.to_latex(**kwargs))

    if caption:
        label = label or path.stem
        body = (
            "\\begin{table}[t]\n\\centering\n"
            f"\\caption{{{_ascii(caption)}}}\n\\label{{tab:{label}}}\n"
            f"{body}\\end{{table}}\n"
        )
    path.write_text(body)
