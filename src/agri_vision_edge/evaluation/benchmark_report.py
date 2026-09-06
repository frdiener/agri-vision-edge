"""Aggregate benchmark artifacts into analysis frames, figures and LaTeX tables.

Reads ``benchmark_results/<platform>/<run>`` artifacts and per-platform
``resize.json`` measurements. Run names encode the evaluation and export
configuration used by :func:`parse_run_name`.
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

# Use the same axis styling as training-curve figures.
from .curves import _prepare_axis

# =========================================================
# Run-name parsing
# =========================================================

_PRECISIONS = {"fp32", "fp16", "int8", "int16", "dynamic"}
_SPLITS = {"val", "test", "train", "eval"}
_CLASSES = {"sc", "mc"}
#: INT8 weight granularity, kept separate from the dataset token.
_GRANULARITIES = {"per-tensor", "per-channel"}
#: Evaluation input regime stored as the leading run-name token.
_EVAL_TILINGS = {"tiled", "untiled"}

ARCH_LABELS = {
    "ssd-mn2": "SSD MobileNetV2",
    "ssd-mn2-fpnlite": "SSD MobileNetV2 FPNLite",
    # Auxiliary detectors share the deployment pipeline but are not primary comparison models.
    "yolov7-tiny": "YOLOv7-tiny",
    "yolox-nano": "YOLOX-Nano",
}

#: Architectures used for controlled comparisons.
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

#: Default export uses class-agnostic fast NMS.
DEFAULT_NMS = "fastnms"

#: Per-class NMS export; also used as the checkpoint-matched control.
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
    """Map a run-name token to a configuration field."""
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
    """Parse a run directory or bare variant name into configuration fields.

    Returns ``None`` when no ``sc`` or ``mc`` class token is present.
    """
    tokens = name.split("_")

    # Remove the evaluation-tiling prefix before parsing the architecture.
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

    # Include training dataset and evaluation regime to keep run groups distinct.
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
    """Extract latency statistics and derive FPS from median latency."""

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
        # Large max/median ratios flag disturbed timing runs.
        median = fields.get("median_latency_ms") or float(np.median(samples))
        if median:
            fields["latency_outlier_ratio"] = float(max(samples)) / float(median)
            fields["fps"] = 1000.0 / float(median)

    return fields


def _runtime_fields(runtime: dict) -> dict:
    """Extract the requested delegate and effective execution backend.

    Older artifacts without an effective backend are reported as ``unknown``.
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
        # Older artifacts without ``format`` are TFLite.
        "format": runtime.get("format", "tflite"),
        "input_dtype": _input_dtype(runtime),
    }


def _input_dtype(runtime: dict) -> str | None:
    """Return the first input dtype from TFLite or SavedModel runtime metadata."""
    details = runtime.get("input_details") or []

    if not details:
        return None

    raw = details[0].get("dtype") or ""
    parts = raw.split("'")

    return parts[-2] if len(parts) >= 2 else (raw or None)


def _faithful_fields(faithful: dict | None, *, classes: str | None) -> dict:
    """Normalize official PhenoBench metrics to 0-1 and add class metrics.

    Older single-class artifacts without ``class_names`` are marked stale. Use
    ``faithful_mAP_plants`` for comparable plant-only aggregate AP.
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

    # Older single-class artifacts predate the label remap and are invalid.
    fields["faithful_stale"] = "class_names" not in faithful and classes == "sc"
    fields["faithful_partial_classes"] = bool(
        predicted is not None and len(predicted) < len(names)
    ) or (predicted is None and classes == "sc")

    return fields


def _pct(value) -> float | None:
    """Convert an upstream percentage to a 0-1 fraction."""
    return None if value is None else float(value) / 100.0


def load_benchmark_results(
    root: str | Path = "benchmark_results",
    exclude_dirs: Iterable[str] = NON_PLATFORM_DIRS,
) -> tuple[pd.DataFrame, list[str]]:
    """Load benchmark runs into one frame.

    Returns ``(df, skipped)``; ``skipped`` records failed, invalid, missing and
    unparsed runs.
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
                # ``metrics_invalid.json`` marks runs rejected by the prediction integrity check.
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
            # ``<board>_cpu`` is the same device with delegation disabled.
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
    """Filter to one NMS variant while retaining rows without an NMS token.

    Use before aggregations that do not group by ``nms``. Pass ``None`` to keep
    all variants.
    """
    if df.empty or nms is None or "nms" not in df.columns:
        return df
    return df[df["nms"].isna() | (df["nms"] == nms)]


# =========================================================
# Preprocessing cost
# =========================================================

#: Per-platform resize measurements written by ``scripts/benchmark_resize.py``.
#: Stored as a file so the run-directory scan ignores it.
RESIZE_ARTIFACT = "resize.json"

#: Latency statistics corrected for resize cost; min/max are not corrected.
_CORRECTED_LATENCY_COLUMNS = {
    "median_latency_ms": "median_latency_ms_net",
    "mean_latency_ms": "mean_latency_ms_net",
    "p95_latency_ms": "p95_latency_ms_net",
}


def load_resize_costs(
    root: str | Path = "benchmark_results",
) -> pd.DataFrame:
    """Load measured resize costs, one row per platform, input regime and size.

    ``size`` is stored as a string to match parsed run metadata.
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
    """Return resize-artifact lookup keys for a results-tree name, exact match first."""
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
    """Attach measured resize cost and latency with that cost removed.

    Unmatched rows remain ``NaN``. SavedModel rows have no external ``cv2.resize``
    cost to subtract.
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

        # Clamp invalid negative corrected latencies to zero.
        out[corrected] = (
            pd.to_numeric(out[column], errors="coerce") - out["resize_ms"]
        ).clip(lower=0.0)

    net = out["median_latency_ms_net"]
    out["fps_net"] = np.where(net > 0, 1000.0 / net, np.nan)

    # Fraction of measured median latency spent resizing.
    if "median_latency_ms" in out.columns:
        measured = pd.to_numeric(out["median_latency_ms"], errors="coerce")
        out["resize_share"] = out["resize_ms"] / measured.where(measured > 0)

    return out


def resize_cost_table(
    costs: pd.DataFrame | str | Path = "benchmark_results",
) -> pd.DataFrame:
    """Format measured resize costs by host, source size and model input size."""
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
    """Apply the shared publication plotting style."""
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
    """Draw grouped bars.

    ``percent`` scales fractional values by 100. ``horizontal`` places categories
    on the y axis. ``fmt`` controls value labels.
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

    # Rotate value labels when vertical grouped bars become too dense.
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

                # Place labels beyond the bar tip, including negative bars.
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
        # Anchor rotated category labels at their ticks.
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
    """Place a legend to the right of the axes and reserve its width.

    Call after ``tight_layout``.
    """
    legend = ax.legend(
        loc="upper left", bbox_to_anchor=(1.01, 1.0), borderaxespad=0.0, **kwargs
    )
    reserve_legend_space(fig)

    return legend


def reserve_legend_space(fig, pad: float = 0.04, passes: int = 3) -> None:
    """Reserve figure width for the widest outside legend."""
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
    """Build labels from architecture and configuration fields that vary."""
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
    """Plot FP32 versus per-tensor PTQ INT8 AP and weed AP.

    Optional ``platform`` and ``eval_tiling`` arguments restrict the comparison.
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


#: CPU reference tree used for cross-device correctness checks.
CPU_REFERENCE_PLATFORM = "x86_cpu"

#: Metrics compared when checking that every CPU tree agrees with the reference.
CPU_REFERENCE_METRICS = ("AP", "AP50", "AP75", "weed_AP", "crop_AP")

#: Tolerance for CPU-reference metric differences.
CPU_REFERENCE_TOLERANCE = 1e-5


def cpu_reference_divergence(
    df: pd.DataFrame,
    *,
    reference: str = CPU_REFERENCE_PLATFORM,
    metrics: Iterable[str] = CPU_REFERENCE_METRICS,
) -> pd.DataFrame:
    """Compare CPU result trees with the reference over shared configurations.

    Returns per-platform metric differences. An empty frame means no comparison was
    possible.
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
        # Replace missing key values so FP32 rows survive index matching.
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
    """Return whether all CPU-reference differences are below ``tolerance``."""
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
    """Plot ``metric`` across export schemes for each model variant.

    The default fixes the CPU reference platform and one NMS variant. Set
    ``platform=None`` only when a cross-platform mean is intended.
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
            + (f" - {platform}" if platform else " - mean over platforms")
        ),
        rotation=20,
    )
    # Keep the legend outside the data region.
    fig.tight_layout()
    _legend_outside(fig, ax, title="scheme")
    return fig


#: Backend comparison colours: CPU reference first, then NPU targets.
BACKEND_COLORS = ("#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3")


def plot_backend_effect(
    df: pd.DataFrame,
    *,
    eval_tiling: str | None = "untiled",
    metric: str = "AP",
    reference: str = CPU_REFERENCE_PLATFORM,
    nms: str | None = DEFAULT_NMS,
):
    """Plot INT8 accuracy on the CPU reference and each NPU delegate."""
    if df.empty or metric not in df.columns or "backend" not in df.columns:
        return None

    sel = select_nms(df[df["precision"] == "int8"].copy(), nms)
    if eval_tiling is not None and "eval_tiling" in sel.columns:
        sel = sel[sel["eval_tiling"] == eval_tiling]
    if sel.empty:
        return None

    sel = add_scheme(sel)

    # One series for the CPU reference and one for each delegated target.
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
    """Plot weed AP for single-class and multi-class models."""
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
    """Plot overall AP by architecture."""
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
    """Plot crop and weed AP for multi-class runs."""
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
    """Plot COCO AP by object area."""
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
    """Plot per-run mean latency with min/max whiskers."""
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
    ax.set_xlabel("mean latency (ms), min/max whiskers")
    ax.set_title("Inference latency per run")
    _prepare_axis(ax)
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    return fig


#: Reference-only trees excluded from deployment latency comparisons.
NON_TARGET_PLATFORMS = ("tf-savedmodel", "tf-savedmodel-nms0", CPU_REFERENCE_PLATFORM)


def platform_label(platform: str) -> str:
    """Convert a results-tree name to a short board/backend label."""
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


#: Marker identifies the full export scheme; colour identifies platform.
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
    """Plot AP versus latency, faceted by architecture.

    Colour encodes platform and marker shape encodes export scheme.
    """
    if "mean_latency_ms" not in df:
        return None

    from matplotlib.lines import Line2D

    sel = add_scheme(select_nms(df, nms))
    sel = sel[~sel["platform"].isin(set(exclude_platforms))]
    # Exclude alternate delegate builds from deployment comparisons.
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
    # Legends are on different axes; adding the first legend as an artist would draw it twice.
    _legend_outside(fig, axes[0], handles=platform_handles, title="platform")
    axes[-1].legend(
        handles=scheme_handles,
        title="export scheme",
        loc="lower left",
        bbox_to_anchor=(1.01, 0.0),
        borderaxespad=0.0,
    )
    # Recompute reservation after both legends exist.
    reserve_legend_space(fig)
    return fig


# =========================================================
# Thesis tables
# =========================================================

#: Published PhenoBench plant-detection baselines, in percentage units.
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
    """Compare local runs with published PhenoBench baselines.

    Uses official metrics and restricts local rows to multi-class, full-frame
    training and full-frame evaluation. Published baselines use the test split.
    """
    if df.empty or "faithful_mAP" not in df.columns:
        return pd.DataFrame()

    # Use one NMS export in leaderboard-style comparisons.
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
    """Build one row per model variant and export scheme.

    Uses pycocotools metrics and includes AP change relative to that variant's FP32
    baseline. NMS is fixed unless ``nms=None``.
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
        # Drop pinned configuration columns that repeat one value.
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
    """Format full COCO metrics for one platform."""
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
    """Format median and p95 latency plus throughput by platform, backend and scheme."""
    if df.empty or "median_latency_ms" not in df.columns:
        return pd.DataFrame()

    sel = add_scheme(df[df["median_latency_ms"].notna()].copy())
    if sel.empty:
        return pd.DataFrame()

    # Keep NMS as a grouping key; the variants are separate graphs.
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

#: Quantized runs below this share of FP32 AP are classified as collapsed.
QUANT_COLLAPSE_FRACTION = 0.5

#: Numerical tolerance for the single-class NMS null control.
NMS_CONTROL_TOLERANCE = 1e-6


def sanity_checks(df: pd.DataFrame) -> pd.DataFrame:
    """Check benchmark data for invalid or suspect runs.

    Checks quantization collapse, backend fallback, FP32 delegation, latency
    outliers, stale or divergent official metrics, and the single-class NMS control.
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

    # Match quantized runs to FP32 baselines with the same NMS variant.
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

        # Only an explicit True marks official metrics as stale.
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

    # Single-class fast and regular NMS are the same algorithm; any delta indicates a bad pair.
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
                "algorithm, so this measures degraded execution, not "
                "post-processing",
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
    """Count sanity-check issues by check and severity."""
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
    """Compare FP32 with per-tensor PTQ INT8 AP for matched configurations."""
    df = select_nms(df, nms)
    df = df[df.get("quant") == "ptq"] if "quant" in df else df
    if "granularity" in df.columns:
        df = df[(df["precision"] == "fp32") | (df["granularity"] == "per-tensor")]
    if df.empty:
        return pd.DataFrame()

    # Group on the complete run identity before selecting one row per configuration.
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

#: Deployability verdicts ordered from worst to best.
DEPLOYABILITY_VERDICTS = ("failed", "unscoreable", "collapsed", "degraded", "ok", "-")

#: AP share below which a delegated run is classified as collapsed.
DEPLOYABILITY_COLLAPSE_FRACTION = 0.5

#: Smaller CPU/delegate AP disagreement is classified as degraded.
DEPLOYABILITY_DEGRADED_FRACTION = 0.9

#: Reference configuration for analyses that do not vary tiling or resolution.
REFERENCE_CONFIG = {
    "classes": "mc",
    "dataset": "phenobench",
    "eval_tiling": "untiled",
    #: Resolution studies also pin the input size when selecting one cell.
    "size": "320",
}

#: Dimensions allowed to collapse for cost metrics.
#: Class count changes only the predictor head; training tiling changes weights, not graph shape;
#: evaluation tiling changes preprocessing but not model compute.
COLLAPSIBLE_DIMENSIONS = ("classes", "dataset", "eval_tiling")

#: Architecture and export scheme remain explicit analysis axes.
KEPT_DIMENSIONS = ("arch", "scheme")

#: Display names for collapse-audit grouping fields.
_GROUP_LABELS = {"platform": "Platform", "arch": "Architecture"}

DIMENSION_LABELS = {
    "classes": "single- vs multi-class",
    "dataset": "trained untiled vs tiled",
    "eval_tiling": "evaluated untiled vs tiled",
    "arch": "ssd-mn2 vs fpnlite",
    "scheme": "export scheme",
}


def reference_config_slice(df: pd.DataFrame, **overrides) -> pd.DataFrame:
    """Filter to :data:`REFERENCE_CONFIG`.

    Pass a field as ``None`` to leave that axis unpinned.
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
    """Return rows whose delegated accuracy collapsed against the matching CPU run."""
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
    """Remove delegated runs classified as failed deployments."""
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
    """Measure variation from collapsing each configuration dimension.

    For each dimension, computes relative within-group spread while holding all
    other configuration fields fixed.
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
    """Return whether every collapsed dimension varies less than every retained dimension."""
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
    """Measure directional bias from each collapsed dimension.

    Changes are relative to the reference configuration; negative values mean the
    off-reference configuration lowers the metric.
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
    """Summarize latency after collapsing selected configuration dimensions.

    Includes the reference-cell latency, collapsed mean, spread and directional
    bias.
    """
    if df.empty or "median_latency_ms" not in df.columns:
        return pd.DataFrame()

    sel = add_scheme(drop_failed_deployments(df)).dropna(subset=["median_latency_ms"])

    # Pin every non-collapsed dimension to its reference value.
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
    """Parse skipped-run strings into ``platform``, ``run`` and ``reason`` columns."""
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
    """Classify each export/platform cell as ``ok``, ``degraded``, ``collapsed`` or ``unscoreable``.

    Accuracy verdicts use the matching CPU run. ``skipped`` supplies runs rejected
    before metrics were written.
    """
    if runs_df.empty:
        return pd.DataFrame()

    sel = add_scheme(select_nms(runs_df, nms))
    if eval_tiling is not None and "eval_tiling" in sel.columns:
        sel = sel[sel["eval_tiling"] == eval_tiling]

    keys = ["arch_label", "classes", "dataset", "size", "scheme"]
    if any(k not in sel.columns for k in keys):
        return pd.DataFrame()

    # Without a CPU reference, accuracy verdicts cannot be classified.
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

    # Apply the caller's scope to skipped runs before adding unscoreable cells.
    in_scope = set(sel[keys].itertuples(index=False, name=None))

    # Add benchmarked runs rejected before metrics were written.
    for _, s in _skipped_frame(skipped).iterrows():
        info = parse_run_name(s["run"])
        if info is None:
            continue
        if eval_tiling is not None and info.get("eval_tiling") != eval_tiling:
            continue
        if nms is not None and info.get("nms") not in (None, nms):
            continue
        # Build skipped-run keys from the same configuration identity.
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
        # Drop scoped columns that repeat one value.
        constant = [
            k
            for k in ("classes", "dataset", "size")
            if k in table.columns and table[k].nunique(dropna=False) <= 1
        ]
        table = table.drop(columns=constant)

    return table


#: Configuration identity used to pair NMS variants.
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
    """Compare current and suffixed delegate builds per architecture.

    ``latency_scope="ok-both"`` uses correct runs in both trees; ``"paired"`` uses
    all configurations timed by both.
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
        """Return median latency per timed configuration, indexed by ``keys``."""
        if subset.empty:
            return pd.Series(dtype=float)
        rows = lat[lat["platform"] == platform].merge(subset[keys], on=keys)
        rows = rows.dropna(subset=["median_latency_ms"])
        if rows.empty:
            return pd.Series(dtype=float)
        return rows.groupby(keys)["median_latency_ms"].median()

    def _paired(older, board, subset):
        """Return paired medians and the number of configurations timed by both frames."""
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
    """Count deployability verdicts per platform."""
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

#: Pairing keys for fast and regular NMS; excludes only ``nms``.
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
    """Compare matched fast-NMS and regular-NMS exports.

    Single-class pairs are a null control and should have zero metric delta.
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

        # Remove rounded negative zero from null-control deltas.
    return (out.round(4) + 0.0).reset_index()


def nms_pair_coverage(df: pd.DataFrame) -> pd.DataFrame:
    """Count configurations containing both NMS variants per platform."""
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
    """Summarize NMS metric deltas by architecture, class regime and input regime.

    Constant grouping columns may be dropped after caller-side filtering.
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
        # Architecture remains explicit because NMS behavior is architecture-dependent.
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
    """Estimate NMS latency cost with a single-class difference-in-differences control.

    ``saving = mean(dLatency | mc) - mean(dLatency | sc)``. Reports standard error,
    sigma and a 95% confidence interval per platform and architecture.
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

        # Standard error for a difference of independent means; undefined with fewer than two samples per arm.
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
    """Plot per-class AP change from fast versus regular NMS."""
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
    # Put the legend outside because all bars extend left from zero.
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
    """Build accuracy and device-latency rows across input resolutions.

    All non-resolution axes are fixed. ``latency_platforms`` adds matched target
    latency columns.
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
        # Join target latency from the unscoped frame; accuracy rows are reference-platform rows.
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
            # Throughput is measured on the target platform.
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


#: SavedModel reference with the same score floor as the TFLite graphs.
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
    """Decompose deployment into conversion, NMS, quantization and delegation deltas.

    Each delta changes one rung while holding the others fixed. Missing reference
    or control rungs remain empty; rows require the deployed TFLite chain.
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
        """Return one rung's metric per configuration.

        ``strict`` requires an explicit matching NMS token.
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
        # One row is expected per config; ``first`` exposes duplicates instead of averaging them.
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

    # Require the deployed chain for each row.
    table = table.dropna(subset=[deployed_label, "int8 CPU"], how="any")
    if table.empty:
        return pd.DataFrame()

    table["conversion"] = table[control_label] - table["SavedModel"]
    table["nms-swap"] = table[deployed_label] - table[control_label]
    table["quantization"] = table["int8 CPU"] - table[deployed_label]
    table["delegation"] = table[npu_label] - table["int8 CPU"]

    # Remove rounded negative zero from reported deltas.
    return table.round(4) + 0.0


# =========================================================
# Deployment analysis stages
# =========================================================
# 1 baseline, 2 conversion/PTQ and NMS, 3 QAT, 4 deployability, 5 device latency, 6 ablations.
# Stages 1-5 use REFERENCE_CONFIG; stage 6 varies one axis at a time.

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
    """Build the float SavedModel baseline table at the reference configuration.

    Uses official PhenoBench metrics. ``sizes=None`` includes every available input
    resolution; optional published baselines use a different test split.
    """
    if df.empty or "faithful_mAP" not in df.columns:
        return pd.DataFrame()

    # Leave ``size`` open; pin the remaining reference axes.
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

    # Sort the string-valued input size numerically.
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
    """Return preparation stages as ``(label, scheme, nms, delta_base)`` tuples.

    INT8 stages are compared with the deployed float TFLite stage, not with one
    another.
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
    """Build conversion and quantization stages at the reference configuration.

    All TFLite stages use the selected ``nms`` variant. ``include_qat=False`` keeps
    only the PTQ path.
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
        # Only successive float stages update the running reference.
        if delta_base is None or delta_base == "prev":
            previous = current

    return pd.DataFrame(rows)


#: Minimum PTQ AP deficit used as the denominator for ``Reclaimed %``.
QAT_RECLAIM_MIN_DEFICIT = 0.1


def qat_reclaim_table(
    df: pd.DataFrame,
    *,
    cpu_platform: str = CPU_REFERENCE_PLATFORM,
    metric: str = "AP",
    percent: bool = True,
    nms: str | None = DEFAULT_NMS,
) -> pd.DataFrame:
    """Compare QAT with PTQ per architecture and weight granularity.

    ``Reclaimed %`` is omitted when the PTQ deficit is below
    :data:`QAT_RECLAIM_MIN_DEFICIT`.
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
                    # Suppress percentage reclaim when the PTQ deficit is below the threshold.
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
    """Return ``(delegate_tree, cpu_tree)`` pairs for boards with both backends.

    The delegated tree must contain runs executed with the delegate; control
    builds such as ``_unpatched`` are excluded.
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
    """Compare CPU and NPU latency on the same board for reference configurations.

    Failed deployments are excluded by default. Includes speedup, FPS and AP delta.
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
    """Plot CPU and NPU median latency by board, architecture and scheme."""
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


#: Reference deviations used by the ablation table.
ABLATION_AXES = (
    ("Reference (mc, trained full, eval full)", {}),
    ("Single-class", {"classes": "sc"}),
    ("Trained tiled", {"dataset": "phenobench-tiled"}),
    ("Evaluated tiled", {"eval_tiling": "tiled"}),
    ("Tiled end to end", {"dataset": "phenobench-tiled", "eval_tiling": "tiled"}),
)

#: Four training/evaluation tiling combinations used to expose their interaction.
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
    """Tabulate all training-tiling and evaluation-tiling combinations.

    Each metric delta is relative to full-frame training and full-frame evaluation
    for the same architecture.
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
    """Compare single-class and tiling deviations across the deployment stages.

    Requires both NMS variants for conversion and NMS-swap columns. Missing rungs
    remain empty.
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
    """Format the full benchmark matrix with thesis column names and rounding."""
    cols = [
        ("platform", "Platform"),
        ("arch_label", "Architecture"),
        ("classes", "Classes"),
        # Keep training dataset and evaluation input as separate columns.
        ("dataset", "Trained on"),
        ("size", "Input"),
        ("eval_tiling", "Eval input"),
        ("precision", "Precision"),
        ("quant", "Quant"),
        ("granularity", "Granularity"),
        # Keep NMS visible because each variant is a separate export.
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

#: Export schemes expected for each model variant in the default-NMS coverage matrix.
DEFAULT_SCHEMES = (
    ("fp32", "ptq", None),
    ("int8", "ptq", "per-tensor"),
    ("int8", "ptq", "per-channel"),
    ("int8", "qat", "per-tensor"),
    ("int8", "qat", "per-channel"),
)

#: Evaluation regimes expected for every model.
DEFAULT_EVAL_TILINGS = ("untiled", "tiled")

#: Platforms a full run targets: the dev host plus the two embedded NPU boards.
DEFAULT_EXPECTED_PLATFORMS = (CPU_REFERENCE_PLATFORM, "frdm-imx8mp", "frdm-imx93")


def scheme_label(scheme: str) -> str:
    """Return a compact display label for an export scheme."""
    if scheme == "fp32_ptq":
        return "fp32"
    if scheme == "fp32_qat":
        return "fp32 (QAT path)"

    return scheme.replace("int8_", "")


def scheme_name(precision, quant, granularity=None) -> str:
    """Return the canonical export-scheme token used in artifact names."""
    parts = [str(precision), str(quant)]
    if granularity and str(granularity) != "nan":
        parts.append(str(granularity))
    return "_".join(parts)


def add_scheme(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of ``df`` with a canonical ``scheme`` column."""
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
    """Enumerate trained variants from ``artifacts/tf`` directory names."""
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


#: Trees outside the TFLite conversion coverage matrix.
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
    """Build expected benchmark coverage across variants, schemes, input regimes and platforms.

    Coverage tracks the default NMS export. ``exclude_platforms`` removes trees
    outside the conversion matrix.
    """
    if variants_df.empty:
        return pd.DataFrame()

    runs_df = select_nms(runs_df, nms)

    def _granularity(value):
        # Normalize missing FP32 granularity to ``None`` for coverage keys.
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
    """Pivot coverage to a variant by platform/scheme grid using ``x`` and ``-``."""
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
    """Summarize completed and expected coverage per platform."""
    if coverage_long.empty:
        return pd.DataFrame()
    g = coverage_long.groupby("platform")["present"]
    out = pd.DataFrame({"done": g.sum(), "total": g.count()})
    out["percent"] = (100 * out["done"] / out["total"]).round(1)
    return out.reset_index()


# =========================================================
# Resource / power sweeps (a separate measurement path)
# =========================================================

#: Resource metrics reported by power sweeps; energy is listed first.
RESOURCE_METRICS = (
    ("net mJ/inf", "Net energy per inference (mJ)", True),
    ("net P (W)", "Net power (W)", True),
    ("lat mean (ms)", "Mean latency (ms)", False),
    ("RSS (MiB)", "Peak resident memory (MiB)", False),
)


def annotate_resource_runs(power_df: pd.DataFrame) -> pd.DataFrame:
    """Parse model configuration fields from resource-run names and add ``scheme``."""
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


def accelerator_latency_table(
    power_frame: pd.DataFrame,
    *,
    nms: str | None = None,
    size: str | None = None,
    schemes: Iterable[str] | None = None,
    verdicts: Iterable[str] | None = ("ok",),
    exclude_devices: str = "_unpatched|_no-concat",
) -> pd.DataFrame:
    """Compare whole-call and graph-invoke CPU/NPU speedups from resource sweeps.

    ``CPU-side (ms)`` is preprocess plus postprocess. Rows without phase timing or
    outside the requested verdict set are excluded.
    """
    needed = {"device", "invoke med (ms)", "lat med (ms)"}
    if power_frame.empty or not needed.issubset(power_frame.columns):
        return pd.DataFrame()

    sel = power_frame
    if exclude_devices:
        sel = sel[~sel["device"].str.contains(exclude_devices, na=False)]
    if nms is not None and "nms" in sel.columns:
        sel = sel[sel["nms"] == nms]
    if size is not None and "size" in sel.columns:
        sel = sel[sel["size"].astype(str) == str(size)]
    if schemes is not None and "scheme" in sel.columns:
        sel = sel[sel["scheme"].isin(set(schemes))]
    if verdicts is not None and "verdict" in sel.columns:
        sel = sel[sel["verdict"].isin(set(verdicts))]
    # Phase comparison requires invoke timing.
    sel = sel[sel["invoke med (ms)"].notna()]
    if sel.empty:
        return pd.DataFrame()

    rows = []
    boards = sorted({d for d in sel["device"].unique() if not str(d).endswith("_cpu")})
    for board in boards:
        npu = sel[sel["device"] == board]
        cpu = sel[sel["device"] == f"{board}_cpu"]
        if npu.empty or cpu.empty:
            continue
        keys = [k for k in ("arch_label", "scheme") if k in sel.columns]
        if not keys:
            continue
        for key, group in npu.groupby(keys, dropna=False):
            key = key if isinstance(key, tuple) else (key,)
            match = cpu
            for column, value in zip(keys, key, strict=False):
                match = match[match[column] == value]
            if match.empty:
                continue
            n_inv = float(group["invoke med (ms)"].median())
            c_inv = float(match["invoke med (ms)"].median())
            n_all = float(group["lat med (ms)"].median())
            c_all = float(match["lat med (ms)"].median())
            # Resize is included in preprocess; CPU-side time is preprocess plus postprocess.
            fixed = sum(
                float(group[col].median())
                for col in ("pre med (ms)", "post med (ms)")
                if col in group.columns and group[col].notna().any()
            )
            row = dict(zip(keys, key, strict=False))
            row.update(
                {
                    "Board": board,
                    "CPU invoke (ms)": round(c_inv, 1),
                    "NPU invoke (ms)": round(n_inv, 1),
                    "Speedup (invoke)": round(c_inv / n_inv, 2) if n_inv else None,
                    "CPU predict (ms)": round(c_all, 1),
                    "NPU predict (ms)": round(n_all, 1),
                    "Speedup (predict)": round(c_all / n_all, 2) if n_all else None,
                    "CPU-side (ms)": round(fixed, 1),
                }
            )
            rows.append(row)

    if not rows:
        return pd.DataFrame()

    out = pd.DataFrame(rows).rename(
        columns={"arch_label": "Architecture", "scheme": "Scheme"}
    )
    lead = [c for c in ("Board", "Architecture", "Scheme") if c in out.columns]
    return (
        out[lead + [c for c in out.columns if c not in lead]]
        .sort_values(lead)
        .reset_index(drop=True)
    )


def plot_resource_summary(
    power_df: pd.DataFrame,
    *,
    metrics: Iterable[tuple[str, str, bool]] = RESOURCE_METRICS,
    verified_only: bool = True,
):
    """Plot resource metrics by export scheme, device and architecture.

    ``verified_only`` applies only to metrics joined to the external power trace.
    """
    if power_df.empty:
        return None

    sel = power_df
    if sel.empty:
        return None

    def _rows_for(joined: bool) -> pd.DataFrame:
        """Filter joined metrics to runs with verified trace alignment."""
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

    # Keep device and architecture as separate series.
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
    """Plot per-class AP across export schemes for multi-class and single-class models."""
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
    """Plot AP against target-device latency across input resolutions.

    One export scheme is used per line; latency uses a logarithmic axis.
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
    ax.set_title(f"Resolution trade-off on device: {scheme_label(scheme)}")
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
    """Plot AP against input resolution by architecture and export scheme."""
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
    """Plot benchmark coverage as a variant by platform/scheme heatmap."""
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


#: Fixed gallery colours for crop and weed boxes.
GALLERY_CLASS_COLORS = {"crop": "#3B8EDE", "weed": "#F2C300"}


def _gallery_boxes(records, image_id, categories, score_threshold):
    """Return score-filtered ``(x, y, w, h, class_name)`` boxes for one image."""
    out = []
    for r in records.get(image_id, ()):
        if r.get("score") is not None and float(r["score"]) < score_threshold:
            continue
        name = categories.get(r.get("category_id"))
        if name is None:
            continue
        out.append((*r["bbox"], name))
    return out


def plot_detection_gallery(
    annotations_path: str | Path,
    image_root: str | Path,
    predictions: dict[str, str | Path],
    *,
    n_images: int = 8,
    score_threshold: float = 0.5,
    max_px: int = 384,
    figure_width: float = 9.0,
):
    """Plot ground truth and detector outputs on the same deterministic frame sample.

    Frames are stratified by weed count. Returns ``None`` if required images,
    annotations or predictions are unavailable.
    """
    annotations_path = Path(annotations_path)
    image_root = Path(image_root)
    if not annotations_path.is_file() or not image_root.is_dir():
        return None

    available = {
        label: Path(path) for label, path in predictions.items() if Path(path).is_file()
    }
    if not available:
        return None

    try:
        import matplotlib.image as mpimg
        from matplotlib.patches import Rectangle
    except ImportError:  # pragma: no cover
        return None

    with open(annotations_path) as handle:
        coco = json.load(handle)

    categories = {c["id"]: c["name"] for c in coco.get("categories", ())}
    images = {img["id"]: img for img in coco.get("images", ())}

    truth: dict[int, list] = {}
    weeds: dict[int, int] = dict.fromkeys(images, 0)
    for ann in coco.get("annotations", ()):
        truth.setdefault(ann["image_id"], []).append(ann)
        if categories.get(ann["category_id"]) == "weed":
            weeds[ann["image_id"]] = weeds.get(ann["image_id"], 0) + 1

    # Only existing image files can be drawn.
    usable = [
        i for i in sorted(images) if (image_root / images[i]["file_name"]).is_file()
    ]
    if not usable:
        return None

    # Stratify on weed count; `id` breaks ties so the choice is reproducible.
    usable.sort(key=lambda i: (weeds.get(i, 0), i))
    n_images = max(1, min(n_images, len(usable)))
    step = len(usable) / n_images
    chosen = [usable[min(len(usable) - 1, int(k * step))] for k in range(n_images)]

    detections: dict[str, dict[int, list]] = {}
    for label, path in available.items():
        with open(path) as handle:
            payload = json.load(handle)
        per_image: dict[int, list] = {}
        for record in payload if isinstance(payload, list) else ():
            per_image.setdefault(record["image_id"], []).append(record)
        detections[label] = per_image

    columns = ["Ground truth", *detections]
    n_cols = len(columns)
    fig, axes = plt.subplots(
        n_images,
        n_cols,
        figsize=(figure_width, figure_width / n_cols * n_images),
        squeeze=False,
    )

    for row, image_id in enumerate(chosen):
        meta = images[image_id]
        frame = mpimg.imread(image_root / meta["file_name"])
        # Plain striding is sufficient for gallery thumbnails.
        stride = max(1, int(max(frame.shape[:2]) / max_px))
        shown = frame[::stride, ::stride]
        scale = 1.0 / stride

        for col, column in enumerate(columns):
            ax = axes[row][col]
            ax.imshow(shown)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_edgecolor("#999999")
                spine.set_linewidth(0.5)

            if column == "Ground truth":
                boxes = [
                    (*a["bbox"], categories.get(a["category_id"]))
                    for a in truth.get(image_id, ())
                ]
            else:
                boxes = _gallery_boxes(
                    detections[column], image_id, categories, score_threshold
                )

            for x, y, w, h, name in boxes:
                ax.add_patch(
                    Rectangle(
                        (x * scale, y * scale),
                        w * scale,
                        h * scale,
                        fill=False,
                        edgecolor=GALLERY_CLASS_COLORS.get(name, "#FFFFFF"),
                        linewidth=0.8,
                    )
                )

            if row == 0:
                ax.set_title(column, fontsize=9)
            if col == 0:
                # Label each row with frame id and class counts.
                _crop = sum(
                    1
                    for a in truth.get(image_id, ())
                    if categories.get(a["category_id"]) == "crop"
                )
                ax.set_ylabel(
                    f"{meta['file_name'].split('_')[-1][:-4]}\n"
                    f"{_crop} crop  {weeds.get(image_id, 0)} weed",
                    fontsize=6,
                )

    handles = [
        plt.Line2D([], [], color=color, linewidth=2, label=name)
        for name, color in GALLERY_CLASS_COLORS.items()
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=len(handles),
        frameon=False,
        fontsize=9,
        bbox_to_anchor=(0.5, 0.0),
    )
    fig.tight_layout(rect=(0, 0.025, 1, 1))
    return fig


def save_figure(
    fig,
    stem: str,
    fig_dir: str | Path,
    formats: Iterable[str] = ("pdf", "png"),
    dpi: float | None = None,
) -> None:
    """Save a figure under each requested format and optional DPI."""
    if fig is None:
        return
    fig_dir = Path(fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)
    for ext in formats:
        # Include legends and other artists placed outside the axes.
        fig.savefig(fig_dir / f"{stem}.{ext}", bbox_inches="tight", dpi=dpi)


# Unicode -> ASCII fallbacks so exported .tex compiles under plain pdflatex.
_ASCII_REPLACEMENTS = {
    "·": "-",  # middle dot
    "→": "->",  # right arrow
    "≤": "<=",  # less-or-equal
    "≥": ">=",  # greater-or-equal
    "Δ": "d",  # capital delta
    "²": "^2",  # superscript two
    "\u2013": "-",  # en dash
    "\u2014": "--",  # em dash
}


def _ascii(value):
    """Convert a string to ASCII using known symbol replacements."""
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
    split_by: str | None = None,
    **to_latex_kwargs,
) -> None:
    """Write a DataFrame as an ASCII-safe booktabs LaTeX table.

    Non-default indexes are preserved as columns. ``split_by`` emits one continued
    table panel per group.
    """
    if df.empty:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    keep_index = not isinstance(df.index, pd.RangeIndex)
    if keep_index:
        # Flatten non-default indexes so LaTeX needs no multirow support.
        df = df.reset_index()
        keep_index = False

    # Sanitize headers and string cells to ASCII before LaTeX escaping.
    df = df.rename(columns=_ascii)
    obj_cols = df.select_dtypes(include="object").columns
    if len(obj_cols):
        df[obj_cols] = df[obj_cols].apply(lambda c: c.map(_ascii))

    if split_by is not None:
        if not caption:
            raise ValueError("split_by requires a caption")
        if split_by not in df.columns:
            raise ValueError(f"split column not found: {split_by}")
        groups = list(df.groupby(split_by, sort=False, dropna=False))
    else:
        groups = [(None, df)]

    kwargs = {
        "index": keep_index,
        "escape": True,
        "na_rep": "--",
        "float_format": "%.4g",
    }
    kwargs.update(to_latex_kwargs)
    table_bodies = [
        (group, _ascii(group_df.to_latex(**kwargs))) for group, group_df in groups
    ]

    if caption:
        label = label or path.stem
        # Scale only tables wider than the text block.
        panels = []
        for panel_index, (group, table_body) in enumerate(table_bodies):
            if panel_index == 0:
                heading = f"\\caption{{{_ascii(caption)}}}\n\\label{{tab:{label}}}\n"
            else:
                group_label = _ascii(str(group)).replace("_", "\\_")
                heading = (
                    "{\\centering\\small\\textbf{\\tablename~\\thetable:} "
                    f"(continued) \\texttt{{{group_label}}}.\\par}}\n"
                    "\\vspace{\\belowcaptionskip}\n"
                )
            panels.append(
                # Allow normal, top, bottom and float-page placement.
                "\\begin{table}[htbp]\n\\centering\n"
                f"{heading}"
                "\\begin{adjustbox}{max width=\\linewidth}\n"
                f"{table_body}"
                "\\end{adjustbox}\n"
                "\\end{table}\n"
            )
        body = "\\clearpage\n".join(panels)
    else:
        body = table_bodies[0][1]
    path.write_text(body)
