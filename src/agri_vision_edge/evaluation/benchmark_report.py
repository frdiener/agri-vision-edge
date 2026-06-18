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

    <arch>_<classes>_<dataset>_<size>_<precision>_<quant>_<nms>_<split>
    ssd-mn2-fpnlite_mc_phenobench-tiled_320_int8_ptq_fastnms_val

Everything is discovered dynamically: new platforms (i.MX 8M Plus, i.MX 93) and
quantization schemes (``qatN``) appear automatically once their artifacts exist.
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

ARCH_LABELS = {
    "ssd-mn2": "SSD MobileNetV2",
    "ssd-mn2-fpnlite": "SSD MobileNetV2 FPNLite",
}
CLASS_LABELS = {
    "sc": "Single-class (weed)",
    "mc": "Multi-class (crop+weed)",
}
PRECISION_LABELS = {"fp32": "FP32", "fp16": "FP16", "int8": "INT8"}

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

    Args:
        name:
            Run directory name, e.g.
            ``ssd-mn2-fpnlite_mc_phenobench-tiled_320_int8_ptq_fastnms_val``.

    Returns:
        Dict of configuration fields plus display labels, or ``None`` if the
        name does not carry a recognizable ``sc``/``mc`` class token.
    """
    tokens = name.split("_")

    cls_idx = next(
        (i for i, t in enumerate(tokens) if t in _CLASSES),
        None,
    )

    if cls_idx is None:
        return None

    info: dict = {
        "run": name,
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

    # Compact label used on chart axes.
    info["config"] = (
        f"{info['arch_label'].replace('SSD MobileNetV2', 'MNv2')}"
        f" | {info['classes'].upper()}"
    )

    return info


# =========================================================
# Loading
# =========================================================


def _read_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


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
        configuration fields, COCO metrics (overall + per-class) and latency,
        and ``skipped`` lists ``platform/run`` entries that were not loadable.
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
                skipped.append(f"{tag} (no metrics)")
                continue

            info = parse_run_name(run_dir.name)
            if info is None:
                skipped.append(f"{tag} (unparsed)")
                continue

            row = dict(info)
            row["platform"] = platform_dir.name

            for key in _METRIC_KEYS:
                row[key] = metrics.get(key)

            for cls, cls_metrics in (metrics.get("per_class") or {}).items():
                row[f"{cls}_AP"] = cls_metrics.get("AP")
                row[f"{cls}_AP50"] = cls_metrics.get("AP50")
                row[f"{cls}_AP75"] = cls_metrics.get("AP75")

            latency = _read_json(run_dir / "latency.json") or {}
            for key in (
                "mean_latency_ms",
                "median_latency_ms",
                "min_latency_ms",
                "max_latency_ms",
            ):
                row[key] = latency.get(key)
            row["n_latency"] = len(latency.get("latencies_ms") or [])
            if row.get("mean_latency_ms"):
                row["fps"] = 1000.0 / row["mean_latency_ms"]

            runtime = _read_json(run_dir / "runtime.json") or {}
            row["delegate"] = runtime.get("delegate")

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
):
    """
    Draw side-by-side bar groups.

    Args:
        values:
            Mapping ``{group_label: [value per category]}``.
    """
    n_groups = max(len(group_labels), 1)
    x = np.arange(len(categories))
    width = 0.8 / n_groups

    for i, glabel in enumerate(group_labels):
        offset = (i - (n_groups - 1) / 2) * width
        vals = values[glabel]

        bars = ax.bar(
            x + offset,
            vals,
            width,
            label=glabel,
            color=colors[i % len(colors)],
            edgecolor="white",
            linewidth=0.5,
        )

        if annotate:
            for bar, v in zip(bars, vals, strict=False):
                if v is None or (isinstance(v, float) and np.isnan(v)):
                    continue
                ax.annotate(
                    fmt.format(v),
                    (bar.get_x() + bar.get_width() / 2, v),
                    ha="center",
                    va="bottom",
                    fontsize=7,
                    xytext=(0, 1),
                    textcoords="offset points",
                )

    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.margins(y=0.15)
    _prepare_axis(ax)
    return ax


def _short(series: pd.Series) -> pd.Series:
    return series.str.replace("SSD MobileNetV2", "MNv2")


# =========================================================
# Figures
# =========================================================


def plot_quantization_effect(df: pd.DataFrame):
    """FP32 vs INT8 on overall AP and weed AP (PTQ runs)."""
    df = df[df.get("quant") == "ptq"].copy() if "quant" in df else df.copy()
    if df.empty or df["precision"].nunique() < 2:
        return None

    df["group"] = df["platform"] + " | " + df["config"]
    precisions = [p for p in ("fp32", "int8") if p in df["precision"].values]
    groups = sorted(df["group"].unique())

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharex=True)
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
        )
        ax.legend(title="precision")
        ax.tick_params(axis="x", labelrotation=20)

    fig.suptitle("FP32 -> INT8 quantization effect", y=1.02)
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
    )
    ax.legend(
        title="regime",
        labels=["single-class", "multi-class"][: len(class_order)],
    )
    ax.tick_params(axis="x", labelrotation=20)
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
    )
    ax.legend(title="architecture")
    ax.tick_params(axis="x", labelrotation=15)
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
    )
    ax.legend(title="class")
    ax.tick_params(axis="x", labelrotation=15)
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
    )
    ax.legend(title="object size", labels=["small", "medium", "large"])
    ax.tick_params(axis="x", labelrotation=25)
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
# Tables
# =========================================================


def quantization_delta_table(df: pd.DataFrame) -> pd.DataFrame:
    """FP32 vs INT8 AP with relative change, one row per config (PTQ runs)."""
    df = df[df.get("quant") == "ptq"] if "quant" in df else df
    if df.empty:
        return pd.DataFrame()

    keys = ["platform", "arch_label", "class_label"]
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


def master_table(df: pd.DataFrame) -> pd.DataFrame:
    """Full benchmark matrix with renamed, rounded thesis-ready columns."""
    cols = [
        ("platform", "Platform"),
        ("arch_label", "Architecture"),
        ("classes", "Classes"),
        ("precision", "Precision"),
        ("quant", "Quant"),
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

#: The (precision, quant) pairs that make up a *full run* for one model variant:
#: an FP32 + INT8 PTQ baseline, plus an INT8 export per QAT scheme. Editable in
#: the notebook to match the planned matrix (QAT is INT8-only by construction).
DEFAULT_FULL_RUN_COMBOS = (
    ("fp32", "ptq"),
    ("int8", "ptq"),
    ("int8", "qat0"),
    ("int8", "qat1"),
    ("int8", "qat2"),
    ("int8", "qat3"),
)

#: Platforms a full run targets: the laptop plus the two embedded NPUs.
DEFAULT_EXPECTED_PLATFORMS = ("theta", "imx8mp", "imx93")

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
    combos: Iterable[tuple[str, str]] = DEFAULT_FULL_RUN_COMBOS,
    platforms: Iterable[str] = DEFAULT_EXPECTED_PLATFORMS,
) -> pd.DataFrame:
    """
    Cross variants × (precision, quant) combos × platforms into a long
    coverage frame, flagging which expected runs are already benchmarked.

    Platforms already present in ``runs_df`` are always included, so unexpected
    platforms still surface. Matching ignores NMS / split.
    """
    if variants_df.empty:
        return pd.DataFrame()

    present_keys = set()
    if not runs_df.empty:
        for _, r in runs_df.iterrows():
            present_keys.add(
                (
                    r["platform"],
                    *(r.get(k) for k in _VARIANT_KEYS),
                    r.get("precision"),
                    r.get("quant"),
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
            for precision, quant in combos:
                key = (platform, *(v[k] for k in _VARIANT_KEYS), precision, quant)
                rows.append(
                    {
                        "platform": platform,
                        "variant": v["variant"],
                        "config": v["config"],
                        "precision": precision,
                        "quant": quant,
                        "scheme": f"{precision}/{quant}",
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
        index="variant",
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
        index="variant",
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
