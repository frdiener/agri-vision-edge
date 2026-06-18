import marimo

__generated_with = "0.23.9"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    # Benchmark Analysis — Weed Detection on Embedded NPUs

    Aggregates the artifacts written by `tflite_conversion.py` and the
    benchmark scripts in `bin/` (`benchmark_tflite.py` → `latency.json` /
    `runtime.json`, `evaluate_coco.py` → `metrics.json`) into
    publication-level tables and figures.

    The heavy lifting lives in
    `agri_vision_edge.evaluation.benchmark_report` (the evaluation-time
    counterpart to `evaluation.curves`); this notebook just loads, displays
    and exports.

    Results are read from
    `benchmark_results/<platform>/<run>/{metrics,latency,runtime}.json`,
    where the run name encodes the configuration:

    ```
    <arch>_<classes>_<dataset>_<size>_<precision>_<quant>_<nms>_<split>
    ssd-mn2-fpnlite_mc_phenobench-tiled_320_int8_ptq_fastnms_val
    ```

    It is **forward-compatible**: every platform (today `theta`, the laptop;
    later the i.MX 8M Plus and i.MX 93) and every quantization scheme (`ptq`
    today, `qat0/1/2` later) is discovered automatically — no edits needed
    when new results land. Figures (PDF + PNG) export to
    `docs/thesis/figures/benchmarks/` and LaTeX tables to
    `docs/thesis/tables/`.
    """)
    return


@app.cell(hide_code=True)
def _():
    from pathlib import Path

    import marimo as mo

    from agri_vision_edge.evaluation import benchmark_report as br

    return Path, br, mo


@app.cell(hide_code=True)
def _(Path, br):
    # --- Locations & options ---------------------------------------------
    def _find_repo_root() -> Path:
        here = Path.cwd().resolve()
        for candidate in (here, *here.parents):
            if (candidate / "benchmark_results").is_dir():
                return candidate
        return here

    REPO_ROOT = _find_repo_root()
    BENCHMARK_ROOT = REPO_ROOT / "benchmark_results"
    ARTIFACTS_TF = REPO_ROOT / "artifacts" / "tf"
    FIG_DIR = REPO_ROOT / "docs" / "thesis" / "figures" / "benchmarks"
    TAB_DIR = REPO_ROOT / "docs" / "thesis" / "tables"

    # Write figures/tables to the repo on run (set False to preview only).
    SAVE_ARTIFACTS = True

    br.apply_publication_style()
    return ARTIFACTS_TF, BENCHMARK_ROOT, FIG_DIR, SAVE_ARTIFACTS, TAB_DIR


@app.cell(hide_code=True)
def _(BENCHMARK_ROOT, br):
    runs_df, skipped_runs = br.load_benchmark_results(BENCHMARK_ROOT)
    return runs_df, skipped_runs


@app.cell(hide_code=True)
def _(FIG_DIR, SAVE_ARTIFACTS, TAB_DIR, br):
    # Small wrappers so each analysis cell stays a one-liner.
    def show_fig(fig, stem, mo):
        if fig is None:
            return mo.md(f"_Not enough data yet for **{stem}** — skipped._")
        if SAVE_ARTIFACTS:
            br.save_figure(fig, stem, FIG_DIR)
        return fig

    def show_table(df, name, mo, caption="", label=""):
        if SAVE_ARTIFACTS and not df.empty:
            br.save_latex_table(
                df, TAB_DIR / f"{name}.tex", caption=caption, label=label or name
            )
        return mo.ui.table(df, label=name, selection=None)

    return show_fig, show_table


@app.cell(hide_code=True)
def _(mo, runs_df, skipped_runs):
    _platforms = sorted(runs_df["platform"].unique()) if not runs_df.empty else []
    _quants = sorted(runs_df["quant"].dropna().unique()) if not runs_df.empty else []
    mo.md(
        f"""
        ## Data overview

        - **{len(runs_df)}** run(s) across **{len(_platforms)}** platform(s):
          {", ".join(_platforms) or "—"}
        - Quantization schemes present: {", ".join(_quants) or "—"}
        - Skipped: {", ".join(skipped_runs) if skipped_runs else "none"}

        > QAT (`qatN`) and the embedded platforms (i.MX 8M Plus, i.MX 93) are
        > not present yet — they will appear automatically once their artifacts
        > are written under `benchmark_results/`.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo, runs_df):
    _cols = [
        c
        for c in (
            "platform",
            "arch_label",
            "class_label",
            "precision",
            "quant",
            "AP",
            "AP50",
            "weed_AP",
            "crop_AP",
            "mean_latency_ms",
            "fps",
        )
        if c in runs_df.columns
    ]
    mo.ui.table(
        runs_df[_cols].round(4) if not runs_df.empty else runs_df,
        label="All benchmarked runs",
        selection=None,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 1 · Effect of INT8 quantization

    The headline question for embedded deployment: how much detection
    quality is lost when the FP32 model is post-training quantized to INT8?
    Shown for overall **AP** and, since this is a weed detector, for the
    **weed AP** specifically.
    """)
    return


@app.cell(hide_code=True)
def _(br, mo, runs_df, show_fig):
    show_fig(br.plot_quantization_effect(runs_df), "quantization_effect", mo)
    return


@app.cell(hide_code=True)
def _(br, mo, runs_df, show_table):
    show_table(
        br.quantization_delta_table(runs_df),
        "quantization_delta",
        mo,
        caption="Detection AP under FP32 vs.\\ INT8 post-training "
        "quantization, with relative change.",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 2 · Single-class vs. multi-class

    Single-class trains a weed-only detector; multi-class adds the crop
    class. We compare **weed AP** between the two regimes (the metric they
    share) to see whether modelling crops as a separate class helps or hurts
    weed localisation.
    """)
    return


@app.cell(hide_code=True)
def _(br, mo, runs_df, show_fig):
    show_fig(br.plot_single_vs_multiclass(runs_df), "single_vs_multiclass", mo)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 3 · Architecture: SSD MobileNetV2 vs. FPNLite

    FPNLite adds a feature-pyramid neck, which should help the small objects
    that dominate PhenoBench. Compared on overall AP here and on the
    per-area breakdown in §5.
    """)
    return


@app.cell(hide_code=True)
def _(br, mo, runs_df, show_fig):
    show_fig(br.plot_architecture_effect(runs_df), "architecture_effect", mo)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 4 · Per-class detection quality (crop vs. weed)

    For multi-class runs, crop is generally the easier class. The gap to
    weed AP quantifies how much harder weed detection is — the central
    concern of the thesis.
    """)
    return


@app.cell(hide_code=True)
def _(br, mo, runs_df, show_fig):
    show_fig(br.plot_per_class_ap(runs_df), "per_class_ap", mo)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 5 · AP by object size (S / M / L)

    PhenoBench weeds are predominantly small. The COCO area breakdown
    (`APS` ≤ 32², `APM`, `APL` > 96²) reveals where the models — and INT8
    quantization — lose the most.
    """)
    return


@app.cell(hide_code=True)
def _(br, mo, runs_df, show_fig):
    show_fig(br.plot_ap_by_area(runs_df), "ap_by_area", mo)
    return


@app.cell(hide_code=True)
def _(mo, runs_df):
    _platforms = sorted(runs_df["platform"].unique()) if not runs_df.empty else []
    mo.md(
        f"""
        ## 6 · Latency and the accuracy / latency trade-off

        Mean inference latency per run, and the AP-vs-latency view.

        > Latency currently reflects **{", ".join(_platforms) or "—"}** only.
        > Once the i.MX 8M Plus (Vivante NPU) and i.MX 93 (Ethos-U65) runs are
        > added, these figures draw the cross-platform comparison — they key on
        > `platform` automatically.
        """
    )
    return


@app.cell(hide_code=True)
def _(br, mo, runs_df, show_fig):
    show_fig(br.plot_latency(runs_df), "latency_per_run", mo)
    return


@app.cell(hide_code=True)
def _(br, mo, runs_df, show_fig):
    show_fig(br.plot_accuracy_latency(runs_df), "accuracy_latency_tradeoff", mo)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 7 · Master results table

    The full matrix, ready for the thesis (also exported to
    `docs/thesis/tables/benchmark_master.tex`).
    """)
    return


@app.cell(hide_code=True)
def _(br, mo, runs_df, show_table):
    show_table(
        br.master_table(runs_df),
        "benchmark_master",
        mo,
        caption="Full benchmark matrix across platforms, architectures, "
        "class regimes and quantization schemes.",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 8 · Coverage — progress toward a full run

    A *full run* benchmarks every trained model variant (the folders in
    `artifacts/tf`) for each `(precision, quant)` combo
    (`br.DEFAULT_FULL_RUN_COMBOS`: FP32+INT8 PTQ baselines plus an INT8 export
    per QAT scheme `qat0..qat3`) on every target platform
    (`br.DEFAULT_EXPECTED_PLATFORMS`: `theta`, `imx8mp`, `imx93`).

    The grid below tracks what is done (`x`) vs. still missing (`-`). Edit the
    combos / platforms in the cell to match the planned matrix.
    """)
    return


@app.cell(hide_code=True)
def _(ARTIFACTS_TF, br, runs_df):
    model_variants = br.discover_model_variants(ARTIFACTS_TF)
    coverage = br.build_coverage(
        runs_df,
        model_variants,
        combos=br.DEFAULT_FULL_RUN_COMBOS,
        platforms=br.DEFAULT_EXPECTED_PLATFORMS,
    )
    return coverage, model_variants


@app.cell(hide_code=True)
def _(br, coverage, mo):
    mo.ui.table(
        br.coverage_summary(coverage), label="Coverage by platform", selection=None
    )
    return


@app.cell(hide_code=True)
def _(br, coverage, mo, show_table):
    show_table(
        br.coverage_matrix(coverage),
        "benchmark_coverage",
        mo,
        caption="Benchmark coverage across the full run matrix "
        "(x = done, - = missing).",
    )
    return


@app.cell(hide_code=True)
def _(br, coverage, mo, show_fig):
    show_fig(br.plot_coverage(coverage), "benchmark_coverage", mo)
    return


@app.cell(hide_code=True)
def _(FIG_DIR, SAVE_ARTIFACTS, TAB_DIR, mo):
    mo.md(f"""
    ---
    **Artifact export:** {"enabled" if SAVE_ARTIFACTS else "disabled"}.
    Figures → `{FIG_DIR}` · tables → `{TAB_DIR}`.
    Set `SAVE_ARTIFACTS = False` in the options cell to preview without
    writing to the repo.
    """)
    return


if __name__ == "__main__":
    app.run()
