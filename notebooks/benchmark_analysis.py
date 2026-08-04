import marimo

__generated_with = "0.23.9"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    # Benchmark Analysis — Weed Detection on Embedded NPUs

    Aggregates the artifacts written by `ave benchmark` (`latency.json` /
    `runtime.json` / `predictions.json`) and `ave evaluate`
    (`metrics.json`, and `metrics_faithful.json` with `--faithful`) into the
    tables and figures the thesis needs. The heavy lifting lives in
    `agri_vision_edge.evaluation.benchmark_report`; this notebook loads,
    checks, displays and exports.

    ## Layout

    ```
    benchmark_results/<platform>/<run>/{metrics,metrics_faithful,latency,runtime}.json

    <tiling>_<arch>_<classes>_<dataset>_<size>_<precision>_<quant>[_<granularity>]_<nms>
    untiled_ssd-mn2-fpnlite_mc_phenobench-tiled_320_int8_qat_per-tensor_fastnms
    ```

    Two independent tiling axes, and keeping them apart is the whole point of
    the sweep:

    | axis | token | meaning |
    |---|---|---|
    | **trained on** | `<dataset>` = `phenobench` / `phenobench-tiled` | the data the model was finetuned on |
    | **evaluated on** | `<tiling>` prefix = `untiled` / `tiled` | the input regime it was benchmarked against |

    Every model is swept over **both** input regimes, so each of the 8 trained
    variants contributes 2 (regimes) × 5 (export schemes) = 10 runs per
    platform.

    ## Two metric families -- should not be mixed

    - **pycocotools** (`AP`, `AP50`, `weed_AP`, …): our own consistent numbers,
      used for every internal comparison.
    - **official PhenoBench** (`faithful_*`): the upstream torchmetrics
      evaluator, used *only* for the comparability table. Rescaled to 0–1 here
      so both families share units.

    Discovery is dynamic: new platforms and schemes appear automatically once
    their artifacts land. Figures export to
    `docs/thesis/figures/benchmarks/`, tables to `docs/thesis/tables/`.
    """)
    return


@app.cell(hide_code=True)
def _():
    from pathlib import Path

    import marimo as mo
    import pandas as pd

    from agri_vision_edge.evaluation import benchmark_report as br

    return Path, br, mo, pd


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

    # The input regime the headline tables are built from. Untiled is the
    # full-frame regime the upstream numbers use; switch to "tiled" to read the
    # same tables for tiled inference.
    PRIMARY_EVAL_TILING = "untiled"

    br.apply_publication_style()
    return (
        ARTIFACTS_TF,
        BENCHMARK_ROOT,
        FIG_DIR,
        PRIMARY_EVAL_TILING,
        SAVE_ARTIFACTS,
        TAB_DIR,
    )


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
        if df is None or df.empty:
            return mo.md(f"_No rows for **{name}** yet — skipped._")
        if SAVE_ARTIFACTS:
            br.save_latex_table(
                df, TAB_DIR / f"{name}.tex", caption=caption, label=label or name
            )
        return mo.ui.table(df, label=name, selection=None)

    return show_fig, show_table


@app.cell(hide_code=True)
def _(br, mo, runs_df, skipped_runs):
    _platforms = sorted(runs_df["platform"].unique()) if not runs_df.empty else []
    _schemes = (
        sorted(br.add_scheme(runs_df)["scheme"].unique()) if not runs_df.empty else []
    )
    _regimes = (
        sorted(runs_df["eval_tiling"].dropna().unique()) if not runs_df.empty else []
    )
    mo.md(
        f"""
        ## Data overview

        - **{len(runs_df)}** run(s) across **{len(_platforms)}** platform(s):
          {", ".join(_platforms) or "—"}
        - Export schemes present: {", ".join(_schemes) or "—"}
        - Evaluated input regimes: {", ".join(_regimes) or "—"}
        - Skipped: {", ".join(skipped_runs) if skipped_runs else "none"}
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 0 · Sanity checks — read this before any table

    Broken runs disappear very effectively into averages and bar charts, so the
    checks come first. Each row is a result that is more likely a bug than a
    finding; an empty table means nothing was flagged, **not** that everything
    is correct.

    | check | severity | what it means |
    |---|---|---|
    | `quant-collapse` | error | an INT8 export scores below half its own FP32 baseline — a broken export, not a quantization cost |
    | `granularity-inversion` | error | per-channel scores materially below per-tensor, which is backwards: the finer granularity is strictly more expressive |
    | `faithful-stale` | error | official metrics predate the crop/weed label remap; re-run `ave evaluate --faithful` |
    | `delegate-fallback` | error | the run asked for a delegate and silently ran on the CPU |
    | `backend-unknown` | warning | artifact predates effective-delegate recording — CPU vs NPU cannot be established |
    | `fp32-on-delegate` | warning | a float graph routed through an INT8 accelerator |
    | `faithful-divergence` | warning | the two evaluators disagree beyond what their implementations explain |
    | `latency-outliers` | info | a disturbed timing run; harmless for the median, fatal for the mean |
    """)
    return


@app.cell(hide_code=True)
def _(br, runs_df):
    issues = br.sanity_checks(runs_df)
    return (issues,)


@app.cell(hide_code=True)
def _(br, issues, mo):
    mo.ui.table(br.sanity_summary(issues), label="Issues by check", selection=None)
    return


@app.cell(hide_code=True)
def _(issues, mo):
    mo.ui.table(
        issues[issues["severity"] == "error"] if not issues.empty else issues,
        label="Errors (runs that should not be reported as results)",
        selection=None,
    )
    return


@app.cell(hide_code=True)
def _(issues, mo):
    mo.ui.table(
        issues[issues["severity"] != "error"] if not issues.empty else issues,
        label="Warnings and notices",
        selection=None,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 1 · Upstream comparability

    Our detectors next to the published PhenoBench baselines. Only a narrow
    slice of runs is eligible, and the filter is deliberate:

    - **multi-class only** — upstream averages mAP over crop *and* weed, so a
      weed-only model is scored on a class it can never predict and lands at
      roughly half its true weed AP;
    - **untiled training data, untiled evaluation** — the upstream number is
      full-frame 1024×1024; our tiled faithful evaluation is explicitly
      tile-wise and not comparable;
    - **official metrics only**, in upstream percentage units.

    > Caveat for the write-up: the published baselines are on the **test**
    > split, ours on **val**. This orients the reader, it is not a ranking.
    """)
    return


@app.cell(hide_code=True)
def _(br, mo, runs_df, show_table):
    show_table(
        br.upstream_comparison_table(runs_df),
        "upstream_comparison",
        mo,
        caption="Plant detection on PhenoBench, our exports next to the "
        "published baselines (official evaluator; baselines on test, ours on "
        "val).",
    )
    return


@app.cell(hide_code=True)
def _(br, mo, runs_df):
    # Guard for §2/§2b: those sections report ONE CPU curve and treat it as the
    # unaccelerated reference for every board. That is only legitimate while the
    # per-board `<board>_cpu` trees actually agree with the reference host, so
    # check it here rather than assuming it.
    _div = br.cpu_reference_divergence(runs_df)

    if _div.empty:
        _verdict = mo.callout(
            mo.md(
                "**Unverified** — no second CPU tree to compare against "
                f"`{br.CPU_REFERENCE_PLATFORM}`. The sections below still show "
                "that host alone; they just cannot claim it stands for the "
                "boards."
            ),
            kind="warn",
        )
    else:
        _ok = br.cpu_reference_holds(_div)
        _verdict = mo.callout(
            mo.md(
                (
                    "**CPU reference holds.** "
                    if _ok
                    else "**CPU reference FAILS — do not collapse the CPU trees.** "
                )
                + f"Worst disagreement with `{br.CPU_REFERENCE_PLATFORM}` across "
                f"{int(_div['configs'].max())} shared configs is "
                f"`{_div['max_abs_diff'].max():.2e}` "
                f"(tolerance `{br.CPU_REFERENCE_TOLERANCE:.0e}`). INT8 "
                "predictions are bit-identical across x86 and ARM; only fp32 "
                "kernels differ, and only by pycocotools accumulation noise."
            ),
            kind="success" if _ok else "danger",
        )

    mo.vstack([_verdict, mo.ui.table(_div, selection=None)])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 2 · PTQ vs. QAT — on the CPU reference

    The core quantization table: every variant against all five deployable
    exports, with `dAP vs fp32` giving each INT8 export's cost against **its
    own** float baseline.

    Scoped to the **CPU reference** so this measures the *export* alone, with
    the accelerator held out — the guard above licenses using one CPU curve for
    every board. What the NPUs then do to these INT8 exports is §2b.

    Expected ordering is `fp32 ≥ per-channel ≥ per-tensor` within a scheme, and
    QAT ≥ PTQ at equal granularity. Rows that break it are exactly what §0
    flags.
    """)
    return


@app.cell(hide_code=True)
def _(PRIMARY_EVAL_TILING, br, mo, runs_df, show_table):
    show_table(
        br.scheme_comparison_table(
            runs_df,
            eval_tiling=PRIMARY_EVAL_TILING,
            platform=br.CPU_REFERENCE_PLATFORM,
        ),
        f"scheme_comparison_{PRIMARY_EVAL_TILING}",
        mo,
        caption="Detection quality per quantization scheme "
        f"({PRIMARY_EVAL_TILING} input) on the CPU reference, with the relative "
        "change against each variant's own FP32 baseline.",
    )
    return


@app.cell(hide_code=True)
def _(PRIMARY_EVAL_TILING, br, mo, runs_df, show_fig):
    show_fig(
        br.plot_scheme_effect(runs_df, eval_tiling=PRIMARY_EVAL_TILING),
        f"scheme_effect_{PRIMARY_EVAL_TILING}",
        mo,
    )
    return


@app.cell(hide_code=True)
def _(PRIMARY_EVAL_TILING, br, mo, runs_df, show_fig):
    show_fig(
        br.plot_scheme_effect(
            runs_df, eval_tiling=PRIMARY_EVAL_TILING, metric="weed_AP"
        ),
        f"scheme_effect_weed_{PRIMARY_EVAL_TILING}",
        mo,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### 2b · What the accelerators do to those INT8 exports

    §2 held the hardware fixed and varied the export; this holds the export
    fixed and varies the hardware. Only INT8 appears — it is the only precision
    a delegate accelerates, and the float baseline is already in §2.

    A delegate that faithfully reproduces its CPU reference draws bars of equal
    height, so **any visible gap is the accelerator changing the result**, not
    the quantization.

    The two axes are deliberately not crossed in one figure: a category would
    then be the product of variant × scheme × board, which is what made the
    previous combined version illegible.
    """)
    return


@app.cell(hide_code=True)
def _(PRIMARY_EVAL_TILING, br, mo, runs_df, show_fig):
    show_fig(
        br.plot_backend_effect(runs_df, eval_tiling=PRIMARY_EVAL_TILING),
        f"backend_effect_{PRIMARY_EVAL_TILING}",
        mo,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### 2c · Degradation ladder — where the accuracy actually goes

    The deployment chain has four rungs, and each delta below isolates one
    transformation by holding the others fixed:

    | rung | delta | what it measures |
    |---|---|---|
    | SavedModel (TF float, TFOD post-processing) | | the trained model |
    | ↓ TFLite fp32 | `conversion` | graph conversion **and** the swap to `TFLite_Detection_PostProcess` |
    | ↓ INT8 on CPU | `quantization` | the precision change alone |
    | ↓ INT8 on NPU | `delegation` | what the accelerator does to it |

    The first rung is the one that was missing until now: without it, conversion
    loss is invisible and gets folded into `quantization`. It is not negligible
    — for `MNv2 | mc | phenobench-tiled` it is **larger** than that config's
    quantization loss.

    Two caveats on `conversion`. It is not purely numerical: the TFLite export
    substitutes a *different NMS implementation*. And it still folds in
    **resampling** — the SavedModel resizes inside its graph
    (`fixed_shape_resizer`) while the TFLite path resizes externally with
    `cv2` — so isolating the post-processing substitution alone would need a
    pre-resized control run.

    The reference is the *floored* export, matching the 0.05 NMS score
    threshold the TFLite graphs bake in, which is what keeps this rung
    like-for-like (see §4.6.1 of the thesis for why the floor stays).
    """)
    return


@app.cell(hide_code=True)
def _(PRIMARY_EVAL_TILING, br, mo, runs_df, show_table):
    show_table(
        br.degradation_ladder_table(runs_df, eval_tiling=PRIMARY_EVAL_TILING),
        f"degradation_ladder_{PRIMARY_EVAL_TILING}",
        mo,
        caption="Detection AP across the four deployment rungs "
        f"({PRIMARY_EVAL_TILING} input, PTQ path), with the loss attributable "
        "to conversion, quantization and delegation separately.",
    )
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
    ## 3 · Tiling: trained-on vs. evaluated-on

    The cross of the two tiling axes. Reading down a column shows what tiled
    inference buys (or costs) a given model; reading across a row shows how
    much a model is specialised to the regime it was trained on.
    """)
    return


@app.cell(hide_code=True)
def _(mo, pd, runs_df):
    _sel = runs_df[runs_df["precision"] == "fp32"] if not runs_df.empty else runs_df
    _pivot = (
        _sel.pivot_table(
            index=["platform", "arch_label", "classes", "dataset"],
            columns="eval_tiling",
            values=["AP", "weed_AP"],
            aggfunc="mean",
        ).round(4)
        if not _sel.empty
        else pd.DataFrame()
    )
    mo.ui.table(
        _pivot.reset_index() if not _pivot.empty else _pivot,
        label="FP32 AP by trained-on dataset x evaluated-on input regime",
        selection=None,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 4 · Single-class vs. multi-class

    Single-class trains a weed-only detector; multi-class adds the crop class.
    Compared on **weed AP**, the metric they share.
    """)
    return


@app.cell(hide_code=True)
def _(br, mo, runs_df, show_fig):
    show_fig(br.plot_single_vs_multiclass(runs_df), "single_vs_multiclass", mo)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 5 · Architecture and object size

    FPNLite adds a feature-pyramid neck, which should help the small objects
    that dominate PhenoBench — the COCO area breakdown (`APS` ≤ 32²,
    `APM`, `APL` > 96²) is where that has to show up.
    """)
    return


@app.cell(hide_code=True)
def _(br, mo, runs_df, show_fig):
    show_fig(br.plot_architecture_effect(runs_df), "architecture_effect", mo)
    return


@app.cell(hide_code=True)
def _(br, mo, runs_df, show_fig):
    show_fig(br.plot_per_class_ap(runs_df), "per_class_ap", mo)
    return


@app.cell(hide_code=True)
def _(br, mo, runs_df, show_fig):
    show_fig(br.plot_ap_by_area(runs_df), "ap_by_area", mo)
    return


@app.cell(hide_code=True)
def _(mo, runs_df):
    _backends = (
        sorted(runs_df["backend"].dropna().unique()) if not runs_df.empty else []
    )
    mo.md(
        f"""
        ## 6 · Latency

        Reported on the **median**, with p95 as the spread: the sweeps pick up
        occasional scheduling outliers an order of magnitude above the typical
        sample (see the `latency-outliers` notices in §0), which the mean
        absorbs and min/max whiskers exaggerate. Throughput is derived from the
        median.

        > Backends present: **{", ".join(_backends) or "—"}**. A CPU-vs-NPU
        > comparison is only meaningful once `backend` is `cpu` / `delegate`
        > — runs recorded as `unknown` predate effective-delegate recording and
        > only stored the delegate that was *requested*, which on a host
        > without the delegate library is not the one that ran.
        """
    )
    return


@app.cell(hide_code=True)
def _(br, mo, runs_df, show_table):
    show_table(
        br.latency_table(runs_df),
        "latency_by_scheme",
        mo,
        caption="Median / p95 inference latency and throughput per platform, "
        "backend and export scheme.",
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
    ## 7 · Per-platform result tables

    One full COCO table per device — the per-device tables sketched in
    `docs/tables_draft.org`. They are generated per platform, so the embedded
    boards appear here automatically as their sweeps land.
    """)
    return


@app.cell(hide_code=True)
def _(mo, runs_df):
    platforms = sorted(runs_df["platform"].unique()) if not runs_df.empty else []
    platform_ui = mo.ui.dropdown(
        options=platforms,
        value=platforms[0] if platforms else None,
        label="platform",
    )
    platform_ui  # noqa: B018 - bare name is how a marimo cell renders a widget
    return platform_ui, platforms


@app.cell(hide_code=True)
def _(br, mo, platform_ui, runs_df, show_table):
    show_table(
        br.platform_metrics_table(runs_df, platform_ui.value)
        if platform_ui.value
        else None,
        f"platform_metrics_{platform_ui.value}",
        mo,
        caption=f"Full COCO metrics for every run on {platform_ui.value}.",
    )
    return


@app.cell(hide_code=True)
def _(br, mo, platforms, runs_df, show_table):
    # Export one table per platform regardless of the dropdown selection, so a
    # full run of the notebook refreshes every device table in docs/thesis.
    for _platform in platforms:
        show_table(
            br.platform_metrics_table(runs_df, _platform),
            f"platform_metrics_{_platform}",
            mo,
            caption=f"Full COCO metrics for every run on {_platform}.",
        )
    mo.md(
        "_Exported per-platform tables: "
        + (", ".join(f"`{p}`" for p in platforms) or "—")
        + "._"
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 8 · Master results table

    The full matrix, ready for the thesis appendix.
    """)
    return


@app.cell(hide_code=True)
def _(br, mo, runs_df, show_table):
    show_table(
        br.master_table(runs_df),
        "benchmark_master",
        mo,
        caption="Full benchmark matrix across platforms, architectures, "
        "class regimes, input regimes and quantization schemes.",
    )
    return


@app.cell(hide_code=True)
def _(br, mo):
    mo.md(f"""
    ## 9 · Coverage — progress toward a full run

    A *full run* benchmarks every trained variant in `artifacts/tf`, for
    every export scheme, on both input regimes, on every target platform:

    - schemes (`br.DEFAULT_SCHEMES`):
      {", ".join(f"`{br.scheme_name(*s)}`" for s in br.DEFAULT_SCHEMES)}
    - input regimes (`br.DEFAULT_EVAL_TILINGS`):
      {", ".join(f"`{t}`" for t in br.DEFAULT_EVAL_TILINGS)}
    - platforms (`br.DEFAULT_EXPECTED_PLATFORMS`):
      {", ".join(f"`{p}`" for p in br.DEFAULT_EXPECTED_PLATFORMS)}

    That is 8 variants x 5 schemes x 2 regimes = **80 runs per platform**.
    The grid tracks done (`x`) vs. missing (`-`).
    """)
    return


@app.cell(hide_code=True)
def _(ARTIFACTS_TF, br, runs_df):
    model_variants = br.discover_model_variants(ARTIFACTS_TF)
    coverage = br.build_coverage(runs_df, model_variants)
    return (coverage,)


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
