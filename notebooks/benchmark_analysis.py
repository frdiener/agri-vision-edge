import marimo

__generated_with = "0.23.9"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    # Benchmark analysis: weed detection on embedded NPUs

    Chapter 6 tables and figures, generated from `ave benchmark` and `ave evaluate`
    artifacts. Analysis functions are imported from
    `agri_vision_edge.evaluation.benchmark_report` as `br`.

    ## Contents

    | § | topic | artifact |
    |---|---|---|
    | 0 | admissible runs | `sanity_checks` |
    | 1 | float baseline | `baseline` |
    | 2 | conversion and PTQ | `preparation_ladder_ptq`, `scheme_comparison` |
    | 3 | QAT and NMS accuracy | `qat_reclaim`, `nms_substitution` |
    | 4 | accelerator correctness | `deployability` |
    | 5 | CPU and NPU cost | `device_latency`, `resource_summary` |
    | 6 | class, tiling, NMS, and resolution ablations | `story_ablation`, `nms_latency_tradeoff` |

    ## Reference configuration

    **Multi-class, trained and evaluated full-frame, 320×320**
    (`br.REFERENCE_CONFIG`). Sections 1–5 use this configuration; section 6 varies
    one axis at a time.

    ## Run grammar

    ```
    benchmark_results/<platform>/<run>/{metrics,metrics_faithful,latency,runtime}.json

    <tiling>_<arch>_<classes>_<dataset>_<size>_<precision>_<quant>[_<granularity>]_<nms>
    untiled_ssd-mn2-fpnlite_mc_phenobench-tiled_320_int8_qat_per-tensor_fastnms
    ```

    | axis | token | meaning |
    |---|---|---|
    | **training data** | `<dataset>` = `phenobench` / `phenobench-tiled` | fine-tuning input |
    | **evaluation input** | `<tiling>` = `untiled` / `tiled` | benchmark input |
    | **export** | `<precision>_<quant>[_<granularity>]` | precision and quantization |
    | **post-processing** | `<nms>` = `fastnms` / `regnms` | NMS in the exported graph |

    `fastnms` and `regnms` use the same checkpoint, graph, and calibration set.
    Each view selects one unless a comparison requires both.

    ## Metric families

    - **pycocotools** (`AP`, `AP50`, `weed_AP`, ...): all comparisons except §1.
    - **PhenoBench** (`faithful_*`): used only for the published-baseline comparison
      in §1; not comparable to pycocotools metrics.

    Figures export to `docs/thesis/figures/benchmarks/`, tables to
    `docs/thesis/tables/`.
    """)
    return


@app.cell(hide_code=True)
def _():
    import json
    from pathlib import Path

    import marimo as mo
    import pandas as pd

    from agri_vision_edge.evaluation import benchmark_report as br
    from agri_vision_edge.evaluation import delegation as dg

    return Path, br, dg, json, mo, pd


@app.cell(hide_code=True)
def _(Path, br):
    # Paths and options
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

    # Power results are optional; §5.3 shows a note when they are absent.
    RESOURCE_ROOT = REPO_ROOT / "resource_results"

    # Optional licensed evaluation data for §8.5. `test_annotations.json` is the
    # internal test half of the PhenoBench validation split; the official test
    # split has no labels.
    _DATASETS = REPO_ROOT.parent.parent / "datasets"
    GT_ANNOTATIONS = _DATASETS / "phenobench_multiclass" / "test_annotations.json"
    IMAGE_ROOT = _DATASETS / "PhenoBench" / "val" / "images"

    # Set to False to preview without writing files.
    SAVE_ARTIFACTS = True

    br.apply_publication_style()
    return (
        ARTIFACTS_TF,
        BENCHMARK_ROOT,
        FIG_DIR,
        GT_ANNOTATIONS,
        IMAGE_ROOT,
        RESOURCE_ROOT,
        SAVE_ARTIFACTS,
        TAB_DIR,
    )


@app.cell(hide_code=True)
def _(BENCHMARK_ROOT, br):
    # Each analysis selects its scope through `view(...)`.
    runs, skipped = br.load_benchmark_results(BENCHMARK_ROOT)

    # Report per-class NMS. `br.DEFAULT_NMS` remains the `ave convert` default.
    NMS = br.REGULAR_NMS

    # Exclude a duplicate SavedModel control and the i.MX93 vendor backend from
    # comparative views. `controls=True` includes them in inventory tables.
    CONTROL_TREES = ("tf-savedmodel-nms0", "frdm-imx93_vendor-stack")


    def view(
        *,
        nms=NMS,
        archs="primary",
        controls=False,
        ref=False,
        deployable=False,
        **where,
    ):
        """Scoped slice of `runs`.

        nms         `regnms`, `fastnms`, or `both`
        archs       `primary`, `aux`, or None
        controls    include CONTROL_TREES
        ref         apply `br.REFERENCE_CONFIG`; a dict overrides selected axes
        deployable  exclude incorrect target executions
        **where     equality filters; sequences use membership
        """
        out = runs
        if not controls:
            out = out[~out["platform"].isin(CONTROL_TREES)]
        if nms != "both":
            out = br.select_nms(out, nms)
        if archs == "primary":
            out = out[out["arch"].isin(br.PRIMARY_ARCHS)]
        elif archs == "aux":
            out = out[~out["arch"].isin(br.PRIMARY_ARCHS)]
        # Apply correctness before platform filters; verdicts require the CPU row.
        if deployable:
            out = br.drop_failed_deployments(out)
        for _col, _val in where.items():
            out = out[
                out[_col].isin(_val)
                if isinstance(_val, (list, tuple, set))
                else out[_col] == _val
            ]
        if ref:
            out = br.reference_config_slice(out, **(ref if isinstance(ref, dict) else {}))
        return out

    return CONTROL_TREES, NMS, runs, skipped, view


@app.cell(hide_code=True)
def _(FIG_DIR, SAVE_ARTIFACTS, TAB_DIR, br):
    def show_fig(fig, stem, mo, **save_kwargs):
        # Galleries override the default vector format through `save_kwargs`.
        if fig is None:
            return mo.md(f"_No data for `{stem}`._")
        if SAVE_ARTIFACTS:
            br.save_figure(fig, stem, FIG_DIR, **save_kwargs)
        return fig

    def show_table(df, name, mo, caption="", label="", **latex_kwargs):
        if df is None or df.empty:
            return mo.md(f"_No rows for `{name}`._")
        if SAVE_ARTIFACTS:
            br.save_latex_table(
                df,
                TAB_DIR / f"{name}.tex",
                caption=caption,
                label=label or name,
                **latex_kwargs,
            )
        return mo.ui.table(df, label=name, selection=None)

    return show_fig, show_table


@app.cell(hide_code=True)
def _(NMS, br, mo, runs, skipped, view):
    _platforms = sorted(runs["platform"].unique())
    _archs = sorted(runs["arch_label"].unique())
    _sizes = sorted(runs["size"].dropna().unique())

    # SavedModel rows have no NMS token and pass through either NMS scope.
    _tokened = runs[runs["nms"].notna()]
    _missing = [
        p
        for p in sorted(_tokened["platform"].unique())
        if not _tokened.loc[_tokened["platform"] == p, "nms"].eq(NMS).any()
    ]

    mo.md(f"""
    ## Data

    **{len(runs)}** runs on **{len(_platforms)}** platforms
    ({", ".join(f"`{p}`" for p in _platforms)}); architectures
    {", ".join(_archs)}; input sizes {", ".join(f"`{s}`" for s in _sizes)}.

    Default scope: **{len(view())}** runs, `{NMS}` post-processing, primary
    architectures, and no control trees. **{len(skipped)}** directories lack
    `metrics.json` (§0.2).

    Reference configuration: **{br.REFERENCE_CONFIG}**.

    {"Missing `" + NMS + "` runs: " + ", ".join(f"`{p}`" for p in _missing) + "."
     if _missing else "All NMS-aware trees include the selected variant."}
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 0 · Integrity gate

    `pycocotools` can accept non-finite boxes at every IoU threshold because a
    `NaN` comparison is false. Affected runs can score spuriously high (observed:
    `AP 85.8`, with `AP == AP50`).

    | check | severity | condition |
    |---|---|---|
    | `quant-collapse` | error | INT8 export below half its own FP32 baseline |
    | `nms-control-broken` | error | single-class NMS pair not identically zero (§2.2) |
    | `faithful-stale` | error | official metrics predate the crop/weed label remap |
    | `delegate-fallback` | error | delegate requested, run executed on CPU |
    | `backend-unknown` | warning | artifact predates effective-delegate recording |
    | `fp32-on-delegate` | warning | float graph routed through an INT8 accelerator |
    | `faithful-divergence` | warning | the two evaluators disagree beyond their implementations |
    | `latency-outliers` | info | disturbed timing run; median/p95 unaffected, mean is not |

    Checks run on the full data set. An empty result means no issue was detected.
    `backend` confirms that the delegate loaded, not which subgraph it accepted.
    """)
    return


@app.cell(hide_code=True)
def _(br, runs):
    issues = br.sanity_checks(runs)
    return (issues,)


@app.cell(hide_code=True)
def _(br, issues, mo):
    mo.ui.table(br.sanity_summary(issues), label="Issues by check", selection=None)
    return


@app.cell(hide_code=True)
def _(issues, mo):
    mo.ui.table(
        issues[issues["severity"] == "error"] if not issues.empty else issues,
        label="Errors (runs that must not be reported as results)",
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
def _(mo, skipped):
    mo.md(f"""
    ### 0.2 · Runs without metrics

    **{len(skipped)}** directories, all float graphs on an i.MX8M Plus delegate.
    `ave evaluate` omits `metrics.json` for non-finite boxes; §4 marks these runs
    `unscoreable`.
    """)
    return


@app.cell(hide_code=True)
def _(mo, pd, skipped):
    _rows = []
    for _entry in skipped:
        _platform, _, _rest = _entry.partition("/")
        _run, _, _reason = _rest.partition(" (")
        _rows.append(
            {"platform": _platform, "run": _run, "reason": _reason.rstrip(")")}
        )
    _skipped = pd.DataFrame(_rows, columns=["platform", "run", "reason"])

    mo.vstack(
        [
            mo.ui.table(
                _skipped.groupby(["platform", "reason"], dropna=False)
                .size()
                .rename("runs")
                .reset_index()
                if not _skipped.empty
                else _skipped,
                label="Not loaded, by platform and reason",
                selection=None,
            ),
            mo.ui.table(_skipped, label="Detail", selection=None),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    # 1 · Float baseline

    Fine-tuned SavedModel accuracy before TFLite conversion, compared with the
    published PhenoBench detectors using official PhenoBench metrics (`faithful_*`).
    The plain detector has 320/512/1024 exports; FPNLite has 320/512.

    Published results use the withheld test split at full resolution. This work
    uses an internal test split from the validation partition; see `Source`.
    """)
    return


@app.cell(hide_code=True)
def _(br, mo, show_table, view):
    show_table(
        br.baseline_table(view()),
        "baseline",
        mo,
        caption="Fine-tuned float accuracy across input resolutions using official "
        "PhenoBench metrics. Published results use the withheld test set; this "
        "work uses an internal split of the validation set, so the values are not "
        "directly comparable.",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    # 2 · Conversion and post-training quantization

    Export effects measured on the CPU reference. Accelerator results follow in
    §4 and §5.
    """)
    return


@app.cell(hide_code=True)
def _(br, mo, view):
    # Verify the shared CPU reference against each board's CPU-only runs.
    _div = br.cpu_reference_divergence(view(archs=None, nms=br.DEFAULT_NMS))

    if _div.empty:
        _verdict = mo.callout(
            mo.md(
                f"**Unverified** — no second CPU tree to compare against "
                f"`{br.CPU_REFERENCE_PLATFORM}`."
            ),
            kind="warn",
        )
    else:
        _ok = br.cpu_reference_holds(_div)
        _verdict = mo.callout(
            mo.md(
                f"**CPU reference {'holds' if _ok else 'FAILS — do not collapse the CPU trees'}.** "
                f"Worst disagreement with `{br.CPU_REFERENCE_PLATFORM}` across "
                f"{int(_div['configs'].max())} shared configs is "
                f"`{_div['max_abs_diff'].max():.2e}`, tolerance "
                f"`{br.CPU_REFERENCE_TOLERANCE:.0e}`."
            ),
            kind="success" if _ok else "danger",
        )

    mo.vstack([_verdict, mo.ui.table(_div, selection=None)])
    return


@app.cell(hide_code=True)
def _(NMS, br, mo):
    mo.md(f"""
    ### 2.1 · The preparation ladder

    All rungs use **{br.NMS_LABELS[NMS].lower()}**, isolating format conversion at
    the float step and precision at the INT8 steps.

    | rung | isolates |
    |---|---|
    | Float SavedModel | the trained checkpoint (§1) |
    | ↓ Float TFLite | conversion, at matched post-processing |
    | ↓ INT8 PTQ, per-channel / per-tensor | precision, at each weight granularity |

    INT8 changes are relative to float TFLite; the INT8 exports are alternatives.
    """)
    return


@app.cell(hide_code=True)
def _(NMS, br, mo, show_table, view):
    show_table(
        br.preparation_ladder_table(view(), include_qat=False, nms=NMS),
        "preparation_ladder_ptq",
        mo,
        caption="Accuracy after conversion and PTQ at the reference configuration. "
        "All rungs use per-class NMS; INT8 changes are relative to float TFLite.",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### 2.2 · Granularity across the matrix

    Quantization schemes across all variants and at the reference configuration.
    Expected order: `fp32 >= per-channel >= per-tensor`.
    """)
    return


@app.cell(hide_code=True)
def _(NMS, br, mo, show_table, view):
    show_table(
        br.scheme_comparison_table(
            view(nms="both", classes="mc", dataset="phenobench"),
            eval_tiling="untiled",
            platform=br.CPU_REFERENCE_PLATFORM,
            nms=NMS,
        ),
        "scheme_comparison_untiled",
        mo,
        caption="Detection quality by quantization scheme and input resolution on "
        "the CPU reference. Changes are relative to each variant's FP32 baseline.",
    )
    return


@app.cell(hide_code=True)
def _(NMS, br, mo, show_fig, view):
    show_fig(
        br.plot_scheme_effect(
            view(nms="both", classes="mc", dataset="phenobench"), nms=NMS
        ),
        "scheme_effect",
        mo,
    )
    return


@app.cell(hide_code=True)
def _(br, mo):
    mo.md(f"""
    # 3 · QAT recovery

    `Reclaimed` is QAT minus PTQ at equal granularity. `Reclaimed %` divides that
    difference by the PTQ deficit; 100% closes the deficit and negative values
    indicate lower QAT accuracy.

    Percent recovery is omitted when the unrounded PTQ deficit is below
    **{br.QAT_RECLAIM_MIN_DEFICIT} AP**. The absolute change remains available in
    `Reclaimed`.

    Reference configuration, per-class NMS, CPU reference.
    """)
    return


@app.cell(hide_code=True)
def _(NMS, br, mo, show_table, view):
    show_table(
        br.qat_reclaim_table(view(), nms=NMS),
        "qat_reclaim",
        mo,
        caption="QAT recovery of the PTQ deficit by architecture and weight "
        "granularity at the reference configuration.",
    )
    return


@app.cell
def _(mo):
    mo.md("""
    ### 3.1 · The full preparation chain

    The §2.1 ladder extended to PTQ and QAT at both weight granularities. INT8
    changes are relative to float TFLite.
    """)
    return


@app.cell(hide_code=True)
def _(NMS, br, mo, show_table, view):
    show_table(
        br.preparation_ladder_table(view(), nms=NMS),
        "preparation_ladder",
        mo,
        caption="Accuracy after conversion, PTQ, and QAT at both weight "
        "granularities. Reference configuration with per-class NMS.",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### 3.2 · NMS accuracy

    `ave convert` defaults to `use_regular_nms=False`: one class-agnostic pass over
    each anchor's highest-scoring class. The checkpoint instead suppresses each
    class separately. Because SSD shares one box per anchor across classes, fast
    NMS removes the lower-scoring class hypothesis.

    Results use the reference configuration. The methods are identical for a
    single class; `nms-control-broken` checks this. §6.3 reports latency.
    """)
    return


@app.cell(hide_code=True)
def _(br, mo, show_table, view):
    show_table(
        br.nms_substitution_summary(
            view(nms="both", ref={"size": None}),
        ),
        "nms_substitution_summary",
        mo,
        caption="Accuracy difference between fast and per-class NMS by architecture "
        "and input resolution at the reference configuration.",
    )
    return


@app.cell(hide_code=True)
def _(br, mo, show_fig, view):
    # Pin resolution because the plot groups only by scheme.
    show_fig(br.plot_nms_substitution(view(nms="both", ref=True)), "nms_substitution", mo)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    # 4 · Accelerator correctness

    Each verdict compares the accelerator with the same file on the CPU reference.

    | verdict | meaning |
    |---|---|
    | `ok` | reproduces the same file on the CPU reference |
    | `degraded` | below 90 % of it |
    | `collapsed` | below 50 % — loads, runs, plausible boxes, near-zero score |
    | `unscoreable` | ran, but the output is not a detection; §0 withheld `metrics.json` |
    | `-` | not benchmarked |

    `unscoreable` runs are recovered from the §0.2 skip list because they have no
    metrics row.
    """)
    return


@app.cell(hide_code=True)
def _(CONTROL_TREES, NMS, br, runs, skipped, view):
    # Select target boards; exclude CPU, unpatched, and alternate-backend trees.
    deploy_boards = [
        p
        for p in sorted(runs["platform"].unique())
        if p.startswith("frdm")
        and not p.endswith(("_cpu", "_unpatched"))
        and p not in CONTROL_TREES
    ]
    deployability = br.deployability_matrix(
        view(nms="both", classes="mc", dataset="phenobench"),
        skipped,
        eval_tiling="untiled",
        platforms=deploy_boards,
        nms=NMS,
    )
    return (deployability,)


@app.cell(hide_code=True)
def _(br, deployability, mo):
    mo.ui.table(
        br.deployability_summary(deployability),
        label="Deployability by platform",
        selection=None,
    )
    return


@app.cell(hide_code=True)
def _(deployability, mo, show_table):
    show_table(
        deployability,
        "deployability",
        mo,
        caption="Export correctness on each target relative to the same file on "
        "the CPU reference. Full-frame input with deployed post-processing.",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### 4.2 · Delegate accuracy

    INT8 only. Equal bar heights indicate agreement with the CPU reference.
    """)
    return


@app.cell(hide_code=True)
def _(NMS, br, mo, show_fig, view):
    # Use the reference resolution available on all boards.
    show_fig(
        br.plot_backend_effect(view(nms="both", ref=True), nms=NMS),
        "backend_effect",
        mo,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### 4.3 · Mesa/Teflon operator support

    `frdm-imx8mp_unpatched` uses the delegate build from before the operator-support
    changes in thesis §5.5. The table reports verdict changes and latency by
    architecture; pooling architectures would hide their different responses.

    `ms before` and `ms after` include every configuration timed by both builds,
    including collapsed runs. Their latency still measures dispatch and CPU–NPU
    transfer, but their accuracy is invalid. `Before` and `After` provide the
    corresponding correctness verdicts.

    Scope: i.MX8M Plus, 320×320, default export. The i.MX93 uses a different
    backend, and the unpatched tree was measured only at 320×320.
    """)
    return


@app.cell(hide_code=True)
def _(br, mo, show_table, skipped, view):
    # Keep the CPU reference for verdicts. `paired` latency includes collapsed
    # runs because their execution time remains valid.
    show_table(
        br.delegate_build_table(
            view(
                platform=[
                    "frdm-imx8mp",
                    "frdm-imx8mp_unpatched",
                    br.CPU_REFERENCE_PLATFORM,
                ],
                size="320",
                nms="both",
                archs=None,
                classes="mc",
                dataset="phenobench",
            ),
            skipped,
            eval_tiling="untiled",
            nms=br.DEFAULT_NMS,
            latency_scope="paired",
            show_board_column=False,
        ),
        "delegate_build_delta",
        mo,
        caption="Effect of the Mesa/Teflon operator-support changes on the i.MX8M "
        "Plus at 320x320: correctness transitions and paired median latency by "
        "architecture and export scheme.",
    )
    return


@app.cell(hide_code=True)
def _(BENCHMARK_ROOT, dg):
    # Parse delegation continuity from `delegate_debug.log`.
    delegation = dg.load_delegation(BENCHMARK_ROOT)
    return (delegation,)


@app.cell(hide_code=True)
def _(br, delegation, dg, mo, show_table):
    # Continuity is constant by architecture and scheme; remove repeated parses
    # across class, NMS, and tiling variants. Exclude auxiliary architectures.
    _primary = [br.ARCH_LABELS[a] for a in br.PRIMARY_ARCHS]

    _captions = {
        "frdm-imx8mp": (
            "Delegation continuity on the i.MX8M Plus with full-frame input and "
            "per-class NMS. $R_{ops}$ is the delegated operation share, $K$ the "
            "number of delegated regions, and $R_{largest}$ the share in the "
            "largest region."
        ),
        "frdm-imx93": (
            "Delegation continuity on the i.MX93 with full-frame input and "
            "per-class NMS. Columns follow \\cref{tab:continuity_imx8mp}. $K=1$ "
            "means the accepted operations form one region. Float rows are not "
            "delegated, so their continuity fields are empty."
        ),
    }

    for _plat, _caption in _captions.items():
        _table = dg.continuity_table(delegation, platform=_plat)
        if not _table.empty:
            _table = _table[_table["Architecture"].isin(_primary)]
            _table = _table.drop_duplicates(["Architecture", "Scheme"])
            # Drop the unstable positional index from parsed logs.
            _table = _table.reset_index(drop=True)
        show_table(
            _table,
            f"continuity_{_plat.replace('frdm-', '').replace('-', '')}",
            mo,
            caption=_caption,
        )

    mo.md("_Exported `continuity_imx8mp` and `continuity_imx93`._")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    # 5 · CPU and NPU cost

    Only exports judged correct in §4 are included. CPU results come from the same
    board with its delegate disabled and XNNPACK enabled. Values are median latency
    with p95, plus `dAP`.

    Scope: default export at the reference configuration. Board CPU trees do not
    yet include per-class NMS runs.
    """)
    return


@app.cell(hide_code=True)
def _(br, mo, show_table, view):
    show_table(
        br.device_latency_table(
            view(nms=br.DEFAULT_NMS, ref=True, deployable=True), nms=br.DEFAULT_NMS
        ),
        "device_latency",
        mo,
        caption="Median latency and throughput for correct exports at the "
        "reference configuration, comparing each board's CPU and NPU.",
    )
    return


@app.cell(hide_code=True)
def _(br, mo, show_fig, view):
    show_fig(
        br.plot_device_latency(view(nms=br.DEFAULT_NMS, ref=True, deployable=True)),
        "device_latency",
        mo,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### 5.2 · Accuracy against latency

    Panels separate architectures because they occupy distinct AP ranges. Colour
    identifies the platform; shape identifies quantization method and granularity.
    The CPU reference is excluded from the deployment comparison.

    The i.MX8M Plus has no float point: its 32 FP32 delegate runs returned
    non-finite boxes and are `unscoreable`, not missing.
    """)
    return


@app.cell(hide_code=True)
def _(br, mo, show_fig, view):
    show_fig(
        br.plot_accuracy_latency(
            view(nms=br.DEFAULT_NMS, ref=True, deployable=True), nms=br.DEFAULT_NMS
        ),
        "accuracy_latency_tradeoff",
        mo,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### 5.3 · Power, energy and resource use

    Resource measurements use a 120 s steady-state run (`ave resources`) joined to
    the FNB58 power trace by `scripts/power_report.py`. Scope matches §5:
    multi-class models trained full-frame.

    Power values require verified clock alignment between the board and lab host.
    `aligned` records the result; unverified values can appear plausible but are
    excluded.
    """)
    return


@app.cell(hide_code=True)
def _(RESOURCE_ROOT, json, pd):
    def load_power_summaries(root) -> pd.DataFrame:
        """Load power summaries and per-phase latency into one frame.

        Energy uses mean latency; run comparisons use median latency. Phase data
        come from each run's `run.json`. Older sweeps leave phase fields empty.
        """
        rows = []
        if not root.is_dir():
            return pd.DataFrame()

        def _phase_stats(run_dir, name, *fields):
            """`latency_phases.<name>.<field>` from a run directory, if recorded."""
            run_json = run_dir / "run.json"
            if not run_json.is_file():
                return {}
            try:
                phases = (json.loads(run_json.read_text()).get("latency_phases") or {})
            except Exception:
                return {}
            block = phases.get(name) or {}
            return {field: block.get(field) for field in fields}

        for summary_path in sorted(root.rglob("power_summary.json")):
            try:
                payload = json.loads(summary_path.read_text())
            except Exception:
                continue

            results = payload.get("runs", payload) if isinstance(payload, dict) else payload
            for result in results if isinstance(results, list) else []:
                if not isinstance(result, dict) or result.get("status") != "ok":
                    continue
                power = result.get("power") or {}
                resources = result.get("resources") or {}
                latency = result.get("latency") or {}
                alignment = result.get("alignment") or {}

                # The run directory is named after the run, beside the summary.
                _run_dir = summary_path.parent / str(result.get("run"))
                _invoke = _phase_stats(_run_dir, "invoke", "median_latency_ms", "mean_latency_ms")
                _net = _phase_stats(_run_dir, "net_of_resize", "median_latency_ms")
                _resize = _phase_stats(_run_dir, "resize", "median_latency_ms")
                _pre = _phase_stats(_run_dir, "preprocess", "median_latency_ms")
                _post = _phase_stats(_run_dir, "postprocess", "median_latency_ms")

                rows.append(
                    {
                        # <device>/<stamp>/power_summary.json
                        "device": summary_path.parent.parent.name,
                        # Distinguish repeated sweeps of the same board.
                        "stamp": summary_path.parent.name,
                        "run": result.get("run"),
                        "backend": result.get("backend"),
                        # Alignment determines whether power values are valid.
                        "aligned": alignment.get("verified"),
                        "state": alignment.get("state"),
                        "residual (s)": alignment.get("max_abs_residual_s"),
                        # Full `predict` pipeline.
                        "lat mean (ms)": latency.get("mean_latency_ms"),
                        "lat med (ms)": latency.get("median_latency_ms"),
                        # Graph execution only.
                        "invoke med (ms)": _invoke.get("median_latency_ms"),
                        "invoke mean (ms)": _invoke.get("mean_latency_ms"),
                        "net-resize med (ms)": _net.get("median_latency_ms"),
                        "resize med (ms)": _resize.get("median_latency_ms"),
                        "pre med (ms)": _pre.get("median_latency_ms"),
                        "post med (ms)": _post.get("median_latency_ms"),
                        "FPS": latency.get("throughput_fps"),
                        "P (W)": power.get("mean_w"),
                        "idle (W)": power.get("idle_w"),
                        "net P (W)": power.get("net_mean_w"),
                        "net mJ/inf": power.get("net_energy_per_inference_mj"),
                        "CPU %": resources.get("cpu_pct_mean"),
                        "RSS (MiB)": (
                            None
                            if resources.get("proc_rss_kb_max") is None
                            else resources["proc_rss_kb_max"] / 1024.0
                        ),
                        "temp max (C)": resources.get("temp_c_max"),
                    }
                )

        frame = pd.DataFrame(rows)

        if frame.empty:
            return frame

        # Keep the latest measurement for runs repeated across timestamped sweeps.
        frame = frame.sort_values(["device", "run", "stamp"], kind="stable")
        frame = frame.drop_duplicates(["device", "run"], keep="last")

        return frame.reset_index(drop=True).round(2)


    power_df = load_power_summaries(RESOURCE_ROOT)
    return (power_df,)


@app.cell
def _(br, power_df):
    # Parse configuration axes from model stems and match the §5 scope.
    power_scoped = br.annotate_resource_runs(power_df)
    if not power_scoped.empty:
        power_scoped = power_scoped[
            (power_scoped["classes"] == "mc") & (power_scoped["dataset"] == "phenobench")
        ]
    return (power_scoped,)


@app.cell
def _(NMS, br, power_scoped, runs, skipped, view):
    # Join correctness verdicts because power sweeps cannot detect plausible but
    # inaccurate output. Include control boards that have power measurements.
    _boards = sorted(
        p for p in runs["platform"].unique() if p.startswith(("frdm", "imx93"))
    )
    _matrix = br.deployability_matrix(
        view(nms="both", classes="mc", dataset="phenobench"),
        skipped,
        eval_tiling="untiled",
        platforms=_boards,
        nms=NMS,
    )

    # Join on configuration keys retained by both frames.
    _keys = [
        c
        for c in ("arch_label", "classes", "dataset", "size", "scheme")
        if c in _matrix.columns and c in power_scoped.columns
    ]

    if power_scoped.empty or _matrix.empty or not _keys:
        power_judged = power_scoped.assign(verdict="unknown")
    else:
        _verdicts = _matrix.melt(
            id_vars=_keys, var_name="device", value_name="verdict"
        )
        power_judged = power_scoped.merge(
            _verdicts, on=[*_keys, "device"], how="left"
        )
        # Mark power runs without an accuracy benchmark explicitly.
        power_judged["verdict"] = power_judged["verdict"].fillna("unknown")
    return (power_judged,)


@app.cell(hide_code=True)
def _(mo, power_scoped):
    # `misaligned` invalidates power values; `unverified` lacks the chirp check.
    # Derive legacy states from `aligned` when necessary.
    if power_scoped.empty:
        _counts, _total = {}, 0
    else:
        _state_col = power_scoped.get("state")
        if _state_col is None or _state_col.isna().all():
            _state_col = power_scoped["aligned"].map(
                {True: "verified", False: "misaligned"}
            )
        _counts = _state_col.fillna("unverified").value_counts().to_dict()
        _total = len(power_scoped)

    _ok = int(_counts.get("verified", 0))
    _bad = int(_counts.get("misaligned", 0))
    _soft = int(_counts.get("unverified", 0))

    if not _total:
        _power_state = mo.callout(
            mo.md(
                "_No sweeps under `resource_results/` — run "
                "`scripts/power_sweep.py --preset arch-matrix` then "
                "`scripts/power_report.py <sweep>`._"
            ),
            kind="neutral",
        )
    else:
        # Escalate when no run is verified or misalignment is prevalent.
        if _ok == 0 or _bad > _ok:
            _kind = "danger"
        elif _bad or _soft:
            _kind = "warn"
        else:
            _kind = "success"

        _power_state = mo.callout(
            mo.md(
                f"**{_ok} of {_total} runs have verified alignment.** "
                + (
                    "The sweeps may predate the check or require a new "
                    "`scripts/power_report.py` run. "
                    if _ok == 0
                    else ""
                )
                + (
                    f"**{_bad} misaligned** and excluded from power results. "
                    if _bad
                    else ""
                )
                + (
                    f"{_soft} unverified; alignment relies on the SSH probe. "
                    if _soft
                    else ""
                )
                + "Board latency, CPU, memory, and temperature do not depend on "
                "trace alignment."
            ),
            kind=_kind,
        )

    _power_state  # noqa: B018 - bare name is how a marimo cell renders a value
    return


@app.cell
def _(NMS, br, mo, power_judged, show_fig):
    # Use the selected NMS and reference resolution.

    show_fig(
        br.plot_resource_summary(
            power_judged[
                (power_judged["nms"] == NMS)
                & (~power_judged["device"].str.contains("_unpatched|_no-concat"))
                & (power_judged["size"].astype(str) == br.REFERENCE_CONFIG["size"])
            ]
        ),
        "resource_summary",
        mo,
    )
    return


@app.cell
def _(mo, power_judged, show_table):
    show_table(
        power_judged[
            (power_judged["aligned"] == True)  # noqa: E712
            & (power_judged["nms"] == "regnms")
            & (power_judged["device"].isin(["frdm-imx8mp", "frdm-imx8mp_unpatched", "frdm-imx93"]))
        ]
        .drop(
            columns=[
                c
                for c in (
                    "run",
                    "stamp",
                    "backend",
                    "state",
                    "classes",
                    "dataset",
                    "nms",
                    "aligned",
                    "arch_label",
                    "precision",
                    "quant",
                    "granularity",
                )
                if c in power_judged
            ]
        )
        .reset_index(drop=True),
        "power_summary",
        mo,
        caption="Steady-state power, energy, and resource use across input "
        "resolutions. The reference is $320\\times320$. Only \\texttt{ok} "
        "verdicts are valid operating points; other rows retain measured power "
        "and latency but do not represent a working detector.",
        split_by="device",
    )
    return


@app.cell
def _(mo):
    mo.md("""
    ### 5.4 &middot; Graph-only latency

    Section 5.1 times the full `predict` call, including resize, input
    quantization, output readback, and rescaling. These CPU-side steps add
    10&ndash;15&nbsp;ms to both CPU and delegated runs.

    `invoke` measures the exported graph, not pure accelerator time.
    `TFLite_Detection_PostProcess` and two dequantize nodes remain on the CPU within
    this phase. `CPU-side (ms)` is `preprocess + postprocess`; `resize` is already
    included in `preprocess`.

    Removing pipeline overhead roughly doubles most speedups. Per-channel speedup
    remains about 0.7, so the delegated graph is slower than the board CPU.
    """)
    return


@app.cell
def _(NMS, br, mo, power_judged, show_table):
    show_table(
        # Exclude timings from incorrect exports.
        br.accelerator_latency_table(
            power_judged, nms=NMS, size=br.REFERENCE_CONFIG["size"]
        ),
        "accelerator_latency",
        mo,
        caption="Delegated and board-CPU latency at the reference configuration, "
        "for the full \\texttt{predict} call and graph-only \\texttt{invoke}. "
        "Invoke includes CPU detection post-processing; \\texttt{CPU-side (ms)} "
        "is \\texttt{preprocess} plus \\texttt{postprocess}.",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    # 6 · Ablations

    The overview varies one axis at a time, plus the matched tiled configuration.
    `Float AP` is absolute; `Conversion`, `NMS swap`, and `PTQ` are changes from the
    preceding rung; `QAT reclaim` is QAT minus PTQ with per-tensor weights.

    `NPU (ms)` is per inference. A tiled inference covers one ninth of a frame, so
    do not compare its latency with full-frame inference.
    """)
    return


@app.cell(hide_code=True)
def _(br, mo, show_table, view):
    show_table(
        br.story_ablation_table(view(nms="both")),
        "story_ablation",
        mo,
        caption="Deployment-stage effects of single-axis deviations from the "
        "reference configuration.",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### 6.1 · Single-class

    Single-class models omit `crop` and detect only the smaller `weed` objects.
    The area breakdown uses COCO thresholds (`APS` ≤ 32²; `APL` > 96²).

    The quantization figure compares crop and weed AP from the multi-class model
    with weed AP from a separately trained single-class model.
    """)
    return


@app.cell(hide_code=True)
def _(br, view):
    arch_df = view(platform=br.CPU_REFERENCE_PLATFORM, ref={"classes": None})
    return (arch_df,)


@app.cell
def _(NMS, br, mo, show_fig, view):
    show_fig(
        br.plot_class_regime_quantization(view(ref={"classes": None}), nms=NMS),
        "qat_class_regime",
        mo,
    )
    return


@app.cell(hide_code=True)
def _(NMS, arch_df, br, mo, show_fig):
    show_fig(br.plot_single_vs_multiclass(arch_df, nms=NMS), "single_vs_multiclass", mo)
    return


@app.cell(hide_code=True)
def _(NMS, arch_df, br, mo, show_fig):
    show_fig(br.plot_per_class_ap(arch_df, nms=NMS), "per_class_ap", mo)
    return


@app.cell(hide_code=True)
def _(NMS, arch_df, br, mo, show_fig):
    show_fig(br.plot_ap_by_area(arch_df, nms=NMS), "ap_by_area", mo)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### 6.2 · Tiling and apparent object scale

    Each tile is a 512² crop of a 1024² frame (3×3, 50% overlap), equivalent to
    2× linear magnification. Training and evaluation tiling therefore model object
    scale rather than end-to-end tiled inference.

    | cell | question |
    |---|---|
    | trained full / eval full | reference |
    | trained full / eval tiled | larger objects only at evaluation |
    | trained tiled / eval full | transfer from larger training objects |
    | trained tiled / eval tiled | matched narrow-field configuration |

    Mismatched cells test scale shift. The matched tiled cell represents a trained
    narrow-field sensor configuration.

    `d` is relative to the same architecture's reference cell. The two axes
    interact, so all four combinations are reported.

    Latency is omitted because a tile covers one ninth of a frame.
    """)
    return


@app.cell(hide_code=True)
def _(br, mo, show_table, view):
    show_table(
        br.tiling_cross_table(view(nms="both")),
        "tiling_cross",
        mo,
        caption="Training and evaluation tiling combinations for multi-class float "
        "TFLite on the CPU reference. Changes are relative to each architecture's "
        "full-frame reference.",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 6.3 · NMS latency

    Fast NMS uses one class-agnostic suppression pass; regular NMS uses one pass per
    class. Their single-class latency difference estimates run-to-run drift. The
    reported saving subtracts this control from the multi-class difference.

    INT8 only. Including FP32 adds substantial timing variance to a millisecond-scale
    effect.

    | column | meaning |
    |---|---|
    | `mc pairs` / `sc pairs` | matched fast/regular NMS pairs |
    | `dLatency mc` | mean fast minus regular latency; negative means fast is quicker |
    | `sc drift` | corresponding single-class difference |
    | `NMS saving` | `dLatency mc` minus `sc drift` |
    | `SE` | standard error of that difference, both arms combined |
    | `sigma` | \|saving\| / SE |
    | `95% CI` | interval for the saving |
    | `resolved` | whether the interval excludes zero |
    | `sc \|drift\| worst` | largest single-pair drift |

    An interval spanning zero does not distinguish the effect from drift.
    """)
    return


@app.cell(hide_code=True)
def _(br, mo, show_table, view):
    show_table(
        br.nms_latency_tradeoff_table(view(nms="both")),
        "nms_latency_tradeoff",
        mo,
        caption="Fast-NMS latency difference by platform and architecture, adjusted "
        "for drift using matched single-class runs.",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### 6.4 · Input resolution

    Compare 320, 512, and 1024 inputs at the reference configuration. Accuracy is
    measured on the CPU reference; latency and FPS are reported per board. Host
    latency is a lower bound, not a device proxy.

    Other axes are fixed to avoid mixing resolution with the scale shift from
    tiling (§6.2). The trade-off figure uses one export scheme and a logarithmic
    latency axis.

    FPNLite at 1024 and i.MX8M Plus runs above 512 are incomplete. Missing device
    measurements remain empty.
    """)
    return


@app.cell(hide_code=True)
def _(NMS, br, mo, show_table, view):
    show_table(
        br.resolution_ladder_table(
            view(),
            nms=NMS,
            latency_platforms=("frdm-imx8mp", "frdm-imx93"),
        ),
        "resolution_ladder",
        mo,
        caption="Detection quality and median latency by input resolution and "
        "export scheme, measured on the CPU reference and each accelerator.",
    )
    return


@app.cell
def _(NMS, br, mo, show_fig, view):
    show_fig(br.plot_resolution_ap(view(), nms=NMS), "resolution_ap", mo)
    return


@app.cell
def _(NMS, br, mo, show_fig, view):
    show_fig(br.plot_resolution_tradeoff(view(), nms=NMS, latency_ticks=(30, 50, 100, 200, 500)), "resolution_tradeoff", mo)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    # 7 · Auxiliary detectors

    YOLOv7-tiny is available only as a tiled-trained 512 export and uses different
    post-processing. It is excluded from the controlled SSD comparison. YOLOX has
    fine-tuned ONNX artifacts but no TFLite benchmarks.
    """)
    return


@app.cell
def _(br, mo, show_table, view):
    show_table(
        br.platform_metrics_table(view(archs="aux"), br.CPU_REFERENCE_PLATFORM),
        "auxiliary_detectors",
        mo,
        caption="Auxiliary YOLO detectors on the CPU reference, excluded from the "
        "controlled SSD comparison.",
    )
    return


@app.cell(hide_code=True)
def _(br, mo):
    mo.md(f"""
    # 8 · Appendix: coverage and full matrices

    A full run covers every trained variant, export scheme, input regime, and
    target platform:

    - schemes: {", ".join(f"`{br.scheme_label(br.scheme_name(*s))}`" for s in br.DEFAULT_SCHEMES)}
    - input regimes: {", ".join(f"`{t}`" for t in br.DEFAULT_EVAL_TILINGS)}
    - platforms: {", ".join(f"`{p}`" for p in br.DEFAULT_EXPECTED_PLATFORMS)}

    Coverage uses the default NMS export; NMS-pair completeness is reported
    separately. `tf-savedmodel` is excluded because it has no quantized export
    schemes, but remains in the per-platform tables.
    """)
    return


@app.cell(hide_code=True)
def _(br, mo, view):
    mo.ui.table(
        # Include all architectures and both NMS variants, excluding controls.
        br.nms_pair_coverage(view(archs=None, nms="both")),
        label="Post-processing pair coverage",
        selection=None,
    )
    return


@app.cell(hide_code=True)
def _(ARTIFACTS_TF, NMS, br, view):
    model_variants = br.discover_model_variants(ARTIFACTS_TF)
    coverage = br.build_coverage(view(archs=None), model_variants, nms=NMS)
    return (coverage,)


@app.cell
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
        "(x: complete; -: missing).",
    )
    return


@app.cell(hide_code=True)
def _(br, coverage, mo, show_fig):
    show_fig(br.plot_coverage(coverage), "benchmark_coverage", mo)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### 8.2 · Per-platform result tables

    One full COCO table per platform for thesis Appendix D.
    """)
    return


@app.cell(hide_code=True)
def _(mo, view):
    platforms = sorted(view(archs=None)["platform"].unique())
    platform_ui = mo.ui.dropdown(
        options=platforms,
        value=platforms[0] if platforms else None,
        label="platform",
    )
    platform_ui  # noqa: B018 - bare name is how a marimo cell renders a widget
    return (platform_ui,)


@app.cell(hide_code=True)
def _(br, mo, platform_ui, view):
    mo.ui.table(
        br.platform_metrics_table(view(archs=None), platform_ui.value)
        if platform_ui.value
        else None,
        label=f"platform_metrics_{platform_ui.value}",
        selection=None,
    )
    return


@app.cell(hide_code=True)
def _(br, mo, show_table, view):
    # Export every platform, including controls, independent of the dropdown.
    _export_platforms = sorted(view(archs=None, controls=True)["platform"].unique())

    for _platform in _export_platforms:
        show_table(
            br.platform_metrics_table(view(archs=None, controls=True), _platform),
            f"platform_metrics_{_platform}",
            mo,
            caption=f"Full COCO metrics for every run on {_platform}.",
        )
    mo.md(
        "_Exported per-platform tables: "
        + (", ".join(f"`{p}`" for p in _export_platforms) or "—")
        + "._"
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### 8.3 · Cross-configuration degradation ladder

    Deployment-stage AP changes for every configuration, extending the §2–§3
    preparation chain through accelerator execution.
    """)
    return


@app.cell(hide_code=True)
def _(br, mo, runs, show_table):
    # Use both NMS variants to compute `nms-swap`.
    show_table(
        br.degradation_ladder_table(runs),
        "degradation_ladder_untiled",
        mo,
        caption="Detection AP by deployment stage for every full-frame PTQ "
        "configuration, separating conversion, NMS, quantization, and delegation.",
    )
    return


@app.cell(hide_code=True)
def _(br, mo, show_table, view):
    show_table(
        br.nms_substitution_table(view(nms="both", archs=None)),
        "nms_substitution_detail",
        mo,
        caption="Per-configuration effect of the post-processing substitution.",
    )
    return


@app.cell(hide_code=True)
def _(br, mo, show_table, view):
    show_table(
        br.latency_table(view(archs=None)),
        "latency_by_scheme",
        mo,
        caption="Median and p95 inference latency and throughput by platform, "
        "backend, and export scheme.",
    )
    return


@app.cell(hide_code=True)
def _(br, mo, show_table, view):
    show_table(
        br.master_table(view(archs=None, nms="both")),
        "benchmark_master",
        mo,
        caption="Full benchmark matrix across platforms, architectures, class and "
        "input regimes, quantization schemes, and NMS variants. Control trees are "
        "excluded.",
    )
    return


@app.cell
def _(mo):
    mo.md("""
    ### 8.5 &middot; Detection examples

    Each row shows ground truth and detector outputs for one frame. Frames are
    sorted by weed count and sampled at fixed intervals for a stable range of scene
    densities.

    Boxes use a display threshold of 0.4; AP still uses all detections down to the
    exported score floor of 0.05. The three galleries use the same frames and
    thesis text width.
    """)
    return


@app.cell(hide_code=True)
def _(BENCHMARK_ROOT, NMS):
    # Match the 418.3 pt thesis text width to preserve label size.
    THESIS_TEXT_WIDTH_IN = 5.79


    def gallery_prediction(platform, arch, scheme, size="320"):
        """`predictions.json` of one untiled multi-class run, for the galleries."""
        return (
            BENCHMARK_ROOT
            / platform
            / f"untiled_{arch}_mc_phenobench_{size}_{scheme}_{NMS}"
            / "predictions.json"
        )


    return THESIS_TEXT_WIDTH_IN, gallery_prediction


@app.cell
def _(
    GT_ANNOTATIONS,
    IMAGE_ROOT,
    THESIS_TEXT_WIDTH_IN,
    br,
    gallery_prediction,
    mo,
    show_fig,
):
    show_fig(
        br.plot_detection_gallery(
            GT_ANNOTATIONS,
            IMAGE_ROOT,
            {
                # Reference export evaluated on the CPU reference.
                _label: gallery_prediction(
                    br.CPU_REFERENCE_PLATFORM, _arch, "int8_qat_per-tensor"
                )
                for _arch, _label in (
                    ("ssd-mn2", "SSD MobileNetV2"),
                    ("ssd-mn2-fpnlite", "SSD MobileNetV2 FPNLite"),
                )
            },
            n_images=4,
            score_threshold=0.4,
            figure_width=THESIS_TEXT_WIDTH_IN,
        ),
        "detection_gallery",
        mo,
        # Raster output is sufficient for 256 px source panels.
        formats=("png",),
        dpi=200,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### 8.6 &middot; Collapsed delegate output

    The same export is shown on the board CPU and before and after the Mesa/Teflon
    operator-support changes. The preceding delegate build returns unrelated boxes;
    the board CPU result isolates the fault to delegation.
    """)
    return


@app.cell(hide_code=True)
def _(
    GT_ANNOTATIONS,
    IMAGE_ROOT,
    THESIS_TEXT_WIDTH_IN,
    br,
    gallery_prediction,
    mo,
    show_fig,
):
    show_fig(
        br.plot_detection_gallery(
            GT_ANNOTATIONS,
            IMAGE_ROOT,
            {
                # Same board with CPU, current delegate, and preceding delegate.
                _label: gallery_prediction(
                    _platform, "ssd-mn2-fpnlite", "int8_ptq_per-tensor"
                )
                for _platform, _label in (
                    ("frdm-imx8mp_cpu", "Board CPU"),
                    ("frdm-imx8mp", "NPU (patched)"),
                    ("frdm-imx8mp_unpatched", "NPU (preceding)"),
                )
            },
            n_images=4,
            score_threshold=0.4,
            figure_width=THESIS_TEXT_WIDTH_IN,
        ),
        "detection_gallery_delegate_build",
        mo,
        formats=("png",),
        dpi=200,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### 8.7 &middot; FPNLite failure at 1024×1024

    The 1024&times;1024 INT8 export delegates as one region without errors but retains
    less than one tenth of its host accuracy. On the i.MX93, 99.3% of detections
    score exactly 0.5 because the class logits are zero. Both boards return almost
    nothing above the display threshold, although the equivalent 512&times;512 export
    is correct.
    """)
    return


@app.cell(hide_code=True)
def _(
    GT_ANNOTATIONS,
    IMAGE_ROOT,
    THESIS_TEXT_WIDTH_IN,
    br,
    gallery_prediction,
    mo,
    show_fig,
):
    show_fig(
        br.plot_detection_gallery(
            GT_ANNOTATIONS,
            IMAGE_ROOT,
            {
                # Compare the same file with its host reference on both NPUs.
                _label: gallery_prediction(
                    _platform, "ssd-mn2-fpnlite", "int8_qat_per-tensor", size="1024"
                )
                for _platform, _label in (
                    (br.CPU_REFERENCE_PLATFORM, "Host reference"),
                    ("frdm-imx93", "i.MX93 NPU"),
                    ("frdm-imx8mp", "i.MX8M Plus NPU"),
                )
            },
            n_images=4,
            score_threshold=0.4,
            figure_width=THESIS_TEXT_WIDTH_IN,
        ),
        "detection_gallery_resolution_failure",
        mo,
        formats=("png",),
        dpi=200,
    )
    return


@app.cell(hide_code=True)
def _(FIG_DIR, SAVE_ARTIFACTS, TAB_DIR, mo):
    mo.md(f"""
    ---
    Artifact export: **{"enabled" if SAVE_ARTIFACTS else "disabled"}**.
    Figures: `{FIG_DIR}` · Tables: `{TAB_DIR}`.
    """)
    return


if __name__ == "__main__":
    app.run()
