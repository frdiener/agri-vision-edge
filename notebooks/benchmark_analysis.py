import marimo

__generated_with = "0.23.9"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    # Benchmark Analysis — Weed Detection on Embedded NPUs

    Tables and figures for Chapter 6, built from the artifacts of `ave benchmark`
    (`latency.json`, `runtime.json`, `predictions.json`) and `ave evaluate`
    (`metrics.json`, and `metrics_faithful.json` under `--faithful`). The analysis
    itself is `agri_vision_edge.evaluation.benchmark_report`, imported as `br`.

    ## The argument

    | § | question | artifact |
    |---|---|---|
    | **0** | which runs are admissible | `sanity_checks` |
    | **1** | float accuracy of the trained detector | `baseline` |
    | **2** | cost of conversion and post-training quantization | `preparation_ladder`, `nms_substitution` |
    | **3** | how much of it QAT reclaims | `qat_reclaim` |
    | **4** | which exports the accelerator executes correctly | `deployability` |
    | **5** | what the survivors cost, CPU against NPU | `device_latency`, `nms_latency_tradeoff` |
    | **6** | whether class count or tiling changes any of it | `story_ablation` |

    §1-§5 hold one configuration fixed; §6 varies it one axis at a time. §7 holds
    measurements not yet joined to the argument.

    ## Reference configuration

    > **multi-class - trained full-frame - evaluated full-frame - 320x320**
    > (`br.REFERENCE_CONFIG`)

    ## Run grammar

    ```
    benchmark_results/<platform>/<run>/{metrics,metrics_faithful,latency,runtime}.json

    <tiling>_<arch>_<classes>_<dataset>_<size>_<precision>_<quant>[_<granularity>]_<nms>
    untiled_ssd-mn2-fpnlite_mc_phenobench-tiled_320_int8_qat_per-tensor_fastnms
    ```

    | axis | token | meaning |
    |---|---|---|
    | **trained on** | `<dataset>` = `phenobench` / `phenobench-tiled` | finetuning data |
    | **evaluated on** | `<tiling>` = `untiled` / `tiled` | input regime at benchmark time |
    | **export scheme** | `<precision>_<quant>[_<granularity>]` | quantization that produced the file |
    | **post-processing** | `<nms>` = `fastnms` / `regnms` | NMS compiled into the exported graph |

    `fastnms` and `regnms` are a matched pair off one checkpoint, graph and
    calibration set, and are therefore identical in every field a table groups on:
    unscoped, 224 of 590 configuration groups carry a silent duplicate. Every view
    below resolves to one of the two.

    ## Metric families

    - **pycocotools** (`AP`, `AP50`, `weed_AP`, ...) - every comparison made here.
    - **official PhenoBench** (`faithful_*`) - the upstream torchmetrics evaluator,
      cited only against the published baselines in §1. A separate metric stack,
      not comparable to the above.

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

    # Where `power_sweep.py` / `power_report.py` land. Absent until the power
    # measurements are run -- §5.3 degrades to a note rather than failing.
    RESOURCE_ROOT = REPO_ROOT / "resource_results"

    # Write figures/tables to the repo on run (set False to preview only).
    SAVE_ARTIFACTS = True

    br.apply_publication_style()
    return (
        ARTIFACTS_TF,
        BENCHMARK_ROOT,
        FIG_DIR,
        RESOURCE_ROOT,
        SAVE_ARTIFACTS,
        TAB_DIR,
    )


@app.cell(hide_code=True)
def _(BENCHMARK_ROOT, br):
    # One frame holds every run. Each table and figure below declares its own
    # scope through `view(...)`; `runs` itself appears only where the scope is
    # literally everything, which is the integrity gate of §0.
    runs, skipped = br.load_benchmark_results(BENCHMARK_ROOT)

    # The configuration this analysis presents: per-class NMS. Both variants are
    # deployable and both were flashed; `br.DEFAULT_NMS` stays what `ave convert`
    # emits unasked (`fastnms`). The two differ in AP and in latency -- §5.3.
    NMS = br.REGULAR_NMS

    # Kept out of every default view: a second copy of the SavedModel rung, which
    # would duplicate it in any aggregation grouped by stage. It carries the NMS
    # score floor removed, and answers a question settled elsewhere.
    CONTROL_TREES = ("tf-savedmodel-nms0",)


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

        nms         `regnms` per-class (presented) | `fastnms` toolchain default | `both`
        archs       `primary` the two compared SSDs | `aux` the YOLO reference | None
        controls    admit CONTROL_TREES
        ref         pin `br.REFERENCE_CONFIG`; pass a dict to override single axes
        deployable  drop what §4 measured as executed incorrectly on its target
        **where     equality filter on any column; a list/tuple/set means membership

        `deployable` is opt-in rather than default so that every comparison states
        in its own scope line what it excluded.
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
        # Before any `where`: the verdict is relative to the CPU reference tree,
        # which a platform filter would have removed.
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

    return NMS, runs, skipped, view


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
def _(NMS, br, mo, runs, skipped, view):
    _platforms = sorted(runs["platform"].unique())
    _archs = sorted(runs["arch_label"].unique())
    _sizes = sorted(runs["size"].dropna().unique())

    # Only a tree that carries NMS tokens can be missing one. The SavedModel rung
    # has none: it *is* the checkpoint's per-class suppressor, with no fast-NMS
    # counterpart to lack, which is why `select_nms` passes untokened rows through
    # either scope untouched.
    _tokened = runs[runs["nms"].notna()]
    _missing = [
        p
        for p in sorted(_tokened["platform"].unique())
        if not _tokened.loc[_tokened["platform"] == p, "nms"].eq(NMS).any()
    ]

    mo.md(f"""
    ## Data

    **{len(runs)}** runs over **{len(_platforms)}** platforms
    ({", ".join(f"`{p}`" for p in _platforms)}); architectures
    {", ".join(_archs)}; input sizes {", ".join(f"`{s}`" for s in _sizes)}.

    `view()` with no arguments resolves to **{len(view())}** of them — `{NMS}`
    post-processing, the two compared architectures, control trees out.
    A further **{len(skipped)}** run directories carried no `metrics.json` (§0.2).

    Reference configuration: **{br.REFERENCE_CONFIG}**.

    {"**Carry only the other post-processing variant:** " + ", ".join(f"`{p}`" for p in _missing) +
     " — sections scoped to those trees render empty until their `" + NMS + "` sweeps land."
     if _missing else "Every tree that distinguishes the two variants carries the presented one."}
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 0 · Integrity gate

    `pycocotools` matches a detection with `if iou < threshold: continue`. A `NaN`
    IoU fails that test, so non-finite boxes are accepted at every threshold and
    the run scores *higher* than a working one — measured `AP 85.8`, fingerprint
    `AP == AP50`.

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

    Evaluated on `runs` unscoped. Empty means nothing was flagged, not that
    everything is correct. `backend` records that the delegate library loaded, not
    which subgraph it accepted.
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

    **{len(skipped)}** run directories, all float graphs on an i.MX8M Plus delegate
    tree. `ave evaluate` withholds `metrics.json` on non-finite boxes; §4 carries
    them as `unscoreable`.
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
    # 1 · The float baseline

    Fine-tuned float accuracy, taken from the checkpoint's own graph before
    TensorFlow Lite is involved (`benchmark_results/tf-savedmodel/`, training-time
    decoding and per-class NMS), beside the detectors published with PhenoBench.
    Official PhenoBench metrics (`faithful_*`), the only table here that does not
    use pycocotools — the published rows were produced with that evaluator.

    Rows are the input-resolution ladder at multi-class, full-frame training and
    evaluation. It is ragged: `artifacts/tf` currently holds 320/512/1024 for the
    plain detector and 320/512 for FPNLite.

    Upstream rows are the withheld official test split at full resolution; *this
    work* is the internal test split derived from the validation partition. The
    `Source` column carries the distinction.
    """)
    return


@app.cell(hide_code=True)
def _(br, mo, show_table, view):
    show_table(
        br.baseline_table(view()),
        "baseline",
        mo,
        caption="Fine-tuned float accuracy of the two detectors across input "
        "resolution, against the baselines published with PhenoBench; multi-class, "
        "full-frame evaluation, official PhenoBench metrics. Upstream figures are "
        "on the withheld official test set, those obtained here on the internal "
        "test split derived from the official validation partition; the comparison "
        "is indicative rather than a leaderboard result.",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    # 2 · Conversion and post-training quantization

    What it costs to get from the checkpoint to a deployable INT8 file, measured on
    the CPU reference with the accelerator held out — a statement about the
    *export*. §4 and §5 bring the hardware in.
    """)
    return


@app.cell(hide_code=True)
def _(br, mo, view):
    # §2, §3 and §5 report one CPU curve as the unaccelerated reference for every
    # board. Legitimate only while the per-board `<board>_cpu` trees agree with it,
    # so it is checked rather than assumed. Scoped to the default export: the
    # `_cpu` trees carry no per-class runs, and agreement is a property of the
    # hardware, not of the post-processing.
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

    Every rung runs **{br.NMS_LABELS[NMS].lower()}**, so the float step prices the
    format conversion alone and the INT8 steps price precision alone.

    | rung | isolates |
    |---|---|
    | Float SavedModel | the trained checkpoint (§1) |
    | ↓ Float TFLite | conversion, at matched post-processing |
    | ↓ INT8 PTQ, per-channel / per-tensor | precision, at each weight granularity |

    INT8 rows are quoted against the float TFLite rung rather than against each
    other: they are alternative exports of one model, not a chain.
    """)
    return


@app.cell(hide_code=True)
def _(NMS, br, mo, show_table, view):
    show_table(
        br.preparation_ladder_table(view(), include_qat=False, nms=NMS),
        "preparation_ladder_ptq",
        mo,
        caption="Accuracy along the conversion and post-training quantization "
        "ladder at the reference configuration, all rungs at per-class NMS. INT8 "
        "rows are quoted against the float TFLite rung.",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### 2.2 · Granularity across the matrix

    The per-scheme view over every variant, beside the single reference cell.
    Expected ordering is `fp32 >= per-channel >= per-tensor`.
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
        caption="Detection quality per quantization scheme on the CPU reference, "
        "multi-class and trained full-frame, across input resolution; relative "
        "change is against each variant's own FP32 baseline.",
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
    # 3 · What QAT reclaims

    `Reclaimed` is QAT minus PTQ at equal granularity; `Reclaimed %` puts that
    against the PTQ deficit, so ~100 % closes it and a negative value means QAT
    lost ground.

    `Reclaimed %` is empty where the PTQ deficit was under
    **{br.QAT_RECLAIM_MIN_DEFICIT} AP** — the denominator is then noise and the
    ratio explodes. The test uses the unrounded cost, so the FPNLite per-channel
    row is suppressed at a displayed `-0.10` (it is -0.0997, and its +0.45 reclaim
    over that would read 451 %). `Reclaimed` still carries the absolute change.

    Reference configuration, per-class NMS, CPU reference.
    """)
    return


@app.cell(hide_code=True)
def _(NMS, br, mo, show_table, view):
    show_table(
        br.qat_reclaim_table(view(), nms=NMS),
        "qat_reclaim",
        mo,
        caption="What quantization-aware training recovers of the post-training "
        "quantization deficit, per architecture and weight granularity, at the "
        "reference configuration.",
    )
    return


@app.cell
def _(mo):
    mo.md("""
    ### 3.1 · The full preparation chain

    §2.1's ladder continued through both quantization methods and both weight
    granularities. INT8 rows are quoted against the float TFLite rung, not against
    each other.
    """)
    return


@app.cell(hide_code=True)
def _(NMS, br, mo, show_table, view):
    show_table(
        br.preparation_ladder_table(view(), nms=NMS),
        "preparation_ladder",
        mo,
        caption="Accuracy along the full preparation chain at the reference "
        "configuration, conversion through both quantization methods and both "
        "weight granularities, all rungs at per-class NMS.",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### 3.2 · The post-processing alternative, in accuracy

    Both §2's conversion rung and §3's QAT rows were measured at per-class NMS.
    The exported default is the other one: `ave convert` emits
    `use_regular_nms=False` unless told otherwise, making a single class-agnostic
    pass over each anchor's argmax class where the checkpoint suppresses per class.
    SSD regresses one box per anchor and shares it across classes, so an anchor's
    crop and weed hypotheses are the *same box* and only the higher-scoring one
    survives — the cost falls on the suppressed class.

    At the reference configuration, multi-class and full-frame. At one class the
    two algorithms coincide and the difference is identically zero; §0's
    `nms-control-broken` check enforces that, and §6 carries the single-class and
    tiled rows.

    Both variants are deployable and both were flashed. What the default buys in
    latency is priced in §5.3; this is the half it costs.
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
        caption="Accuracy cost of the exported post-processing substitution (fast "
        "NMS minus per-class NMS) per architecture and input resolution, at the "
        "reference configuration.",
    )
    return


@app.cell(hide_code=True)
def _(br, mo, show_fig, view):
    # Pinned to the reference cell: the figure groups by scheme, so a frame
    # spanning resolutions would average them into each bar without saying so.
    show_fig(br.plot_nms_substitution(view(nms="both", ref=True)), "nms_substitution", mo)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    # 4 · What the accelerators execute correctly

    Verdicts are relative to the **same file on the CPU reference**, which
    separates "this accelerator broke it" from "this export was always weak".

    | verdict | meaning |
    |---|---|
    | `ok` | reproduces the same file on the CPU reference |
    | `degraded` | below 90 % of it |
    | `collapsed` | below 50 % — loads, runs, plausible boxes, near-zero score |
    | `unscoreable` | ran, but the output is not a detection; §0 withheld `metrics.json` |
    | `-` | not benchmarked |

    `unscoreable` is recovered from §0.2's skip list: a refused run leaves no row in
    the results frame, so without that recovery the cell would read as "never
    benchmarked".
    """)
    return


@app.cell(hide_code=True)
def _(NMS, br, runs, skipped, view):
    # Boards only -- a `<board>_cpu` tree is the comparison, not a target, and the
    # `_unpatched` trees are a different delegate build (§4.3). Scoped to the
    # reference training configuration, as §2 and §3 are.
    deploy_boards = [
        p
        for p in sorted(runs["platform"].unique())
        if p.startswith("frdm") and not p.endswith(("_cpu", "_unpatched"))
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
        caption="Which exports each target executes correctly (full-frame "
        "input, deployed post-processing), judged against the same file on the "
        "CPU reference.",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### 4.2 · What the delegates do to the exports that survive

    INT8 only — the sole precision a delegate accelerates. Equal bar heights mean
    the accelerator reproduced its CPU reference.
    """)
    return


@app.cell(hide_code=True)
def _(NMS, br, mo, show_fig, view):
    # Reference cell: the boards carry 320 only, so a wider scope adds rows with a
    # CPU bar and no NPU bar to a CPU-vs-NPU figure.
    show_fig(
        br.plot_backend_effect(view(nms="both", ref=True), nms=NMS),
        "backend_effect",
        mo,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### 4.3 · The Mesa/Teflon operator-support delta

    `frdm-imx8mp_unpatched` is the same board with the delegate built before the
    operator support added in thesis §5.5, so the pair prices that work in two
    currencies: configurations that changed verdict, and what the ones that already
    worked now cost.

    Rows are resolved **per architecture**. The plain head and FPNLite do not
    respond to the rebuild alike — on per-tensor it moved `ssd-mn2` by −30.5 % and
    FPNLite by −60.6 % — so a pooled median would belong to neither.

    `ms before` / `ms after` cover every configuration **both builds timed**, not
    only the ones both executed correctly. A collapsed run's wall clock is still a
    real measurement of dispatch and CPU↔NPU tensor movement, which is exactly the
    cost this rebuild shifts; it is the *accuracy* of a collapsed run that is
    worthless, not its clock. Read the two medians against `Before` / `After`,
    which say what correctness each side had — and note that FPNLite, which
    fragments hardest under the older build, is also what the rebuild reclaims most
    from.

    Scoped to the i.MX8M Plus at **320×320**, default export. The i.MX93 runs a
    different backend (Ethos-U rather than Vivante) whose rebuild is a separate
    question, and the resolution ladder is §6.4's axis — the `_unpatched` tree was
    only ever swept at 320 in any case.
    """)
    return


@app.cell(hide_code=True)
def _(br, mo, show_table, skipped, view):
    # Scope: the i.MX8M Plus pair only (the i.MX93 delegate is a different backend
    # and its rebuild is not this question), 320 only (the resolution ladder is
    # §6.4's axis, not this one), default export (`_unpatched` carries no per-class
    # runs).
    #
    # `br.CPU_REFERENCE_PLATFORM` stays in the frame although it is never a column:
    # every verdict is scored against it, and `deployability_matrix` falls back to
    # `ok` for a configuration it cannot find a baseline for. Drop it and the whole
    # table reads `ok` -> `ok`, `Fixed = 0` -- the rebuild appears to have done
    # nothing.
    #
    # `paired` latency, not `ok-both`: a collapsed run's wall clock is still a real
    # measurement of dispatch and CPU<->NPU tensor movement, which is the cost the
    # rebuild is meant to shift.
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
        caption="Effect of the Mesa/Teflon operator-support work on the i.MX8M "
        "Plus at 320x320, per architecture and export scheme: verdict transitions, "
        "and median latency over the configurations both delegate builds timed.",
    )
    return


@app.cell(hide_code=True)
def _(BENCHMARK_ROOT, dg):
    # Continuity is read from the delegate's own verbose diagnostics
    # (`delegate_debug.log`), not from `metrics.json`, so it needs its own parse
    # rather than a slice of `runs`.
    delegation = dg.load_delegation(BENCHMARK_ROOT)
    return (delegation,)


@app.cell(hide_code=True)
def _(br, delegation, dg, mo, show_table):
    # One row per (architecture, scheme). Continuity is a property of the graph
    # the delegate was handed, so it cannot vary with class regime, NMS variant
    # or evaluation tiling -- and does not: within every (architecture, scheme)
    # group of this frame, `Ops`, `Delegated`, `K` and `Largest` each take
    # exactly one distinct value. The repeats are the same measurement seen
    # again, so one is kept rather than averaged.
    #
    # The auxiliary YOLO exports are dropped for the same reason they are
    # everywhere else: they are a reference point, not part of the controlled
    # SSD comparison.
    _primary = [br.ARCH_LABELS[a] for a in br.PRIMARY_ARCHS]

    _captions = {
        "frdm-imx8mp": (
            "Delegation continuity on the i.MX8M Plus, full-frame input, "
            "per-class post-processing. $R_{ops}$ is the delegated share of all "
            "operations; $K$ the number of delegated regions; $R_{largest}$ the "
            "share of delegated operations in the largest region. Runtime "
            "partitioning makes $K$ a measured quantity on this backend."
        ),
        "frdm-imx93": (
            "Delegation continuity on the i.MX93, full-frame input, per-class "
            "post-processing. Columns are as in "
            "\\cref{tab:continuity_imx8mp}. $K$ is measured here too, being read "
            "off the same per-region partition dump: $K=1$ records that the "
            "accepted set is contiguous, not a limitation of the backend, which "
            "splits the graph into two regions for the auxiliary YOLOv7-tiny "
            "exports. The float rows are delegated nowhere, so their continuity "
            "columns are empty."
        ),
    }

    for _plat, _caption in _captions.items():
        _table = dg.continuity_table(delegation, platform=_plat)
        if not _table.empty:
            _table = _table[_table["Architecture"].isin(_primary)]
            _table = _table.drop_duplicates(["Architecture", "Scheme"])
            # Positional index of the parsed log, i.e. how many files happened to
            # sort ahead of this one. It changes whenever logs are re-collected
            # and means nothing to a reader, so it is not carried into the table.
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
    # 5 · What the survivors cost — CPU vs NPU

    Restricted to what §4 found each accelerator executes correctly: a delegate
    returning `NaN` boxes still produces real timings, and usually fast ones.

    The CPU column is the **same board with the delegate switched off**, not a host,
    so the speedup is a property of the accelerator. That baseline is
    XNNPACK-accelerated, which makes it conservative. `dAP` travels with every row.

    Median, with p95 as spread. Multi-class, trained and evaluated full-frame,
    320x320. Scoped to the default export: the `<board>_cpu` trees carry no
    per-class runs yet.
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
        caption="Median inference latency and throughput at the reference "
        "configuration, each board's CPU against its own NPU delegate, for the "
        "exports the delegate executes correctly.",
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

    One panel per detector: accuracy is set by architecture and export scheme,
    latency by platform and granularity, so pooling the two leaves most of the axis
    empty between two AP bands. Colour is the platform, shape the whole export
    scheme — a scheme is a method *and* a granularity, and both are needed to
    identify a point.

    Deployment targets only. The CPU reference is excluded: it exists to establish
    that a device run computed the right answer, and its milliseconds compare two
    machine classes rather than two deployments.

    **The i.MX8M Plus NPU has no float point.** Its 32 fp32 runs produced non-finite
    boxes under the delegate, so §0 withheld their metrics and §4 records them as
    `unscoreable`. The gap is that result, not a missing measurement.
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

    A separate measurement path from `ave benchmark`: that asks what a model
    *predicts*, this asks what it *costs to run*, which needs a steady state long
    enough for supply current, core frequency and die temperature to settle
    (`ave resources MODEL IMAGES --seconds 120`, joined to the FNB58 trace by
    `scripts/power_report.py`).

    Multi-class, trained full-frame, as in §5. Resource cost does not depend on
    class count or training tiling, so the sweep collapses to one model per
    (architecture x scheme) anyway; the scope keeps this section comparable with
    the rest rather than adding information.

    **`aligned` gates everything below it.** Three hosts, three clocks: the trace is
    timestamped in the lab server's `CLOCK_MONOTONIC`, the board's artifacts in the
    board's epoch clock. `ave resources` saturates the cores either side of the
    measured loop so the resulting step can be matched against a locally recorded
    time, and the residual of that match is the check. An unverified run still
    reports entirely plausible watts.
    """)
    return


@app.cell(hide_code=True)
def _(RESOURCE_ROOT, json, pd):
    def load_power_summaries(root) -> pd.DataFrame:
        """Flatten every `power_summary.json` under `root` into one frame."""
        rows = []
        if not root.is_dir():
            return pd.DataFrame()

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
                rows.append(
                    {
                        # <device>/<stamp>/power_summary.json
                        "device": summary_path.parent.parent.name,
                        # Kept so repeated sweeps of one board stay distinguishable.
                        # Without it two sweeps under the same device directory
                        # merge into duplicate rows -- silently, and straight into
                        # any mean or plot drawn from the frame.
                        "stamp": summary_path.parent.name,
                        "run": result.get("run"),
                        "backend": result.get("backend"),
                        # Leads on purpose: an unaligned run's power figures are
                        # plausible and wrong.
                        "aligned": alignment.get("verified"),
                        "state": alignment.get("state"),
                        "residual (s)": alignment.get("max_abs_residual_s"),
                        "lat mean (ms)": latency.get("mean_latency_ms"),
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

        # A board can be swept more than once -- to extend coverage to further
        # resolutions, or to re-measure after the instrumentation gained its
        # per-phase latency split. Both land under the same device directory, so
        # the same `run` arrives twice and every aggregate below would weight it
        # twice. Timestamped sweep directories sort chronologically, so keeping
        # the last occurrence keeps the newest measurement of each run while
        # leaving runs the newer sweep never repeated untouched.
        frame = frame.sort_values(["device", "run", "stamp"], kind="stable")
        frame = frame.drop_duplicates(["device", "run"], keep="last")

        return frame.reset_index(drop=True).round(2)


    power_df = load_power_summaries(RESOURCE_ROOT)
    return (power_df,)


@app.cell
def _(br, power_df):
    # The sweep records model stems, so the config axes come from the name. Scoped
    # to §5's reference cell; latency and memory would survive a wider scope, but mixing
    # them with power a reader cannot quote is worse than a smaller table.
    power_scoped = br.annotate_resource_runs(power_df)
    if not power_scoped.empty:
        power_scoped = power_scoped[
            (power_scoped["classes"] == "mc") & (power_scoped["dataset"] == "phenobench")
        ]
    return (power_scoped,)


@app.cell
def _(NMS, br, power_scoped, runs, skipped, view):
    # A power row is an *operating point* only if its export was executed
    # correctly, and the sweep cannot tell. `output_integrity` catches output
    # that is non-finite or out of range -- the fp32 graphs under Teflon -- but
    # a collapse that still emits plausible boxes is visible only against the
    # CPU reference, which is what `deployability_matrix` scores. Without this
    # join the table presents a collapsed run's watts as an ordinary reading:
    # real numbers that are not the cost of a working detector.
    #
    # Scored over every board carrying power rows, including the `_unpatched`
    # and `_cpu` trees that §4's table leaves out of its columns.
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

    # `deployability_matrix` drops the keys that are constant in its scope, so
    # the join is on whichever of them survived.
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
        # A tree never benchmarked for accuracy has no verdict to join. Say so:
        # a blank cell reads as `ok` to anyone skimming the column.
        power_judged["verdict"] = power_judged["verdict"].fillna("unknown")
    return (power_judged,)


@app.cell(hide_code=True)
def _(mo, power_scoped):
    # Three states, and only one of them condemns the numbers. `misaligned` means
    # the chirp was found somewhere other than where the board recorded it, so the
    # power is joined to the wrong part of the trace. `unverified` means no edge was
    # detectable at all -- weaker evidence, not a contradiction, and the ssh probe
    # behind it measures under 0.1 s of uncertainty against a 120 s window.
    #
    # Summaries written before the three-state split carry only `aligned`, so the
    # state is derived when it is absent rather than counting every run as nothing.
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
        # No matched chirp anywhere is the *worst* outcome, not the best: it means
        # the check never ran, and an earlier version fell through to "success".
        # Some misaligned runs among many good ones is a caveat, not an alarm --
        # they are excluded below, so what is displayed stays sound.
        if _ok == 0 or _bad > _ok:
            _kind = "danger"
        elif _bad or _soft:
            _kind = "warn"
        else:
            _kind = "success"

        _power_state = mo.callout(
            mo.md(
                f"**{_ok} of {_total} runs in scope have a matched chirp.** "
                + (
                    "None do — either the sweeps predate the alignment check or "
                    "`scripts/power_report.py` has not been re-run over them. "
                    if _ok == 0
                    else ""
                )
                + (
                    f"**{_bad} misaligned** — chirp found away from where the board "
                    "recorded it, so their power is joined to the wrong part of the "
                    "trace and is excluded below. "
                    if _bad
                    else ""
                )
                + (
                    f"{_soft} unverified: no edge was detectable, so the join rests "
                    "on the ssh probe alone. "
                    if _soft
                    else ""
                )
                + "Latency, CPU, memory and temperature come from the board and do "
                "not depend on the join at all."
            ),
            kind=_kind,
        )

    _power_state
    return


@app.cell
def _(NMS, br, mo, power_judged, show_fig):
    # Pinned to the presented NMS like every other figure. The sweeps do not all
    # cover the same variants -- some carry both, some only `regnms` -- so an
    # unscoped frame averages a board's fast-NMS runs into its per-class ones
    # wherever both exist, and silently changes which board that happens to.
    # The two differ in latency by enough to have their own table (§6.3).
    # Diagnostic delegate builds are dropped here rather than in `power_judged`,
    # which the table below still needs: `_unpatched` is §6.4's axis and
    # `_no-concat` a forced-fallback control, so neither is an operating point.
    # The `_cpu` trees stay: both boards carry one, and the delegate-off series
    # is the baseline the energy argument is made against.
    #
    # `size` is pinned to the reference rung. `plot_resource_summary` groups by
    # (device, arch, scheme) and takes a mean, so an unpinned frame averages
    # 320/512/1024 into one bar -- roughly 3x the reference latency and energy,
    # and not a configuration anyone deploys. The ladder is §6.6's axis and has
    # its own table.
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
        caption="Steady-state power, energy per inference and resource use, per "
        "run across the input-resolution ladder; the reference rung is "
        "$320\\times320$ and the higher rungs belong to the input-resolution "
        "ablation. "
        "\\texttt{verdict} carries each export's "
        "correctness on that board from \\cref{tab:deployability}: only "
        "\\texttt{ok} rows are valid operating points. A \\texttt{collapsed} or "
        "\\texttt{unscoreable} run executes the graph and its power and timings "
        "are real measurements; what they cannot be read as is the cost of a "
        "working detector.",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    # 6 · Three ablations

    §1–§5 hold three things fixed. Each is varied here in turn: **class regime**
    (§6.1), **apparent object scale via tiling** (§6.2), and the **exported
    post-processing** (§6.3).

    The overview below deviates one axis at a time, plus the tiled configuration
    where both tiling axes move together — they are not independent, and §6.2 is
    about why. Column kinds differ: `Float AP` is a level; `Conversion`, `NMS swap`
    and `PTQ` are costs against the preceding rung; `QAT reclaim` is QAT minus PTQ
    at per-tensor weights.

    > `NPU (ms)` is **per inference**, and a tiled inference covers a ninth of the
    > frame. Read it down a column within one evaluation regime, never across the
    > tiled and full-frame rows.
    """)
    return


@app.cell(hide_code=True)
def _(br, mo, show_table, view):
    show_table(
        br.story_ablation_table(view(nms="both")),
        "story_ablation",
        mo,
        caption="Effect of each single-axis deviation from the reference "
        "configuration on every step of the deployment chain.",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### 6.1 · Single-class

    Dropping `crop` and detecting only `weed`. The task keeps the harder class:
    weeds are the small objects, so the COCO area breakdown (`APS` <= 32², `APM`,
    `APL` > 96²) is where an architecture's answer to them shows.

    The first figure asks whether quantization behaves the same in both regimes.
    Three series per scheme: the two classes a multi-class model reports, and the
    weed AP of the *single-class* model trained on the same data. The last is not a
    slice of the first — it is a different model, hence its own series.
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
    ### 6.2 · Tiling — apparent object scale

    A tile is a 512² crop of a 1024² frame (3x3, overlap 0.5), so it shows the same
    scene at **2x linear magnification**. The two tiling axes are therefore not an
    inference strategy; they are a proxy for how large objects appear to the model —
    sensor focus at train time and at deployment time. Each of the four cells is a
    different question about scale:

    | cell | question |
    |---|---|
    | trained full / eval full | the reference |
    | trained full / eval tiled | how the model reacts to **halving camera distance** in deployment — out of domain at inference |
    | trained tiled / eval full | whether **training on strictly magnified features** transfers back to the wide view |
    | trained tiled / eval tiled | the coherent **narrow-focus** configuration, in domain at both ends |

    The mismatched cells measure out-of-domain robustness in scale; only the matched
    one is a deployable configuration, and its gain is a statement about **sensor
    choice**: a narrower field of view, trained for, is worth substantially more
    detection accuracy than the same detector aimed wide.

    `d` is against the reference cell of the same architecture. The deltas do not
    add — each axis alone loses several AP while moving both together gains — which
    is why the square is reported rather than two one-axis rows.

    Latency is deliberately absent: a tiled inference covers a ninth of the area, so
    per-inference times are not comparable across these cells and the question here
    is scale, not throughput.
    """)
    return


@app.cell(hide_code=True)
def _(br, mo, show_table, view):
    show_table(
        br.tiling_cross_table(view(nms="both")),
        "tiling_cross",
        mo,
        caption="All four training x evaluation tiling combinations at the float "
        "TFLite rung on the CPU reference, multi-class, with each cell's change "
        "against its architecture's reference.",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 6.3 · The exported post-processing

    `ave convert` emits `use_regular_nms=False` unless told otherwise: one
    class-agnostic suppression pass instead of one per class. §3.2 priced that in
    accuracy; this is its latency side, and §6.1's dimension is what makes it
    measurable. At one class the two algorithms **are** the same algorithm, so the
    latency difference within a single-class pair is drift between two separately
    benchmarked runs and nothing else — subtracting it leaves the algorithm.

    **INT8 only.** The quantity is a roughly fixed millisecond cost, so pooling a
    340 ms fp32 regime with a 33 ms INT8 one adds variance without signal: the fp32
    pairs carry ~40x the absolute timing noise, and four of twenty were setting
    every interval's width. They also moved a point estimate by an order of
    magnitude (`x86_cpu · MNv2`, -0.73 ms pooled against -0.06 ms here).

    | column | meaning |
    |---|---|
    | `mc pairs` / `sc pairs` | matched `fastnms`/`regnms` run pairs entering each arm |
    | `dLatency mc` | mean (fast − per-class) over multi-class pairs; negative = fast is quicker |
    | `sc drift` | the same over single-class pairs, where the true difference is zero — pure measurement drift |
    | `NMS saving` | `dLatency mc` − `sc drift`: the difference in differences, the estimate |
    | `SE` | standard error of that difference, both arms combined |
    | `sigma` | \|saving\| / SE — how far the estimate sits from zero |
    | `95% CI` | interval on the saving; one containing zero has **not** resolved a saving from drift |
    | `resolved` | whether the interval excludes zero |
    | `sc \|drift\| worst` | largest single-pair drift, as a sanity check on the arm |

    A large point estimate whose interval spans zero is an *unmeasured* effect, not
    a small one.
    """)
    return


@app.cell(hide_code=True)
def _(br, mo, show_table, view):
    show_table(
        br.nms_latency_tradeoff_table(view(nms="both")),
        "nms_latency_tradeoff",
        mo,
        caption="Latency saved by the fast NMS substitution, estimated against the "
        "single-class null control (difference in differences), per platform and "
        "architecture.",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### 6.4 · Input resolution

    320 / 512 / 1024 at the reference configuration. Accuracy is measured once on
    the CPU reference; `FPS` and `(ms)` are per board, because a rung is
    disqualified by what it costs **on the device it would ship on**. The `x86 (ms)`
    column is the reference host and is a lower bound, not a proxy — it runs 320 in
    ~14 ms against 30–50 ms on the accelerators.

    Resolution is the axis under study, so class regime, training set and input
    regime are pinned rather than collapsed: folding tiled runs in would confound
    input *resolution* with apparent object *scale* (§6.2).

    The trade-off figure is pinned to one export scheme and log-scaled in latency.
    Granularity alone moves latency eightfold on the Vivante NPU, so a line mixing
    schemes would trace that instead of resolution; and the ladder spans 30 ms to
    several seconds, which a linear axis collapses.

    Still filling in: FPNLite at 1024 (a multi-session Kaggle run) and the i.MX8M
    Plus above 512. A rung not yet benchmarked on a board leaves its cell empty
    rather than falling back to the host.
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
        caption="Detection quality against input resolution at the reference "
        "configuration, every export scheme, with median latency on the CPU "
        "reference and on each accelerator.",
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

    `yolov7-tiny` is in the frame; YOLOX has ONNX finetune artifacts
    (`artifacts/onnx/yolox-nano_*`) but no TFLite exports or benchmark results.

    A different family, kept out of the controlled comparison: different
    post-processing (no `TFLite_Detection_PostProcess`, so §6.3 does not apply to
    it) and a partial matrix — `yolov7-tiny` exists only as a tiled-trained 512
    export. `br.PRIMARY_ARCHS` keeps it out of the default `view()`; `archs="aux"`
    lets it back in.
    """)
    return


@app.cell
def _(br, mo, show_table, view):
    show_table(
        br.platform_metrics_table(view(archs="aux"), br.CPU_REFERENCE_PLATFORM),
        "auxiliary_detectors",
        mo,
        caption="Auxiliary reference detectors (YOLO family) on the CPU "
        "reference. Not part of the controlled SSD comparison.",
    )
    return


@app.cell(hide_code=True)
def _(br, mo):
    mo.md(f"""
    # 8 · Appendix — coverage and full matrices

    A *full run* benchmarks every trained variant in `artifacts/tf`, for every
    export scheme, on both input regimes, on every target platform:

    - schemes: {", ".join(f"`{br.scheme_label(br.scheme_name(*s))}`" for s in br.DEFAULT_SCHEMES)}
    - input regimes: {", ".join(f"`{t}`" for t in br.DEFAULT_EVAL_TILINGS)}
    - platforms: {", ".join(f"`{p}`" for p in br.DEFAULT_EXPECTED_PLATFORMS)}

    Coverage counts the **default export** only. The per-class variant is equally
    deployable and was swept on the boards, but not on the CPU-only and unpatched
    control trees, so counting it would invent gaps that are deliberate — and
    letting it *satisfy* a cell would let a per-class run stand in for the default.
    The NMS axis gets its own completeness view instead.

    `tf-savedmodel` is not in this matrix. It holds the checkpoint's raw fp32
    graph — nothing there was quantized, so no export scheme is something it can be
    missing. It appears in §8.2's per-platform tables.
    """)
    return


@app.cell(hide_code=True)
def _(br, mo, runs):
    mo.ui.table(
        br.nms_pair_coverage(runs),
        label="Post-processing pair coverage (where both variants exist)",
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
        "(x = done, - = missing).",
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

    One full COCO table per device (thesis Appendix D). Generated per platform,
    so boards appear automatically as their sweeps land.
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
    return platform_ui, platforms


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
def _(br, mo, platforms, show_table, view):
    # Export one table per platform regardless of the dropdown selection, so a
    # full run of the notebook refreshes every device table in docs/thesis.
    for _platform in platforms:
        show_table(
            br.platform_metrics_table(view(archs=None), _platform),
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
    ### 8.3 · Cross-configuration degradation ladder

    §2 and §3 report the preparation chain at the reference configuration.
    This is the same decomposition across *every* configuration and continuing
    onto the accelerator — the transpose of §2.1, and the frame §6's ablation
    is drawn from.
    """)
    return


@app.cell(hide_code=True)
def _(br, mo, runs, show_table):
    # Unscoped by construction: this table's `nms-swap` column is built from both
    # post-processing variants, so it needs the pair rather than one of them.
    show_table(
        br.degradation_ladder_table(runs),
        "degradation_ladder_untiled",
        mo,
        caption="Detection AP across the deployment rungs (full-frame input, PTQ "
        "path) for every configuration, with the loss attributable to conversion, "
        "the post-processing substitution, quantization and delegation separately.",
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
        caption="Median / p95 inference latency and throughput per platform, "
        "backend and export scheme, uncollapsed.",
    )
    return


@app.cell(hide_code=True)
def _(br, mo, runs, show_table):
    show_table(
        br.master_table(runs),
        "benchmark_master",
        mo,
        caption="Full benchmark matrix across platforms, architectures, class "
        "regimes, input regimes, quantization schemes and exported "
        "post-processing. Unscoped: both NMS variants and every tree.",
    )
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
