#!/usr/bin/env bash
#
# Evaluate every benchmarked model in a results directory, inferring the matching
# test-bundle annotations (sc/mc, tiled/untiled) from each entry's directory
# name. Tiling is encoded by the tiled_ / untiled_ result-directory prefix.
#
# Usage:
#   scripts/evaluate_all.sh [--faithful [--only-relevant]] [target-dir]
#
# target-dir defaults to benchmark_results/<hostname>. Each immediate subdir is
# expected to contain a predictions.json (as written by `ave benchmark`);
# dirs without one (e.g. failed runs holding error.json) are skipped. metrics.json
# is written beside each predictions.json.
#
# Pass --faithful to ALSO run the official PhenoBench evaluator
# (`ave evaluate --faithful`, writing metrics_faithful.json) alongside the
# lightweight pycocotools metrics. It points --phenobench-dir at the raw dataset
# that matches each run: datasets/phenobench_raw_full for untiled_ runs,
# datasets/phenobench_raw_tiled for tiled_ runs (override with the
# PHENOBENCH_RAW_FULL / PHENOBENCH_RAW_TILED env vars). Needs the 'faithful-eval'
# extra (torch/torchvision/torchmetrics); failures are reported per model and the
# sweep continues.
#
# Pass --only-relevant (after --faithful) to run the faithful evaluator ONLY on
# the runs whose upstream number is actually comparable: multi-class models
# evaluated on untiled_ (full 1024 frames), for BOTH the untiled- and the
# tiled-finetuned models. The lightweight pycocotools eval still runs for every
# model. The excluded runs are excluded because upstream cannot express them:
#
#   * tiled_ runs  -- upstream is applied per 512 tile, which is internally
#     consistent but NOT the official full-frame leaderboard number (that needs
#     tile predictions stitched back to 1024 first); evaluation/faithful.py
#     warns about this on every non-1024 run.
#   * sc runs      -- upstream always averages over crop AND weed, so a
#     weed-only model is scored on a class it structurally cannot predict.
#
# Faithful eval is by far the slowest step here (torchmetrics + a staged
# ground-truth tree per image), so this is the flag to use when only the
# leaderboard-comparable numbers are wanted.
#
# datasets/phenobench_raw_tiled must be cut with the same grid as the exported
# bundles (currently 3x3, overlap=0.5 -- notebooks 03/04); regenerate it with
# scripts/materialize_raw_tiled.py. A mismatched grid is rejected by
# `ave evaluate --faithful` rather than silently scored.

set -uo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/.." && pwd)"

bundle_dir="${repo_root}/datasets/test-bundle"

# Raw PhenoBench datasets for the faithful (official upstream) evaluator.
raw_full_dir="${PHENOBENCH_RAW_FULL:-${repo_root}/datasets/phenobench_raw_full}"
raw_tiled_dir="${PHENOBENCH_RAW_TILED:-${repo_root}/datasets/phenobench_raw_tiled}"

faithful=0
only_relevant=0
target_dir=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --faithful)
            faithful=1
            ;;
        --only-relevant)
            only_relevant=1
            ;;
        *)
            target_dir="$1"
            ;;
    esac
    shift
done
target_dir="${target_dir:-${repo_root}/benchmark_results/$(hostname)}"

# --only-relevant only narrows the faithful step; on its own it would silently
# do nothing, which for a long sweep is worth failing over.
if [[ ${only_relevant} -eq 1 && ${faithful} -eq 0 ]]; then
    echo "--only-relevant only applies to --faithful; pass both" >&2
    exit 2
fi

# The runs whose upstream (faithful) number is comparable: multi-class, scored
# on full 1024 frames. Both finetunes qualify -- `phenobench` (untiled) and
# `phenobench-tiled` (tiled) models alike, as long as they are *evaluated*
# untiled. See the header for why the rest are excluded.
is_relevant_for_faithful() {
    [[ "$1" == untiled_* && "$1" == *_mc_* ]]
}

if [[ ! -d "${target_dir}" ]]; then
    echo "target dir not found: ${target_dir}" >&2
    exit 1
fi

echo "evaluating runs in: ${target_dir}"
echo

evaluated=0
skipped=0
faithful_run=0
faithful_skipped=0

for model_dir in "${target_dir}"/*/; do
    [[ -d "${model_dir}" ]] || continue

    name="$(basename "${model_dir}")"
    predictions="${model_dir}predictions.json"

    if [[ ! -f "${predictions}" ]]; then
        echo "[skip] ${name}: no predictions.json"
        skipped=$((skipped + 1))
        continue
    fi

    # sc vs mc (single-class vs multi-class)
    if [[ "${name}" == *_mc_* ]]; then
        cls="mc"
    elif [[ "${name}" == *_sc_* ]]; then
        cls="sc"
    else
        echo "[skip] ${name}: cannot infer sc/mc from name" >&2
        skipped=$((skipped + 1))
        continue
    fi

    # tiled vs untiled, encoded by the benchmark result-directory prefix
    if [[ "${name}" == tiled_* ]]; then
        annotations="${bundle_dir}/annotations_${cls}_tiled.json"
        raw_dir="${raw_tiled_dir}"
    elif [[ "${name}" == untiled_* ]]; then
        annotations="${bundle_dir}/annotations_${cls}.json"
        raw_dir="${raw_full_dir}"
    else
        echo "[skip] ${name}: cannot infer tiled/untiled prefix" >&2
        skipped=$((skipped + 1))
        continue
    fi

    echo "[eval] ${name}  (annotations=$(basename "${annotations}"))"
    "${script_dir}/ave" evaluate \
        "${annotations}" \
        "${predictions}"

    # Optional: official upstream (faithful) evaluation, using the raw dataset
    # that matches this model's tiling.
    if [[ ${faithful} -eq 1 ]]; then
        if [[ ${only_relevant} -eq 1 ]] && ! is_relevant_for_faithful "${name}"; then
            echo "[faithful-skip] ${name}: not leaderboard-comparable" \
                 "(--only-relevant keeps untiled_ + mc only)"
            faithful_skipped=$((faithful_skipped + 1))
        elif [[ ! -d "${raw_dir}" ]]; then
            echo "[warn] ${name}: raw dataset not found at ${raw_dir}" >&2
        else
            echo "[faithful] ${name}  (phenobench-dir=$(basename "${raw_dir}"))"
            "${script_dir}/ave" evaluate \
                "${annotations}" \
                "${predictions}" \
                --faithful \
                --phenobench-dir "${raw_dir}" \
                && faithful_run=$((faithful_run + 1)) \
                || echo "[warn] faithful eval failed for ${name} -- see the" \
                        "error above (corrupt predictions, a tiling mismatch," \
                        "or the missing 'faithful-eval' extra:" \
                        "torch/torchvision/torchmetrics)" >&2
        fi
    fi

    evaluated=$((evaluated + 1))
done

echo
echo "done: ${evaluated} evaluated, ${skipped} skipped"

if [[ ${faithful} -eq 1 ]]; then
    echo "faithful: ${faithful_run} run, ${faithful_skipped} skipped"
fi
