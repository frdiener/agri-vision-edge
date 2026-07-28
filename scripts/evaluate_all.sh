#!/usr/bin/env bash
#
# Evaluate every benchmarked model in a results directory, inferring the matching
# test-bundle annotations (sc/mc, tiled/untiled) from each entry's directory
# name. Tiling is encoded by the tiled_ / untiled_ result-directory prefix.
#
# Usage:
#   scripts/evaluate_all.sh [--faithful] [target-dir]
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

set -uo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/.." && pwd)"

bundle_dir="${repo_root}/datasets/test-bundle"

# Raw PhenoBench datasets for the faithful (official upstream) evaluator.
raw_full_dir="${PHENOBENCH_RAW_FULL:-${repo_root}/datasets/phenobench_raw_full}"
raw_tiled_dir="${PHENOBENCH_RAW_TILED:-${repo_root}/datasets/phenobench_raw_tiled}"

faithful=0
target_dir=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --faithful)
            faithful=1
            ;;
        *)
            target_dir="$1"
            ;;
    esac
    shift
done
target_dir="${target_dir:-${repo_root}/benchmark_results/$(hostname)}"

if [[ ! -d "${target_dir}" ]]; then
    echo "target dir not found: ${target_dir}" >&2
    exit 1
fi

echo "evaluating runs in: ${target_dir}"
echo

evaluated=0
skipped=0

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
        if [[ ! -d "${raw_dir}" ]]; then
            echo "[warn] ${name}: raw dataset not found at ${raw_dir}" >&2
        else
            echo "[faithful] ${name}  (phenobench-dir=$(basename "${raw_dir}"))"
            "${script_dir}/ave" evaluate \
                "${annotations}" \
                "${predictions}" \
                --faithful \
                --phenobench-dir "${raw_dir}" \
                || echo "[warn] faithful eval failed for ${name} (needs the" \
                        "'faithful-eval' extra: torch/torchvision/torchmetrics)" >&2
        fi
    fi

    evaluated=$((evaluated + 1))
done

echo
echo "done: ${evaluated} evaluated, ${skipped} skipped"
