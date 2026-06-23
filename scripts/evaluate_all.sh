#!/usr/bin/env bash
#
# Evaluate every benchmarked model in a results directory, inferring the matching
# test-bundle annotations (sc/mc, tiled/non-tiled) from each entry's directory
# name (which is the model stem).
#
# Usage:
#   bin/evaluate_all.sh [target-dir]
#
# target-dir defaults to benchmark_results/<hostname>. Each immediate subdir is
# expected to contain a predictions.json (as written by bin/benchmark_tflite.py);
# dirs without one (e.g. failed runs holding error.json) are skipped. metrics.json
# is written beside each predictions.json.

set -uo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/.." && pwd)"

bundle_dir="${repo_root}/datasets/test-bundle"
target_dir="${1:-${repo_root}/benchmark_results/$(hostname)}"

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

    # tiled vs non-tiled
    if [[ "${name}" == *phenobench-tiled* ]]; then
        annotations="${bundle_dir}/annotations_${cls}_tiled.json"
    else
        annotations="${bundle_dir}/annotations_${cls}.json"
    fi

    echo "[eval] ${name}  (annotations=$(basename "${annotations}"))"
    python3 "${script_dir}/evaluate_coco.py" \
        "${annotations}" \
        "${predictions}"
    evaluated=$((evaluated + 1))
done

echo
echo "done: ${evaluated} evaluated, ${skipped} skipped"
