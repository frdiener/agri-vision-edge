#!/usr/bin/env bash
#
# Benchmark every TFLite model in artifacts/tflite against the matching
# test-bundle split, inferring sc/mc and tiled/non-tiled from the model name.
#
# Output goes to benchmark_results/<hostname>/<model-stem>/. By default a model
# whose output already contains latency.json is skipped; pass --override to
# re-run it. Any extra arguments (e.g. --delegate ...) are forwarded to
# bin/benchmark_tflite.py.

set -uo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/.." && pwd)"

models_dir="${repo_root}/artifacts/tflite"
bundle_dir="${repo_root}/datasets/test-bundle"
output_root="${repo_root}/benchmark_results/$(hostname)"

override=0
forward_args=()

for arg in "$@"; do
    if [[ "${arg}" == "--override" ]]; then
        override=1
    else
        forward_args+=("${arg}")
    fi
done

shopt -s nullglob
models=("${models_dir}"/*.tflite)
shopt -u nullglob

if [[ ${#models[@]} -eq 0 ]]; then
    echo "no .tflite models found in ${models_dir}" >&2
    exit 1
fi

echo "found ${#models[@]} model(s) in ${models_dir}"
echo "output root: ${output_root}"
echo

ran=0
skipped=0

for model in "${models[@]}"; do
    name="$(basename "${model}")"
    stem="${name%.tflite}"

    # sc vs mc (single-class vs multi-class)
    if [[ "${name}" == *_mc_* ]]; then
        cls="mc"
    elif [[ "${name}" == *_sc_* ]]; then
        cls="sc"
    else
        echo "[skip] ${name}: cannot infer sc/mc from name" >&2
        continue
    fi

    # tiled vs non-tiled
    if [[ "${name}" == *phenobench-tiled* ]]; then
        images="${bundle_dir}/images_tiled"
        annotations="${bundle_dir}/annotations_${cls}_tiled.json"
    else
        images="${bundle_dir}/images"
        annotations="${bundle_dir}/annotations_${cls}.json"
    fi

    if [[ ${override} -eq 0 && -f "${output_root}/${stem}/latency.json" ]]; then
        echo "[skip] ${name}: already benchmarked (use --override to re-run)"
        skipped=$((skipped + 1))
        continue
    fi

    echo "[run]  ${name}  (cls=${cls}, images=$(basename "${images}"))"
    python3 "${script_dir}/benchmark_tflite.py" \
        "${model}" \
        "${images}" \
        --annotations "${annotations}" \
        --output-dir "${output_root}" \
        ${forward_args[@]+"${forward_args[@]}"}
    ran=$((ran + 1))
done

echo
echo "done: ${ran} run, ${skipped} skipped"
