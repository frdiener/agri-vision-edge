#!/usr/bin/env bash
#
# Benchmark every TFLite model in artifacts/tflite against the matching
# test-bundle split, inferring sc/mc and tiled/non-tiled from the model name.
#
# Output goes to benchmark_results/<hostname>/<model-stem>/. By default a model
# whose output already contains latency.json is skipped; pass --override to
# re-run it.
#
# All models use the NPU delegate (--delegate, default /usr/lib/libteflon.so).
# Note: the Teflon delegate targets INT8 — routing an fp32 graph through it
# reports support for float conv ops and silently degrades results, so use --cpu
# for trustworthy fp32 (and CPU-reference int8) numbers.
#
# Pass --cpu to disable the delegate for every model and write the results to
# benchmark_results/<hostname>_cpu/ instead, for a clean CPU-only run alongside
# the delegated one.
#
# Pass --faithful to additionally run the official PhenoBench evaluator
# (`ave evaluate --faithful`) on each model's predictions right after it is
# benchmarked, pointing --phenobench-dir at the raw dataset that matches the
# model: datasets/phenobench_raw_full for untiled models, datasets/
# phenobench_raw_tiled for *phenobench-tiled* models. Override those locations
# with the PHENOBENCH_RAW_FULL / PHENOBENCH_RAW_TILED env vars. This needs the
# host 'faithful-eval' extra (torch/torchvision/torchmetrics); if it is not
# installed the faithful step is reported as failed and the sweep continues.
#
# Any other extra arguments are forwarded to `ave benchmark`.

set -uo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/.." && pwd)"

models_dir="${repo_root}/artifacts/tflite"
bundle_dir="${repo_root}/datasets/test-bundle"

override=0
cpu_only=0
faithful=0
delegate="/usr/lib/libteflon.so"
forward_args=()

# Raw PhenoBench datasets for the faithful (official upstream) evaluator, keyed
# by tiling. Overridable via env so a device with the data mounted elsewhere can
# point at it.
raw_full_dir="${PHENOBENCH_RAW_FULL:-${repo_root}/datasets/phenobench_raw_full}"
raw_tiled_dir="${PHENOBENCH_RAW_TILED:-${repo_root}/datasets/phenobench_raw_tiled}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --override)
            override=1
            ;;
        --cpu)
            cpu_only=1
            ;;
        --faithful)
            faithful=1
            ;;
        --delegate)
            delegate="$2"
            shift
            ;;
        --delegate=*)
            delegate="${1#*=}"
            ;;
        *)
            forward_args+=("$1")
            ;;
    esac
    shift
done

# CPU-only runs go to a separate <hostname>_cpu/ tree so they don't clobber the
# delegated results.
if [[ ${cpu_only} -eq 1 ]]; then
    output_root="${repo_root}/benchmark_results/$(hostname)_cpu"
else
    output_root="${repo_root}/benchmark_results/$(hostname)"
fi

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
        raw_dir="${raw_tiled_dir}"
    else
        images="${bundle_dir}/images"
        annotations="${bundle_dir}/annotations_${cls}.json"
        raw_dir="${raw_full_dir}"
    fi

    # --cpu disables the delegate for every model; otherwise all models use it.
    if [[ ${cpu_only} -eq 1 ]]; then
        model_delegate="none"
    else
        model_delegate="${delegate}"
    fi

    if [[ ${override} -eq 0 && -f "${output_root}/${stem}/latency.json" ]]; then
        echo "[skip] ${name}: already benchmarked (use --override to re-run)"
        skipped=$((skipped + 1))
        continue
    fi

    echo "[run]  ${name}  (cls=${cls}, images=$(basename "${images}"), delegate=${model_delegate})"
    "${script_dir}/ave" benchmark \
        "${model}" \
        "${images}" \
        --annotations "${annotations}" \
        --output-dir "${output_root}" \
        --delegate "${model_delegate}" \
        ${forward_args[@]+"${forward_args[@]}"}
    ran=$((ran + 1))

    # Optional: official upstream (faithful) evaluation on the fresh predictions,
    # using the raw dataset that matches this model's tiling.
    if [[ ${faithful} -eq 1 ]]; then
        predictions="${output_root}/${stem}/predictions.json"
        if [[ ! -f "${predictions}" ]]; then
            echo "[warn] ${name}: no predictions.json to faithfully evaluate"
        elif [[ ! -d "${raw_dir}" ]]; then
            echo "[warn] ${name}: raw dataset not found at ${raw_dir}" >&2
        else
            echo "[faithful] ${name}  (phenobench-dir=$(basename "${raw_dir}"))"
            "${script_dir}/ave" evaluate \
                "${annotations}" \
                "${predictions}" \
                --faithful \
                --phenobench-dir "${raw_dir}" \
                || echo "[warn] faithful eval failed for ${name} (needs the host" \
                        "'faithful-eval' extra: torch/torchvision/torchmetrics)" >&2
        fi
    fi
done

echo
echo "done: ${ran} run, ${skipped} skipped"
