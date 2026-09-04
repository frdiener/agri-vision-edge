#!/usr/bin/env bash
#
# Benchmark every TFLite model in artifacts/tflite against both tiled and
# untiled test-bundle splits, inferring only sc/mc from the model name.
#
# Output goes to benchmark_results/<hostname>/<tiled|untiled>_<model-stem>/.
# By default a run whose output already contains latency.json is skipped; pass
# --override to re-run it. --filter '<glob>' narrows the sweep to matching model
# stems, for recollecting one subset in a single coherent batch.
#
# All models use the NPU delegate (--delegate, default /usr/lib/libteflon.so).
# Note: the Teflon delegate targets INT8 — routing an fp32 graph through it
# reports support for float conv ops and silently degrades results. Use --cpu
# for runs without delegate.
#
# Pass --cpu to disable the delegate for every model and write the results to
# benchmark_results/<hostname>_cpu/ instead.
#
# The platform directory is $(hostname), plus --suffix for a tree that is the
# same board under a different delegate build, plus _cpu when --cpu is given:
#   --suffix unpatched            -> benchmark_results/<hostname>_unpatched/
#   --suffix unpatched --cpu      -> benchmark_results/<hostname>_unpatched_cpu/
#
# This script only *measures*: it writes predictions.json / latency.json /
# runtime.json and stops there. Scoring is a separate step -- run
# scripts/evaluate_all.sh afterwards. Keeping the two apart matters on a device:
# evaluation is pure host-side post-processing over predictions.json, so mixing
# it into the sweep would only add heat, memory pressure and wall time between
# two latency measurements without changing a single number.
#
# The tiled split pairs test-bundle/images_tiled with annotations_*_tiled.json,
# and both must come from the SAME tile geometry (currently 3x3, overlap=0.5 --
# see notebooks 03/04). Nothing here can detect a mismatch: tile
# file names are identical across grids, so inference would silently run on the
# wrong crops.
#
# Any other extra arguments are forwarded to `ave benchmark`.

set -uo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/.." && pwd)"

models_dir="${repo_root}/artifacts/tflite"
bundle_dir="${repo_root}/datasets/test-bundle"

override=0
cpu_only=0
delegate="/usr/lib/libteflon.so"
suffix=""
filter=""
forward_args=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --override)
            override=1
            ;;
        --cpu)
            cpu_only=1
            ;;
        --delegate)
            delegate="$2"
            shift
            ;;
        --delegate=*)
            delegate="${1#*=}"
            ;;
        --suffix)
            suffix="$2"
            shift
            ;;
        --suffix=*)
            suffix="${1#*=}"
            ;;
        --filter)
            filter="$2"
            shift
            ;;
        --filter=*)
            filter="${1#*=}"
            ;;
        *)
            forward_args+=("$1")
            ;;
    esac
    shift
done

# CPU-only runs go to a separate <hostname>_cpu/ tree so they don't clobber the
# delegated results; --suffix separates whole trees measured on the same board
# under a different delegate build, and `_cpu` stays last so it names the
# unaccelerated variant of whichever tree that is.
platform="$(hostname)"
if [[ -n "${suffix}" ]]; then
    platform="${platform}_${suffix#_}"
fi

if [[ ${cpu_only} -eq 1 ]]; then
    platform="${platform}_cpu"
fi

output_root="${repo_root}/benchmark_results/${platform}"

shopt -s nullglob
models=("${models_dir}"/*.tflite)
shopt -u nullglob

# --filter narrows the sweep to model stems matching a glob.
if [[ -n "${filter}" ]]; then
    selected=()
    for model in "${models[@]}"; do
        stem="$(basename "${model}" .tflite)"
        # shellcheck disable=SC2053  -- the glob is the point.
        [[ "${stem}" == ${filter} ]] && selected+=("${model}")
    done
    models=(${selected[@]+"${selected[@]}"})
    echo "filter: ${filter}"
fi

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

    # --cpu disables the delegate for every model; otherwise all models use it.
    if [[ ${cpu_only} -eq 1 ]]; then
        model_delegate="none"
    else
        model_delegate="${delegate}"
    fi

    for tiling in untiled tiled; do
        if [[ "${tiling}" == "tiled" ]]; then
            images="${bundle_dir}/images_tiled"
            annotations="${bundle_dir}/annotations_${cls}_tiled.json"
        else
            images="${bundle_dir}/images"
            annotations="${bundle_dir}/annotations_${cls}.json"
        fi

        output_stem="${tiling}_${stem}"

        if [[ ${override} -eq 0 && -f "${output_root}/${output_stem}/latency.json" ]]; then
            echo "[skip] ${tiling} ${name}: already benchmarked (use --override to re-run)"
            skipped=$((skipped + 1))
            continue
        fi

        echo "[run]  ${tiling} ${name}  (cls=${cls}, images=$(basename "${images}"), delegate=${model_delegate})"
        "${script_dir}/ave" benchmark \
            "${model}" \
            "${images}" \
            --annotations "${annotations}" \
            --output-dir "${output_root}" \
            --output-prefix "${tiling}_" \
            --delegate "${model_delegate}" \
            ${forward_args[@]+"${forward_args[@]}"}
        ran=$((ran + 1))
    done
done

echo
echo "done: ${ran} run, ${skipped} skipped"
