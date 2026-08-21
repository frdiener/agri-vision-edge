#!/usr/bin/env bash
#
# Collect the delegate's verbose graph-partitioning log for TFLite models, one
# inference per model, into that model's benchmark_results run directory:
#
#   benchmark_results/<hostname>[_cpu]/<tiling>_<model-stem>/delegate_debug.log
#   benchmark_results/<hostname>[_cpu]/<tiling>_<model-stem>/delegate_debug.json
#
# The TEFLON_UNSUPPORTED_* variables change partitioning, so whatever is set in
# the environment is forwarded and recorded in the sidecar JSON.
#
# This script only collects. Nothing here parses the log.
#
# The platform directory is $(hostname), plus --suffix for a tree that is the
# same board under a different delegate build, plus _cpu when --cpu is given:
#   --suffix unpatched            -> benchmark_results/<hostname>_unpatched/
#   --suffix unpatched --cpu      -> benchmark_results/<hostname>_unpatched_cpu/
#
# Usage:
#   scripts/collect_delegate_debug.sh                    # every model, delegated
#   scripts/collect_delegate_debug.sh --cpu              # CPU reference tree
#   scripts/collect_delegate_debug.sh --suffix unpatched # older delegate build
#   scripts/collect_delegate_debug.sh model.tflite ...   # named models only
#   TEFLON_UNSUPPORTED_OPS=83 scripts/collect_delegate_debug.sh model.tflite

set -uo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/.." && pwd)"

models_dir="${repo_root}/artifacts/tflite"
bundle_dir="${repo_root}/datasets/test-bundle"

override=0
cpu_only=0
delegate="/usr/lib/libteflon.so"
tiling="untiled"
image=""
suffix=""
models=()

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
        --tiling)
            tiling="$2"
            shift
            ;;
        --tiling=*)
            tiling="${1#*=}"
            ;;
        --image)
            image="$2"
            shift
            ;;
        --image=*)
            image="${1#*=}"
            ;;
        --suffix)
            suffix="$2"
            shift
            ;;
        --suffix=*)
            suffix="${1#*=}"
            ;;
        --models-dir)
            models_dir="$2"
            shift
            ;;
        -h|--help)
            awk 'NR == 1 { next } /^#/ { sub(/^# ?/, ""); print; next } { exit }' \
                "${BASH_SOURCE[0]}"
            exit 0
            ;;
        *)
            models+=("$1")
            ;;
    esac
    shift
done

case "${tiling}" in
    untiled|tiled) ;;
    *)
        echo "--tiling must be 'untiled' or 'tiled', got '${tiling}'" >&2
        exit 1
        ;;
esac

# The graph is identical for both input regimes -- the model's input size is
# fixed, and tiling only changes what is resized into it -- so partitioning is
# collected under one regime rather than duplicated across both.
#
# The platform directory mirrors benchmark_all.sh: $(hostname), then --suffix
# for a tree that is the same board under a different delegate build
# (`_unpatched`), then `_cpu` last for the unaccelerated variant of whichever
# tree that is.
platform="$(hostname)"
if [[ -n "${suffix}" ]]; then
    platform="${platform}_${suffix#_}"
fi

if [[ ${cpu_only} -eq 1 ]]; then
    platform="${platform}_cpu"
    model_delegate="none"
else
    model_delegate="${delegate}"
fi

output_root="${repo_root}/benchmark_results/${platform}"

if [[ -z "${image}" ]]; then
    if [[ "${tiling}" == "tiled" ]]; then
        image_dir="${bundle_dir}/images_tiled"
    else
        image_dir="${bundle_dir}/images"
    fi
    shopt -s nullglob
    candidates=("${image_dir}"/*.png "${image_dir}"/*.jpg)
    shopt -u nullglob
    if [[ ${#candidates[@]} -eq 0 ]]; then
        echo "no images found in ${image_dir}" >&2
        exit 1
    fi
    # Sorted glob: the same frame every time, so two logs stay comparable.
    image="${candidates[0]}"
fi

if [[ ! -f "${image}" ]]; then
    echo "image not found: ${image}" >&2
    exit 1
fi

if [[ ${#models[@]} -eq 0 ]]; then
    shopt -s nullglob
    models=("${models_dir}"/*.tflite)
    shopt -u nullglob
fi

if [[ ${#models[@]} -eq 0 ]]; then
    echo "no .tflite models found in ${models_dir}" >&2
    exit 1
fi

echo "found ${#models[@]} model(s)"
echo "output root: ${output_root}"
echo "delegate:    ${model_delegate}"
echo "image:       ${image}"
echo "tiling:      ${tiling}"
echo

ran=0
skipped=0
failed=0

for model in "${models[@]}"; do
    if [[ ! -f "${model}" ]]; then
        echo "[skip] ${model}: not a file" >&2
        skipped=$((skipped + 1))
        continue
    fi

    stem="$(basename "${model}")"
    stem="${stem%.tflite}"

    run_dir="${output_root}/${tiling}_${stem}"
    log="${run_dir}/delegate_debug.log"
    meta="${run_dir}/delegate_debug.json"

    if [[ -s "${log}" && ${override} -eq 0 ]]; then
        echo "[skip] ${stem}: delegate_debug.log exists (--override to redo)"
        skipped=$((skipped + 1))
        continue
    fi

    mkdir -p "${run_dir}"

    # `ave infer` writes an annotated image; it is a by-product here, so keep it
    # out of the results tree.
    out_dir="$(mktemp -d)"

    TEFLON_DEBUG=verbose \
    ETNA_MESA_DEBUG=ml_msgs \
    "${script_dir}/ave" infer \
        --delegate "${model_delegate}" \
        --output-dir "${out_dir}" \
        "${model}" \
        "${image}" \
        >"${log}" 2>&1
    status=$?

    rm -rf "${out_dir}"

    # The flags are part of the measurement: TEFLON_UNSUPPORTED_* move the
    # partition boundary, so a log is only interpretable together with them.
    cat >"${meta}" <<EOF
{
  "model": "$(basename "${model}")",
  "run": "${tiling}_${stem}",
  "platform": "$(basename "${output_root}")",
  "hostname": "$(hostname)",
  "delegate": "${model_delegate}",
  "image": "$(basename "${image}")",
  "tiling": "${tiling}",
  "exit_status": ${status},
  "collected_utc": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "env": {
    "TEFLON_DEBUG": "verbose",
    "ETNA_MESA_DEBUG": "ml_msgs",
    "ETHOSU_DEBUG=dbg_msgs",
    "TEFLON_UNSUPPORTED_OPS": "${TEFLON_UNSUPPORTED_OPS-}",
    "TEFLON_UNSUPPORTED_NODES": "${TEFLON_UNSUPPORTED_NODES-}"
  }
}
EOF

    if [[ ${status} -ne 0 ]]; then
        echo "[fail] ${stem}: exit ${status} (log kept: ${log})" >&2
        failed=$((failed + 1))
        continue
    fi

    lines="$(wc -l <"${log}")"
    echo "[ok]   ${stem}: ${lines} log lines"
    ran=$((ran + 1))
done

echo
echo "collected ${ran}, skipped ${skipped}, failed ${failed}"
