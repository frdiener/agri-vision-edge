#!/usr/bin/env bash
# setup.sh — Reconstruct the YOCTO.BSP-AVE source workspace
#
# Reads sources.lock and clones / checks out each external layer at the
# recorded pinned commit.  Safe to re-run on an already-prepared tree.
#
# Usage:
#   ./setup.sh
#
# After completion, source oe-init-build-env to configure a build:
#   . ./oe-init-build-env [build-dir]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOCK="${SCRIPT_DIR}/sources.lock"

if [[ ! -f "${LOCK}" ]]; then
    echo "ERROR: sources.lock not found in ${SCRIPT_DIR}" >&2
    exit 1
fi

clone_or_update() {
    local name="$1"
    local url="$2"
    local sha="$3"
    local dest="${SCRIPT_DIR}/${name}"

    if [[ -d "${dest}/.git" ]]; then
        # Directory already exists and is a real git repo — check current SHA
        local current
        current=$(git -C "${dest}" rev-parse HEAD 2>/dev/null || echo "unknown")

        if [[ "${current}" == "${sha}" ]]; then
            echo "  [ok]   ${name} already at ${sha:0:12}"
            return
        fi

        # Check for local modifications before touching anything
        if ! git -C "${dest}" diff --quiet 2>/dev/null || \
           ! git -C "${dest}" diff --cached --quiet 2>/dev/null; then
            echo "  [WARN] ${name} has local modifications — skipping (manual action required)" >&2
            return
        fi

        echo "  [upd]  ${name}: ${current:0:12} → ${sha:0:12}"
        # Fetch the required commit; it may not be reachable without fetching
        git -C "${dest}" fetch --quiet origin 2>/dev/null || true
        git -C "${dest}" fetch --quiet origin "${sha}" 2>/dev/null || true
        git -C "${dest}" checkout --quiet --detach "${sha}"

    else
        echo "  [clone] ${name} → ${url}"
        git clone --quiet "${url}" "${dest}"
        git -C "${dest}" fetch --quiet origin "${sha}" 2>/dev/null || true
        git -C "${dest}" checkout --quiet --detach "${sha}"
    fi
}

echo "=== YOCTO.BSP-AVE workspace setup ==="
echo "Pinning external layers from: ${LOCK}"
echo ""

# Parse sources.lock (skip blank lines and comments)
while IFS= read -r line; do
    # Strip comments and leading/trailing whitespace
    line="${line%%#*}"
    line="${line#"${line%%[![:space:]]*}"}"
    line="${line%"${line##*[![:space:]]}"}"
    [[ -z "${line}" ]] && continue

    read -r name url sha <<< "${line}"
    clone_or_update "${name}" "${url}" "${sha}"
done < "${LOCK}"

echo ""
echo "=== Done ==="
echo ""
echo "To start a build:"
echo "  . ${SCRIPT_DIR}/oe-init-build-env [build-dir]"
echo "  MACHINE=frdm-imx8mp bitbake ave-base-image"
echo "  MACHINE=frdm-imx93  bitbake ave-base-image"
