#!/usr/bin/env bash
# RLF Vast.ai 24GB GPU — Interactive Chat Runner
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="${SCRIPT_DIR}"

BACKEND="cuda"
CHECKPOINT="${ROOT_DIR}/models/vast_frontier_24g_1b_master.rlfsp"

EXECUTABLE="${ROOT_DIR}/build/3090-release/solstice"
if [ ! -f "${EXECUTABLE}" ]; then
    EXECUTABLE="${ROOT_DIR}/build/windows-msvc-cuda-release/Release/solstice.exe"
fi
if [ ! -f "${EXECUTABLE}" ]; then
    EXECUTABLE="${ROOT_DIR}/build/ubuntu-cuda-release/solstice"
fi

if [ ! -f "${CHECKPOINT}" ]; then
    echo "[-] Error: Missing checkpoint ${CHECKPOINT}. Run ./TRAIN_VAST_24G_1B.sh first."
    exit 1
fi

echo "========================================================================="
echo " Launching RLF Frontier 24G Interactive Chat (${BACKEND})               "
echo "========================================================================="

"${EXECUTABLE}" chat \
    --checkpoint "${CHECKPOINT}" \
    --profile frontier-24g \
    --backend "${BACKEND}" \
    --enforce-profile \
    --no-tools \
    --max-tokens 512
