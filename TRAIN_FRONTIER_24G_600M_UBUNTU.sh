#!/usr/bin/env bash
# RLF Ubuntu 24GB VRAM — 600M+ Token Frontier Training Campaign Runner
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="${SCRIPT_DIR}"

BACKEND="cuda"
CHECKPOINT="${ROOT_DIR}/models/frontier_24g_600m_master.rlfsp"
DATA_DIR="${ROOT_DIR}/demo_data/frontier_600m"

echo "========================================================================="
echo "  RLF Ubuntu 24GB VRAM — 600M+ Token Frontier Training Campaign         "
echo "  Target Architecture: Frontier-24G (65,536 Vocab, 2M Episodes)          "
echo "========================================================================="

# Compile executable if missing
EXECUTABLE="${ROOT_DIR}/build/windows-msvc-cuda-release/Release/solstice.exe"
if [ ! -f "${EXECUTABLE}" ]; then
    EXECUTABLE="${ROOT_DIR}/build/3090-release/solstice"
fi
if [ ! -f "${EXECUTABLE}" ]; then
    EXECUTABLE="${ROOT_DIR}/build/ubuntu-cuda-release/solstice"
fi

if [ ! -f "${EXECUTABLE}" ]; then
    echo "[+] Compiling RLF Solstice CUDA release binary for Ubuntu..."
    if [ -f "${ROOT_DIR}/BUILD_UBUNTU_3090.sh" ]; then
        bash "${ROOT_DIR}/BUILD_UBUNTU_3090.sh"
    elif [ -f "${ROOT_DIR}/scripts/build_ubuntu_3090.sh" ]; then
        bash "${ROOT_DIR}/scripts/build_ubuntu_3090.sh"
    fi
fi

if [ ! -f "${EXECUTABLE}" ]; then
    echo "[-] Error: Solstice binary not found. Build manually with cmake --preset 3090-release."
    exit 1
fi

echo "[+] Using Solstice executable: ${EXECUTABLE}"

# Ingest/download dataset
echo "[+] Fetching and formatting 600M+ token CoT reasoning dataset..."
python3 "${ROOT_DIR}/scripts/download_and_build_600m_cot_dataset.py" --output-dir "${DATA_DIR}" --max-samples 1500000

# Ensure dialogues are present
if [ -f "${ROOT_DIR}/scripts/windows/create_preview_conversation_data.ps1" ]; then
    powershell.exe -NoProfile -ExecutionPolicy Bypass -File "${ROOT_DIR}/scripts/windows/create_preview_conversation_data.ps1" -OutputDirectory "${DATA_DIR}" || true
fi

mkdir -p "$(dirname "${CHECKPOINT}")"

echo "========================================================================="
echo " Starting 600M+ Token CUDA Training Pass on 24GB GPU                    "
echo "========================================================================="

"${EXECUTABLE}" train-text \
    --checkpoint "${CHECKPOINT}" \
    --profile frontier-24g \
    --backend "${BACKEND}" \
    --enforce-profile \
    --blank \
    --input "${DATA_DIR}/corpus.txt"

"${EXECUTABLE}" train-instructions \
    --checkpoint "${CHECKPOINT}" \
    --profile frontier-24g \
    --backend "${BACKEND}" \
    --enforce-profile \
    --manifest "${DATA_DIR}/instructions.tsv"

"${EXECUTABLE}" verify \
    --checkpoint "${CHECKPOINT}" \
    --profile frontier-24g \
    --backend "${BACKEND}" \
    --enforce-profile

echo ""
echo "========================================================================="
echo " 600M+ Token Ubuntu Training Campaign Complete!                          "
echo " Master Checkpoint: ${CHECKPOINT}                                       "
echo " Launch Chat: ./CHAT_FRONTIER_24G_UBUNTU.sh                              "
echo "========================================================================="
