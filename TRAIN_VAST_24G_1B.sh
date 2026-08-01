#!/usr/bin/env bash
# RLF Vast.ai 24GB VRAM — 1-Billion Token Master Training Campaign
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="${SCRIPT_DIR}"

BACKEND="cuda"
CHECKPOINT="${ROOT_DIR}/models/vast_frontier_24g_1b_master.rlfsp"
DATA_DIR="${ROOT_DIR}/demo_data/vast_1b"

echo "========================================================================="
echo "  RLF Vast.ai 24GB GPU — 1-Billion Token Master Campaign                "
echo "  Target Architecture: Frontier-24G (65,536 Vocab, 2M Episodes, 20M Contexts)"
echo "========================================================================="

# 1. Load .env file for HF_TOKEN if available
if [ -f "${ROOT_DIR}/.env" ]; then
    echo "[+] Loading environment variables from .env"
    set -a
    source "${ROOT_DIR}/.env"
    set +a
fi

# 2. Check NVIDIA CUDA readiness
if command -v nvidia-smi &> /dev/null; then
    echo "[+] GPU Environment Detected:"
    nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader || true
else
    echo "[-] Warning: nvidia-smi not found. Ensure NVIDIA CUDA drivers are loaded."
fi

# 3. Install build tools if missing (Vast.ai ubuntu instance)
if ! command -v cmake &> /dev/null || ! command -v g++ &> /dev/null; then
    echo "[+] Installing build essentials (cmake, g++, ninja)..."
    apt-get update && apt-get install -y cmake build-essential ninja-build python3-pip || true
fi

# 4. Compile executable if missing
EXECUTABLE="${ROOT_DIR}/build/3090-release/solstice"
if [ ! -f "${EXECUTABLE}" ]; then
    EXECUTABLE="${ROOT_DIR}/build/windows-msvc-cuda-release/Release/solstice.exe"
fi
if [ ! -f "${EXECUTABLE}" ]; then
    EXECUTABLE="${ROOT_DIR}/build/ubuntu-cuda-release/solstice"
fi

if [ ! -f "${EXECUTABLE}" ]; then
    echo "[+] Building Solstice CUDA release executable for Vast.ai..."
    mkdir -p "${ROOT_DIR}/build/3090-release"
    cd "${ROOT_DIR}"
    cmake -B build/3090-release -S . -DCMAKE_BUILD_TYPE=Release -DRLF_ENABLE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="80;86;89;90" -DRLF_WARNINGS_AS_ERRORS=OFF -DBUILD_TESTING=OFF
    cmake --build build/3090-release --target solstice -j"$(nproc)"
    EXECUTABLE="${ROOT_DIR}/build/3090-release/solstice"
fi

if [ ! -f "${EXECUTABLE}" ]; then
    echo "[-] Error: Failed to build solstice executable."
    exit 1
fi

echo "[+] Using Solstice executable: ${EXECUTABLE}"

# 5. Download and stream CoT dataset from HF
echo "[+] Fetching and processing CoT reasoning dataset from Hugging Face..."
python3 "${ROOT_DIR}/scripts/download_and_build_1b_cot_dataset.py" --output-dir "${DATA_DIR}" --max-samples 2500000
if [ $? -ne 0 ]; then
    echo "[-] Error: Dataset download failed!"
    exit 1
fi

# Verify data files exist and are non-empty
if [ ! -s "${DATA_DIR}/corpus.txt" ]; then
    echo "[-] Error: corpus.txt is empty or missing!"
    exit 1
fi
if [ ! -s "${DATA_DIR}/instructions.tsv" ]; then
    echo "[-] Error: instructions.tsv is empty or missing!"
    exit 1
fi

echo "[+] Dataset files verified:"
wc -l "${DATA_DIR}/corpus.txt" "${DATA_DIR}/instructions.tsv"

mkdir -p "$(dirname "${CHECKPOINT}")"

echo "========================================================================="
echo " Starting CUDA Training Pass on 24GB Vast.ai GPU                        "
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

"${EXECUTABLE}" verify-checkpoint \
    --checkpoint "${CHECKPOINT}" \
    --profile frontier-24g \
    --backend "${BACKEND}" \
    --enforce-profile

echo ""
echo "========================================================================="
echo " Vast.ai Campaign Complete!                                              "
echo " Master Checkpoint: ${CHECKPOINT}                                       "
echo " Launch Chat: ./CHAT_VAST_24G.sh                                         "
echo "========================================================================="

