#!/usr/bin/env bash
# Magnum 5 — 5.0 Billion Token Multi-Modal Master Campaign
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="${SCRIPT_DIR}"

BACKEND="cuda"
BOOTSTRAP_CHECKPOINT="${ROOT_DIR}/models/vast_frontier_24g_1b_master.rlfsp"
CHECKPOINT="${ROOT_DIR}/models/vast_frontier_24g_5b_master.rlfsp"
DATA_DIR="${ROOT_DIR}/demo_data/vast_5b"

echo "========================================================================="
echo "  Magnum 5 — 5.0 Billion Token Multi-Modal Master Campaign               "
echo "  Developer: Magnum 5 by Mekan Bahmanjeh © 2026                          "
echo "  Profile: Frontier-24G (65,536 Vocab, 2M Episodes, 20M Contexts)        "
echo "========================================================================="

if [ -f "${ROOT_DIR}/.env" ]; then
    echo "[+] Loading environment variables from .env"
    set -a
    source "${ROOT_DIR}/.env"
    set +a
fi

if command -v nvidia-smi &> /dev/null; then
    echo "[+] GPU Environment Detected:"
    nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader || true
fi

EXECUTABLE="${ROOT_DIR}/build/3090-release/solstice"
if [ ! -f "${EXECUTABLE}" ]; then
    echo "[+] Building Solstice CUDA release executable..."
    mkdir -p "${ROOT_DIR}/build/3090-release"
    cd "${ROOT_DIR}"
    cmake -B build/3090-release -S . -DCMAKE_BUILD_TYPE=Release -DRLF_ENABLE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="80;86;89;90" -DRLF_WARNINGS_AS_ERRORS=OFF -DBUILD_TESTING=OFF
    cmake --build build/3090-release --target solstice -j"$(nproc)"
fi

echo "[+] Building 5.0 Billion Token Multi-Modal & Multi-Domain Dataset Suite..."
python3 "${ROOT_DIR}/scripts/download_and_build_5b_multimodal_dataset.py" --output-dir "${DATA_DIR}" --max-samples 5000000

echo "[+] Injecting Magnum 5 Brand Identity, Anti-Slop WebGL & Autonomous Debug Seeds..."
python3 "${ROOT_DIR}/scripts/inject_magnum5_identity.py" --output-dir "${DATA_DIR}"
python3 "${ROOT_DIR}/scripts/inject_anti_slop_seeds.py" --output-dir "${DATA_DIR}"
python3 "${ROOT_DIR}/scripts/inject_unattended_autonomous_seeds.py" --output-dir "${DATA_DIR}"
python3 "${ROOT_DIR}/scripts/inject_magnum5_branding.py" --output-dir "${DATA_DIR}"

mkdir -p "$(dirname "${CHECKPOINT}")"

if [ -f "${BOOTSTRAP_CHECKPOINT}" ]; then
    echo "[+] Bootstrapping from 1.0B Checkpoint: ${BOOTSTRAP_CHECKPOINT}"
    cp "${BOOTSTRAP_CHECKPOINT}" "${CHECKPOINT}"
fi

echo "========================================================================="
echo " Starting 5.0 Billion Token CUDA Attractor Training Pass on RTX 3090      "
echo "========================================================================="

"${EXECUTABLE}" train-text \
    --checkpoint "${CHECKPOINT}" \
    --profile frontier-24g \
    --backend "${BACKEND}" \
    --enforce-profile \
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
echo " Magnum 5 (5.0B Token) Master Campaign Complete!                         "
echo " Master Checkpoint Saved: ${CHECKPOINT}                                 "
echo " Copyright: Magnum 5 by Mekan Bahmanjeh © 2026                           "
echo "========================================================================="
