#!/usr/bin/env bash
# TRAIN_ULTRA_FAST_FABLE_5.sh
# Ultra-Fast GPU Attractor Training Script for Magnum 5.1 (Claude Fable 5 Tier)
# Bypasses the 2-hour CPU bottleneck by using 100% GPU train-dialogue execution.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

BUILD_DIR="build/3090-release"
CHECKPOINT="models/vast_frontier_24g_8b_flagship.rlfsp"
PROFILE="frontier-24g"
BACKEND="cuda"

echo "========================================================================="
echo "  Magnum 5.1 — Ultra-Fast Claude Fable 5 Tier GPU Training Campaign     "
echo "  Developer: Magnum 5.1 by Mekan Bahmanjeh © 2026                        "
echo "  Capability Target: Claude Fable 5 & Mythos Level Intelligence         "
echo "  Profile: Frontier-24G (20 Million Attractor Context Memory Nodes)       "
echo "========================================================================="

if command -v nvidia-smi &> /dev/null; then
    echo "[+] GPU Environment Detected:"
    nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
fi

# 1. Build Multi-Domain Dialogue Dataset (102,000 Rows)
echo "[+] Building Re-balanced Multi-Domain Dialogue Suite..."
python3 scripts/build_magnum_5_1_dialogue_dataset.py

DIALOGUE_FILE="/workspace/RLF/demo_data/vast_8b_dialogue/dialogue.tsv"

# 2. Build Solstice C++ Engine Executable
echo "[+] Compiling Solstice C++ Engine Executable..."
cmake -B "${BUILD_DIR}" -S . \
    -DCMAKE_BUILD_TYPE=Release \
    -DRLF_ENABLE_CUDA=ON \
    -DCMAKE_CUDA_ARCHITECTURES="80;86;89;90" \
    -DRLF_WARNINGS_AS_ERRORS=OFF \
    -DBUILD_TESTING=OFF

cmake --build "${BUILD_DIR}" --target solstice -j$(nproc)

mkdir -p models

# 3. Execute 100% GPU Attractor Matrix Training (Fast ~3-5 Minutes!)
echo "========================================================================="
echo " Starting 100% GPU Attractor Training Pass on RTX 3090 Tensor Cores     "
echo "========================================================================="
./"${BUILD_DIR}"/solstice train-dialogue \
    --checkpoint "${CHECKPOINT}" \
    --profile "${PROFILE}" \
    --backend "${BACKEND}" \
    --enforce-profile \
    --manifest "${DIALOGUE_FILE}"

# 4. Verify Final Checkpoint
echo "[+] Verifying Magnum 5.1 8.0B Checkpoint..."
./"${BUILD_DIR}"/solstice verify-checkpoint --checkpoint "${CHECKPOINT}"

echo ""
echo "========================================================================="
echo " Magnum 5.1 (8.0B Token) Fable Tier Training Pass Complete!              "
echo " Master Checkpoint Saved: ${CHECKPOINT}                                 "
echo "========================================================================="
