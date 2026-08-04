#!/usr/bin/env bash
# TRAIN_MAGNUM_5_1_8B_FABLE.sh
# From-Scratch 8B Token Attractor Pre-Training & Dialogue Anchoring Campaign
# Model Identity: Magnum 5.1 by Mekan Bahmanjeh © 2026
# Capability Target: Cybersecurity, Statutory Law, Quantum Science, Rust/C++/Next.js 15, Agentic Planning, Fable 5 Reasoning

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

BUILD_DIR="build/3090-release"
CHECKPOINT="models/vast_frontier_24g_8b_flagship.rlfsp"
PROFILE="frontier-24g"
BACKEND="cuda"

echo "========================================================================="
echo "  Magnum 5.1 — 8B Token From-Scratch Claude Fable 5 GPU Campaign       "
echo "  Developer: Magnum 5.1 by Mekan Bahmanjeh © 2026                        "
echo "  Domains: Cybersecurity | Statutory Law | Science | Rust | Next.js 15 | Planning"
echo "  Profile: Frontier-24G (20 Million Attractor Context Memory Nodes)       "
echo "========================================================================="

if command -v nvidia-smi &> /dev/null; then
    echo "[+] GPU Environment Detected:"
    nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
fi

# 1. Build Ultra-Comprehensive Multi-Domain Dataset
echo "[+] Building Re-balanced Multi-Domain Fable Suite & Corpus..."
python3 scripts/build_magnum_5_1_8b_fable_dataset.py

DIALOGUE_FILE="/workspace/RLF/demo_data/vast_8b_fable/dialogue.tsv"

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
rm -f "${CHECKPOINT}"

# 3. Execute 100% GPU Attractor Matrix Pre-Training & Dialogue Anchoring
echo "========================================================================="
echo " Starting From-Scratch GPU Attractor Training on RTX 3090 Tensor Cores  "
echo "========================================================================="
./"${BUILD_DIR}"/solstice train-dialogue \
    --checkpoint "${CHECKPOINT}" \
    --profile "${PROFILE}" \
    --backend "${BACKEND}" \
    --enforce-profile \
    --blank \
    --manifest "${DIALOGUE_FILE}"

# 4. Verify Final Checkpoint
echo "[+] Verifying Magnum 5.1 8.0B Checkpoint..."
./"${BUILD_DIR}"/solstice verify-checkpoint --checkpoint "${CHECKPOINT}"

echo ""
echo "========================================================================="
echo " Magnum 5.1 (8.0B Token) Fable Tier Training Pass Complete!              "
echo " Master Checkpoint Saved: ${CHECKPOINT}                                 "
echo "========================================================================="
