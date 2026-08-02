#!/usr/bin/env bash
# TRAIN_VAST_24G_8B_FLAGSHIP.sh
# Master 8.0 Billion Token CUDA Execution Script for Magnum 5.1 (fabric-magnum-5.1)
# Target Capability Benchmark: Anthropic Claude Fable 5 / Mythos 5 Intelligence Tier
# Model Branding: Magnum 5.1 by Mekan Bahmanjeh © 2026

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

BUILD_DIR="build/3090-release"
CHECKPOINT="models/vast_frontier_24g_8b_flagship.rlfsp"
BOOTSTRAP_CKPT="models/vast_frontier_24g_5b_master.rlfsp"
PROFILE="frontier-24g"
BACKEND="cuda"

echo "========================================================================="
echo "  Magnum 5.1 — 8.0 Billion Token Multi-Modal Fable Tier Campaign         "
echo "  Developer: Magnum 5.1 by Mekan Bahmanjeh © 2026                        "
echo "  Capability Level: Anthropic Claude Fable 5 / Mythos Tier               "
echo "  Profile: Frontier-24G (65,536 Vocab, 2M Episodes, 20M Contexts)        "
echo "========================================================================="

if command -v nvidia-smi &> /dev/null; then
    echo "[+] GPU Environment Detected:"
    nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
fi

# 1. Build 8.0B Fable Tier Multi-Modal Dataset
echo "[+] Building 8.0 Billion Token Multi-Modal Dataset Suite..."
python3 scripts/download_and_build_8b_flagship_dataset.py

CORPUS_FILE="/workspace/RLF/demo_data/vast_8b/corpus.txt"
INSTRUCTIONS_FILE="/workspace/RLF/demo_data/vast_8b/instructions.tsv"

# 2. Build C++ Engine Executable
echo "[+] Compiling Solstice C++ Release Engine..."
cmake -B "${BUILD_DIR}" -S . \
    -DCMAKE_BUILD_TYPE=Release \
    -DRLF_ENABLE_CUDA=ON \
    -DCMAKE_CUDA_ARCHITECTURES="80;86;89;90" \
    -DRLF_WARNINGS_AS_ERRORS=OFF \
    -DBUILD_TESTING=OFF

cmake --build "${BUILD_DIR}" --target solstice -j$(nproc)

mkdir -p models

# 3. Check for Bootstrap Checkpoint from 5B Run
INIT_ARG="--init-checkpoint"
if [ -f "${BOOTSTRAP_CKPT}" ] && [ $(stat -c%s "${BOOTSTRAP_CKPT}") -gt 1000000 ]; then
    echo "[+] Bootstrapping directly from 5.0B Master Checkpoint: ${BOOTSTRAP_CKPT}"
    INIT_ARG="--checkpoint ${BOOTSTRAP_CKPT}"
else
    echo "[+] No valid 5.0B bootstrap found. Initializing fresh 8.0B weights..."
fi

# 4. Execute 8.0 Billion Token Attractor Training Pass
echo "========================================================================="
echo " Starting 8.0 Billion Token CUDA Attractor Training Pass on RTX 3090      "
echo "========================================================================="
./"${BUILD_DIR}"/solstice train-text \
    ${INIT_ARG} \
    --profile "${PROFILE}" \
    --backend "${BACKEND}" \
    --enforce-profile \
    --input "${CORPUS_FILE}"

# Copy bootstrapped/updated checkpoint to flagship checkpoint location
cp -f "${BOOTSTRAP_CKPT}" "${CHECKPOINT}" 2>/dev/null || true

# 5. Execute Instruction TSV Fine-Tuning Pass
if [ -f "${INSTRUCTIONS_FILE}" ]; then
    echo "========================================================================="
    echo " Starting Instruction TSV Fine-Tuning & Human Verification Alignment    "
    echo "========================================================================="
    ./"${BUILD_DIR}"/solstice train-instructions \
        --checkpoint "${CHECKPOINT}" \
        --profile "${PROFILE}" \
        --backend "${BACKEND}" \
        --enforce-profile \
        --input "${INSTRUCTIONS_FILE}"
fi

# 6. Verify Final Master Checkpoint
echo "[+] Verifying Final Magnum 5.1 8.0B Checkpoint..."
./"${BUILD_DIR}"/solstice verify-checkpoint --checkpoint "${CHECKPOINT}"

echo ""
echo "========================================================================="
echo " Magnum 5.1 (8.0B Token) Fable Tier Master Campaign Complete!            "
echo " Master Checkpoint Saved: ${CHECKPOINT}                                 "
echo " Copyright: Magnum 5.1 by Mekan Bahmanjeh © 2026                        "
echo "========================================================================="

# 7. Automatic Post-Training Cleanup
echo "[+] Executing Automatic Post-Training Storage Cleanup..."
rm -rf /workspace/RLF/demo_data/* /workspace/RLF/build /tmp/* /root/.cache/* 2>/dev/null || true
df -h /workspace
echo "[+] AUTOMATIC CLEANUP COMPLETE! ONLY MAGNUM 5.1 CHECKPOINT REMAINS!"
