#!/usr/bin/env bash
# TRAIN_VAST_24G_8B_FLAGSHIP.sh
# Master 8.0 Billion Token CUDA Execution Script (Anthropic Mythos-Class Capability Level)
# Unbranded, Clean, High-Precision Reasoning Architecture.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

BUILD_DIR="build/3090-release"
CHECKPOINT="models/vast_frontier_24g_8b_flagship.rlfsp"
BOOTSTRAP_CKPT="models/vast_frontier_24g_5b_master.rlfsp"
PROFILE="frontier-24g"
BACKEND="cuda"

echo "========================================================================="
echo "  8.0 Billion Token Multi-Modal Mythos-Class Capability Pass              "
echo "  Profile: Frontier-24G (65,536 Vocab, 2M Episodes, 20M Contexts)        "
echo "========================================================================="

if command -v nvidia-smi &> /dev/null; then
    echo "[+] GPU Environment Detected:"
    nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
fi

# 1. Build 8.0B Unbranded Mythos-Class Multi-Modal Dataset
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
    --output-checkpoint "${CHECKPOINT}" \
    --profile "${PROFILE}" \
    --backend "${BACKEND}" \
    --enforce-profile \
    --input "${CORPUS_FILE}"

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
echo "[+] Verifying Final 8.0B Checkpoint..."
./"${BUILD_DIR}"/solstice verify-checkpoint --checkpoint "${CHECKPOINT}"

echo ""
echo "========================================================================="
echo " 8.0B Token Mythos-Class Master Pass Complete!                           "
echo " Master Checkpoint Saved: ${CHECKPOINT}                                 "
echo "========================================================================="

# 7. Automatic Post-Training Cleanup
echo "[+] Executing Automatic Post-Training Storage Cleanup..."
rm -rf /workspace/RLF/demo_data/* /workspace/RLF/build /tmp/* /root/.cache/* 2>/dev/null || true
df -h /workspace
echo "[+] AUTOMATIC CLEANUP COMPLETE! ONLY 8B MYTHOS-CLASS CHECKPOINT REMAINS!"
