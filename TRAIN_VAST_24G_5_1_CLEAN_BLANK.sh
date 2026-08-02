#!/usr/bin/env bash
# TRAIN_VAST_24G_5_1_CLEAN_BLANK.sh
# Master Clean (--blank) Training Execution Script for Magnum 5.1

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

BUILD_DIR="build/3090-release"
CHECKPOINT="models/vast_frontier_24g_8b_flagship.rlfsp"
PROFILE="frontier-24g"
BACKEND="cuda"

echo "========================================================================="
echo "  Magnum 5.1 — Clean (--blank) Multi-Domain Attractor Training Pass      "
echo "========================================================================="

# 1. Build Multi-Domain Dataset (102,000 Rows)
python3 scripts/build_magnum_5_1_dialogue_dataset.py

CORPUS_FILE="/workspace/RLF/demo_data/vast_8b_dialogue/corpus.txt"
DIALOGUE_FILE="/workspace/RLF/demo_data/vast_8b_dialogue/dialogue.tsv"

# 2. Build C++ Solstice Engine Executable
cmake -B "${BUILD_DIR}" -S . \
    -DCMAKE_BUILD_TYPE=Release \
    -DRLF_ENABLE_CUDA=ON \
    -DCMAKE_CUDA_ARCHITECTURES="80;86;89;90" \
    -DRLF_WARNINGS_AS_ERRORS=OFF \
    -DBUILD_TESTING=OFF

cmake --build "${BUILD_DIR}" --target solstice -j$(nproc)

# Remove old polluted checkpoint file if present
rm -f "${CHECKPOINT}" 2>/dev/null || true
mkdir -p models

# 3. Execute Clean (--blank) Attractor Training Pass
echo "========================================================================="
echo " Starting Fresh Clean (--blank) Attractor Matrix Initialization on GPU   "
echo "========================================================================="
./"${BUILD_DIR}"/solstice train-text \
    --checkpoint "${CHECKPOINT}" \
    --profile "${PROFILE}" \
    --backend "${BACKEND}" \
    --enforce-profile \
    --blank \
    --input "${CORPUS_FILE}"

# 4. Execute 2-Column Dialogue TSV Attractor Fine-Tuning Pass
echo "========================================================================="
echo " Starting 2-Column Dialogue Q&A Attractor Anchoring Pass                 "
echo "========================================================================="
./"${BUILD_DIR}"/solstice train-dialogue \
    --checkpoint "${CHECKPOINT}" \
    --profile "${PROFILE}" \
    --backend "${BACKEND}" \
    --enforce-profile \
    --manifest "${DIALOGUE_FILE}"

# 5. Verify Clean Master Checkpoint
echo "[+] Verifying Clean Magnum 5.1 8.0B Checkpoint..."
./"${BUILD_DIR}"/solstice verify-checkpoint --checkpoint "${CHECKPOINT}"

echo ""
echo "========================================================================="
echo " Clean Magnum 5.1 (8.0B Token) Master Training Pass Complete!            "
echo " Saved Checkpoint: ${CHECKPOINT}                                        "
echo "========================================================================="
