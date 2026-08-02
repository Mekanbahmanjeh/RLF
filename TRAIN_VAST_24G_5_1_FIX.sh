#!/usr/bin/env bash
# TRAIN_VAST_24G_5_1_FIX.sh
# Fine-Tuning Execution Script to Fix Open-Ended Text Generation in Magnum 5.1

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

BUILD_DIR="build/3090-release"
CHECKPOINT="models/vast_frontier_24g_8b_flagship.rlfsp"
PROFILE="frontier-24g"
BACKEND="cuda"

echo "========================================================================="
echo "  Magnum 5.1 — Re-balanced Open-Ended Generation Fine-Tuning Pass        "
echo "========================================================================="

# 1. Generate Re-balanced Dialogue Dataset
python3 scripts/build_magnum_5_1_dialogue_dataset.py

CORPUS_FILE="/workspace/RLF/demo_data/vast_8b_dialogue/corpus.txt"
DIALOGUE_FILE="/workspace/RLF/demo_data/vast_8b_dialogue/dialogue.tsv"

# 2. Build Solstice C++ Executable
cmake -B "${BUILD_DIR}" -S . \
    -DCMAKE_BUILD_TYPE=Release \
    -DRLF_ENABLE_CUDA=ON \
    -DCMAKE_CUDA_ARCHITECTURES="80;86;89;90" \
    -DRLF_WARNINGS_AS_ERRORS=OFF \
    -DBUILD_TESTING=OFF

cmake --build "${BUILD_DIR}" --target solstice -j$(nproc)

# 3. Fine-Tune on Re-balanced Text Corpus
echo "[+] Executing Text Corpus Fine-Tuning Pass..."
./"${BUILD_DIR}"/solstice train-text \
    --checkpoint "${CHECKPOINT}" \
    --profile "${PROFILE}" \
    --backend "${BACKEND}" \
    --enforce-profile \
    --input "${CORPUS_FILE}"

# 4. Fine-Tune with 2-Column Dialogue TSV
echo "[+] Executing 2-Column Dialogue TSV Attractor Fine-Tuning Pass..."
./"${BUILD_DIR}"/solstice train-dialogue \
    --checkpoint "${CHECKPOINT}" \
    --profile "${PROFILE}" \
    --backend "${BACKEND}" \
    --enforce-profile \
    --manifest "${DIALOGUE_FILE}"

# 5. Verify Updated Checkpoint
echo "[+] Verifying Updated Checkpoint..."
./"${BUILD_DIR}"/solstice verify-checkpoint --checkpoint "${CHECKPOINT}"

echo "========================================================================="
echo " Magnum 5.1 Fine-Tuning Complete!                                       "
echo "========================================================================="
