#!/usr/bin/env bash
# TRAIN_MAGNUM_5_1_8B_FABLE.sh
# From-Scratch 8B Token Pre-Training & Instruction Anchoring
# Model: Magnum 5.1 by Mekan Bahmanjeh © 2026
# Architecture: train-text (free-form LLM generation) + train-dialogue (instruction following)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

BUILD_DIR="build/3090-release"
CHECKPOINT="models/vast_frontier_24g_8b_flagship.rlfsp"
PROFILE="frontier-24g"
BACKEND="cuda"

echo "========================================================================="
echo "  Magnum 5.1 — 8B Token From-Scratch Claude Fable 5 GPU Campaign       "
echo "  Developer: Mekan Bahmanjeh © 2026                                      "
echo "  Architecture: train-text (LLM generation) + train-dialogue (Q&A)      "
echo "  Domains: Cybersecurity | Law | Science | Coding | Math | Planning     "
echo "  Profile: Frontier-24G (20M Attractor Context Memory Nodes)            "
echo "========================================================================="

if command -v nvidia-smi &> /dev/null; then
    echo "[+] GPU Environment Detected:"
    nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
fi

# 1. Build Multi-Domain Dataset (dialogue.tsv + corpus.txt)
echo "[+] Building Multi-Domain Fable Suite & Corpus..."
python3 scripts/build_magnum_5_1_8b_fable_dataset.py
if [ $? -ne 0 ]; then
    echo "[!] FATAL: Dataset build failed. Aborting."
    exit 1
fi

DIALOGUE_FILE="/workspace/RLF/demo_data/vast_8b_fable/dialogue.tsv"
CORPUS_FILE="/workspace/RLF/demo_data/vast_8b_fable/corpus.txt"

echo "[+] Dataset files:"
ls -lh "${DIALOGUE_FILE}" "${CORPUS_FILE}"
echo "[+] Dialogue row count: $(wc -l < "${DIALOGUE_FILE}")"
echo "[+] Corpus size: $(wc -c < "${CORPUS_FILE}" | numfmt --to=iec-i)B"

# 2. Compile Solstice C++ Engine
echo "[+] Compiling Solstice C++ Engine..."
cmake -B "${BUILD_DIR}" -S . \
    -DCMAKE_BUILD_TYPE=Release \
    -DRLF_ENABLE_CUDA=ON \
    -DCMAKE_CUDA_ARCHITECTURES="80;86;89;90" \
    -DRLF_WARNINGS_AS_ERRORS=OFF \
    -DBUILD_TESTING=OFF

cmake --build "${BUILD_DIR}" --target solstice -j$(nproc)

mkdir -p models
rm -f "${CHECKPOINT}"

# 3. PHASE 1: train-text — Build Deep Predictive Context Tree (Free-Form LLM Generation)
echo "========================================================================="
echo " PHASE 1: train-text — Building Deep N-Gram Context Tree on CUDA GPU   "
echo " This enables FREE-FORM LLM text generation on ANY topic!              "
echo "========================================================================="
./"${BUILD_DIR}"/solstice train-text \
    --checkpoint "${CHECKPOINT}" \
    --profile "${PROFILE}" \
    --backend "${BACKEND}" \
    --enforce-profile \
    --blank \
    --corpus "${CORPUS_FILE}"

echo "[+] Phase 1 Complete! Checkpoint after train-text:"
ls -lh "${CHECKPOINT}"

# 4. PHASE 2: train-dialogue — Anchor Instruction-Following Episodes
echo "========================================================================="
echo " PHASE 2: train-dialogue — Anchoring Instruction Q&A Episodes on GPU   "
echo " This enables precise answers to known domain questions!                "
echo "========================================================================="
./"${BUILD_DIR}"/solstice train-dialogue \
    --checkpoint "${CHECKPOINT}" \
    --profile "${PROFILE}" \
    --backend "${BACKEND}" \
    --enforce-profile \
    --manifest "${DIALOGUE_FILE}"

echo "[+] Phase 2 Complete! Final checkpoint:"
ls -lh "${CHECKPOINT}"

# 5. Verify Final Checkpoint
echo "[+] Verifying Magnum 5.1 8.0B Checkpoint..."
./"${BUILD_DIR}"/solstice verify-checkpoint --checkpoint "${CHECKPOINT}"

echo ""
echo "========================================================================="
echo " Magnum 5.1 (8B Token) Fable 5 Level Training Complete!                "
echo " Master Checkpoint: ${CHECKPOINT}                                       "
echo "========================================================================="
