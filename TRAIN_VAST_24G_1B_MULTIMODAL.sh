#!/usr/bin/env bash
################################################################################
# RLF 1.0 Billion Token Multi-Modal Master Campaign Execution Script
# Target: Vast.ai 24GB VRAM GPU (NVIDIA RTX 3090 / RTX 4090 / A5000)
################################################################################
set -e

echo "========================================================================="
echo " Starting 1.0B Token Multi-Modal Master Campaign on Vast.ai 24GB GPU      "
echo "========================================================================="

# 1. Single-Instance Enforcement: Kill any leftover duplicate solstice workers
echo "[+] Enforcing Single-Instance Execution Policy..."
CURRENT_PID=$$
OTHER_PIDS=$(pgrep -f "solstice train-text" | grep -v "^${CURRENT_PID}$" || true)
if [ -n "$OTHER_PIDS" ]; then
    echo "[!] Found duplicate worker processes ($OTHER_PIDS). Terminating..."
    kill -9 $OTHER_PIDS 2>/dev/null || true
    sleep 2
    echo "[+] Duplicate processes successfully terminated."
fi

# 2. Build CMake CUDA Release Executable if missing
BUILD_DIR="/workspace/RLF/build/3090-release"
if [ ! -f "$BUILD_DIR/solstice" ]; then
    echo "[+] Compiling Solstice CUDA Engine in $BUILD_DIR..."
    mkdir -p "$BUILD_DIR"
    cd "$BUILD_DIR"
    cmake ../.. -DCMAKE_BUILD_TYPE=Release -DRLF_BUILD_CUDA=ON -DRLF_WARNINGS_AS_ERRORS=OFF -DBUILD_TESTING=OFF
    make -j$(nproc) solstice
    cd /workspace/RLF
fi

SOLSTICE_BIN="$BUILD_DIR/solstice"
echo "[+] Solstice CUDA Binary: $SOLSTICE_BIN"

# 3. Assemble 1.0 Billion Token Multi-Modal Master Dataset Suite
echo "[+] Assembling 1.0B Multi-Modal Master Dataset Suite..."
PYTHONUNBUFFERED=1 python3 scripts/download_and_build_1b_multimodal_master.py

CORPUS_PATH="/workspace/RLF/demo_data/vast_1b_multimodal/corpus.txt"
MANIFEST_PATH="/workspace/RLF/demo_data/vast_1b_multimodal/instructions.tsv"
MODEL_CHECKPOINT="/workspace/RLF/models/vast_frontier_24g_1b_master.rlfsp"

mkdir -p /workspace/RLF/models

echo "========================================================================="
echo " Starting CUDA Pre-Training Pass on 24GB Vast.ai GPU                    "
echo "========================================================================="

# 4. CUDA Pre-Training Pass on Text & Reasoning Corpus
PYTHONUNBUFFERED=1 "$SOLSTICE_BIN" train-text \
    --checkpoint "$MODEL_CHECKPOINT" \
    --profile frontier-24g \
    --backend cuda \
    --enforce-profile \
    --blank \
    --input "$CORPUS_PATH"

echo "========================================================================="
echo " Starting CUDA Attractor Learning Pass on Multi-Domain Manifest        "
echo "========================================================================="

# 5. CUDA Instruction Attractor Learning Pass
PYTHONUNBUFFERED=1 "$SOLSTICE_BIN" train-instructions \
    --checkpoint "$MODEL_CHECKPOINT" \
    --profile frontier-24g \
    --backend cuda \
    --manifest "$MANIFEST_PATH"

echo "========================================================================="
echo " Verifying 1.0B Multi-Modal Master Model Checkpoint Integrity           "
echo "========================================================================="

# 6. Verify Master Checkpoint Integrity
"$SOLSTICE_BIN" verify-checkpoint "$MODEL_CHECKPOINT"

echo "========================================================================="
echo " 1.0B Token Multi-Modal Master Campaign Execution Successfully Completed! "
echo " Master Checkpoint: $MODEL_CHECKPOINT                                    "
echo "========================================================================="
