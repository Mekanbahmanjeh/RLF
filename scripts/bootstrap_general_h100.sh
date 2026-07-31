#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BIN="${ROOT}/build/ubuntu-h100-cuda/solstice"
CHECKPOINT="${1:-${ROOT}/models/solstice_general_h100_bootstrap.rlfsp}"
[[ -x "${BIN}" ]] || { echo "Build the H100 target first." >&2; exit 1; }
export CUDA_MODULE_LOADING=LAZY
export MALLOC_ARENA_MAX="${MALLOC_ARENA_MAX:-4}"
"${BIN}" bootstrap --profile general-h100 --backend cuda --enforce-profile --checkpoint "${CHECKPOINT}"
"${BIN}" train-text --profile general-h100 --backend cuda --enforce-profile --input "${ROOT}/examples/solstice/corpus.txt" --checkpoint "${CHECKPOINT}"
"${BIN}" train-dialogue --profile general-h100 --backend cuda --enforce-profile --manifest "${ROOT}/examples/solstice/dialogues.tsv" --checkpoint "${CHECKPOINT}"
"${BIN}" train-instructions --profile general-h100 --backend cuda --enforce-profile --manifest "${ROOT}/examples/solstice/general_instructions.tsv" --checkpoint "${CHECKPOINT}"
"${BIN}" train-preferences --profile general-h100 --backend cuda --enforce-profile --manifest "${ROOT}/examples/solstice/preferences.tsv" --checkpoint "${CHECKPOINT}"
"${BIN}" train-vision --profile general-h100 --backend cuda --enforce-profile --manifest "${ROOT}/examples/solstice/vision.tsv" --checkpoint "${CHECKPOINT}"
"${BIN}" train-tools --profile general-h100 --backend cuda --enforce-profile --manifest "${ROOT}/examples/solstice/tools.tsv" --checkpoint "${CHECKPOINT}"
"${BIN}" train-facts --profile general-h100 --backend cuda --enforce-profile --manifest "${ROOT}/examples/solstice/facts.tsv" --checkpoint "${CHECKPOINT}"
"${BIN}" train-rules --profile general-h100 --backend cuda --enforce-profile --manifest "${ROOT}/examples/solstice/rules.tsv" --checkpoint "${CHECKPOINT}"
"${BIN}" verify-checkpoint --checkpoint "${CHECKPOINT}" --profile general-h100 --enforce-profile
echo "Created ${CHECKPOINT}"
