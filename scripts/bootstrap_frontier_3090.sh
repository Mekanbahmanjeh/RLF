#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BIN="${ROOT}/build/ubuntu-rtx3090-cuda/solstice"
CHECKPOINT="${1:-${ROOT}/models/solstice_frontier_3090_bootstrap.rlfsp}"

if [[ ! -x "${BIN}" ]]; then
  echo "CUDA executable not found. Run scripts/build_ubuntu_3090.sh first." >&2
  exit 1
fi

export CUDA_MODULE_LOADING=LAZY
export MALLOC_ARENA_MAX="${MALLOC_ARENA_MAX:-4}"

"${BIN}" bootstrap \
  --profile frontier-24g \
  --backend cuda \
  --enforce-profile \
  --checkpoint "${CHECKPOINT}"

"${BIN}" train-text \
  --profile frontier-24g \
  --backend cuda \
  --enforce-profile \
  --input "${ROOT}/examples/solstice/corpus.txt" \
  --checkpoint "${CHECKPOINT}"

"${BIN}" train-dialogue \
  --profile frontier-24g \
  --backend cuda \
  --enforce-profile \
  --manifest "${ROOT}/examples/solstice/dialogues.tsv" \
  --checkpoint "${CHECKPOINT}"

"${BIN}" train-vision \
  --profile frontier-24g \
  --backend cuda \
  --enforce-profile \
  --manifest "${ROOT}/examples/solstice/vision.tsv" \
  --checkpoint "${CHECKPOINT}"

"${BIN}" train-tools \
  --profile frontier-24g \
  --backend cuda \
  --enforce-profile \
  --manifest "${ROOT}/examples/solstice/tools.tsv" \
  --checkpoint "${CHECKPOINT}"

"${BIN}" train-facts \
  --profile frontier-24g \
  --backend cuda \
  --enforce-profile \
  --manifest "${ROOT}/examples/solstice/facts.tsv" \
  --checkpoint "${CHECKPOINT}"

"${BIN}" train-rules \
  --profile frontier-24g \
  --backend cuda \
  --enforce-profile \
  --manifest "${ROOT}/examples/solstice/rules.tsv" \
  --checkpoint "${CHECKPOINT}"

"${BIN}" verify-checkpoint --checkpoint "${CHECKPOINT}" \
  --profile frontier-24g --enforce-profile
echo "Created ${CHECKPOINT}"
