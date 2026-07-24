#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BIN="${ROOT}/build/ubuntu-h100-cuda/solstice"
CHECKPOINT="${1:-${ROOT}/models/solstice_frontier_h100_bootstrap.rlfsp}"

[[ -x "${BIN}" ]] || {
  echo "H100 executable not found. Run scripts/build_ubuntu_h100.sh first." >&2
  exit 1
}

export CUDA_MODULE_LOADING=LAZY
export MALLOC_ARENA_MAX="${MALLOC_ARENA_MAX:-4}"

"${BIN}" bootstrap --profile frontier-h100 --backend cuda --enforce-profile --checkpoint "${CHECKPOINT}"
"${BIN}" train-text --profile frontier-h100 --backend cuda \
  --enforce-profile --input "${ROOT}/examples/solstice/corpus.txt" --checkpoint "${CHECKPOINT}"
"${BIN}" train-dialogue --profile frontier-h100 --backend cuda \
  --enforce-profile --manifest "${ROOT}/examples/solstice/dialogues.tsv" --checkpoint "${CHECKPOINT}"
"${BIN}" train-vision --profile frontier-h100 --backend cuda \
  --enforce-profile --manifest "${ROOT}/examples/solstice/vision.tsv" --checkpoint "${CHECKPOINT}"
"${BIN}" train-tools --profile frontier-h100 --backend cuda \
  --enforce-profile --manifest "${ROOT}/examples/solstice/tools.tsv" --checkpoint "${CHECKPOINT}"
"${BIN}" train-facts --profile frontier-h100 --backend cuda \
  --enforce-profile --manifest "${ROOT}/examples/solstice/facts.tsv" --checkpoint "${CHECKPOINT}"
"${BIN}" train-rules --profile frontier-h100 --backend cuda \
  --enforce-profile --manifest "${ROOT}/examples/solstice/rules.tsv" --checkpoint "${CHECKPOINT}"
"${BIN}" verify-checkpoint --checkpoint "${CHECKPOINT}" --profile frontier-h100 --enforce-profile

echo "Created ${CHECKPOINT}"
