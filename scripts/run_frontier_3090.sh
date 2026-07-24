#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BIN="${ROOT}/build/ubuntu-rtx3090-cuda/solstice"
CHECKPOINT="${1:-${ROOT}/models/solstice_frontier_3090_bootstrap.rlfsp}"
TOOL_ROOT="${2:-${ROOT}}"

if [[ ! -x "${BIN}" ]]; then
  echo "CUDA executable not found. Run scripts/build_ubuntu_3090.sh first." >&2
  exit 1
fi
if [[ ! -f "${CHECKPOINT}" ]]; then
  echo "Checkpoint not found. Run scripts/bootstrap_frontier_3090.sh first." >&2
  exit 1
fi

export CUDA_MODULE_LOADING=LAZY
export MALLOC_ARENA_MAX="${MALLOC_ARENA_MAX:-4}"

exec "${BIN}" chat \
  --backend cuda \
  --profile frontier-24g \
  --checkpoint "${CHECKPOINT}" \
  --tool-root "${TOOL_ROOT}" \
  --max-tokens 1024 \
  --top-k 64 \
  --temperature 0.75
