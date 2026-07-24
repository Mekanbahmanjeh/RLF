#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BIN="${ROOT}/build/ubuntu-h100-cuda/solstice"
CHECKPOINT="${1:-${ROOT}/models/solstice_frontier_h100_bootstrap.rlfsp}"
TOOL_ROOT="${2:-${ROOT}}"

[[ -x "${BIN}" ]] || {
  echo "H100 executable not found. Run scripts/build_ubuntu_h100.sh first." >&2
  exit 1
}
[[ -f "${CHECKPOINT}" ]] || {
  echo "Checkpoint not found. Run scripts/bootstrap_frontier_h100.sh first." >&2
  exit 1
}

export CUDA_MODULE_LOADING=LAZY
export MALLOC_ARENA_MAX="${MALLOC_ARENA_MAX:-4}"

exec "${BIN}" chat --backend cuda --profile frontier-h100 \
  --checkpoint "${CHECKPOINT}" --tool-root "${TOOL_ROOT}" \
  --max-tokens 2048 --top-k 128 --temperature 0.72
