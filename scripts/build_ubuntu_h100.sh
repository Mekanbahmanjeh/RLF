#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
JOBS="${RLF_BUILD_JOBS:-$(nproc)}"

command -v nvcc >/dev/null 2>&1 || {
  echo "nvcc was not found. Install CUDA Toolkit 12.x or newer." >&2
  exit 1
}
command -v nvidia-smi >/dev/null 2>&1 || {
  echo "nvidia-smi was not found. Install/activate the NVIDIA driver." >&2
  exit 1
}

cd "${ROOT}"
cmake --preset ubuntu-h100-cuda
cmake --build --preset ubuntu-h100-cuda --parallel "${JOBS}"
ctest --preset ubuntu-h100-cuda --parallel "${JOBS}" --output-on-failure

"${ROOT}/build/ubuntu-h100-cuda/solstice" device-info \
  --backend cuda --profile frontier-h100
"${ROOT}/build/ubuntu-h100-cuda/rlf_frontier_benchmark" \
  --output "${ROOT}/results/frontier_research_benchmark_h100.json"

echo "Build complete: ${ROOT}/build/ubuntu-h100-cuda/solstice"
