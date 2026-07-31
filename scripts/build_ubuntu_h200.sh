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
cmake --preset ubuntu-h200-cuda
cmake --build --preset ubuntu-h200-cuda --parallel "${JOBS}"
ctest --preset ubuntu-h200-cuda --parallel "${JOBS}" --output-on-failure
"${ROOT}/build/ubuntu-h200-cuda/solstice" profile-info \
  --profile general-h200-141g-30t
"${ROOT}/build/ubuntu-h200-cuda/solstice" device-info \
  --backend cuda --profile general-h200-141g-30t

echo "Build complete: ${ROOT}/build/ubuntu-h200-cuda/solstice"
