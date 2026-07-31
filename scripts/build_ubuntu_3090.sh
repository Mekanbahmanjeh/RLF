#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
JOBS="${RLF_BUILD_JOBS:-$(nproc)}"

if ! command -v nvcc >/dev/null 2>&1; then
  echo "nvcc was not found. Run scripts/setup_ubuntu_3090.sh and install CUDA Toolkit." >&2
  exit 1
fi
if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "nvidia-smi was not found. Install/activate the NVIDIA driver." >&2
  exit 1
fi

cd "${ROOT}"
cmake --preset ubuntu-rtx3090-cuda
cmake --build --preset ubuntu-rtx3090-cuda --parallel "${JOBS}"
ctest --preset ubuntu-rtx3090-cuda --parallel "${JOBS}" --output-on-failure

"${ROOT}/build/ubuntu-rtx3090-cuda/solstice" device-info \
  --backend cuda \
  --profile frontier-24g

echo
echo "Build complete: ${ROOT}/build/ubuntu-rtx3090-cuda/solstice"
