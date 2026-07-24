#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"
cmake --preset ubuntu-h100-cuda
cmake --build --preset ubuntu-h100-cuda --parallel "${RLF_BUILD_JOBS:-$(nproc)}"
ctest --preset ubuntu-h100-cuda --output-on-failure
"${ROOT}/build/ubuntu-h100-cuda/solstice" profile-info --profile general-h100
"${ROOT}/build/ubuntu-h100-cuda/solstice" device-info --profile general-h100 --backend cuda
