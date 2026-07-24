#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"
cmake --preset ubuntu-general-cuda-compat
cmake --build --preset ubuntu-general-cuda-compat --parallel "${RLF_BUILD_JOBS:-$(nproc)}"
ctest --preset ubuntu-general-cuda-compat --output-on-failure
"${ROOT}/build/ubuntu-general-cuda-compat/solstice" profile-info --profile general-40g
"${ROOT}/build/ubuntu-general-cuda-compat/solstice" profile-info --profile general-v100-32g
"${ROOT}/build/ubuntu-general-cuda-compat/solstice" profile-info --profile general-v100-32g-text
"${ROOT}/build/ubuntu-general-cuda-compat/solstice" profile-info --profile video-v100-32g
"${ROOT}/build/ubuntu-general-cuda-compat/solstice" device-info \
  --profile "${RLF_GENERAL_CUDA_PROFILE:-general-40g}" --backend cuda
