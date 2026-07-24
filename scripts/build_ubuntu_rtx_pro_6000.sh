#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"
cmake --preset ubuntu-rtx-pro-6000-cuda
cmake --build --preset ubuntu-rtx-pro-6000-cuda --parallel "${RLF_BUILD_JOBS:-$(nproc)}"
ctest --preset ubuntu-rtx-pro-6000-cuda --output-on-failure
for profile in rtx-pro-6000-96g general-rtx-pro-6000-96g general-rtx-pro-6000-96g-text video-rtx-pro-6000-96g; do
  "${ROOT}/build/ubuntu-rtx-pro-6000-cuda/solstice" profile-info --profile "${profile}"
done
"${ROOT}/build/ubuntu-rtx-pro-6000-cuda/solstice" device-info \
  --profile "${RLF_RTX_PRO_6000_PROFILE:-rtx-pro-6000-96g}" --backend cuda
