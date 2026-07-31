#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD="${1:-${ROOT}/build-release-validation}"

cmake -S "${ROOT}" -B "${BUILD}" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DRLF_BUILD_TESTS=ON \
  -DRLF_ENABLE_CUDA=OFF
cmake --build "${BUILD}" -j2
ctest --test-dir "${BUILD}" --output-on-failure

"${BUILD}/rlf" experiment \
  --name rlf7_full \
  --config "${ROOT}/configs/rlf7_frontier.conf" \
  --output "${ROOT}/results/rlf7/rlf7_full.json"

"${BUILD}/rlf" experiment \
  --name frontier_full \
  --config "${ROOT}/configs/frontier_release.conf" \
  --agent-gate \
  --output "${ROOT}/results/frontier/frontier_full.json"

"${BUILD}/rlf" verify-checkpoint \
  --checkpoint "${ROOT}/results/rlf7/rlf7_model.rlf"
"${BUILD}/rlf" verify-checkpoint \
  --checkpoint "${ROOT}/results/frontier/frontier_model.rlf"

printf 'RLF-7 and RLF-Frontier release validation completed.\n'
