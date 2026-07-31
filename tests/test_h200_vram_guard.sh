#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORK="$(mktemp -d)"
trap 'rm -rf -- "${WORK}"' EXIT
export RLF_NVIDIA_SMI="${ROOT}/tests/fixtures/fake_nvidia_smi.sh"
export RLF_FAKE_GPU_NAME="NVIDIA H200"
export RLF_FAKE_TOTAL_MIB=143771

RLF_FAKE_VRAM_MIB=130000 "${ROOT}/scripts/run_h200_with_vram_guard.sh" \
  --trace "${WORK}/within.csv" --summary "${WORK}/within.json" \
  --interval-ms 10 -- bash -c 'sleep 0.04'
grep -Fq '"schema": "rlf-h200-vram-v1"' "${WORK}/within.json"
grep -Fq '"within_limit": true' "${WORK}/within.json"

set +e
RLF_FAKE_VRAM_MIB=135169 "${ROOT}/scripts/run_h200_with_vram_guard.sh" \
  --trace "${WORK}/exceeded.csv" --summary "${WORK}/exceeded.json" \
  --interval-ms 10 -- bash -c 'sleep 0.04'
STATUS=$?
set -e
[[ "${STATUS}" -eq 4 ]]
grep -Fq '"within_limit": false' "${WORK}/exceeded.json"
