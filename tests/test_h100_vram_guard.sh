#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORK="$(mktemp -d)"
trap 'rm -rf -- "${WORK}"' EXIT
export RLF_NVIDIA_SMI="${ROOT}/tests/fixtures/fake_nvidia_smi.sh"

RLF_FAKE_VRAM_MIB=70000 "${ROOT}/scripts/run_h100_with_vram_guard.sh" \
  --trace "${WORK}/within.csv" --summary "${WORK}/within.json" \
  --interval-ms 10 -- bash -c 'sleep 0.04'
grep -q '"within_limit": true' "${WORK}/within.json"
grep -q '"sampler_ok": true' "${WORK}/within.json"
grep -q '"peak_memory_mib": 70000' "${WORK}/within.json"

set +e
RLF_FAKE_VRAM_MIB=77825 "${ROOT}/scripts/run_h100_with_vram_guard.sh" \
  --trace "${WORK}/exceeded.csv" --summary "${WORK}/exceeded.json" \
  --interval-ms 10 -- bash -c 'sleep 0.04'
STATUS=$?
set -e
[[ "${STATUS}" -eq 4 ]]
grep -q '"within_limit": false' "${WORK}/exceeded.json"
grep -q '"peak_memory_mib": 77825' "${WORK}/exceeded.json"
