#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

fail=0
check_cmd() {
  local name="$1"
  if command -v "${name}" >/dev/null 2>&1; then
    printf '[ok] %-12s %s\n' "${name}" "$(command -v "${name}")"
  else
    printf '[missing] %s\n' "${name}"
    fail=1
  fi
}

printf 'RLF one-H100 Codex campaign preflight\n'
printf 'root: %s\n' "${ROOT}"
for required in README.md CMakeLists.txt; do
  if [[ -f "${required}" ]]; then
    printf '[ok] file       %s\n' "${required}"
  else
    printf '[missing] file  %s\n' "${required}"
    fail=1
  fi
done

check_cmd cmake
check_cmd ctest
check_cmd g++
check_cmd ninja
check_cmd nvcc
check_cmd nvidia-smi
check_cmd codex
check_cmd git
check_cmd sha256sum

printf '\nSystem:\n'
uname -a || true
printf '\nGPU:\n'
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader || true
  gpu_name="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -n 1 || true)"
  gpu_count="$(nvidia-smi --query-gpu=count --format=csv,noheader 2>/dev/null | head -n 1 || true)"
  if [[ "${gpu_name}" != *H100* ]]; then
    printf '[warning] first visible GPU is not reported as an H100: %s\n' "${gpu_name:-unknown}"
  fi
  if [[ -n "${gpu_count}" && "${gpu_count}" != "1" ]]; then
    printf '[warning] visible GPU count is %s; the claimed campaign must use exactly one H100.\n' "${gpu_count}"
  fi
fi

printf '\nTool versions:\n'
cmake --version 2>/dev/null | head -n 1 || true
g++ --version 2>/dev/null | head -n 1 || true
nvcc --version 2>/dev/null | tail -n 1 || true
codex --version 2>/dev/null || true

mkdir -p results/codex_campaign
if [[ "${fail}" -ne 0 ]]; then
  printf '\nPreflight found missing required tools or files.\n' >&2
  exit 1
fi
printf '\nPreflight completed. Warnings do not imply frontier readiness.\n'
