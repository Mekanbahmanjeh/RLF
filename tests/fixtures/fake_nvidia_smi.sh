#!/usr/bin/env bash
set -euo pipefail
case "$*" in
  *"--query-gpu=index,name,uuid,memory.total,memory.free,compute_cap,driver_version"*)
    printf '0, %s, GPU-TEST-0001, %s, %s, %s, 999.1\n' \
      "${RLF_FAKE_GPU_NAME:-NVIDIA H100 80GB HBM3}" \
      "${RLF_FAKE_TOTAL_MIB:-81559}" \
      "${RLF_FAKE_FREE_MIB:-80000}" \
      "${RLF_FAKE_COMPUTE_CAP:-9.0}"
    ;;
  *"--query-gpu=name,uuid,memory.total,driver_version"*)
    printf '%s, GPU-TEST-0001, %s, 999.1\n' \
      "${RLF_FAKE_GPU_NAME:-NVIDIA H100 80GB HBM3}" \
      "${RLF_FAKE_TOTAL_MIB:-81559}"
    ;;
  *"--query-gpu=name,uuid,memory.total,compute_cap,driver_version"*)
    printf '%s, GPU-TEST-0001, %s, %s, 999.1\n' \
      "${RLF_FAKE_GPU_NAME:-NVIDIA H100 80GB HBM3}" \
      "${RLF_FAKE_TOTAL_MIB:-81559}" \
      "${RLF_FAKE_COMPUTE_CAP:-9.0}"
    ;;
  *"--query-gpu=memory.used,utilization.gpu,power.draw"*)
    printf '%s, %s, %s\n' "${RLF_FAKE_VRAM_MIB:-70000}" \
      "${RLF_FAKE_GPU_UTILIZATION:-50}" "${RLF_FAKE_POWER_DRAW:-250.0}"
    ;;
  *"--query-gpu=memory.used"*)
    printf '%s\n' "${RLF_FAKE_VRAM_MIB:-70000}"
    ;;
  *)
    echo "unsupported fake nvidia-smi invocation: $*" >&2
    exit 2
    ;;
esac
