#!/usr/bin/env bash
set -uo pipefail

usage() {
  cat <<'EOF'
Usage: check_imagegen_v100_readiness.sh --output DIR [options]

Verifies, without training, the CUDA-12 Tesla V100 32GB build used by the
profile-bound non-neural resonant image trainer. Real evidence requires at
least 30 GiB free VRAM, 512 GiB host RAM, 1 TiB local disk, the full test
suite, and a CUDA backend that reports available. Test doubles always emit
ready=false.
EOF
}

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUTPUT=""; BUILD_DIR="${ROOT}/build/ubuntu-general-cuda-compat"; GPU_INDEX=0
MIN_HOST_RAM_GIB=512; MIN_DISK_GIB=1024
while (($# > 0)); do
  case "$1" in
    --output) OUTPUT="${2:?--output requires a directory}"; shift 2 ;;
    --build-dir) BUILD_DIR="${2:?--build-dir requires a directory}"; shift 2 ;;
    --gpu-index) GPU_INDEX="${2:?--gpu-index requires an integer}"; shift 2 ;;
    --min-host-ram-gib) MIN_HOST_RAM_GIB="${2:?requires an integer}"; shift 2 ;;
    --min-disk-gib) MIN_DISK_GIB="${2:?requires an integer}"; shift 2 ;;
    --help|-h) usage; exit 0 ;;
    *) echo "unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done
[[ -n "${OUTPUT}" ]] || { echo "--output is required" >&2; exit 2; }
[[ "${GPU_INDEX}" =~ ^[0-9]+$ && "${MIN_HOST_RAM_GIB}" =~ ^[0-9]+$ && \
   "${MIN_DISK_GIB}" =~ ^[0-9]+$ ]] || { echo "invalid numeric option" >&2; exit 2; }
if [[ "${RLF_ALLOW_TEST_DOUBLES:-0}" != 1 ]] && \
   ((MIN_HOST_RAM_GIB < 512 || MIN_DISK_GIB < 1024)); then
  echo "real imagegen V100 readiness may not lower the 512 GiB RAM / 1024 GiB disk floor" >&2
  exit 2
fi
if [[ -n "${RLF_NVIDIA_SMI:-}" && "${RLF_ALLOW_TEST_DOUBLES:-0}" != 1 ]]; then
  echo "RLF_NVIDIA_SMI is forbidden for real readiness" >&2
  exit 2
fi

NVCC_LINE="$(nvcc --version 2>/dev/null | grep -E 'release [0-9]+\.' | tail -n 1 || true)"
CUDA_MAJOR="$(sed -nE 's/.*release ([0-9]+)\..*/\1/p' <<<"${NVCC_LINE}")"
[[ "${CUDA_MAJOR}" == 12 ]] || {
  echo "Tesla V100 image training requires parseable CUDA Toolkit major 12; found ${CUDA_MAJOR:-unknown}" >&2
  exit 2
}

mkdir -p "${OUTPUT}"
OUTPUT="$(realpath "${OUTPUT}")"
CHECKS="${OUTPUT}/readiness_checks.tsv"
printf 'check\tstatus\tdetail\n' >"${CHECKS}"
FAILURES=0
record() {
  local name="$1" status="$2" detail="$3"
  detail="${detail//$'\t'/ }"; detail="${detail//$'\n'/ }"
  printf '%s\t%s\t%s\n' "${name}" "${status}" "${detail}" >>"${CHECKS}"
  [[ "${status}" == pass ]] || FAILURES=$((FAILURES + 1))
}
trim() {
  local value="$1"
  value="${value#"${value%%[![:space:]]*}"}"
  value="${value%"${value##*[![:space:]]}"}"
  printf '%s' "${value}"
}

for command_name in cmake ctest nvcc sha256sum awk grep df realpath; do
  command -v "${command_name}" >/dev/null 2>&1 && \
    record "command_${command_name}" pass "$(command -v "${command_name}")" || \
    record "command_${command_name}" fail missing
done
NVIDIA_SMI="${RLF_NVIDIA_SMI:-nvidia-smi}"
command -v "${NVIDIA_SMI}" >/dev/null 2>&1 && \
  record command_nvidia_smi pass "$(command -v "${NVIDIA_SMI}")" || \
  record command_nvidia_smi fail missing
[[ -r /etc/os-release ]] && grep -Eq '^ID=ubuntu$' /etc/os-release && \
  record operating_system pass ubuntu || record operating_system fail 'Ubuntu is required'

BIN="${BUILD_DIR}/solstice"; CACHE="${BUILD_DIR}/CMakeCache.txt"
[[ -x "${BIN}" ]] && record solstice_binary pass "${BIN}" || \
  record solstice_binary fail missing
[[ -f "${CACHE}" ]] && record cmake_cache pass "${CACHE}" || \
  record cmake_cache fail missing
if [[ -f "${CACHE}" ]]; then
  grep -Eq '^RLF_ENABLE_CUDA:BOOL=ON$' "${CACHE}" && \
    record cuda_build pass enabled || record cuda_build fail disabled
  grep -Eq '^CMAKE_CUDA_ARCHITECTURES:[^=]+=(70;80|80;70)$' "${CACHE}" && \
    record cuda_architectures pass sm_70_sm_80 || \
    record cuda_architectures fail 'expected 70;80'
  grep -Eq '^RLF_WARNINGS_AS_ERRORS:BOOL=ON$' "${CACHE}" && \
    record warnings_as_errors pass enabled || record warnings_as_errors fail disabled
  grep -Eq '^CMAKE_BUILD_TYPE:STRING=Release$' "${CACHE}" && \
    record release_build pass Release || record release_build fail 'not Release'
fi

GPU_ROW=""; GPU_UUID=""; GPU_NAME=""; TOTAL_MIB=0; FREE_MIB=0; COMPUTE_CAP=""
if command -v "${NVIDIA_SMI}" >/dev/null 2>&1; then
  GPU_ROW="$(${NVIDIA_SMI} --id="${GPU_INDEX}" \
    --query-gpu=index,name,uuid,memory.total,memory.free,compute_cap,driver_version \
    --format=csv,noheader,nounits 2>"${OUTPUT}/nvidia_smi_error.txt")"
  GPU_STATUS=$?
  if ((GPU_STATUS == 0)) && [[ -n "${GPU_ROW}" && "${GPU_ROW}" != *$'\n'* ]]; then
    IFS=',' read -r FOUND_INDEX GPU_NAME GPU_UUID TOTAL_MIB FREE_MIB COMPUTE_CAP DRIVER_VERSION <<<"${GPU_ROW}"
    FOUND_INDEX="$(trim "${FOUND_INDEX}")"; GPU_NAME="$(trim "${GPU_NAME}")"
    GPU_UUID="$(trim "${GPU_UUID}")"; TOTAL_MIB="$(trim "${TOTAL_MIB}")"
    FREE_MIB="$(trim "${FREE_MIB}")"; COMPUTE_CAP="$(trim "${COMPUTE_CAP}")"
    [[ "${FOUND_INDEX}" == "${GPU_INDEX}" ]] && record gpu_index pass "${FOUND_INDEX}" || record gpu_index fail "${FOUND_INDEX}"
    [[ "${GPU_NAME}" == *V100* ]] && record gpu_model pass "${GPU_NAME}" || record gpu_model fail "${GPU_NAME}"
    [[ "${COMPUTE_CAP}" == 7.0 ]] && record compute_capability pass 7.0 || record compute_capability fail "${COMPUTE_CAP}"
    [[ "${TOTAL_MIB}" =~ ^[0-9]+$ ]] && ((TOTAL_MIB >= 30720)) && record total_vram pass "${TOTAL_MIB} MiB" || record total_vram fail "${TOTAL_MIB} MiB; need 30720"
    [[ "${FREE_MIB}" =~ ^[0-9]+$ ]] && ((FREE_MIB >= 30720)) && record free_vram pass "${FREE_MIB} MiB" || record free_vram fail "${FREE_MIB} MiB; need 30720"
  else
    record gpu_query fail "nvidia-smi exit ${GPU_STATUS} or ambiguous device"
  fi
fi

AVAILABLE_KIB="$(df -Pk "${OUTPUT}" 2>/dev/null | awk 'NR == 2 {print $4}')"
REQUIRED_KIB=$((MIN_DISK_GIB * 1024 * 1024))
[[ "${AVAILABLE_KIB}" =~ ^[0-9]+$ ]] && ((AVAILABLE_KIB >= REQUIRED_KIB)) && \
  record disk_space pass "${AVAILABLE_KIB} KiB" || record disk_space fail "${AVAILABLE_KIB:-unknown} KiB; need ${REQUIRED_KIB}"
HOST_RAM_KIB="$(awk '/^MemTotal:/ {print $2}' /proc/meminfo 2>/dev/null)"
REQUIRED_HOST_RAM_KIB=$((MIN_HOST_RAM_GIB * 1024 * 1024))
[[ "${HOST_RAM_KIB}" =~ ^[0-9]+$ ]] && ((HOST_RAM_KIB >= REQUIRED_HOST_RAM_KIB)) && \
  record host_ram pass "${HOST_RAM_KIB} KiB" || record host_ram fail "${HOST_RAM_KIB:-unknown} KiB; need ${REQUIRED_HOST_RAM_KIB}"

if [[ -x "${BIN}" ]]; then
  "${BIN}" imagegen-profile-info --profile imagegen-v100-32g \
    >"${OUTPUT}/profile_info.txt" 2>&1
  PROFILE_STATUS=$?
  ((PROFILE_STATUS == 0)) && \
    grep -Eq '^gpu_working_set_gib=30$' "${OUTPUT}/profile_info.txt" && \
    grep -Eq '^maximum_modes=48000000$' "${OUTPUT}/profile_info.txt" && \
    record imagegen_profile pass imagegen-v100-32g || \
    record imagegen_profile fail "exit ${PROFILE_STATUS} or profile mismatch"
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="${GPU_INDEX}" \
    "${BIN}" device-info --profile general-v100-32g --backend cuda \
    >"${OUTPUT}/device_info.txt" 2>&1
  DEVICE_STATUS=$?
  ((DEVICE_STATUS == 0)) && grep -Eq '^backend=cuda-persistent$' "${OUTPUT}/device_info.txt" && \
    grep -Eq '^available=true$' "${OUTPUT}/device_info.txt" && \
    record cuda_backend pass available || record cuda_backend fail "exit ${DEVICE_STATUS}"
fi
if command -v ctest >/dev/null 2>&1 && [[ -d "${BUILD_DIR}" ]]; then
  ctest --test-dir "${BUILD_DIR}" --output-on-failure >"${OUTPUT}/ctest.txt" 2>&1
  TEST_STATUS=$?
  ((TEST_STATUS == 0)) && record full_test_suite pass passed || \
    record full_test_suite fail "exit ${TEST_STATUS}"
fi

SOURCE_SHA=""; BINARY_SHA=""
"${ROOT}/scripts/create_source_manifest.sh" "${OUTPUT}/source_manifest.tsv" \
  >"${OUTPUT}/source_manifest_stdout.txt" 2>&1
SOURCE_STATUS=$?
if ((SOURCE_STATUS == 0)); then
  SOURCE_SHA="$(sha256sum -- "${OUTPUT}/source_manifest.tsv" | awk '{print $1}')"
  record source_manifest pass "${SOURCE_SHA}"
else
  record source_manifest fail "exit ${SOURCE_STATUS}"
fi
if [[ -f "${BIN}" ]]; then BINARY_SHA="$(sha256sum -- "${BIN}" | awk '{print $1}')"; fi

SYNTHETIC_CHECKS_PASSED=false
((FAILURES == 0)) && SYNTHETIC_CHECKS_PASSED=true
READY=false
if [[ "${SYNTHETIC_CHECKS_PASSED}" == true && "${RLF_ALLOW_TEST_DOUBLES:-0}" != 1 ]]; then
  READY=true
fi
cat >"${OUTPUT}/readiness.json" <<EOF
{
  "schema": "rlf-imagegen-v100-readiness-v1",
  "ready": ${READY},
  "synthetic_checks_passed": ${SYNTHETIC_CHECKS_PASSED},
  "test_doubles": $([[ "${RLF_ALLOW_TEST_DOUBLES:-0}" == 1 ]] && echo true || echo false),
  "profile": "imagegen-v100-32g",
  "gpu_index": ${GPU_INDEX},
  "gpu_uuid": "${GPU_UUID}",
  "cuda_major": 12,
  "peak_vram_limit_mib": 30720,
  "source_manifest_sha256": "${SOURCE_SHA}",
  "solstice_binary_sha256": "${BINARY_SHA}",
  "training_performed": false,
  "frontier_claim_authorized": false
}
EOF
[[ "${READY}" == true || "${RLF_ALLOW_TEST_DOUBLES:-0}" == 1 ]] || exit 1
((FAILURES == 0)) || exit 1
