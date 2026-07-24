#!/usr/bin/env bash
set -uo pipefail

usage() {
  cat <<'EOF'
Usage: check_general_h100_readiness.sh --ledger FILE --output DIR [options]

Options:
  --build-dir DIR       H100 CUDA build (default: build/ubuntu-h100-cuda)
  --gpu-index N         Physical H100 selected for the later run (default: 0)
  --min-free-mib N      Required free VRAM (default: 77824 = 76 GiB)
  --min-host-ram-gib N  Required physical host RAM (default: 1024)
  --min-disk-gib N      Required output-filesystem space (default: 2048)
  --max-audit-records N Maximum records held by contamination audit (default: 10000000)
  --max-text-shard-bytes N  Maximum one text shard (default: 2147483648)
  --max-train-shard-bytes N Maximum shard plus media before checkpoint (default: 4294967296)

This command performs build, test, device, data-audit, and storage checks only.
It never bootstraps or trains a model. Test doubles are rejected unless
RLF_ALLOW_TEST_DOUBLES=1 is explicitly set by the shell regression test.
EOF
}

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${ROOT}/build/ubuntu-h100-cuda"
LEDGER=""
OUTPUT=""
GPU_INDEX=0
MIN_FREE_MIB=77824
MIN_HOST_RAM_GIB=1024
MIN_DISK_GIB=2048
MAX_AUDIT_RECORDS=10000000
MAX_TEXT_SHARD_BYTES=2147483648
MAX_TRAIN_SHARD_BYTES=4294967296

while (($# > 0)); do
  case "$1" in
    --ledger) LEDGER="${2:?--ledger requires a file}"; shift 2 ;;
    --output) OUTPUT="${2:?--output requires a directory}"; shift 2 ;;
    --build-dir) BUILD_DIR="${2:?--build-dir requires a directory}"; shift 2 ;;
    --gpu-index) GPU_INDEX="${2:?--gpu-index requires an integer}"; shift 2 ;;
    --min-free-mib) MIN_FREE_MIB="${2:?--min-free-mib requires an integer}"; shift 2 ;;
    --min-host-ram-gib) MIN_HOST_RAM_GIB="${2:?--min-host-ram-gib requires an integer}"; shift 2 ;;
    --min-disk-gib) MIN_DISK_GIB="${2:?--min-disk-gib requires an integer}"; shift 2 ;;
    --max-audit-records) MAX_AUDIT_RECORDS="${2:?--max-audit-records requires an integer}"; shift 2 ;;
    --max-text-shard-bytes) MAX_TEXT_SHARD_BYTES="${2:?--max-text-shard-bytes requires an integer}"; shift 2 ;;
    --max-train-shard-bytes) MAX_TRAIN_SHARD_BYTES="${2:?--max-train-shard-bytes requires an integer}"; shift 2 ;;
    --help|-h) usage; exit 0 ;;
    *) echo "unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

[[ -n "${LEDGER}" && -f "${LEDGER}" ]] || { echo "--ledger must name a regular file" >&2; exit 2; }
[[ -n "${OUTPUT}" ]] || { echo "--output is required" >&2; exit 2; }
[[ "${GPU_INDEX}" =~ ^[0-9]+$ && "${MIN_FREE_MIB}" =~ ^[0-9]+$ && \
   "${MIN_HOST_RAM_GIB}" =~ ^[0-9]+$ && "${MIN_DISK_GIB}" =~ ^[0-9]+$ && \
   "${MAX_AUDIT_RECORDS}" =~ ^[1-9][0-9]*$ && \
   "${MAX_TEXT_SHARD_BYTES}" =~ ^[1-9][0-9]*$ && \
   "${MAX_TRAIN_SHARD_BYTES}" =~ ^[1-9][0-9]*$ ]] || {
  echo "GPU, memory, and disk values must be non-negative integers" >&2; exit 2;
}

if [[ -n "${RLF_NVIDIA_SMI:-}" && "${RLF_ALLOW_TEST_DOUBLES:-0}" != "1" ]]; then
  echo "RLF_NVIDIA_SMI overrides are forbidden for real readiness evidence" >&2
  exit 2
fi

mkdir -p "${OUTPUT}"
OUTPUT="$(realpath "${OUTPUT}")"
LEDGER="$(realpath "${LEDGER}")"
LEDGER_SHA256="$(sha256sum -- "${LEDGER}" | awk '{print $1}')"
CHECKS="${OUTPUT}/readiness_checks.tsv"
ENVIRONMENT="${OUTPUT}/environment.txt"
printf 'check\tstatus\tdetail\n' >"${CHECKS}"
FAILURES=0

record() {
  local name="$1" status="$2" detail="$3"
  detail="${detail//$'\t'/ }"
  detail="${detail//$'\n'/ }"
  printf '%s\t%s\t%s\n' "${name}" "${status}" "${detail}" >>"${CHECKS}"
  [[ "${status}" == pass ]] || FAILURES=$((FAILURES + 1))
}

for command_name in bash cmake ctest ninja g++ nvcc sha256sum awk grep sort find stat df realpath; do
  if command -v "${command_name}" >/dev/null 2>&1; then
    record "command_${command_name}" pass "$(command -v "${command_name}")"
  else
    record "command_${command_name}" fail missing
  fi
done

NVIDIA_SMI="${RLF_NVIDIA_SMI:-nvidia-smi}"
if command -v "${NVIDIA_SMI}" >/dev/null 2>&1; then
  record command_nvidia_smi pass "$(command -v "${NVIDIA_SMI}")"
else
  record command_nvidia_smi fail missing
fi

if [[ -r /etc/os-release ]] && grep -Eq '^ID=ubuntu$' /etc/os-release; then
  record operating_system pass ubuntu
else
  record operating_system fail 'Ubuntu is required'
fi

for required in README.md CMakeLists.txt CMakePresets.json; do
  [[ -f "${ROOT}/${required}" ]] && record "file_${required}" pass present || record "file_${required}" fail missing
done

BIN="${BUILD_DIR}/solstice"
CACHE="${BUILD_DIR}/CMakeCache.txt"
[[ -x "${BIN}" ]] && record solstice_binary pass "${BIN}" || record solstice_binary fail missing
[[ -f "${CACHE}" ]] && record cmake_cache pass "${CACHE}" || record cmake_cache fail missing
if [[ -f "${CACHE}" ]]; then
  grep -Eq '^RLF_ENABLE_CUDA:BOOL=ON$' "${CACHE}" && record cuda_build pass enabled || record cuda_build fail disabled
  grep -Eq '^CMAKE_CUDA_ARCHITECTURES:[^=]+=90$' "${CACHE}" && record cuda_architecture pass sm_90 || record cuda_architecture fail 'expected sm_90'
  grep -Eq '^RLF_WARNINGS_AS_ERRORS:BOOL=ON$' "${CACHE}" && record warnings_as_errors pass enabled || record warnings_as_errors fail disabled
  grep -Eq '^CMAKE_BUILD_TYPE:STRING=Release$' "${CACHE}" && record release_build pass Release || record release_build fail 'not Release'
fi

GPU_ROW=""
if command -v "${NVIDIA_SMI}" >/dev/null 2>&1; then
  GPU_ROW="$(${NVIDIA_SMI} --id="${GPU_INDEX}" --query-gpu=index,name,uuid,memory.total,memory.free,compute_cap,driver_version --format=csv,noheader,nounits 2>"${OUTPUT}/nvidia_smi_error.txt")"
  GPU_STATUS=$?
  if ((GPU_STATUS == 0)) && [[ -n "${GPU_ROW}" && "${GPU_ROW}" != *$'\n'* ]]; then
    IFS=',' read -r FOUND_INDEX GPU_NAME GPU_UUID TOTAL_MIB FREE_MIB COMPUTE_CAP DRIVER_VERSION <<<"${GPU_ROW}"
    trim() { local value="$1"; value="${value#"${value%%[![:space:]]*}"}"; value="${value%"${value##*[![:space:]]}"}"; printf '%s' "${value}"; }
    FOUND_INDEX="$(trim "${FOUND_INDEX}")"; GPU_NAME="$(trim "${GPU_NAME}")"; GPU_UUID="$(trim "${GPU_UUID}")"
    TOTAL_MIB="$(trim "${TOTAL_MIB}")"; FREE_MIB="$(trim "${FREE_MIB}")"; COMPUTE_CAP="$(trim "${COMPUTE_CAP}")"; DRIVER_VERSION="$(trim "${DRIVER_VERSION}")"
    [[ "${FOUND_INDEX}" == "${GPU_INDEX}" ]] && record gpu_index pass "${FOUND_INDEX}" || record gpu_index fail "${FOUND_INDEX}"
    [[ "${GPU_NAME}" == *H100* ]] && record gpu_model pass "${GPU_NAME}" || record gpu_model fail "${GPU_NAME}"
    [[ "${COMPUTE_CAP}" == 9.0 ]] && record compute_capability pass "${COMPUTE_CAP}" || record compute_capability fail "${COMPUTE_CAP}"
    [[ "${TOTAL_MIB}" =~ ^[0-9]+$ ]] && ((TOTAL_MIB >= 77824)) && record total_vram pass "${TOTAL_MIB} MiB" || record total_vram fail "${TOTAL_MIB} MiB"
    [[ "${FREE_MIB}" =~ ^[0-9]+$ ]] && ((FREE_MIB >= MIN_FREE_MIB)) && record free_vram pass "${FREE_MIB} MiB" || record free_vram fail "${FREE_MIB} MiB; need ${MIN_FREE_MIB}"
  else
    record gpu_query fail "nvidia-smi exit ${GPU_STATUS} or ambiguous device"
  fi
fi

AVAILABLE_KIB="$(df -Pk "${OUTPUT}" 2>/dev/null | awk 'NR == 2 {print $4}')"
REQUIRED_KIB=$((MIN_DISK_GIB * 1024 * 1024))
[[ "${AVAILABLE_KIB}" =~ ^[0-9]+$ ]] && ((AVAILABLE_KIB >= REQUIRED_KIB)) && record disk_space pass "${AVAILABLE_KIB} KiB" || record disk_space fail "${AVAILABLE_KIB:-unknown} KiB; need ${REQUIRED_KIB}"

HOST_RAM_KIB="$(awk '/^MemTotal:/ {print $2}' /proc/meminfo 2>/dev/null)"
REQUIRED_HOST_RAM_KIB=$((MIN_HOST_RAM_GIB * 1024 * 1024))
[[ "${HOST_RAM_KIB}" =~ ^[0-9]+$ ]] && ((HOST_RAM_KIB >= REQUIRED_HOST_RAM_KIB)) && record host_ram pass "${HOST_RAM_KIB} KiB" || record host_ram fail "${HOST_RAM_KIB:-unknown} KiB; need ${REQUIRED_HOST_RAM_KIB}"

ATOMIC_SOURCE="$(mktemp "${OUTPUT}/.atomic-source.XXXXXX")"
ATOMIC_TARGET="${ATOMIC_SOURCE}.renamed"
printf 'atomic-write-probe\n' >"${ATOMIC_SOURCE}"
if mv -- "${ATOMIC_SOURCE}" "${ATOMIC_TARGET}" && [[ -f "${ATOMIC_TARGET}" ]]; then
  record atomic_rename pass "${OUTPUT}"
  rm -f -- "${ATOMIC_TARGET}"
else
  record atomic_rename fail "${OUTPUT}"
fi

{
  printf 'schema=rlf-h100-readiness-environment-v1\n'
  printf 'test_doubles=%s\n' "${RLF_ALLOW_TEST_DOUBLES:-0}"
  printf 'timestamp_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  printf 'repository_root=%s\n' "${ROOT}"
  uname -a 2>&1 || true
  cmake --version 2>&1 | head -n 1 || true
  g++ --version 2>&1 | head -n 1 || true
  nvcc --version 2>&1 | tail -n 1 || true
  printf 'gpu=%s\n' "${GPU_ROW}"
} >"${ENVIRONMENT}"

if [[ -x "${BIN}" ]]; then
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="${GPU_INDEX}" "${BIN}" profile-info --profile general-h100 >"${OUTPUT}/profile_info.txt" 2>&1
  PROFILE_STATUS=$?
  ((PROFILE_STATUS == 0)) && grep -Eq '^gpu_working_set_gib=' "${OUTPUT}/profile_info.txt" && record profile_info pass general-h100 || record profile_info fail "exit ${PROFILE_STATUS}"
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="${GPU_INDEX}" "${BIN}" device-info --profile general-h100 --backend cuda >"${OUTPUT}/device_info.txt" 2>&1
  DEVICE_STATUS=$?
  if ((DEVICE_STATUS == 0)) && grep -Eq '^backend=cuda-persistent$' "${OUTPUT}/device_info.txt" && grep -Eq '^available=true$' "${OUTPUT}/device_info.txt" && grep -Eq '^profile_fits_device=true$' "${OUTPUT}/device_info.txt"; then
    record cuda_backend pass available
  else
    record cuda_backend fail "exit ${DEVICE_STATUS}"
  fi
  "${BIN}" audit-data --ledger "${LEDGER}" --output "${OUTPUT}/data_audit.json" \
    --max-audit-records "${MAX_AUDIT_RECORDS}" \
    --max-text-shard-bytes "${MAX_TEXT_SHARD_BYTES}" \
    --max-train-shard-bytes "${MAX_TRAIN_SHARD_BYTES}" \
    >"${OUTPUT}/data_audit_stdout.txt" 2>&1
  AUDIT_STATUS=$?
  ((AUDIT_STATUS == 0)) && record data_audit pass "${LEDGER}" || record data_audit fail "exit ${AUDIT_STATUS}"
fi

if command -v ctest >/dev/null 2>&1 && [[ -d "${BUILD_DIR}" ]]; then
  ctest --test-dir "${BUILD_DIR}" --output-on-failure >"${OUTPUT}/ctest.txt" 2>&1
  TEST_STATUS=$?
  ((TEST_STATUS == 0)) && record full_test_suite pass passed || record full_test_suite fail "exit ${TEST_STATUS}"
fi

"${ROOT}/scripts/create_source_manifest.sh" "${OUTPUT}/source_manifest.tsv" >"${OUTPUT}/source_manifest_stdout.txt" 2>&1
SOURCE_STATUS=$?
((SOURCE_STATUS == 0)) && record source_manifest pass created || record source_manifest fail "exit ${SOURCE_STATUS}"

CHECKS_PASS=false
((FAILURES == 0)) && CHECKS_PASS=true
TEST_DOUBLES=false
[[ "${RLF_ALLOW_TEST_DOUBLES:-0}" == 1 ]] && TEST_DOUBLES=true
READY="${CHECKS_PASS}"
[[ "${TEST_DOUBLES}" == true ]] && READY=false
printf '{\n  "schema": "rlf-general-h100-readiness-v1",\n  "ready": %s,\n  "synthetic_checks_passed": %s,\n  "training_performed": false,\n  "test_doubles": %s,\n  "profile": "general-h100",\n  "ledger_sha256": "%s",\n  "gpu_index": %s,\n  "minimum_free_memory_mib": %s,\n  "minimum_host_ram_gib": %s,\n  "minimum_disk_gib": %s,\n  "maximum_audit_records": %s,\n  "maximum_text_shard_bytes": %s,\n  "maximum_train_shard_bytes": %s,\n  "failed_checks": %s\n}\n' \
  "${READY}" "${CHECKS_PASS}" "${TEST_DOUBLES}" "${LEDGER_SHA256}" "${GPU_INDEX}" "${MIN_FREE_MIB}" "${MIN_HOST_RAM_GIB}" "${MIN_DISK_GIB}" "${MAX_AUDIT_RECORDS}" "${MAX_TEXT_SHARD_BYTES}" "${MAX_TRAIN_SHARD_BYTES}" "${FAILURES}" >"${OUTPUT}/readiness.json"

if [[ "${CHECKS_PASS}" != true ]]; then
  echo "H100 readiness failed closed with ${FAILURES} failed checks. No training was performed." >&2
  exit 1
fi
if [[ "${TEST_DOUBLES}" == true ]]; then
  echo "Synthetic readiness test passed. This is not H100 evidence. No training was performed."
else
  echo "Single-H100 readiness passed. No training was performed."
fi
