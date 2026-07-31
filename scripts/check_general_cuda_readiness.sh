#!/usr/bin/env bash
set -uo pipefail

usage() {
  cat <<'EOF'
Usage: check_general_cuda_readiness.sh --profile PROFILE --ledger FILE --output DIR [options]

PROFILE may be general-40g, general-v100-32g, general-v100-32g-text,
video-v100-32g,
general-v100-32g-500m (the isolated 1,586-hour campaign),
rtx-pro-6000-96g,
general-rtx-pro-6000-96g, general-rtx-pro-6000-96g-text, or
video-rtx-pro-6000-96g. The first requires 38 GiB free VRAM, V100 profiles
require 30 GiB, the exact target profile requires 88 GiB, and the legacy
RTX PRO 6000 Blackwell profiles require 90 GiB.
The staged V100 campaign requires at least 1 TiB host RAM, 2 TiB local disk,
and CUDA Toolkit 12.x for its initial 50M calibration. Larger-stage resource
projections and 20% headroom are enforced by the separate promotion gate;
readiness alone never authorizes 200M or 500M.
The command verifies the matching CUDA architecture, full tests, data, and
storage without training. Test doubles can only produce ready=false evidence.
EOF
}

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR=""
PROFILE=""; LEDGER=""; OUTPUT=""; CHECKPOINT=""; GPU_INDEX=0
MIN_HOST_RAM_GIB=""; MIN_DISK_GIB=""
MAX_AUDIT_RECORDS=10000000
MAX_TEXT_SHARD_BYTES=2147483648
MAX_TRAIN_SHARD_BYTES=4294967296
while (($# > 0)); do
  case "$1" in
    --profile) PROFILE="${2:?--profile requires a name}"; shift 2 ;;
    --ledger) LEDGER="${2:?--ledger requires a file}"; shift 2 ;;
    --output) OUTPUT="${2:?--output requires a directory}"; shift 2 ;;
    --checkpoint) CHECKPOINT="${2:?--checkpoint requires a path}"; shift 2 ;;
    --build-dir) BUILD_DIR="${2:?--build-dir requires a directory}"; shift 2 ;;
    --gpu-index) GPU_INDEX="${2:?--gpu-index requires an integer}"; shift 2 ;;
    --min-host-ram-gib) MIN_HOST_RAM_GIB="${2:?requires an integer}"; shift 2 ;;
    --min-disk-gib) MIN_DISK_GIB="${2:?requires an integer}"; shift 2 ;;
    --max-audit-records) MAX_AUDIT_RECORDS="${2:?requires an integer}"; shift 2 ;;
    --max-text-shard-bytes) MAX_TEXT_SHARD_BYTES="${2:?requires an integer}"; shift 2 ;;
    --max-train-shard-bytes) MAX_TRAIN_SHARD_BYTES="${2:?requires an integer}"; shift 2 ;;
    --help|-h) usage; exit 0 ;;
    *) echo "unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done
case "${PROFILE}" in
  general-40g) MIN_FREE_MIB=38912; EXPECTED_ARCHITECTURES='70;80'; REQUIRED_HOST_RAM_GIB=1024; REQUIRED_DISK_GIB=2048 ;;
  general-v100-32g|general-v100-32g-text|video-v100-32g) MIN_FREE_MIB=30720; EXPECTED_ARCHITECTURES='70;80'; TRAINING_WALL_BUDGET_SECONDS=900000; REQUIRED_HOST_RAM_GIB=1024; REQUIRED_DISK_GIB=2048 ;;
  general-v100-32g-500m) MIN_FREE_MIB=30720; EXPECTED_ARCHITECTURES='70;80'; TRAINING_WALL_BUDGET_SECONDS=5709600; REQUIRED_HOST_RAM_GIB=1024; REQUIRED_DISK_GIB=2048 ;;
  rtx-pro-6000-96g) MIN_FREE_MIB=90112; EXPECTED_ARCHITECTURES=120; TRAINING_WALL_BUDGET_SECONDS=0; REQUIRED_HOST_RAM_GIB=1024; REQUIRED_DISK_GIB=2048 ;;
  general-rtx-pro-6000-96g|general-rtx-pro-6000-96g-text|video-rtx-pro-6000-96g) MIN_FREE_MIB=92160; EXPECTED_ARCHITECTURES=120; TRAINING_WALL_BUDGET_SECONDS=0; REQUIRED_HOST_RAM_GIB=1024; REQUIRED_DISK_GIB=2048 ;;
  *) echo "unsupported general CUDA profile" >&2; exit 2 ;;
esac
[[ -n "${MIN_HOST_RAM_GIB}" ]] || MIN_HOST_RAM_GIB="${REQUIRED_HOST_RAM_GIB}"
[[ -n "${MIN_DISK_GIB}" ]] || MIN_DISK_GIB="${REQUIRED_DISK_GIB}"
[[ -n "${TRAINING_WALL_BUDGET_SECONDS:-}" ]] || TRAINING_WALL_BUDGET_SECONDS=0
if [[ -z "${BUILD_DIR}" ]]; then
  if [[ "${PROFILE}" == *rtx-pro-6000-96g* ]]; then BUILD_DIR="${ROOT}/build/ubuntu-rtx-pro-6000-cuda"; else BUILD_DIR="${ROOT}/build/ubuntu-general-cuda-compat"; fi
fi
[[ -n "${LEDGER}" && -f "${LEDGER}" ]] || { echo "--ledger must name a regular file" >&2; exit 2; }
[[ -n "${OUTPUT}" ]] || { echo "--output is required" >&2; exit 2; }
if [[ "${PROFILE}" == general-v100-32g-500m ]]; then
  [[ -n "${CHECKPOINT}" ]] || { echo "general-v100-32g-500m requires --checkpoint for its campaign binding" >&2; exit 2; }
elif [[ -n "${CHECKPOINT}" ]]; then
  echo "--checkpoint is reserved for general-v100-32g-500m readiness" >&2
  exit 2
fi
[[ "${GPU_INDEX}" =~ ^[0-9]+$ && "${MIN_HOST_RAM_GIB}" =~ ^[0-9]+$ && \
   "${MIN_DISK_GIB}" =~ ^[0-9]+$ && "${MAX_AUDIT_RECORDS}" =~ ^[1-9][0-9]*$ && \
   "${MAX_TEXT_SHARD_BYTES}" =~ ^[1-9][0-9]*$ && \
   "${MAX_TRAIN_SHARD_BYTES}" =~ ^[1-9][0-9]*$ ]] || { echo "invalid numeric option" >&2; exit 2; }
CUDA_TOOLKIT_MAJOR=""
if [[ "${PROFILE}" == general-v100-32g || "${PROFILE}" == general-v100-32g-text || "${PROFILE}" == general-v100-32g-500m || "${PROFILE}" == video-v100-32g ]]; then
  NVCC_VERSION_OUTPUT="$(nvcc --version 2>&1)" || { echo "V100 profiles require a working CUDA Toolkit 12.x nvcc" >&2; exit 2; }
  if [[ "${NVCC_VERSION_OUTPUT}" =~ release[[:space:]]+([0-9]+)(\.[0-9]+)? ]]; then
    CUDA_TOOLKIT_MAJOR="${BASH_REMATCH[1]}"
  fi
  [[ "${CUDA_TOOLKIT_MAJOR}" == 12 ]] || { echo "V100 profiles require parseable CUDA Toolkit major 12 (found ${CUDA_TOOLKIT_MAJOR:-unknown})" >&2; exit 2; }
fi
if [[ -n "${RLF_NVIDIA_SMI:-}" && "${RLF_ALLOW_TEST_DOUBLES:-0}" != 1 ]]; then
  echo "RLF_NVIDIA_SMI overrides are forbidden for real readiness evidence" >&2; exit 2
fi
if [[ "${RLF_ALLOW_TEST_DOUBLES:-0}" != 1 ]] && \
   ((MIN_HOST_RAM_GIB < REQUIRED_HOST_RAM_GIB || MIN_DISK_GIB < REQUIRED_DISK_GIB)); then
  echo "real ${PROFILE} readiness may not lower the ${REQUIRED_HOST_RAM_GIB} GiB RAM / ${REQUIRED_DISK_GIB} GiB disk floor" >&2
  exit 2
fi

mkdir -p "${OUTPUT}"
OUTPUT="$(realpath "${OUTPUT}")"; LEDGER="$(realpath "${LEDGER}")"
LEDGER_SHA256="$(sha256sum -- "${LEDGER}" | awk '{print $1}')"
TRAINING_WALL_BINDING_SHA256=""
if [[ "${PROFILE}" == general-v100-32g-500m ]]; then
  CHECKPOINT="$(realpath -m "${CHECKPOINT}")"
  # The cumulative campaign binding must survive the deliberate 50M -> 200M
  # ledger expansion. Each readiness report still binds its exact ledger hash
  # independently, while the wall-budget state binds profile/checkpoint/budget.
  TRAINING_WALL_BINDING_SHA256="$(printf '%s\n%s\n%s\n' "${PROFILE}" "${CHECKPOINT}" "${TRAINING_WALL_BUDGET_SECONDS}" | sha256sum | awk '{print $1}')"
fi
CHECKS="${OUTPUT}/readiness_checks.tsv"; ENVIRONMENT="${OUTPUT}/environment.txt"
printf 'check\tstatus\tdetail\n' >"${CHECKS}"; FAILURES=0
record() { local name="$1" status="$2" detail="$3"; detail="${detail//$'\t'/ }"; detail="${detail//$'\n'/ }"; printf '%s\t%s\t%s\n' "${name}" "${status}" "${detail}" >>"${CHECKS}"; [[ "${status}" == pass ]] || FAILURES=$((FAILURES + 1)); }
trim() { local value="$1"; value="${value#"${value%%[![:space:]]*}"}"; value="${value%"${value##*[![:space:]]}"}"; printf '%s' "${value}"; }

for command_name in bash cmake ctest ninja g++ nvcc sha256sum awk grep sort find stat df realpath timeout flock; do
  command -v "${command_name}" >/dev/null 2>&1 && record "command_${command_name}" pass "$(command -v "${command_name}")" || record "command_${command_name}" fail missing
done
[[ -z "${CUDA_TOOLKIT_MAJOR}" ]] || record cuda_toolkit_major pass "${CUDA_TOOLKIT_MAJOR}"
NVIDIA_SMI="${RLF_NVIDIA_SMI:-nvidia-smi}"
command -v "${NVIDIA_SMI}" >/dev/null 2>&1 && record command_nvidia_smi pass "$(command -v "${NVIDIA_SMI}")" || record command_nvidia_smi fail missing
[[ -r /etc/os-release ]] && grep -Eq '^ID=ubuntu$' /etc/os-release && record operating_system pass ubuntu || record operating_system fail 'Ubuntu is required'
for required in README.md CMakeLists.txt CMakePresets.json; do
  [[ -f "${ROOT}/${required}" ]] && record "file_${required}" pass present || record "file_${required}" fail missing
done

BIN="${BUILD_DIR}/solstice"; CACHE="${BUILD_DIR}/CMakeCache.txt"
SOLSTICE_BINARY_SHA256=""
[[ -x "${BIN}" ]] && record solstice_binary pass "${BIN}" || record solstice_binary fail missing
if [[ -f "${BIN}" ]]; then
  SOLSTICE_BINARY_SHA256="$(sha256sum -- "${BIN}" | awk '{print $1}')"
fi
[[ -f "${CACHE}" ]] && record cmake_cache pass "${CACHE}" || record cmake_cache fail missing
if [[ -f "${CACHE}" ]]; then
  grep -Eq '^RLF_ENABLE_CUDA:BOOL=ON$' "${CACHE}" && record cuda_build pass enabled || record cuda_build fail disabled
  if [[ "${EXPECTED_ARCHITECTURES}" == 120 ]]; then
    grep -Eq '^CMAKE_CUDA_ARCHITECTURES:[^=]+=120$' "${CACHE}" && record cuda_architectures pass sm_120 || record cuda_architectures fail 'expected 120'
  else
    grep -Eq '^CMAKE_CUDA_ARCHITECTURES:[^=]+=(70;80|80;70)$' "${CACHE}" && record cuda_architectures pass sm_70_sm_80 || record cuda_architectures fail 'expected 70;80'
  fi
  grep -Eq '^RLF_WARNINGS_AS_ERRORS:BOOL=ON$' "${CACHE}" && record warnings_as_errors pass enabled || record warnings_as_errors fail disabled
  grep -Eq '^CMAKE_BUILD_TYPE:STRING=Release$' "${CACHE}" && record release_build pass Release || record release_build fail 'not Release'
fi

GPU_ROW=""; GPU_NAME=""; GPU_UUID=""; TOTAL_MIB=0; FREE_MIB=0; COMPUTE_CAP=""; DRIVER_VERSION=""
if command -v "${NVIDIA_SMI}" >/dev/null 2>&1; then
  GPU_ROW="$(${NVIDIA_SMI} --id="${GPU_INDEX}" --query-gpu=index,name,uuid,memory.total,memory.free,compute_cap,driver_version --format=csv,noheader,nounits 2>"${OUTPUT}/nvidia_smi_error.txt")"
  GPU_STATUS=$?
  if ((GPU_STATUS == 0)) && [[ -n "${GPU_ROW}" && "${GPU_ROW}" != *$'\n'* ]]; then
    IFS=',' read -r FOUND_INDEX GPU_NAME GPU_UUID TOTAL_MIB FREE_MIB COMPUTE_CAP DRIVER_VERSION <<<"${GPU_ROW}"
    FOUND_INDEX="$(trim "${FOUND_INDEX}")"; GPU_NAME="$(trim "${GPU_NAME}")"; GPU_UUID="$(trim "${GPU_UUID}")"
    TOTAL_MIB="$(trim "${TOTAL_MIB}")"; FREE_MIB="$(trim "${FREE_MIB}")"; COMPUTE_CAP="$(trim "${COMPUTE_CAP}")"; DRIVER_VERSION="$(trim "${DRIVER_VERSION}")"
    [[ "${FOUND_INDEX}" == "${GPU_INDEX}" ]] && record gpu_index pass "${FOUND_INDEX}" || record gpu_index fail "${FOUND_INDEX}"
    if [[ "${PROFILE}" == general-v100-32g || "${PROFILE}" == general-v100-32g-text || "${PROFILE}" == general-v100-32g-500m || "${PROFILE}" == video-v100-32g ]]; then
      [[ "${GPU_NAME}" == *V100* ]] && record gpu_model pass "${GPU_NAME}" || record gpu_model fail "${GPU_NAME}"
      [[ "${COMPUTE_CAP}" == 7.0 ]] && record compute_capability pass 7.0 || record compute_capability fail "${COMPUTE_CAP}"
    elif [[ "${PROFILE}" == general-40g ]]; then
      [[ "${GPU_NAME}" == *A100* || "${GPU_NAME}" == *V100* ]] && record gpu_model pass "${GPU_NAME}" || record gpu_model fail "${GPU_NAME}"
      [[ "${COMPUTE_CAP}" == 7.0 || "${COMPUTE_CAP}" == 8.0 ]] && record compute_capability pass "${COMPUTE_CAP}" || record compute_capability fail "${COMPUTE_CAP}"
    else
      [[ "${GPU_NAME}" == *"RTX PRO 6000 Blackwell"* ]] && record gpu_model pass "${GPU_NAME}" || record gpu_model fail "${GPU_NAME}"
      [[ "${COMPUTE_CAP}" == 12.0 ]] && record compute_capability pass 12.0 || record compute_capability fail "${COMPUTE_CAP}"
    fi
    [[ "${TOTAL_MIB}" =~ ^[0-9]+$ ]] && ((TOTAL_MIB >= MIN_FREE_MIB)) && record total_vram pass "${TOTAL_MIB} MiB" || record total_vram fail "${TOTAL_MIB} MiB; need ${MIN_FREE_MIB}"
    [[ "${FREE_MIB}" =~ ^[0-9]+$ ]] && ((FREE_MIB >= MIN_FREE_MIB)) && record free_vram pass "${FREE_MIB} MiB" || record free_vram fail "${FREE_MIB} MiB; need ${MIN_FREE_MIB}"
  else
    record gpu_query fail "nvidia-smi exit ${GPU_STATUS} or ambiguous device"
  fi
fi

AVAILABLE_KIB="$(df -Pk "${OUTPUT}" 2>/dev/null | awk 'NR == 2 {print $4}')"; REQUIRED_KIB=$((MIN_DISK_GIB * 1024 * 1024))
[[ "${AVAILABLE_KIB}" =~ ^[0-9]+$ ]] && ((AVAILABLE_KIB >= REQUIRED_KIB)) && record disk_space pass "${AVAILABLE_KIB} KiB" || record disk_space fail "${AVAILABLE_KIB:-unknown} KiB; need ${REQUIRED_KIB}"
HOST_RAM_KIB="$(awk '/^MemTotal:/ {print $2}' /proc/meminfo 2>/dev/null)"; REQUIRED_HOST_RAM_KIB=$((MIN_HOST_RAM_GIB * 1024 * 1024))
[[ "${HOST_RAM_KIB}" =~ ^[0-9]+$ ]] && ((HOST_RAM_KIB >= REQUIRED_HOST_RAM_KIB)) && record host_ram pass "${HOST_RAM_KIB} KiB" || record host_ram fail "${HOST_RAM_KIB:-unknown} KiB; need ${REQUIRED_HOST_RAM_KIB}"
ATOMIC_SOURCE="$(mktemp "${OUTPUT}/.atomic-source.XXXXXX")"; ATOMIC_TARGET="${ATOMIC_SOURCE}.renamed"; printf 'probe\n' >"${ATOMIC_SOURCE}"
if mv -- "${ATOMIC_SOURCE}" "${ATOMIC_TARGET}" && [[ -f "${ATOMIC_TARGET}" ]]; then record atomic_rename pass "${OUTPUT}"; rm -f -- "${ATOMIC_TARGET}"; else record atomic_rename fail "${OUTPUT}"; fi

{
  printf 'schema=rlf-general-cuda-readiness-environment-v1\nprofile=%s\ntest_doubles=%s\n' "${PROFILE}" "${RLF_ALLOW_TEST_DOUBLES:-0}"
  printf 'timestamp_utc=%s\nrepository_root=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${ROOT}"
  uname -a 2>&1 || true; cmake --version 2>&1 | head -n 1 || true; g++ --version 2>&1 | head -n 1 || true; nvcc --version 2>&1 | tail -n 1 || true; printf 'gpu=%s\n' "${GPU_ROW}"
} >"${ENVIRONMENT}"
if [[ -x "${BIN}" ]]; then
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="${GPU_INDEX}" "${BIN}" profile-info --profile "${PROFILE}" >"${OUTPUT}/profile_info.txt" 2>&1; PROFILE_STATUS=$?
  PROFILE_POLICY_OK=false
  if grep -Eq "^profile=${PROFILE}$" "${OUTPUT}/profile_info.txt"; then
    if [[ "${PROFILE}" == general-v100-32g-text || "${PROFILE}" == general-rtx-pro-6000-96g-text ]]; then
      grep -Eq '^vision_training_enabled=false$' "${OUTPUT}/profile_info.txt" && grep -Eq '^video_training_enabled=false$' "${OUTPUT}/profile_info.txt" && PROFILE_POLICY_OK=true
    elif [[ "${PROFILE}" == video-rtx-pro-6000-96g || "${PROFILE}" == video-v100-32g ]]; then
      grep -Eq '^vision_training_enabled=true$' "${OUTPUT}/profile_info.txt" && grep -Eq '^video_training_enabled=true$' "${OUTPUT}/profile_info.txt" && PROFILE_POLICY_OK=true
    else
      grep -Eq '^vision_training_enabled=true$' "${OUTPUT}/profile_info.txt" && grep -Eq '^video_training_enabled=false$' "${OUTPUT}/profile_info.txt" && PROFILE_POLICY_OK=true
    fi
  fi
  ((PROFILE_STATUS == 0)) && [[ "${PROFILE_POLICY_OK}" == true ]] && record profile_info pass "${PROFILE}" || record profile_info fail "exit ${PROFILE_STATUS} or policy mismatch"
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="${GPU_INDEX}" "${BIN}" device-info --profile "${PROFILE}" --backend cuda >"${OUTPUT}/device_info.txt" 2>&1; DEVICE_STATUS=$?
  if ((DEVICE_STATUS == 0)) && grep -Eq '^backend=cuda-persistent$' "${OUTPUT}/device_info.txt" && grep -Eq '^available=true$' "${OUTPUT}/device_info.txt" && grep -Eq '^profile_fits_device=true$' "${OUTPUT}/device_info.txt"; then record cuda_backend pass available; else record cuda_backend fail "exit ${DEVICE_STATUS}"; fi
  "${BIN}" audit-data --ledger "${LEDGER}" --output "${OUTPUT}/data_audit.json" --require-media-hashes --max-audit-records "${MAX_AUDIT_RECORDS}" --max-text-shard-bytes "${MAX_TEXT_SHARD_BYTES}" --max-train-shard-bytes "${MAX_TRAIN_SHARD_BYTES}" >"${OUTPUT}/data_audit_stdout.txt" 2>&1; AUDIT_STATUS=$?
  ((AUDIT_STATUS == 0)) && record data_audit pass "${LEDGER}" || record data_audit fail "exit ${AUDIT_STATUS}"
fi
if command -v ctest >/dev/null 2>&1 && [[ -d "${BUILD_DIR}" ]]; then
  ctest --test-dir "${BUILD_DIR}" --output-on-failure >"${OUTPUT}/ctest.txt" 2>&1; TEST_STATUS=$?
  ((TEST_STATUS == 0)) && record full_test_suite pass passed || record full_test_suite fail "exit ${TEST_STATUS}"
fi
"${ROOT}/scripts/create_source_manifest.sh" "${OUTPUT}/source_manifest.tsv" >"${OUTPUT}/source_manifest_stdout.txt" 2>&1; SOURCE_STATUS=$?
SOURCE_MANIFEST_SHA256=""
if ((SOURCE_STATUS == 0)) && [[ -f "${OUTPUT}/source_manifest.tsv" ]]; then
  SOURCE_MANIFEST_SHA256="$(sha256sum -- "${OUTPUT}/source_manifest.tsv" | awk '{print $1}')"
  record source_manifest pass "${SOURCE_MANIFEST_SHA256}"
else
  record source_manifest fail "exit ${SOURCE_STATUS}"
fi

CHECKS_PASS=false; ((FAILURES == 0)) && CHECKS_PASS=true
TEST_DOUBLES=false; [[ "${RLF_ALLOW_TEST_DOUBLES:-0}" == 1 ]] && TEST_DOUBLES=true
READY="${CHECKS_PASS}"; [[ "${TEST_DOUBLES}" == true ]] && READY=false
printf '{\n  "schema": "rlf-general-cuda-readiness-v1",\n  "ready": %s,\n  "synthetic_checks_passed": %s,\n  "training_performed": false,\n  "test_doubles": %s,\n  "profile": "%s",\n  "ledger_sha256": "%s",\n  "source_manifest_sha256": "%s",\n  "solstice_binary_sha256": "%s",\n  "gpu_index": %s,\n  "gpu_name": "%s",\n  "gpu_uuid": "%s",\n  "compute_capability": "%s",\n  "minimum_free_memory_mib": %s,\n  "minimum_host_ram_gib": %s,\n  "minimum_disk_gib": %s,\n  "maximum_audit_records": %s,\n  "maximum_text_shard_bytes": %s,\n  "maximum_train_shard_bytes": %s,\n  "training_wall_budget_seconds": %s,\n  "training_wall_binding_sha256": "%s",\n  "require_media_hashes": true,\n  "failed_checks": %s\n}\n' \
  "${READY}" "${CHECKS_PASS}" "${TEST_DOUBLES}" "${PROFILE}" "${LEDGER_SHA256}" "${SOURCE_MANIFEST_SHA256}" "${SOLSTICE_BINARY_SHA256}" "${GPU_INDEX}" "${GPU_NAME}" "${GPU_UUID}" "${COMPUTE_CAP}" "${MIN_FREE_MIB}" "${MIN_HOST_RAM_GIB}" "${MIN_DISK_GIB}" "${MAX_AUDIT_RECORDS}" "${MAX_TEXT_SHARD_BYTES}" "${MAX_TRAIN_SHARD_BYTES}" "${TRAINING_WALL_BUDGET_SECONDS}" "${TRAINING_WALL_BINDING_SHA256}" "${FAILURES}" >"${OUTPUT}/readiness.json"
if [[ "${CHECKS_PASS}" != true ]]; then echo "${PROFILE} readiness failed closed with ${FAILURES} failed checks. No training was performed." >&2; exit 1; fi
if [[ "${TEST_DOUBLES}" == true ]]; then echo "Synthetic ${PROFILE} readiness checks passed; this is not hardware evidence."; else echo "${PROFILE} readiness passed. No training was performed."; fi
