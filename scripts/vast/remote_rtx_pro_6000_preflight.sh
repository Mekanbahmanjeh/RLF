#!/usr/bin/env bash
set -uo pipefail

usage() {
  cat <<'EOF'
Usage: remote_rtx_pro_6000_preflight.sh --output DIR [options]

Runs a non-training, fail-closed hardware and capacity preflight. Real evidence
requires exactly one visible NVIDIA RTX PRO 6000 Blackwell Workstation Edition,
compute capability 12.0, at least 97,000 MiB total and 90,112 MiB free VRAM,
MIG disabled/not supported with no MIG devices, Ubuntu, 1 TiB host RAM, and
2 TiB free disk at --work-root.

Options:
  --work-root DIR          Filesystem whose free capacity is checked (default: .)
  --min-host-ram-gib N     May only increase the real 1024 GiB floor
  --min-disk-gib N         May only increase the real 2048 GiB floor

RLF_ALLOW_TEST_DOUBLES=1 permits lower floors and RLF_NVIDIA_SMI for regression
tests, but always writes ready=false and can never create hardware evidence.
EOF
}

OUTPUT=""
WORK_ROOT="."
MIN_HOST_RAM_GIB=1024
MIN_DISK_GIB=2048
while (($# > 0)); do
  case "$1" in
    --output) OUTPUT="${2:?--output requires a directory}"; shift 2 ;;
    --work-root) WORK_ROOT="${2:?--work-root requires a directory}"; shift 2 ;;
    --min-host-ram-gib) MIN_HOST_RAM_GIB="${2:?requires an integer}"; shift 2 ;;
    --min-disk-gib) MIN_DISK_GIB="${2:?requires an integer}"; shift 2 ;;
    --help|-h) usage; exit 0 ;;
    *) echo "unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

[[ -n "${OUTPUT}" && "${MIN_HOST_RAM_GIB}" =~ ^[0-9]+$ && \
   "${MIN_DISK_GIB}" =~ ^[0-9]+$ ]] || { usage >&2; exit 2; }
[[ -d "${WORK_ROOT}" ]] || { echo "--work-root must be an existing directory" >&2; exit 2; }
TEST_DOUBLES=false
if [[ "${RLF_ALLOW_TEST_DOUBLES:-0}" == 1 ]]; then
  TEST_DOUBLES=true
elif [[ -n "${RLF_NVIDIA_SMI:-}" ]]; then
  echo "RLF_NVIDIA_SMI is forbidden for real preflight evidence" >&2
  exit 2
elif ((MIN_HOST_RAM_GIB < 1024 || MIN_DISK_GIB < 2048)); then
  echo "real preflight may not lower the 1 TiB RAM / 2 TiB disk floors" >&2
  exit 2
fi
[[ ! -e "${OUTPUT}" ]] || { echo "--output must not already exist" >&2; exit 2; }
mkdir -p "${OUTPUT}"
OUTPUT="$(realpath "${OUTPUT}")"
WORK_ROOT="$(realpath "${WORK_ROOT}")"

CHECKS="${OUTPUT}/checks.tsv"
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
trim() {
  local value="$1"
  value="${value#"${value%%[![:space:]]*}"}"
  value="${value%"${value##*[![:space:]]}"}"
  printf '%s' "${value}"
}
json_escape() {
  local value="$1"
  value="${value//\\/\\\\}"
  value="${value//\"/\\\"}"
  value="${value//$'\n'/\\n}"
  value="${value//$'\r'/\\r}"
  value="${value//$'\t'/\\t}"
  printf '%s' "${value}"
}

for command_name in bash awk df grep realpath sha256sum stat uname; do
  if command -v "${command_name}" >/dev/null 2>&1; then
    record "command_${command_name}" pass "$(command -v "${command_name}")"
  else
    record "command_${command_name}" fail missing
  fi
done
if [[ -r /etc/os-release ]] && grep -Eq '^ID=ubuntu$' /etc/os-release; then
  record operating_system pass ubuntu
else
  record operating_system fail 'Ubuntu is required'
fi

NVIDIA_SMI="${RLF_NVIDIA_SMI:-nvidia-smi}"
if command -v "${NVIDIA_SMI}" >/dev/null 2>&1; then
  record command_nvidia_smi pass "$(command -v "${NVIDIA_SMI}")"
else
  record command_nvidia_smi fail missing
fi

GPU_NAME=""
GPU_UUID=""
TOTAL_MIB=0
FREE_MIB=0
COMPUTE_CAP=""
GPU_QUERY=""
GPU_QUERY_STATUS=1
if command -v "${NVIDIA_SMI}" >/dev/null 2>&1; then
  GPU_QUERY="$(${NVIDIA_SMI} \
    --query-gpu=index,name,uuid,memory.total,memory.free,compute_cap \
    --format=csv,noheader,nounits 2>"${OUTPUT}/nvidia_smi_query_error.txt")"
  GPU_QUERY_STATUS=$?
fi
GPU_COUNT="$(printf '%s\n' "${GPU_QUERY}" | awk 'NF { count += 1 } END { print count + 0 }')"
if ((GPU_QUERY_STATUS == 0 && GPU_COUNT == 1)); then
  IFS=',' read -r GPU_INDEX GPU_NAME GPU_UUID TOTAL_MIB FREE_MIB COMPUTE_CAP <<<"${GPU_QUERY}"
  GPU_INDEX="$(trim "${GPU_INDEX}")"
  GPU_NAME="$(trim "${GPU_NAME}")"
  GPU_UUID="$(trim "${GPU_UUID}")"
  TOTAL_MIB="$(trim "${TOTAL_MIB}")"
  FREE_MIB="$(trim "${FREE_MIB}")"
  COMPUTE_CAP="$(trim "${COMPUTE_CAP}")"
  [[ "${GPU_INDEX}" == 0 ]] && record visible_gpu_count pass one || record visible_gpu_count fail "index=${GPU_INDEX}"
  CANONICAL_GPU_NAME="${GPU_NAME#NVIDIA }"
  [[ "${CANONICAL_GPU_NAME}" == 'RTX PRO 6000 Blackwell Workstation Edition' ]] && \
    record gpu_model pass "${GPU_NAME}" || record gpu_model fail "${GPU_NAME}"
  [[ "${COMPUTE_CAP}" == 12.0 ]] && record compute_capability pass 12.0 || \
    record compute_capability fail "${COMPUTE_CAP}"
  [[ "${TOTAL_MIB}" =~ ^[0-9]+$ ]] && ((TOTAL_MIB >= 97000)) && \
    record total_vram pass "${TOTAL_MIB} MiB" || \
    record total_vram fail "${TOTAL_MIB} MiB; need at least 97000"
  [[ "${FREE_MIB}" =~ ^[0-9]+$ ]] && ((FREE_MIB >= 90112)) && \
    record free_vram pass "${FREE_MIB} MiB" || \
    record free_vram fail "${FREE_MIB} MiB; need at least 90112"
else
  record visible_gpu_count fail "query_status=${GPU_QUERY_STATUS}; rows=${GPU_COUNT}"
fi

GPU_LISTING=""
GPU_LISTING_STATUS=1
if command -v "${NVIDIA_SMI}" >/dev/null 2>&1; then
  GPU_LISTING="$(${NVIDIA_SMI} -L 2>"${OUTPUT}/nvidia_smi_list_error.txt")"
  GPU_LISTING_STATUS=$?
fi
LIST_GPU_COUNT="$(printf '%s\n' "${GPU_LISTING}" | grep -Ec '^GPU [0-9]+:' || true)"
MIG_DEVICE_COUNT="$(printf '%s\n' "${GPU_LISTING}" | grep -Ec '^[[:space:]]*MIG ' || true)"
if ((GPU_LISTING_STATUS == 0 && LIST_GPU_COUNT == 1 && MIG_DEVICE_COUNT == 0)); then
  record gpu_partition_listing pass 'one physical GPU and no MIG devices'
else
  record gpu_partition_listing fail "status=${GPU_LISTING_STATUS}; gpu=${LIST_GPU_COUNT}; mig=${MIG_DEVICE_COUNT}"
fi

MIG_MODE=""
MIG_STATUS=1
if command -v "${NVIDIA_SMI}" >/dev/null 2>&1; then
  MIG_MODE="$(${NVIDIA_SMI} --id=0 --query-gpu=mig.mode.current \
    --format=csv,noheader,nounits 2>"${OUTPUT}/nvidia_smi_mig_error.txt")"
  MIG_STATUS=$?
fi
MIG_MODE="$(trim "${MIG_MODE}")"
case "${MIG_MODE,,}" in
  disabled|n/a|'[n/a]'|'not supported') MIG_ACCEPTED=true ;;
  *) MIG_ACCEPTED=false ;;
esac
if ((MIG_STATUS == 0)) && [[ "${MIG_ACCEPTED}" == true ]]; then
  record mig_mode pass "${MIG_MODE}"
else
  record mig_mode fail "status=${MIG_STATUS}; mode=${MIG_MODE:-unknown}"
fi

AVAILABLE_KIB="$(df -Pk "${WORK_ROOT}" 2>/dev/null | awk 'NR == 2 { print $4 }')"
REQUIRED_DISK_KIB=$((MIN_DISK_GIB * 1024 * 1024))
if [[ "${AVAILABLE_KIB}" =~ ^[0-9]+$ ]] && ((AVAILABLE_KIB >= REQUIRED_DISK_KIB)); then
  record disk_space pass "${AVAILABLE_KIB} KiB"
else
  record disk_space fail "${AVAILABLE_KIB:-unknown} KiB; need ${REQUIRED_DISK_KIB}"
fi
HOST_RAM_KIB="$(awk '/^MemTotal:/ { print $2 }' /proc/meminfo 2>/dev/null)"
REQUIRED_RAM_KIB=$((MIN_HOST_RAM_GIB * 1024 * 1024))
if [[ "${HOST_RAM_KIB}" =~ ^[0-9]+$ ]] && ((HOST_RAM_KIB >= REQUIRED_RAM_KIB)); then
  record host_ram pass "${HOST_RAM_KIB} KiB"
else
  record host_ram fail "${HOST_RAM_KIB:-unknown} KiB; need ${REQUIRED_RAM_KIB}"
fi

{
  printf 'schema=rlf-vast-rtx-pro-6000-environment-v1\n'
  printf 'timestamp_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  printf 'test_doubles=%s\nwork_root=%s\n' "${TEST_DOUBLES}" "${WORK_ROOT}"
  uname -a 2>&1 || true
  printf 'gpu_query=%s\n' "${GPU_QUERY}"
  printf 'gpu_listing=%s\n' "${GPU_LISTING//$'\n'/; }"
  printf 'mig_mode=%s\n' "${MIG_MODE}"
  printf 'host_ram_kib=%s\navailable_disk_kib=%s\n' "${HOST_RAM_KIB}" "${AVAILABLE_KIB}"
} >"${ENVIRONMENT}"

CHECKS_PASS=false
((FAILURES == 0)) && CHECKS_PASS=true
READY="${CHECKS_PASS}"
[[ "${TEST_DOUBLES}" == true ]] && READY=false
printf '{\n' >"${OUTPUT}/preflight.json"
printf '  "schema": "rlf-vast-rtx-pro-6000-preflight-v1",\n' >>"${OUTPUT}/preflight.json"
printf '  "ready": %s,\n' "${READY}" >>"${OUTPUT}/preflight.json"
printf '  "synthetic_checks_passed": %s,\n' "${CHECKS_PASS}" >>"${OUTPUT}/preflight.json"
printf '  "test_doubles": %s,\n' "${TEST_DOUBLES}" >>"${OUTPUT}/preflight.json"
printf '  "training_performed": false,\n' >>"${OUTPUT}/preflight.json"
printf '  "gpu_name": "%s",\n' "$(json_escape "${GPU_NAME}")" >>"${OUTPUT}/preflight.json"
printf '  "gpu_uuid": "%s",\n' "$(json_escape "${GPU_UUID}")" >>"${OUTPUT}/preflight.json"
printf '  "compute_capability": "%s",\n' "$(json_escape "${COMPUTE_CAP}")" >>"${OUTPUT}/preflight.json"
printf '  "total_vram_mib": %s,\n' "${TOTAL_MIB:-0}" >>"${OUTPUT}/preflight.json"
printf '  "free_vram_mib": %s,\n' "${FREE_MIB:-0}" >>"${OUTPUT}/preflight.json"
printf '  "mig_mode": "%s",\n' "$(json_escape "${MIG_MODE}")" >>"${OUTPUT}/preflight.json"
printf '  "minimum_host_ram_gib": %s,\n' "${MIN_HOST_RAM_GIB}" >>"${OUTPUT}/preflight.json"
printf '  "minimum_disk_gib": %s,\n' "${MIN_DISK_GIB}" >>"${OUTPUT}/preflight.json"
printf '  "failed_checks": %s\n' "${FAILURES}" >>"${OUTPUT}/preflight.json"
printf '}\n' >>"${OUTPUT}/preflight.json"
(cd "${OUTPUT}" && sha256sum checks.tsv environment.txt preflight.json >evidence.sha256)

if [[ "${CHECKS_PASS}" != true ]]; then
  echo "Vast RTX PRO 6000 preflight failed closed with ${FAILURES} failed checks. No training was performed." >&2
  exit 1
fi
if [[ "${TEST_DOUBLES}" == true ]]; then
  echo "Synthetic preflight checks passed; ready=false, so this is not hardware evidence."
else
  echo "Exact RTX PRO 6000 hardware preflight passed. No training was performed."
fi
