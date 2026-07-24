#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: run_h100_with_vram_guard.sh --trace FILE --summary FILE [options] -- COMMAND [ARGS...]

Options:
  --gpu-index N       Physical GPU index to expose (default: 0)
  --interval-ms N     Sampling interval in milliseconds (default: 100)
  --limit-mib N       Fail when peak device use exceeds N MiB (default/max: 77824)

The wrapper requires an NVIDIA H100, exposes only the selected device to the
child process, records total device memory use, and exits 4 after the child if
the 76 GiB campaign limit was exceeded. RLF_NVIDIA_SMI may name a test double.
EOF
}

TRACE=""
SUMMARY=""
GPU_INDEX=0
INTERVAL_MS=100
LIMIT_MIB="${RLF_DEFAULT_VRAM_LIMIT_MIB:-77824}"
MAXIMUM_LIMIT_MIB="${RLF_MAXIMUM_VRAM_LIMIT_MIB:-77824}"
REQUIRED_GPU_NAME="${RLF_REQUIRED_GPU_NAME:-H100}"
RESOURCE_SCHEMA="${RLF_VRAM_RESOURCE_SCHEMA:-rlf-h100-vram-v1}"

while (($# > 0)); do
  case "$1" in
    --trace) TRACE="${2:?--trace requires a path}"; shift 2 ;;
    --summary) SUMMARY="${2:?--summary requires a path}"; shift 2 ;;
    --gpu-index) GPU_INDEX="${2:?--gpu-index requires an integer}"; shift 2 ;;
    --interval-ms) INTERVAL_MS="${2:?--interval-ms requires an integer}"; shift 2 ;;
    --limit-mib) LIMIT_MIB="${2:?--limit-mib requires an integer}"; shift 2 ;;
    --) shift; break ;;
    --help|-h) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

[[ -n "${TRACE}" ]] || { echo "--trace is required" >&2; exit 2; }
[[ -n "${SUMMARY}" ]] || { echo "--summary is required" >&2; exit 2; }
(($# > 0)) || { echo "a command is required after --" >&2; exit 2; }
[[ "${GPU_INDEX}" =~ ^[0-9]+$ ]] || { echo "invalid GPU index" >&2; exit 2; }
[[ "${INTERVAL_MS}" =~ ^[1-9][0-9]*$ ]] || { echo "invalid sampling interval" >&2; exit 2; }
[[ "${LIMIT_MIB}" =~ ^[1-9][0-9]*$ ]] || { echo "invalid VRAM limit" >&2; exit 2; }
((LIMIT_MIB <= MAXIMUM_LIMIT_MIB)) || {
  echo "VRAM limit may not exceed ${MAXIMUM_LIMIT_MIB} MiB" >&2
  exit 2
}
[[ "${TRACE}" != "${SUMMARY}" ]] || { echo "trace and summary paths must differ" >&2; exit 2; }

NVIDIA_SMI="${RLF_NVIDIA_SMI:-nvidia-smi}"
command -v "${NVIDIA_SMI}" >/dev/null 2>&1 || {
  echo "nvidia-smi was not found" >&2
  exit 2
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

GPU_ROW="$(${NVIDIA_SMI} --id="${GPU_INDEX}" \
  --query-gpu=name,uuid,memory.total,driver_version \
  --format=csv,noheader,nounits)" || {
  echo "unable to inspect GPU ${GPU_INDEX}" >&2
  exit 2
}
[[ "${GPU_ROW}" != *$'\n'* ]] || { echo "GPU selector returned multiple devices" >&2; exit 2; }
IFS=',' read -r GPU_NAME GPU_UUID TOTAL_MIB DRIVER_VERSION <<<"${GPU_ROW}"
GPU_NAME="$(trim "${GPU_NAME}")"
GPU_UUID="$(trim "${GPU_UUID}")"
TOTAL_MIB="$(trim "${TOTAL_MIB}")"
DRIVER_VERSION="$(trim "${DRIVER_VERSION}")"
[[ "${GPU_NAME}" == *"${REQUIRED_GPU_NAME}"* ]] || {
  echo "selected device is not an NVIDIA ${REQUIRED_GPU_NAME}: ${GPU_NAME}" >&2
  exit 2
}
[[ "${TOTAL_MIB}" =~ ^[0-9]+$ ]] || { echo "invalid total GPU memory" >&2; exit 2; }
((TOTAL_MIB >= LIMIT_MIB)) || {
  echo "selected ${REQUIRED_GPU_NAME} has less memory than the configured limit" >&2
  exit 2
}

mkdir -p "$(dirname "${TRACE}")" "$(dirname "${SUMMARY}")"
printf 'timestamp_utc,memory_used_mib\n' >"${TRACE}"

CONTROL_DIR="$(mktemp -d)"
STOP_FILE="${CONTROL_DIR}/stop"
ERROR_FILE="${CONTROL_DIR}/sampler_error"
SAMPLER_PID=""

cleanup() {
  if [[ -n "${SAMPLER_PID}" ]] && kill -0 "${SAMPLER_PID}" 2>/dev/null; then
    : >"${STOP_FILE}"
    wait "${SAMPLER_PID}" 2>/dev/null || true
  fi
  rm -rf -- "${CONTROL_DIR}"
}
trap cleanup EXIT

INTERVAL_SECONDS="$(awk -v milliseconds="${INTERVAL_MS}" 'BEGIN { printf "%.3f", milliseconds / 1000.0 }')"
(
  while [[ ! -e "${STOP_FILE}" ]]; do
    if ! USED_MIB="$(${NVIDIA_SMI} --id="${GPU_INDEX}" \
        --query-gpu=memory.used --format=csv,noheader,nounits)"; then
      printf 'nvidia-smi sampling failed\n' >"${ERROR_FILE}"
      exit 1
    fi
    USED_MIB="$(trim "${USED_MIB}")"
    if [[ ! "${USED_MIB}" =~ ^[0-9]+$ ]]; then
      printf 'invalid sampled memory value: %s\n' "${USED_MIB}" >"${ERROR_FILE}"
      exit 1
    fi
    printf '%s,%s\n' "$(date -u +%Y-%m-%dT%H:%M:%S.%3NZ)" "${USED_MIB}" >>"${TRACE}"
    sleep "${INTERVAL_SECONDS}"
  done
) &
SAMPLER_PID=$!

START_UTC="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
START_NS="$(date +%s%N)"
printf -v COMMAND_TEXT '%q ' "$@"
set +e
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="${GPU_INDEX}" "$@"
COMMAND_EXIT=$?
set -e
END_NS="$(date +%s%N)"
END_UTC="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

: >"${STOP_FILE}"
SAMPLER_EXIT=0
wait "${SAMPLER_PID}" || SAMPLER_EXIT=$?
SAMPLER_PID=""

SAMPLES="$(awk 'NR > 1 { count += 1 } END { print count + 0 }' "${TRACE}")"
PEAK_MIB="$(awk 'NR > 1 && $2 + 0 > peak { peak = $2 + 0 } END { print peak + 0 }' FS=',' "${TRACE}")"
DURATION_MS=$(((END_NS - START_NS) / 1000000))
PEAK_GIB="$(awk -v peak="${PEAK_MIB}" 'BEGIN { printf "%.6f", peak / 1024.0 }')"
WITHIN_LIMIT=true
((SAMPLES > 0)) || WITHIN_LIMIT=false
((PEAK_MIB <= LIMIT_MIB)) || WITHIN_LIMIT=false
SAMPLER_OK=true
if ((SAMPLER_EXIT != 0)) || [[ -s "${ERROR_FILE}" ]]; then
  SAMPLER_OK=false
fi

cat >"${SUMMARY}" <<EOF
{
  "schema": "${RESOURCE_SCHEMA}",
  "start_utc": "$(json_escape "${START_UTC}")",
  "end_utc": "$(json_escape "${END_UTC}")",
  "duration_ms": ${DURATION_MS},
  "gpu_index": ${GPU_INDEX},
  "gpu_name": "$(json_escape "${GPU_NAME}")",
  "gpu_uuid": "$(json_escape "${GPU_UUID}")",
  "driver_version": "$(json_escape "${DRIVER_VERSION}")",
  "total_memory_mib": ${TOTAL_MIB},
  "limit_memory_mib": ${LIMIT_MIB},
  "sample_interval_ms": ${INTERVAL_MS},
  "samples": ${SAMPLES},
  "peak_memory_mib": ${PEAK_MIB},
  "peak_memory_gib": ${PEAK_GIB},
  "within_limit": ${WITHIN_LIMIT},
  "sampler_ok": ${SAMPLER_OK},
  "command": "$(json_escape "${COMMAND_TEXT% }")",
  "command_exit_code": ${COMMAND_EXIT}
}
EOF

if [[ "${SAMPLER_OK}" != true ]]; then
  [[ -s "${ERROR_FILE}" ]] && cat "${ERROR_FILE}" >&2
  echo "VRAM telemetry failed closed" >&2
  exit 5
fi
if [[ "${WITHIN_LIMIT}" != true ]]; then
  echo "VRAM guard failed: peak ${PEAK_MIB} MiB exceeds ${LIMIT_MIB} MiB or no samples were recorded" >&2
  exit 4
fi
exit "${COMMAND_EXIT}"
