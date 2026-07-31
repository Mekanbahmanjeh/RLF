#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: run_general_cuda_with_vram_guard.sh --profile PROFILE --trace FILE \
  --summary FILE [options] -- COMMAND [ARGS...]

PROFILE is general-40g (38 GiB maximum), general-v100-32g (30 GiB,
multimodal), general-v100-32g-text (30 GiB, image input disabled),
general-v100-32g-500m (30 GiB, isolated 1,586-hour campaign),
video-v100-32g (30 GiB, frame sequences enabled),
imagegen-v100-32g (30 GiB, resonant image trainer),
rtx-pro-6000-96g (88 GiB, multimodal),
general-rtx-pro-6000-96g (90 GiB, multimodal),
general-rtx-pro-6000-96g-text (90 GiB, image input disabled), or
video-rtx-pro-6000-96g (90 GiB, frame sequences enabled).
The wrapper exposes one selected GPU, samples total device use every 100 ms,
and fails closed on the wrong device class, missing telemetry, or an exceeded
profile ceiling. RLF_NVIDIA_SMI is reserved for shell regression tests.
EOF
}

PROFILE=""
TRACE=""
SUMMARY=""
GPU_INDEX=0
INTERVAL_MS=100
LIMIT_MIB=""
EXPECTED_UUID=""
while (($# > 0)); do
  case "$1" in
    --profile) PROFILE="${2:?--profile requires a name}"; shift 2 ;;
    --trace) TRACE="${2:?--trace requires a path}"; shift 2 ;;
    --summary) SUMMARY="${2:?--summary requires a path}"; shift 2 ;;
    --gpu-index) GPU_INDEX="${2:?--gpu-index requires an integer}"; shift 2 ;;
    --interval-ms) INTERVAL_MS="${2:?--interval-ms requires an integer}"; shift 2 ;;
    --limit-mib) LIMIT_MIB="${2:?--limit-mib requires an integer}"; shift 2 ;;
    --expected-uuid) EXPECTED_UUID="${2:?--expected-uuid requires a value}"; shift 2 ;;
    --) shift; break ;;
    --help|-h) usage; exit 0 ;;
    *) echo "unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

case "${PROFILE}" in
  general-40g) MAXIMUM_LIMIT_MIB=38912 ;;
  general-v100-32g|general-v100-32g-text|general-v100-32g-500m|video-v100-32g|imagegen-v100-32g) MAXIMUM_LIMIT_MIB=30720 ;;
  rtx-pro-6000-96g) MAXIMUM_LIMIT_MIB=90112 ;;
  general-rtx-pro-6000-96g|general-rtx-pro-6000-96g-text|video-rtx-pro-6000-96g) MAXIMUM_LIMIT_MIB=92160 ;;
  *) echo "unsupported general CUDA profile" >&2; exit 2 ;;
esac
[[ -n "${LIMIT_MIB}" ]] || LIMIT_MIB="${MAXIMUM_LIMIT_MIB}"
[[ -n "${TRACE}" && -n "${SUMMARY}" && "${TRACE}" != "${SUMMARY}" ]] || {
  echo "distinct --trace and --summary paths are required" >&2; exit 2;
}
(($# > 0)) || { echo "a command is required after --" >&2; exit 2; }
[[ "${GPU_INDEX}" =~ ^[0-9]+$ && "${INTERVAL_MS}" =~ ^[1-9][0-9]*$ && \
   "${LIMIT_MIB}" =~ ^[1-9][0-9]*$ ]] || { echo "invalid numeric option" >&2; exit 2; }
((LIMIT_MIB <= MAXIMUM_LIMIT_MIB)) || {
  echo "VRAM limit exceeds the ${PROFILE} maximum of ${MAXIMUM_LIMIT_MIB} MiB" >&2
  exit 2
}

NVIDIA_SMI="${RLF_NVIDIA_SMI:-nvidia-smi}"
command -v "${NVIDIA_SMI}" >/dev/null 2>&1 || { echo "nvidia-smi was not found" >&2; exit 2; }
trim() { local value="$1"; value="${value#"${value%%[![:space:]]*}"}"; value="${value%"${value##*[![:space:]]}"}"; printf '%s' "${value}"; }
json_escape() { local value="$1"; value="${value//\\/\\\\}"; value="${value//\"/\\\"}"; value="${value//$'\n'/\\n}"; value="${value//$'\r'/\\r}"; value="${value//$'\t'/\\t}"; printf '%s' "${value}"; }

GPU_ROW="$(${NVIDIA_SMI} --id="${GPU_INDEX}" \
  --query-gpu=name,uuid,memory.total,compute_cap,driver_version \
  --format=csv,noheader,nounits)" || { echo "unable to inspect GPU ${GPU_INDEX}" >&2; exit 2; }
[[ "${GPU_ROW}" != *$'\n'* ]] || { echo "GPU selector returned multiple devices" >&2; exit 2; }
IFS=',' read -r GPU_NAME GPU_UUID TOTAL_MIB COMPUTE_CAP DRIVER_VERSION <<<"${GPU_ROW}"
GPU_NAME="$(trim "${GPU_NAME}")"; GPU_UUID="$(trim "${GPU_UUID}")"
TOTAL_MIB="$(trim "${TOTAL_MIB}")"; COMPUTE_CAP="$(trim "${COMPUTE_CAP}")"
DRIVER_VERSION="$(trim "${DRIVER_VERSION}")"
[[ -z "${EXPECTED_UUID}" || "${GPU_UUID}" == "${EXPECTED_UUID}" ]] || {
  echo "selected GPU UUID does not match the readiness report" >&2; exit 2;
}
[[ "${TOTAL_MIB}" =~ ^[0-9]+$ ]] || { echo "invalid total GPU memory" >&2; exit 2; }
if [[ "${PROFILE}" == general-v100-32g || "${PROFILE}" == general-v100-32g-text || "${PROFILE}" == general-v100-32g-500m || "${PROFILE}" == video-v100-32g || "${PROFILE}" == imagegen-v100-32g ]]; then
  [[ "${GPU_NAME}" == *V100* && "${COMPUTE_CAP}" == 7.0 ]] || {
    echo "general-v100-32g requires an NVIDIA V100 compute 7.0 device" >&2; exit 2;
  }
elif [[ "${PROFILE}" == general-40g ]]; then
  [[ "${GPU_NAME}" == *A100* || "${GPU_NAME}" == *V100* ]] || {
    echo "general-40g requires an approved A100/V100-class device" >&2; exit 2;
  }
  [[ "${COMPUTE_CAP}" == 7.0 || "${COMPUTE_CAP}" == 8.0 ]] || {
    echo "general-40g requires compute capability 7.0 or 8.0" >&2; exit 2;
  }
else
  [[ "${GPU_NAME}" == *"RTX PRO 6000 Blackwell"* && "${COMPUTE_CAP}" == 12.0 ]] || {
    echo "${PROFILE} requires RTX PRO 6000 Blackwell compute 12.0" >&2; exit 2;
  }
fi
((TOTAL_MIB >= LIMIT_MIB)) || {
  echo "selected device has less memory than the ${PROFILE} limit" >&2; exit 2;
}

mkdir -p "$(dirname "${TRACE}")" "$(dirname "${SUMMARY}")"
printf 'timestamp_utc,memory_used_mib,gpu_utilization_percent,power_draw_watts\n' >"${TRACE}"
CONTROL_DIR="$(mktemp -d)"
STOP_FILE="${CONTROL_DIR}/stop"; ERROR_FILE="${CONTROL_DIR}/sampler_error"; SAMPLER_PID=""
cleanup() {
  if [[ -n "${SAMPLER_PID}" ]] && kill -0 "${SAMPLER_PID}" 2>/dev/null; then
    : >"${STOP_FILE}"; wait "${SAMPLER_PID}" 2>/dev/null || true
  fi
  rm -rf -- "${CONTROL_DIR}"
}
trap cleanup EXIT
INTERVAL_SECONDS="$(awk -v milliseconds="${INTERVAL_MS}" 'BEGIN { printf "%.3f", milliseconds / 1000.0 }')"
(
  while [[ ! -e "${STOP_FILE}" ]]; do
    if ! SAMPLE="$(${NVIDIA_SMI} --id="${GPU_INDEX}" --query-gpu=memory.used,utilization.gpu,power.draw --format=csv,noheader,nounits)"; then
      printf 'nvidia-smi sampling failed\n' >"${ERROR_FILE}"; exit 1
    fi
    [[ "${SAMPLE}" != *$'\n'* ]] || { printf 'ambiguous sample\n' >"${ERROR_FILE}"; exit 1; }
    IFS=',' read -r USED_MIB GPU_UTILIZATION POWER_DRAW <<<"${SAMPLE}"
    USED_MIB="$(trim "${USED_MIB}")"; GPU_UTILIZATION="$(trim "${GPU_UTILIZATION}")"; POWER_DRAW="$(trim "${POWER_DRAW}")"
    [[ "${USED_MIB}" =~ ^[0-9]+$ && "${GPU_UTILIZATION}" =~ ^[0-9]+([.][0-9]+)?$ ]] || { printf 'invalid memory/utilization sample: %s\n' "${SAMPLE}" >"${ERROR_FILE}"; exit 1; }
    awk -v value="${GPU_UTILIZATION}" 'BEGIN { exit !(value >= 0 && value <= 100) }' || { printf 'utilization outside 0..100\n' >"${ERROR_FILE}"; exit 1; }
    if [[ "${POWER_DRAW}" == N/A || "${POWER_DRAW}" == "[N/A]" || "${POWER_DRAW}" == "Not Supported" ]]; then
      POWER_DRAW=""
    elif [[ ! "${POWER_DRAW}" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
      printf 'invalid power sample: %s\n' "${SAMPLE}" >"${ERROR_FILE}"; exit 1
    fi
    printf '%s,%s,%s,%s\n' "$(date -u +%Y-%m-%dT%H:%M:%S.%3NZ)" "${USED_MIB}" "${GPU_UTILIZATION}" "${POWER_DRAW}" >>"${TRACE}"
    sleep "${INTERVAL_SECONDS}"
  done
) &
SAMPLER_PID=$!
START_UTC="$(date -u +%Y-%m-%dT%H:%M:%SZ)"; START_NS="$(date +%s%N)"
printf -v COMMAND_TEXT '%q ' "$@"
set +e
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="${GPU_INDEX}" "$@"
COMMAND_EXIT=$?
set -e
END_NS="$(date +%s%N)"; END_UTC="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
: >"${STOP_FILE}"; SAMPLER_EXIT=0; wait "${SAMPLER_PID}" || SAMPLER_EXIT=$?; SAMPLER_PID=""
SAMPLES="$(awk 'NR > 1 { count += 1 } END { print count + 0 }' "${TRACE}")"
PEAK_MIB="$(awk 'NR > 1 && $2 + 0 > peak { peak = $2 + 0 } END { print peak + 0 }' FS=',' "${TRACE}")"
AVERAGE_UTILIZATION="$(awk 'NR > 1 { total += $3; count += 1 } END { if (count) printf "%.6f", total / count; else print 0 }' FS=',' "${TRACE}")"
GPU_ACTIVE_SECONDS="$(awk -F',' -v interval_ms="${INTERVAL_MS}" 'NR > 1 && $3 + 0 > 0 { active += interval_ms / 1000.0 } END { printf "%.6f", active + 0 }' "${TRACE}")"
POWER_SAMPLES="$(awk -F',' 'NR > 1 && $4 != "" { count += 1 } END { print count + 0 }' "${TRACE}")"
if ((SAMPLES > 0 && POWER_SAMPLES == SAMPLES)); then
  ENERGY_JOULES="$(awk -F',' -v interval_ms="${INTERVAL_MS}" 'NR > 1 { energy += $4 * interval_ms / 1000.0 } END { printf "%.6f", energy + 0 }' "${TRACE}")"
else
  ENERGY_JOULES=null
fi
DURATION_MS=$(((END_NS - START_NS) / 1000000))
PEAK_GIB="$(awk -v peak="${PEAK_MIB}" 'BEGIN { printf "%.6f", peak / 1024.0 }')"
WITHIN_LIMIT=true; ((SAMPLES > 0 && PEAK_MIB <= LIMIT_MIB)) || WITHIN_LIMIT=false
SAMPLER_OK=true; ((SAMPLER_EXIT == 0)) && [[ ! -s "${ERROR_FILE}" ]] || SAMPLER_OK=false
cat >"${SUMMARY}" <<EOF
{
  "schema": "rlf-general-cuda-vram-v1",
  "profile": "${PROFILE}",
  "start_utc": "$(json_escape "${START_UTC}")",
  "end_utc": "$(json_escape "${END_UTC}")",
  "duration_ms": ${DURATION_MS},
  "gpu_index": ${GPU_INDEX},
  "gpu_name": "$(json_escape "${GPU_NAME}")",
  "gpu_uuid": "$(json_escape "${GPU_UUID}")",
  "compute_capability": "$(json_escape "${COMPUTE_CAP}")",
  "driver_version": "$(json_escape "${DRIVER_VERSION}")",
  "total_memory_mib": ${TOTAL_MIB},
  "limit_memory_mib": ${LIMIT_MIB},
  "sample_interval_ms": ${INTERVAL_MS},
  "samples": ${SAMPLES},
  "peak_memory_mib": ${PEAK_MIB},
  "peak_memory_gib": ${PEAK_GIB},
  "average_gpu_utilization_percent": ${AVERAGE_UTILIZATION},
  "gpu_active_seconds_estimate": ${GPU_ACTIVE_SECONDS},
  "power_samples": ${POWER_SAMPLES},
  "energy_joules_estimate": ${ENERGY_JOULES},
  "sampling_estimate_note": "Utilization-active time and energy are rectangular integrations at the declared sampling interval.",
  "within_limit": ${WITHIN_LIMIT},
  "sampler_ok": ${SAMPLER_OK},
  "command": "$(json_escape "${COMMAND_TEXT% }")",
  "command_exit_code": ${COMMAND_EXIT}
}
EOF
[[ "${SAMPLER_OK}" == true ]] || { [[ -s "${ERROR_FILE}" ]] && cat "${ERROR_FILE}" >&2; echo "VRAM telemetry failed closed" >&2; exit 5; }
[[ "${WITHIN_LIMIT}" == true ]] || { echo "VRAM guard failed: peak ${PEAK_MIB} MiB, limit ${LIMIT_MIB} MiB" >&2; exit 4; }
exit "${COMMAND_EXIT}"
