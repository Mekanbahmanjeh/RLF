#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORK="$(mktemp -d)"; trap 'rm -rf -- "${WORK}"' EXIT
export RLF_NVIDIA_SMI="${ROOT}/tests/fixtures/fake_nvidia_smi.sh"
RLF_FAKE_GPU_NAME='NVIDIA A100-SXM4-40GB' RLF_FAKE_TOTAL_MIB=40536 \
RLF_FAKE_COMPUTE_CAP=8.0 RLF_FAKE_VRAM_MIB=38000 \
  "${ROOT}/scripts/run_general_cuda_with_vram_guard.sh" --profile general-40g \
  --trace "${WORK}/within.csv" --summary "${WORK}/within.json" --interval-ms 10 -- bash -c 'sleep 0.04'
grep -q '"profile": "general-40g"' "${WORK}/within.json"
grep -q '"peak_memory_mib": 38000' "${WORK}/within.json"
grep -q '"within_limit": true' "${WORK}/within.json"
grep -q '"average_gpu_utilization_percent": 50.000000' "${WORK}/within.json"
grep -Eq '"energy_joules_estimate": [0-9]' "${WORK}/within.json"
set +e
RLF_FAKE_GPU_NAME='NVIDIA A100-SXM4-40GB' RLF_FAKE_TOTAL_MIB=40536 \
RLF_FAKE_COMPUTE_CAP=8.0 RLF_FAKE_VRAM_MIB=38913 \
  "${ROOT}/scripts/run_general_cuda_with_vram_guard.sh" --profile general-40g \
  --trace "${WORK}/exceeded.csv" --summary "${WORK}/exceeded.json" --interval-ms 10 -- bash -c 'sleep 0.04'
STATUS=$?
set -e
[[ "${STATUS}" -eq 4 ]]
grep -q '"within_limit": false' "${WORK}/exceeded.json"

printf 'pre-attempt-checkpoint\n' >"${WORK}/transaction.rlfsp"
TRANSACTION_SHA="$(sha256sum -- "${WORK}/transaction.rlfsp" | awk '{print $1}')"
set +e
RLF_FAKE_GPU_NAME='NVIDIA A100-SXM4-40GB' RLF_FAKE_TOTAL_MIB=40536 \
RLF_FAKE_COMPUTE_CAP=8.0 RLF_FAKE_VRAM_MIB=38913 \
  "${ROOT}/scripts/run_checkpoint_transaction.sh" \
  --checkpoint "${WORK}/transaction.rlfsp" --failure-root "${WORK}/failed-attempts" \
  --preserve "${WORK}/transaction-summary.json" -- \
  "${ROOT}/scripts/run_general_cuda_with_vram_guard.sh" --profile general-40g \
  --trace "${WORK}/transaction.csv" --summary "${WORK}/transaction-summary.json" \
  --interval-ms 10 -- bash -c 'printf "over-limit-mutation\n" >"$1.tmp"; mv -f -- "$1.tmp" "$1"; sleep 0.04' _ "${WORK}/transaction.rlfsp"
STATUS=$?
set -e
[[ "${STATUS}" -eq 4 ]]
[[ "$(sha256sum -- "${WORK}/transaction.rlfsp" | awk '{print $1}')" == "${TRANSACTION_SHA}" ]]
grep -Rqx 'reason=command-failed' "${WORK}/failed-attempts"
grep -Rq '"within_limit": false' "${WORK}/failed-attempts"
RLF_FAKE_GPU_NAME='Tesla V100-SXM2-32GB' RLF_FAKE_TOTAL_MIB=32510 \
RLF_FAKE_COMPUTE_CAP=7.0 RLF_FAKE_VRAM_MIB=30000 \
  "${ROOT}/scripts/run_general_cuda_with_vram_guard.sh" --profile general-v100-32g \
  --expected-uuid GPU-TEST-0001 --trace "${WORK}/v100.csv" --summary "${WORK}/v100.json" --interval-ms 10 -- bash -c 'sleep 0.04'
grep -q '"profile": "general-v100-32g"' "${WORK}/v100.json"
RLF_FAKE_GPU_NAME='Tesla V100-SXM2-32GB' RLF_FAKE_TOTAL_MIB=32510 \
RLF_FAKE_COMPUTE_CAP=7.0 RLF_FAKE_VRAM_MIB=30000 \
  "${ROOT}/scripts/run_general_cuda_with_vram_guard.sh" --profile general-v100-32g-500m \
  --expected-uuid GPU-TEST-0001 --trace "${WORK}/v100-500m.csv" \
  --summary "${WORK}/v100-500m.json" --interval-ms 10 -- bash -c 'sleep 0.04'
grep -q '"profile": "general-v100-32g-500m"' "${WORK}/v100-500m.json"
grep -q '"limit_memory_mib": 30720' "${WORK}/v100-500m.json"
RLF_FAKE_GPU_NAME='Tesla V100-SXM2-32GB' RLF_FAKE_TOTAL_MIB=32510 \
RLF_FAKE_COMPUTE_CAP=7.0 RLF_FAKE_VRAM_MIB=30000 \
  "${ROOT}/scripts/run_general_cuda_with_vram_guard.sh" --profile video-v100-32g \
  --expected-uuid GPU-TEST-0001 --trace "${WORK}/v100-video.csv" \
  --summary "${WORK}/v100-video.json" --interval-ms 10 -- bash -c 'sleep 0.04'
grep -q '"profile": "video-v100-32g"' "${WORK}/v100-video.json"
grep -q '"limit_memory_mib": 30720' "${WORK}/v100-video.json"
RLF_FAKE_GPU_NAME='Tesla V100-SXM2-32GB' RLF_FAKE_TOTAL_MIB=32510 \
RLF_FAKE_COMPUTE_CAP=7.0 RLF_FAKE_VRAM_MIB=30000 \
  "${ROOT}/scripts/run_general_cuda_with_vram_guard.sh" --profile imagegen-v100-32g \
  --expected-uuid GPU-TEST-0001 --trace "${WORK}/v100-imagegen.csv" \
  --summary "${WORK}/v100-imagegen.json" --interval-ms 10 -- bash -c 'sleep 0.04'
grep -q '"profile": "imagegen-v100-32g"' "${WORK}/v100-imagegen.json"
grep -q '"limit_memory_mib": 30720' "${WORK}/v100-imagegen.json"
set +e
RLF_FAKE_GPU_NAME='Tesla V100-SXM2-32GB' RLF_FAKE_TOTAL_MIB=32510 \
RLF_FAKE_COMPUTE_CAP=7.0 RLF_FAKE_VRAM_MIB=30000 \
  "${ROOT}/scripts/run_general_cuda_with_vram_guard.sh" --profile general-v100-32g-500M \
  --trace "${WORK}/wrong-v100-500m.csv" --summary "${WORK}/wrong-v100-500m.json" \
  --interval-ms 10 -- bash -c 'sleep 0.04' >/dev/null 2>&1
WRONG_PROFILE_STATUS=$?
set -e
[[ "${WRONG_PROFILE_STATUS}" -eq 2 ]]
RLF_FAKE_GPU_NAME='NVIDIA RTX PRO 6000 Blackwell Workstation Edition' \
RLF_FAKE_TOTAL_MIB=97887 RLF_FAKE_COMPUTE_CAP=12.0 RLF_FAKE_VRAM_MIB=90000 \
  "${ROOT}/scripts/run_general_cuda_with_vram_guard.sh" \
  --profile video-rtx-pro-6000-96g --trace "${WORK}/video.csv" \
  --summary "${WORK}/video.json" --interval-ms 10 -- bash -c 'sleep 0.04'
grep -q '"profile": "video-rtx-pro-6000-96g"' "${WORK}/video.json"
grep -q '"within_limit": true' "${WORK}/video.json"
for pro_profile in general-rtx-pro-6000-96g general-rtx-pro-6000-96g-text; do
  summary="${WORK}/${pro_profile}.json"
  RLF_FAKE_GPU_NAME='NVIDIA RTX PRO 6000 Blackwell Workstation Edition' \
  RLF_FAKE_TOTAL_MIB=97887 RLF_FAKE_COMPUTE_CAP=12.0 RLF_FAKE_VRAM_MIB=90000 \
    "${ROOT}/scripts/run_general_cuda_with_vram_guard.sh" \
    --profile "${pro_profile}" --trace "${WORK}/${pro_profile}.csv" \
    --summary "${summary}" --interval-ms 10 -- bash -c 'sleep 0.04'
  grep -q "\"profile\": \"${pro_profile}\"" "${summary}"
  grep -q '"within_limit": true' "${summary}"
done
RLF_FAKE_GPU_NAME='NVIDIA RTX PRO 6000 Blackwell Workstation Edition' \
RLF_FAKE_TOTAL_MIB=97887 RLF_FAKE_COMPUTE_CAP=12.0 RLF_FAKE_VRAM_MIB=88000 \
  "${ROOT}/scripts/run_general_cuda_with_vram_guard.sh" \
  --profile rtx-pro-6000-96g --trace "${WORK}/exact-target.csv" \
  --summary "${WORK}/exact-target.json" --interval-ms 10 -- bash -c 'sleep 0.04'
grep -q '"profile": "rtx-pro-6000-96g"' "${WORK}/exact-target.json"
grep -q '"limit_memory_mib": 90112' "${WORK}/exact-target.json"
