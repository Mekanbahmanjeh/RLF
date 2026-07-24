#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORK="$(mktemp -d)"
trap 'rm -rf -- "${WORK}"' EXIT
CONTROLLER="${ROOT}/scripts/h200/h200_12month_controller.sh"
BINDING=bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb

bash -n "${CONTROLLER}"
bash "${CONTROLLER}" --plan >"${WORK}/plan.txt"
grep -Fq 'profile=general-h200-141g-30t' "${WORK}/plan.txt"
grep -Fq 'total_budget_seconds=31536000' "${WORK}/plan.txt"
grep -Fq 'primary_target_tokens=15000000000000' "${WORK}/plan.txt"
grep -Fq 'conditional_target_tokens=30000000000000' "${WORK}/plan.txt"
PLANNED_HOURS="$(awk '$1 ~ /^(hardware-preflight|cuda12-sm90-build-test|data-provenance-audit|token-census-contamination-audit|physical-throughput-resume-probe|train-1t-calibration|recovery-frozen-eval-1t|train-5t|recovery-frozen-eval-5t|train-15t-primary|recovery-frozen-eval-15t|train-30t-promoted|final-external-evaluation|artifact-export|recovery-reserve)$/ {total += $2} END {print total + 0}' "${WORK}/plan.txt")"
[[ "${PLANNED_HOURS}" -eq 8760 ]]

bash "${CONTROLLER}" --start --state-dir "${WORK}/state" \
  --binding "${BINDING}" >/dev/null
bash "${CONTROLLER}" --status --state-dir "${WORK}/state" >"${WORK}/status.txt"
grep -Fq 'remaining_seconds=31536000' "${WORK}/status.txt"

bash "${CONTROLLER}" --run-stage hardware-preflight \
  --state-dir "${WORK}/state" --binding "${BINDING}" -- \
  bash -c '[[ "$RLF_TRAINING_AUTHORIZED" == 0 ]]'

set +e
bash "${CONTROLLER}" --run-stage train-1t-calibration \
  --state-dir "${WORK}/state" --binding "${BINDING}" \
  >"${WORK}/training.out" 2>"${WORK}/training.err"
STATUS=$?
set -e
[[ "${STATUS}" -eq 5 ]]
grep -Fq 'physical token promotion gate' "${WORK}/training.err"
