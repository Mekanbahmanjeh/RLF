#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONTROLLER="${ROOT}/scripts/v100/imagegen_40h_controller.sh"
WORK="$(mktemp -d)"; trap 'rm -rf -- "${WORK}"' EXIT

bash -n "${CONTROLLER}"
bash "${CONTROLLER}" --plan >"${WORK}/plan.txt"
grep -Fqx 'total_budget_hours=40' "${WORK}/plan.txt"
grep -Fqx 'total_budget_seconds=144000' "${WORK}/plan.txt"
grep -Eq '^TOTAL[[:space:]]+40$' "${WORK}/plan.txt"
grep -Eq '^prompt-semantic-training[[:space:]]+6[[:space:]]' "${WORK}/plan.txt"

BINDING="$(printf imagegen-40h-fixture | sha256sum | awk '{print $1}')"
printf 'fixture ledger\n' >"${WORK}/language-ledger.tsv"
bash "${CONTROLLER}" --start --state-dir "${WORK}/state" --binding "${BINDING}" \
  --language-ledger "${WORK}/language-ledger.tsv" \
  >"${WORK}/start.txt"
bash "${CONTROLLER}" --status --state-dir "${WORK}/state" >"${WORK}/status.txt"
grep -Fqx 'total_budget_seconds=144000' "${WORK}/status.txt"
grep -Fqx 'remaining_seconds=144000' "${WORK}/status.txt"

set +e
bash "${CONTROLLER}" --run-stage controlled-training \
  --state-dir "${WORK}/state" --binding "${BINDING}" -- true \
  >/dev/null 2>"${WORK}/training.err"
STATUS=$?
set -e
[[ "${STATUS}" -eq 2 ]]
grep -q 'training rejects arbitrary commands' "${WORK}/training.err"

set +e
bash "${CONTROLLER}" --run-stage prompt-semantic-training \
  --state-dir "${WORK}/state" --binding "${BINDING}" -- true \
  >/dev/null 2>"${WORK}/prompt-training.err"
PROMPT_STATUS=$?
set -e
[[ "${PROMPT_STATUS}" -eq 2 ]]
grep -q 'prompt training rejects arbitrary commands' \
  "${WORK}/prompt-training.err"
printf 'imagegen_40h_controller_test=pass\nphysical_training_performed=false\n'
