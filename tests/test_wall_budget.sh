#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORK="$(mktemp -d)"; trap 'rm -rf -- "${WORK}"' EXIT
BINDING="$(printf 'campaign-a' | sha256sum | awk '{print $1}')"
"${ROOT}/scripts/run_with_wall_budget.sh" --budget-seconds 2 --binding "${BINDING}" --state "${WORK}/state" -- bash -c 'true' >"${WORK}/first.txt"
grep -Eq '^consumed_seconds=1$' "${WORK}/state"
exec 8>"${WORK}/state.lock"
flock -n 8
set +e
"${ROOT}/scripts/run_with_wall_budget.sh" --budget-seconds 2 --binding "${BINDING}" --state "${WORK}/state" -- true >/dev/null 2>&1
STATUS=$?
set -e
[[ "${STATUS}" -eq 7 ]]
flock -u 8
set +e
"${ROOT}/scripts/run_with_wall_budget.sh" --budget-seconds 2 --binding "${BINDING}" --state "${WORK}/state" -- bash -c 'sleep 5' >"${WORK}/second.txt" 2>&1
STATUS=$?
set -e
[[ "${STATUS}" -eq 124 ]]
grep -Eq '^consumed_seconds=2$' "${WORK}/state"
set +e
"${ROOT}/scripts/run_with_wall_budget.sh" --budget-seconds 2 --binding "${BINDING}" --state "${WORK}/state" -- true >/dev/null 2>&1
STATUS=$?
set -e
[[ "${STATUS}" -eq 6 ]]

RECOVERY_STATE="${WORK}/recovery-state"
STARTED=$(( $(date +%s) - 3 ))
printf 'schema=rlf-wall-budget-v2\nbudget_seconds=10\nbinding_sha256=%s\nconsumed_seconds=10\nactive=true\nactive_started_epoch=%s\nactive_base_consumed=2\n' \
  "${BINDING}" "${STARTED}" >"${RECOVERY_STATE}"
"${ROOT}/scripts/run_with_wall_budget.sh" --budget-seconds 10 --binding "${BINDING}" \
  --state "${RECOVERY_STATE}" -- true >/dev/null
RECOVERED="$(awk -F= '$1 == "consumed_seconds" { print $2 }' "${RECOVERY_STATE}")"
[[ "${RECOVERED}" =~ ^[0-9]+$ && "${RECOVERED}" -ge 6 && "${RECOVERED}" -le 10 ]]

set +e
"${ROOT}/scripts/run_with_wall_budget.sh" --budget-seconds 10 \
  --binding "$(printf 'campaign-b' | sha256sum | awk '{print $1}')" \
  --state "${RECOVERY_STATE}" -- true >/dev/null 2>&1
STATUS=$?
set -e
[[ "${STATUS}" -eq 2 ]]
