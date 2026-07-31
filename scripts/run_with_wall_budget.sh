#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "Usage: run_with_wall_budget.sh --budget-seconds N --binding SHA256 --state FILE -- COMMAND [ARGS...]" >&2
}
BUDGET=""; BINDING=""; STATE=""
while (($# > 0)); do
  case "$1" in
    --budget-seconds) BUDGET="${2:?}"; shift 2 ;;
    --binding) BINDING="${2:?}"; shift 2 ;;
    --state) STATE="${2:?}"; shift 2 ;;
    --) shift; break ;;
    --help|-h) usage; exit 0 ;;
    *) echo "unknown option: $1" >&2; usage; exit 2 ;;
  esac
done
[[ "${BUDGET}" =~ ^[1-9][0-9]*$ && "${BINDING}" =~ ^[0-9a-f]{64}$ && \
   -n "${STATE}" && $# -gt 0 ]] || { usage; exit 2; }
command -v timeout >/dev/null 2>&1 || { echo "timeout was not found" >&2; exit 2; }
command -v flock >/dev/null 2>&1 || { echo "flock was not found" >&2; exit 2; }
mkdir -p "$(dirname "${STATE}")"
exec 9>"${STATE}.lock"
flock -n 9 || { echo "another process holds the training wall-time budget" >&2; exit 7; }

state_value() { awk -F= -v key="$2" '$1 == key { print substr($0, length(key) + 2) }' "$1"; }
write_state() {
  local consumed="$1" active="$2" started="$3" base="$4"
  local temporary
  temporary="$(mktemp "$(dirname "${STATE}")/.wall-budget.XXXXXX")"
  {
    printf 'schema=rlf-wall-budget-v2\n'
    printf 'budget_seconds=%s\n' "${BUDGET}"
    printf 'binding_sha256=%s\n' "${BINDING}"
    printf 'consumed_seconds=%s\n' "${consumed}"
    printf 'active=%s\n' "${active}"
    printf 'active_started_epoch=%s\n' "${started}"
    printf 'active_base_consumed=%s\n' "${base}"
  } >"${temporary}"
  mv -f -- "${temporary}" "${STATE}"
}

CONSUMED=0; ACTIVE=false; ACTIVE_STARTED=0; ACTIVE_BASE=0
if [[ -e "${STATE}" ]]; then
  [[ -f "${STATE}" ]] || { echo "wall-budget state is not a regular file" >&2; exit 2; }
  [[ "$(state_value "${STATE}" schema)" == rlf-wall-budget-v2 ]] || { echo "invalid wall-budget schema" >&2; exit 2; }
  [[ "$(state_value "${STATE}" budget_seconds)" == "${BUDGET}" ]] || { echo "wall-budget total changed" >&2; exit 2; }
  [[ "$(state_value "${STATE}" binding_sha256)" == "${BINDING}" ]] || { echo "wall-budget campaign binding changed" >&2; exit 2; }
  CONSUMED="$(state_value "${STATE}" consumed_seconds)"
  ACTIVE="$(state_value "${STATE}" active)"
  ACTIVE_STARTED="$(state_value "${STATE}" active_started_epoch)"
  ACTIVE_BASE="$(state_value "${STATE}" active_base_consumed)"
  [[ "${CONSUMED}" =~ ^[0-9]+$ && "${ACTIVE_STARTED}" =~ ^[0-9]+$ && \
     "${ACTIVE_BASE}" =~ ^[0-9]+$ && ("${ACTIVE}" == true || "${ACTIVE}" == false) ]] || {
    echo "invalid wall-budget state" >&2; exit 2;
  }
fi

NOW="$(date +%s)"
if [[ "${ACTIVE}" == true ]]; then
  ((NOW >= ACTIVE_STARTED)) || { echo "system clock moved before active budget lease" >&2; exit 2; }
  RECOVERED=$((ACTIVE_BASE + NOW - ACTIVE_STARTED))
  ((RECOVERED <= BUDGET)) || RECOVERED="${BUDGET}"
  CONSUMED="${RECOVERED}"
  write_state "${CONSUMED}" false 0 0
fi
((CONSUMED < BUDGET)) || { echo "training wall-time budget exhausted (${BUDGET} seconds)" >&2; exit 6; }
REMAINING=$((BUDGET - CONSUMED))
START_EPOCH="$(date +%s)"; BASE_CONSUMED="${CONSUMED}"
# Reserve the remaining budget before launch. A crash leaves an active lease
# that is conservatively reconciled from wall time on the next invocation.
write_state "${BUDGET}" true "${START_EPOCH}" "${BASE_CONSUMED}"
set +e
timeout --signal=TERM --kill-after=300s "${REMAINING}s" "$@"
COMMAND_STATUS=$?
set -e
END_EPOCH="$(date +%s)"
((END_EPOCH >= START_EPOCH)) || { echo "system clock moved backwards during budgeted run" >&2; exit 2; }
ELAPSED=$((END_EPOCH - START_EPOCH)); ((ELAPSED > 0)) || ELAPSED=1
CONSUMED=$((BASE_CONSUMED + ELAPSED)); ((CONSUMED <= BUDGET)) || CONSUMED="${BUDGET}"
write_state "${CONSUMED}" false 0 0
printf 'wall_budget_seconds=%s\nwall_consumed_seconds=%s\nwall_remaining_seconds=%s\n' \
  "${BUDGET}" "${CONSUMED}" "$((BUDGET - CONSUMED))"
exit "${COMMAND_STATUS}"
