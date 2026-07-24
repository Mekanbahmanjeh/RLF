#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  phase1_118h_controller.sh --plan
  phase1_118h_controller.sh --start --state-dir DIR --binding SHA256 \
    [--rental-start-epoch EPOCH]
  phase1_118h_controller.sh --status --state-dir DIR
  phase1_118h_controller.sh --run-stage NAME --state-dir DIR --binding SHA256 \
    [--preflight-json FILE] -- COMMAND [ARGS...]

The default action is --plan and performs no work. --start creates an immutable
118-hour wall-clock deadline measured from the supplied rental start (or now).
--run-stage requires an explicit command and
enforces both the fixed campaign deadline and cumulative per-stage/overall
command budgets. It does not create or destroy Vast instances.
EOF
}

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TOTAL_SECONDS=424800
ACTION=plan
STAGE=""
STATE_DIR=""
BINDING=""
PREFLIGHT=""
RENTAL_START_EPOCH=""
COMMAND=()
while (($# > 0)); do
  case "$1" in
    --plan) ACTION=plan; shift ;;
    --start) ACTION=start; shift ;;
    --status) ACTION=status; shift ;;
    --run-stage) ACTION=run; STAGE="${2:?--run-stage requires a name}"; shift 2 ;;
    --state-dir) STATE_DIR="${2:?--state-dir requires a directory}"; shift 2 ;;
    --binding) BINDING="${2:?--binding requires SHA-256}"; shift 2 ;;
    --rental-start-epoch) RENTAL_START_EPOCH="${2:?requires an epoch}"; shift 2 ;;
    --preflight-json) PREFLIGHT="${2:?--preflight-json requires a file}"; shift 2 ;;
    --) shift; COMMAND=("$@"); break ;;
    --help|-h) usage; exit 0 ;;
    *) echo "unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

print_plan() {
  cat <<'EOF'
stage                         hours  purpose
hardware-preflight               4  exact host proof, transfer checks, capacity sizing
build-validation                 8  sm_120 build, full suite, CUDA/CPU equivalence
ingest-efficiency                14  repeated small-scale baseline/optimized A/B timing
controlled-image                 10  RLF-native image tests and small audited image runs
50m-candidates                   44  staged candidates; checkpoint and resume validation
recovery-ablations               18  failure injection, parser/policy ablations, reruns
frozen-dev-evaluation            12  frozen phase-one candidate and development evaluation
artifact-export                   8  hashes, transfer, local verification before destroy
adaptive-contingency              0  may consume only time left by earlier stages
TOTAL                           118  leaves two hours outside this controller as rental margin

This allocation is a ceiling, not evidence of capability. It neither asserts a
100,000x result nor authorizes a frontier claim.
EOF
}

stage_seconds() {
  case "$1" in
    hardware-preflight) echo $((4 * 3600)) ;;
    build-validation) echo $((8 * 3600)) ;;
    ingest-efficiency) echo $((14 * 3600)) ;;
    controlled-image) echo $((10 * 3600)) ;;
    50m-candidates) echo $((44 * 3600)) ;;
    recovery-ablations) echo $((18 * 3600)) ;;
    frozen-dev-evaluation) echo $((12 * 3600)) ;;
    artifact-export) echo $((8 * 3600)) ;;
    adaptive-contingency) echo "${TOTAL_SECONDS}" ;;
    *) return 1 ;;
  esac
}

state_value() {
  awk -F= -v key="$2" '$1 == key { print substr($0, length(key) + 2); exit }' "$1"
}

if [[ "${ACTION}" == plan ]]; then
  print_plan
  exit 0
fi
[[ -n "${STATE_DIR}" ]] || { echo "--state-dir is required" >&2; exit 2; }
STATE_DIR="$(realpath -m "${STATE_DIR}")"
META="${STATE_DIR}/campaign.meta"

if [[ "${ACTION}" == start ]]; then
  [[ "${BINDING}" =~ ^[0-9a-f]{64}$ ]] || { echo "--binding must be lowercase SHA-256" >&2; exit 2; }
  [[ ! -e "${META}" ]] || { echo "campaign state already exists" >&2; exit 2; }
  mkdir -p "${STATE_DIR}"
  NOW_EPOCH="$(date +%s)"
  [[ -n "${RENTAL_START_EPOCH}" ]] || RENTAL_START_EPOCH="${NOW_EPOCH}"
  [[ "${RENTAL_START_EPOCH}" =~ ^[0-9]+$ && "${RENTAL_START_EPOCH}" -le "${NOW_EPOCH}" ]] || {
    echo "--rental-start-epoch must be a non-future Unix epoch" >&2; exit 2;
  }
  START_EPOCH="${RENTAL_START_EPOCH}"
  DEADLINE_EPOCH=$((START_EPOCH + TOTAL_SECONDS))
  ((NOW_EPOCH < DEADLINE_EPOCH)) || { echo "118-hour rental window already expired" >&2; exit 6; }
  TEMPORARY="$(mktemp "${STATE_DIR}/.campaign-meta.XXXXXX")"
  {
    printf 'schema=rlf-vast-phase1-wall-v1\n'
    printf 'binding_sha256=%s\n' "${BINDING}"
    printf 'started_epoch=%s\n' "${START_EPOCH}"
    printf 'deadline_epoch=%s\n' "${DEADLINE_EPOCH}"
    printf 'total_budget_seconds=%s\n' "${TOTAL_SECONDS}"
  } >"${TEMPORARY}"
  mv -f -- "${TEMPORARY}" "${META}"
  printf 'campaign_started_epoch=%s\ncampaign_deadline_epoch=%s\n' "${START_EPOCH}" "${DEADLINE_EPOCH}"
  exit 0
fi

[[ -f "${META}" ]] || { echo "campaign was not started" >&2; exit 2; }
[[ "$(state_value "${META}" schema)" == rlf-vast-phase1-wall-v1 ]] || { echo "invalid campaign state" >&2; exit 2; }
META_BINDING="$(state_value "${META}" binding_sha256)"
DEADLINE_EPOCH="$(state_value "${META}" deadline_epoch)"
[[ "$(state_value "${META}" total_budget_seconds)" == "${TOTAL_SECONDS}" && \
   "${META_BINDING}" =~ ^[0-9a-f]{64}$ && "${DEADLINE_EPOCH}" =~ ^[0-9]+$ ]] || {
  echo "invalid campaign accounting fields" >&2; exit 2;
}
NOW="$(date +%s)"
REMAINING=0
((NOW < DEADLINE_EPOCH)) && REMAINING=$((DEADLINE_EPOCH - NOW))

if [[ "${ACTION}" == status ]]; then
  printf 'binding_sha256=%s\ndeadline_epoch=%s\nwall_remaining_seconds=%s\n' \
    "${META_BINDING}" "${DEADLINE_EPOCH}" "${REMAINING}"
  for stage_name in hardware-preflight build-validation ingest-efficiency \
    controlled-image 50m-candidates recovery-ablations frozen-dev-evaluation \
    artifact-export adaptive-contingency; do
    state_file="${STATE_DIR}/${stage_name}.wall.state"
    consumed=0
    [[ -f "${state_file}" ]] && consumed="$(state_value "${state_file}" consumed_seconds)"
    printf 'stage_%s_consumed_seconds=%s\n' "${stage_name}" "${consumed:-0}"
  done
  exit 0
fi

[[ "${ACTION}" == run && "${BINDING}" == "${META_BINDING}" ]] || {
  echo "run binding does not match immutable campaign state" >&2; exit 2;
}
STAGE_SECONDS="$(stage_seconds "${STAGE}")" || { echo "unsupported phase-one stage" >&2; exit 2; }
((${#COMMAND[@]} > 0)) || { echo "--run-stage requires a command after --" >&2; exit 2; }
((REMAINING > 0)) || { echo "fixed 118-hour campaign deadline has expired" >&2; exit 6; }
if [[ "${STAGE}" != hardware-preflight && "${STAGE}" != artifact-export ]]; then
  [[ -f "${PREFLIGHT}" ]] || { echo "this stage requires --preflight-json" >&2; exit 2; }
  grep -Eq '"schema"[[:space:]]*:[[:space:]]*"rlf-vast-rtx-pro-6000-preflight-v1"' "${PREFLIGHT}" || {
    echo "unsupported hardware preflight" >&2; exit 2;
  }
  grep -Eq '"ready"[[:space:]]*:[[:space:]]*true' "${PREFLIGHT}" || {
    echo "stage requires real ready=true hardware evidence" >&2; exit 2;
  }
  grep -Eq '"test_doubles"[[:space:]]*:[[:space:]]*false' "${PREFLIGHT}" || {
    echo "test-double hardware evidence is rejected" >&2; exit 2;
  }
fi

# Artifact hashing must see a quiescent result tree. Running this stage through
# the ordinary accounting wrappers would mutate their state files after the
# manifest was written. The immutable rental deadline remains authoritative,
# and each export invocation is additionally capped at its eight-hour slot.
if [[ "${STAGE}" == artifact-export ]]; then
  EXPORT_LIMIT="${STAGE_SECONDS}"
  ((REMAINING < EXPORT_LIMIT)) && EXPORT_LIMIT="${REMAINING}"
  set +e
  timeout --signal=TERM --kill-after=300s "${EXPORT_LIMIT}s" "${COMMAND[@]}"
  STATUS=$?
  set -e
  exit "${STATUS}"
fi

STAGE_BINDING="$(printf '%s\t%s\n' "${BINDING}" "${STAGE}" | sha256sum | awk '{ print $1 }')"
GLOBAL_STATE="${STATE_DIR}/overall.wall.state"
STAGE_STATE="${STATE_DIR}/${STAGE}.wall.state"
set +e
timeout --signal=TERM --kill-after=300s "${REMAINING}s" \
  "${ROOT}/scripts/run_with_wall_budget.sh" \
    --budget-seconds "${TOTAL_SECONDS}" --binding "${BINDING}" --state "${GLOBAL_STATE}" -- \
  "${ROOT}/scripts/run_with_wall_budget.sh" \
    --budget-seconds "${STAGE_SECONDS}" --binding "${STAGE_BINDING}" --state "${STAGE_STATE}" -- \
  "${COMMAND[@]}"
STATUS=$?
set -e
if ((STATUS == 124 || STATUS == 137)); then
  echo "campaign command reached the fixed 118-hour deadline" >&2
fi
exit "${STATUS}"
