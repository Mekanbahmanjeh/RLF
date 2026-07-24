#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  v100_1586h_controller.sh --plan
  v100_1586h_controller.sh --start --state-dir DIR --binding SHA256
  v100_1586h_controller.sh --status --state-dir DIR
  v100_1586h_controller.sh --run-stage NAME --state-dir DIR --binding SHA256 \
    [--evidence-dir DIR --manifest FILE --authorization FILE] \
    [--ledger FILE --checkpoint FILE --result-dir DIR --readiness-report FILE] \
    [-- COMMAND [ARGS...]]

The default action is --plan and executes nothing. Training stages require a
current authorization ticket plus the hashed physical evidence bundle from
which it was issued. Exact training stages invoke train_general_cuda_audited.sh
directly and do not accept an arbitrary command. Non-training stages require a
command after --. This controller neither rents hardware nor claims frontier
capability.
EOF
}

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PROFILE=general-v100-32g-500m
TOTAL_SECONDS=5709600
ACTION=plan
STAGE=""
STATE_DIR=""
BINDING=""
EVIDENCE_DIR=""
MANIFEST=""
AUTHORIZATION=""
LEDGER=""
CHECKPOINT=""
RESULT_DIR=""
READINESS_REPORT=""
COMMAND=()

while (($# > 0)); do
  case "$1" in
    --plan) ACTION=plan; shift ;;
    --start) ACTION=start; shift ;;
    --status) ACTION=status; shift ;;
    --run-stage) ACTION=run; STAGE="${2:?--run-stage requires a name}"; shift 2 ;;
    --state-dir) STATE_DIR="${2:?--state-dir requires a directory}"; shift 2 ;;
    --binding) BINDING="${2:?--binding requires SHA-256}"; shift 2 ;;
    --evidence-dir) EVIDENCE_DIR="${2:?--evidence-dir requires a directory}"; shift 2 ;;
    --manifest) MANIFEST="${2:?--manifest requires a file}"; shift 2 ;;
    --authorization) AUTHORIZATION="${2:?--authorization requires a file}"; shift 2 ;;
    --ledger) LEDGER="${2:?--ledger requires a file}"; shift 2 ;;
    --checkpoint) CHECKPOINT="${2:?--checkpoint requires a file}"; shift 2 ;;
    --result-dir) RESULT_DIR="${2:?--result-dir requires a directory}"; shift 2 ;;
    --readiness-report) READINESS_REPORT="${2:?--readiness-report requires a file}"; shift 2 ;;
    --) shift; COMMAND=("$@"); break ;;
    --help|-h) usage; exit 0 ;;
    *) echo "unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

print_plan() {
  cat <<'EOF'
profile=general-v100-32g-500m
total_budget_hours=1586
total_budget_seconds=5709600
primary_target_records=200000000
conditional_target_records=500000000
training_authorized=false

stage                              hours  records      authorization
hardware-preflight                    12  0            none; no training
cuda12-sm70-build-test                24  0            none; no training
data-provenance-audit                 96  0            none; no training
physical-throughput-probe             24  0            authorized bounded scratch probes only
train-50m                            250  50000000     hashed physical probe evidence
recovery-frozen-eval-50m              80  0            no scale promotion
train-200m-primary                   600  200000000    completed 50M evidence
recovery-frozen-eval-200m            120  0            no scale promotion
train-500m-promoted                  300  500000000    completed 50M or 200M evidence
final-external-evaluation             50  0            no frontier claim by default
artifact-export                       30  0            verify hashes before teardown
TOTAL                               1586

The 200M run is the primary target. The reserved 500M stage is conditional and
remains inaccessible unless the promotion gate verifies zero capacity skips,
checkpoint/resume recovery, <=30 GiB peak VRAM, non-regressing frozen quality,
20% timing/resource headroom, exact stage counts, and cross-bound hashes.
EOF
}

stage_seconds() {
  case "$1" in
    hardware-preflight) echo $((12 * 3600)) ;;
    cuda12-sm70-build-test) echo $((24 * 3600)) ;;
    data-provenance-audit) echo $((96 * 3600)) ;;
    physical-throughput-probe) echo $((24 * 3600)) ;;
    train-50m) echo $((250 * 3600)) ;;
    recovery-frozen-eval-50m) echo $((80 * 3600)) ;;
    train-200m-primary) echo $((600 * 3600)) ;;
    recovery-frozen-eval-200m) echo $((120 * 3600)) ;;
    train-500m-promoted) echo $((300 * 3600)) ;;
    final-external-evaluation) echo $((50 * 3600)) ;;
    artifact-export) echo $((30 * 3600)) ;;
    *) return 1 ;;
  esac
}

target_records() {
  case "$1" in
    train-50m) echo 50000000 ;;
    train-200m-primary) echo 200000000 ;;
    train-500m-promoted) echo 500000000 ;;
    *) return 1 ;;
  esac
}

state_value() {
  awk -F= -v key="$2" '$1 == key { print substr($0, length(key) + 2); exit }' "$1"
}

if [[ "${ACTION}" == plan ]]; then
  ((${#COMMAND[@]} == 0)) || { echo "--plan never accepts a command" >&2; exit 2; }
  print_plan
  exit 0
fi

[[ -n "${STATE_DIR}" ]] || { echo "--state-dir is required" >&2; exit 2; }
STATE_DIR="$(realpath -m "${STATE_DIR}")"
META="${STATE_DIR}/campaign.meta"

if [[ "${ACTION}" == start ]]; then
  ((${#COMMAND[@]} == 0)) || { echo "--start never accepts a command" >&2; exit 2; }
  [[ "${BINDING}" =~ ^[0-9a-f]{64}$ ]] || { echo "--binding must be lowercase SHA-256" >&2; exit 2; }
  [[ ! -e "${META}" ]] || { echo "campaign state already exists" >&2; exit 2; }
  mkdir -p "${STATE_DIR}"
  TEMPORARY="$(mktemp "${STATE_DIR}/.v100-campaign.XXXXXX")"
  {
    printf 'schema=rlf-v100-1586h-campaign-v1\n'
    printf 'profile=%s\n' "${PROFILE}"
    printf 'binding_sha256=%s\n' "${BINDING}"
    printf 'total_budget_seconds=%s\n' "${TOTAL_SECONDS}"
    printf 'primary_target_records=200000000\n'
    printf 'conditional_target_records=500000000\n'
    printf 'started_epoch=%s\n' "$(date +%s)"
    printf 'frontier_claim_authorized=false\n'
  } >"${TEMPORARY}"
  mv -f -- "${TEMPORARY}" "${META}"
  printf 'campaign_started=true\nprofile=%s\ntotal_budget_seconds=%s\n' "${PROFILE}" "${TOTAL_SECONDS}"
  exit 0
fi

[[ -f "${META}" && ! -L "${META}" ]] || { echo "campaign was not started" >&2; exit 2; }
[[ "$(state_value "${META}" schema)" == rlf-v100-1586h-campaign-v1 && \
   "$(state_value "${META}" profile)" == "${PROFILE}" && \
   "$(state_value "${META}" total_budget_seconds)" == "${TOTAL_SECONDS}" ]] || {
  echo "invalid V100 campaign state" >&2; exit 2;
}
META_BINDING="$(state_value "${META}" binding_sha256)"
[[ "${META_BINDING}" =~ ^[0-9a-f]{64}$ ]] || { echo "invalid campaign binding" >&2; exit 2; }

consumed_or_zero() {
  local file="$1" value=0
  if [[ -f "${file}" ]]; then
    value="$(state_value "${file}" consumed_seconds)"
    [[ "${value}" =~ ^[0-9]+$ ]] || { echo "invalid wall-budget state" >&2; exit 2; }
  fi
  printf '%s' "${value}"
}

if [[ "${ACTION}" == status ]]; then
  ((${#COMMAND[@]} == 0)) || { echo "--status never accepts a command" >&2; exit 2; }
  OVERALL_CONSUMED="$(consumed_or_zero "${STATE_DIR}/overall.wall.state")"
  ((OVERALL_CONSUMED <= TOTAL_SECONDS)) || { echo "overall budget state exceeds its ceiling" >&2; exit 2; }
  printf 'profile=%s\nbinding_sha256=%s\ntotal_budget_seconds=%s\nconsumed_seconds=%s\nremaining_seconds=%s\n' \
    "${PROFILE}" "${META_BINDING}" "${TOTAL_SECONDS}" "${OVERALL_CONSUMED}" \
    "$((TOTAL_SECONDS - OVERALL_CONSUMED))"
  for stage_name in hardware-preflight cuda12-sm70-build-test data-provenance-audit \
    physical-throughput-probe train-50m recovery-frozen-eval-50m \
    train-200m-primary recovery-frozen-eval-200m train-500m-promoted \
    final-external-evaluation artifact-export; do
    printf 'stage_%s_consumed_seconds=%s\n' "${stage_name}" \
      "$(consumed_or_zero "${STATE_DIR}/${stage_name}.wall.state")"
  done
  exit 0
fi

[[ "${ACTION}" == run ]] || { usage >&2; exit 2; }
[[ "${BINDING}" == "${META_BINDING}" ]] || { echo "run binding does not match campaign" >&2; exit 2; }
STAGE_SECONDS="$(stage_seconds "${STAGE}")" || { echo "unsupported V100 campaign stage" >&2; exit 2; }
TRAINING_AUTHORIZED=0
AUTHORIZED_RECORDS=0
if target_records "${STAGE}" >/dev/null 2>&1; then
  AUTHORIZED_RECORDS="$(target_records "${STAGE}")"
  ((${#COMMAND[@]} == 0)) || {
    echo "training stages do not accept an arbitrary command" >&2; exit 2;
  }
  [[ -n "${EVIDENCE_DIR}" && -n "${MANIFEST}" && -n "${AUTHORIZATION}" ]] || {
    echo "training stage requires evidence, manifest, and authorization ticket" >&2; exit 5;
  }
  [[ -f "${LEDGER}" && -f "${READINESS_REPORT}" && -n "${CHECKPOINT}" && -n "${RESULT_DIR}" ]] || {
    echo "training stage requires --ledger, --checkpoint, --result-dir, and --readiness-report" >&2
    exit 2
  }
  "${ROOT}/scripts/v100/v100_scale_promotion_gate.sh" verify \
    --campaign-state "${STATE_DIR}" --evidence-dir "${EVIDENCE_DIR}" \
    --manifest "${MANIFEST}" --target-records "${AUTHORIZED_RECORDS}" \
    --ticket "${AUTHORIZATION}" >/dev/null
  TRAINING_AUTHORIZED=1
  COMMAND=("${ROOT}/scripts/train_general_cuda_audited.sh" \
    "${LEDGER}" "${CHECKPOINT}" "${RESULT_DIR}" "${READINESS_REPORT}")
else
  [[ -z "${EVIDENCE_DIR}${MANIFEST}${AUTHORIZATION}" ]] || {
    echo "authorization inputs are accepted only for exact training stages" >&2; exit 2;
  }
  [[ -z "${LEDGER}${CHECKPOINT}${RESULT_DIR}${READINESS_REPORT}" ]] || {
    echo "audited training inputs are accepted only for exact training stages" >&2; exit 2;
  }
  ((${#COMMAND[@]} > 0)) || { echo "non-training stage requires a command after --" >&2; exit 2; }
fi

STAGE_BINDING="$(printf '%s\t%s\n' "${BINDING}" "${STAGE}" | sha256sum | awk '{ print $1 }')"
GLOBAL_STATE="${STATE_DIR}/overall.wall.state"
STAGE_STATE="${STATE_DIR}/${STAGE}.wall.state"
set +e
"${ROOT}/scripts/run_with_wall_budget.sh" \
  --budget-seconds "${TOTAL_SECONDS}" --binding "${BINDING}" --state "${GLOBAL_STATE}" -- \
  "${ROOT}/scripts/run_with_wall_budget.sh" \
    --budget-seconds "${STAGE_SECONDS}" --binding "${STAGE_BINDING}" --state "${STAGE_STATE}" -- \
  env RLF_TRAINING_AUTHORIZED="${TRAINING_AUTHORIZED}" \
    RLF_CAMPAIGN_PROFILE="${PROFILE}" \
    RLF_AUTHORIZED_RECORDS="${AUTHORIZED_RECORDS}" \
    RLF_CAMPAIGN_BINDING_SHA256="${BINDING}" \
    RLF_CAMPAIGN_STATE_DIR="${STATE_DIR}" \
    RLF_CAMPAIGN_STAGE="${STAGE}" \
    "${COMMAND[@]}"
STATUS=$?
set -e
exit "${STATUS}"
