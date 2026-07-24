#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  h200_12month_controller.sh --plan
  h200_12month_controller.sh --start --state-dir DIR --binding SHA256
  h200_12month_controller.sh --status --state-dir DIR
  h200_12month_controller.sh --run-stage NAME --state-dir DIR --binding SHA256 -- COMMAND...

The controller binds a single-H200, 8,760-hour campaign. Training stages are
deliberately fail-closed until a physical token-throughput promotion gate and
an audited H200 trainer authorize the exact cumulative tokenizer-piece target.
EOF
}

PROFILE=general-h200-141g-30t
TOTAL_SECONDS=31536000
ACTION=plan
STAGE=""
STATE_DIR=""
BINDING=""
COMMAND=()

while (($# > 0)); do
  case "$1" in
    --plan) ACTION=plan; shift ;;
    --start) ACTION=start; shift ;;
    --status) ACTION=status; shift ;;
    --run-stage) ACTION=run; STAGE="${2:?stage required}"; shift 2 ;;
    --state-dir) STATE_DIR="${2:?directory required}"; shift 2 ;;
    --binding) BINDING="${2:?SHA-256 required}"; shift 2 ;;
    --) shift; COMMAND=("$@"); break ;;
    --help|-h) usage; exit 0 ;;
    *) echo "unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

print_plan() {
  cat <<'EOF'
profile=general-h200-141g-30t
total_budget_hours=8760
total_budget_seconds=31536000
primary_target_tokens=15000000000000
conditional_target_tokens=30000000000000
training_authorized=false

stage                                  hours  cumulative_tokens  authorization
hardware-preflight                        24  0                  none; no training
cuda12-sm90-build-test                    48  0                  none; no training
data-provenance-audit                    480  0                  none; no training
token-census-contamination-audit         360  0                  none; no training
physical-throughput-resume-probe          72  0                  bounded probes only
train-1t-calibration                     500  1000000000000      physical token evidence
recovery-frozen-eval-1t                  120  0                  no scale promotion
train-5t                                1500  5000000000000      completed 1T evidence
recovery-frozen-eval-5t                  180  0                  no scale promotion
train-15t-primary                       2400  15000000000000     completed 5T evidence
recovery-frozen-eval-15t                 240  0                  no scale promotion
train-30t-promoted                      1800  30000000000000     completed 15T evidence
final-external-evaluation                480  0                  no frontier claim by default
artifact-export                          216  0                  verify and export
recovery-reserve                         340  0                  failures and reprocessing
TOTAL                                   8760

The 15T stage is the primary goal. The 30T stage remains conditional on zero
capacity skips, byte-identical resume, <=132 GiB peak VRAM, non-regressing
frozen multimodal quality, complete provenance, and measured timing headroom.
EOF
}

stage_seconds() {
  case "$1" in
    hardware-preflight) echo $((24 * 3600)) ;;
    cuda12-sm90-build-test) echo $((48 * 3600)) ;;
    data-provenance-audit) echo $((480 * 3600)) ;;
    token-census-contamination-audit) echo $((360 * 3600)) ;;
    physical-throughput-resume-probe) echo $((72 * 3600)) ;;
    train-1t-calibration) echo $((500 * 3600)) ;;
    recovery-frozen-eval-1t) echo $((120 * 3600)) ;;
    train-5t) echo $((1500 * 3600)) ;;
    recovery-frozen-eval-5t) echo $((180 * 3600)) ;;
    train-15t-primary) echo $((2400 * 3600)) ;;
    recovery-frozen-eval-15t) echo $((240 * 3600)) ;;
    train-30t-promoted) echo $((1800 * 3600)) ;;
    final-external-evaluation) echo $((480 * 3600)) ;;
    artifact-export) echo $((216 * 3600)) ;;
    recovery-reserve) echo $((340 * 3600)) ;;
    *) return 1 ;;
  esac
}

is_training_stage() {
  case "$1" in
    train-1t-calibration|train-5t|train-15t-primary|train-30t-promoted) return 0 ;;
    *) return 1 ;;
  esac
}

value() {
  awk -F= -v key="$2" '$1 == key {print substr($0,length(key)+2); exit}' "$1"
}

if [[ "${ACTION}" == plan ]]; then
  ((${#COMMAND[@]} == 0)) || { echo "--plan accepts no command" >&2; exit 2; }
  print_plan
  exit 0
fi

[[ -n "${STATE_DIR}" ]] || { echo "--state-dir is required" >&2; exit 2; }
STATE_DIR="$(realpath -m "${STATE_DIR}")"
META="${STATE_DIR}/campaign.meta"

if [[ "${ACTION}" == start ]]; then
  [[ "${BINDING}" =~ ^[0-9a-f]{64}$ && ! -e "${META}" ]] || {
    echo "unique lowercase SHA-256 binding required" >&2
    exit 2
  }
  mkdir -p "${STATE_DIR}"
  TEMP="$(mktemp "${STATE_DIR}/.h200-campaign.XXXXXX")"
  {
    echo schema=rlf-h200-12month-campaign-v1
    echo profile=${PROFILE}
    echo binding_sha256=${BINDING}
    echo total_budget_seconds=${TOTAL_SECONDS}
    echo primary_target_tokens=15000000000000
    echo conditional_target_tokens=30000000000000
    echo started_epoch="$(date +%s)"
    echo frontier_claim_authorized=false
  } >"${TEMP}"
  mv -f -- "${TEMP}" "${META}"
  echo campaign_started=true
  exit 0
fi

[[ -f "${META}" && ! -L "${META}" ]] || {
  echo "campaign was not started" >&2
  exit 2
}
[[ "$(value "${META}" schema)" == rlf-h200-12month-campaign-v1 &&
   "$(value "${META}" profile)" == "${PROFILE}" &&
   "$(value "${META}" total_budget_seconds)" == "${TOTAL_SECONDS}" ]] || {
  echo "invalid H200 campaign state" >&2
  exit 2
}

if [[ "${ACTION}" == status ]]; then
  CONSUMED=0
  [[ ! -f "${STATE_DIR}/overall.wall.state" ]] ||
    CONSUMED="$(value "${STATE_DIR}/overall.wall.state" consumed_seconds)"
  [[ "${CONSUMED}" =~ ^[0-9]+$ && "${CONSUMED}" -le "${TOTAL_SECONDS}" ]] || exit 2
  echo profile=${PROFILE}
  echo total_budget_seconds=${TOTAL_SECONDS}
  echo consumed_seconds=${CONSUMED}
  echo remaining_seconds=$((TOTAL_SECONDS - CONSUMED))
  exit 0
fi

[[ "${ACTION}" == run && "${BINDING}" == "$(value "${META}" binding_sha256)" ]] || {
  echo "run binding does not match campaign" >&2
  exit 2
}
STAGE_SECONDS="$(stage_seconds "${STAGE}")" || {
  echo "unsupported H200 campaign stage" >&2
  exit 2
}
if is_training_stage "${STAGE}"; then
  echo "H200 training stages require the physical token promotion gate, which is not yet implemented" >&2
  exit 5
fi
((${#COMMAND[@]} > 0)) || {
  echo "non-training stage requires -- COMMAND" >&2
  exit 2
}
STAGE_BINDING="$(printf '%s\t%s\n' "${BINDING}" "${STAGE}" | sha256sum | awk '{print $1}')"
bash "$(dirname "${BASH_SOURCE[0]}")/../run_with_wall_budget.sh" \
  --budget-seconds "${TOTAL_SECONDS}" --binding "${BINDING}" \
  --state "${STATE_DIR}/overall.wall.state" -- \
  bash "$(dirname "${BASH_SOURCE[0]}")/../run_with_wall_budget.sh" \
  --budget-seconds "${STAGE_SECONDS}" --binding "${STAGE_BINDING}" \
  --state "${STATE_DIR}/${STAGE}.wall.state" -- \
  env RLF_TRAINING_AUTHORIZED=0 RLF_CAMPAIGN_PROFILE="${PROFILE}" \
    RLF_CAMPAIGN_BINDING_SHA256="${BINDING}" RLF_CAMPAIGN_STAGE="${STAGE}" \
    "${COMMAND[@]}"
