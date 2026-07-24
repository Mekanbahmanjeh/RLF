#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  imagegen_40h_controller.sh --plan
  imagegen_40h_controller.sh --start --state-dir DIR --binding SHA256 \
    --language-ledger FILE
  imagegen_40h_controller.sh --status --state-dir DIR
  imagegen_40h_controller.sh --run-stage NAME --state-dir DIR --binding SHA256 \
    [--pair-manifest FILE --result-dir DIR --readiness-report FILE \
     --audit-dir DIR --resume-report FILE --language-ledger FILE \
     --language-records N] [-- COMMAND...]

The default is --plan and performs no work. Only controlled-training invokes
the audited image trainer. Other stages require an explicit command after --.
The cumulative physical wall-clock ceiling is exactly 144000 seconds.
EOF
}

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PROFILE=imagegen-v100-32g
TOTAL_SECONDS=144000
ACTION=plan; STAGE=""; STATE_DIR=""; BINDING=""
PAIR_MANIFEST=""; RESULT_DIR=""; READINESS=""; AUDIT_DIR=""; RESUME=""
LANGUAGE_LEDGER=""; LANGUAGE_RECORDS=""
COMMAND=()
while (($# > 0)); do
  case "$1" in
    --plan) ACTION=plan; shift ;;
    --start) ACTION=start; shift ;;
    --status) ACTION=status; shift ;;
    --run-stage) ACTION=run; STAGE="${2:?stage required}"; shift 2 ;;
    --state-dir) STATE_DIR="${2:?directory required}"; shift 2 ;;
    --binding) BINDING="${2:?SHA-256 required}"; shift 2 ;;
    --pair-manifest) PAIR_MANIFEST="${2:?file required}"; shift 2 ;;
    --result-dir) RESULT_DIR="${2:?directory required}"; shift 2 ;;
    --readiness-report) READINESS="${2:?file required}"; shift 2 ;;
    --audit-dir) AUDIT_DIR="${2:?directory required}"; shift 2 ;;
    --resume-report) RESUME="${2:?file required}"; shift 2 ;;
    --language-ledger) LANGUAGE_LEDGER="${2:?file required}"; shift 2 ;;
    --language-records) LANGUAGE_RECORDS="${2:?count required}"; shift 2 ;;
    --) shift; COMMAND=("$@"); break ;;
    --help|-h) usage; exit 0 ;;
    *) echo "unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

plan() {
  cat <<'EOF'
profile=imagegen-v100-32g
total_budget_hours=40
total_budget_seconds=144000
training_authorized=false

stage                    hours  purpose
hardware-preflight           2  physical identity, storage, CUDA 12/sm_70
cuda-build-full-tests         3  Release build and complete test suite
native-data-audit             4  provenance, licenses, exact/near/perceptual dedup
resume-vram-probe             4  physical CUDA, <=30 GiB, byte-identical resume
prompt-semantic-training      6  audited exact-target text-context learning
controlled-training          11  audited 5K-or-smaller image-pair experiment
frozen-evaluation             6  held-out quality/diversity/composition/memorization
artifact-export               4  hashes, checkpoint verification, off-host copy
TOTAL                        40

This is a bounded non-neural image-transformation experiment. It does not
authorize diffusion parity, frontier image generation, or an efficiency claim.
EOF
}
stage_seconds() {
  case "$1" in
    hardware-preflight) echo 7200 ;;
    cuda-build-full-tests) echo 10800 ;;
    native-data-audit) echo 14400 ;;
    resume-vram-probe) echo 14400 ;;
    prompt-semantic-training) echo 21600 ;;
    controlled-training) echo 39600 ;;
    frozen-evaluation) echo 21600 ;;
    artifact-export) echo 14400 ;;
    *) return 1 ;;
  esac
}
value() { awk -F= -v key="$2" '$1 == key {print substr($0,length(key)+2); exit}' "$1"; }
consumed() { [[ -f "$1" ]] && value "$1" consumed_seconds || echo 0; }

if [[ "${ACTION}" == plan ]]; then
  ((${#COMMAND[@]} == 0)) || { echo "--plan accepts no command" >&2; exit 2; }
  plan; exit 0
fi
[[ -n "${STATE_DIR}" ]] || { echo "--state-dir is required" >&2; exit 2; }
STATE_DIR="$(realpath -m "${STATE_DIR}")"; META="${STATE_DIR}/campaign.meta"
if [[ "${ACTION}" == start ]]; then
  [[ "${BINDING}" =~ ^[0-9a-f]{64}$ && ! -e "${META}" &&
     -f "${LANGUAGE_LEDGER}" && ! -L "${LANGUAGE_LEDGER}" ]] || {
    echo "unique binding and regular prompt-language ledger required" >&2; exit 2;
  }
  LANGUAGE_LEDGER_SHA="$(sha256sum -- "${LANGUAGE_LEDGER}" | awk '{print $1}')"
  mkdir -p "${STATE_DIR}"
  TEMP="$(mktemp "${STATE_DIR}/.imagegen-40h.XXXXXX")"
  {
    echo schema=rlf-imagegen-v100-40h-campaign-v1
    echo profile=${PROFILE}
    echo binding_sha256=${BINDING}
    echo prompt_language_ledger_sha256=${LANGUAGE_LEDGER_SHA}
    echo total_budget_seconds=${TOTAL_SECONDS}
    echo started_epoch="$(date +%s)"
    echo frontier_claim_authorized=false
  } >"${TEMP}"
  mv -f -- "${TEMP}" "${META}"
  echo campaign_started=true; echo total_budget_seconds=${TOTAL_SECONDS}; exit 0
fi
[[ -f "${META}" && ! -L "${META}" ]] || { echo "campaign was not started" >&2; exit 2; }
[[ "$(value "${META}" schema)" == rlf-imagegen-v100-40h-campaign-v1 &&
   "$(value "${META}" profile)" == "${PROFILE}" &&
   "$(value "${META}" prompt_language_ledger_sha256)" =~ ^[0-9a-f]{64}$ &&
   "$(value "${META}" total_budget_seconds)" == "${TOTAL_SECONDS}" ]] || {
  echo "invalid image campaign state" >&2; exit 2;
}
META_BINDING="$(value "${META}" binding_sha256)"
if [[ "${ACTION}" == status ]]; then
  USED="$(consumed "${STATE_DIR}/overall.wall.state")"
  [[ "${USED}" =~ ^[0-9]+$ ]] && ((USED <= TOTAL_SECONDS)) || exit 2
  echo total_budget_seconds=${TOTAL_SECONDS}; echo consumed_seconds=${USED}
  echo remaining_seconds=$((TOTAL_SECONDS - USED)); exit 0
fi
[[ "${ACTION}" == run && "${BINDING}" == "${META_BINDING}" ]] || {
  echo "run binding does not match campaign" >&2; exit 2;
}
SECONDS_FOR_STAGE="$(stage_seconds "${STAGE}")" || {
  echo "unsupported image campaign stage" >&2; exit 2;
}
if [[ "${STAGE}" == prompt-semantic-training ]]; then
  ((${#COMMAND[@]} == 0)) || { echo "prompt training rejects arbitrary commands" >&2; exit 2; }
  [[ -f "${LANGUAGE_LEDGER}" && -f "${READINESS}" && -n "${RESULT_DIR}" &&
     "${LANGUAGE_RECORDS}" =~ ^[1-9][0-9]*$ ]] || {
    echo "prompt training requires language ledger, exact records, readiness, and result dir" >&2
    exit 2
  }
  [[ "$(sha256sum -- "${LANGUAGE_LEDGER}" | awk '{print $1}')" == \
     "$(value "${META}" prompt_language_ledger_sha256)" ]] || {
    echo "prompt-language ledger changed after campaign start" >&2; exit 1;
  }
  [[ -z "${PAIR_MANIFEST}${AUDIT_DIR}${RESUME}" ]] || {
    echo "image training inputs are forbidden during prompt training" >&2; exit 2;
  }
  COMMAND=(bash "${ROOT}/scripts/v100/train_imagegen_prompt_semantics.sh"
    "${LANGUAGE_LEDGER}" "${LANGUAGE_RECORDS}" "${RESULT_DIR}" "${READINESS}")
elif [[ "${STAGE}" == controlled-training ]]; then
  ((${#COMMAND[@]} == 0)) || { echo "training rejects arbitrary commands" >&2; exit 2; }
  [[ -f "${PAIR_MANIFEST}" && -f "${READINESS}" && -d "${AUDIT_DIR}" &&
     -f "${RESUME}" && -n "${RESULT_DIR}" ]] || {
    echo "training requires pair manifest, readiness, audits, resume report, and result dir" >&2
    exit 2
  }
  COMMAND=(bash "${ROOT}/scripts/train_imagegen_v100_audited.sh"
    "${PAIR_MANIFEST}" "${RESULT_DIR}" "${READINESS}" "${AUDIT_DIR}" "${RESUME}")
  [[ -z "${LANGUAGE_LEDGER}${LANGUAGE_RECORDS}" ]] || {
    echo "prompt training inputs are forbidden during image training" >&2; exit 2;
  }
else
  [[ -z "${PAIR_MANIFEST}${RESULT_DIR}${READINESS}${AUDIT_DIR}${RESUME}${LANGUAGE_LEDGER}${LANGUAGE_RECORDS}" ]] || {
    echo "training inputs are accepted only by controlled-training" >&2; exit 2;
  }
  ((${#COMMAND[@]} > 0)) || { echo "non-training stage requires -- COMMAND" >&2; exit 2; }
fi
STAGE_BINDING="$(printf '%s\t%s\n' "${BINDING}" "${STAGE}" | sha256sum | awk '{print $1}')"
"${ROOT}/scripts/run_with_wall_budget.sh" \
  --budget-seconds "${TOTAL_SECONDS}" --binding "${BINDING}" \
  --state "${STATE_DIR}/overall.wall.state" -- \
  "${ROOT}/scripts/run_with_wall_budget.sh" \
    --budget-seconds "${SECONDS_FOR_STAGE}" --binding "${STAGE_BINDING}" \
    --state "${STATE_DIR}/${STAGE}.wall.state" -- \
  env RLF_TRAINING_AUTHORIZED=$([[ "${STAGE}" == controlled-training || "${STAGE}" == prompt-semantic-training ]] && echo 1 || echo 0) \
    RLF_CAMPAIGN_PROFILE="${PROFILE}" RLF_CAMPAIGN_BINDING_SHA256="${BINDING}" \
    RLF_CAMPAIGN_STATE_DIR="${STATE_DIR}" RLF_CAMPAIGN_STAGE="${STAGE}" \
    "${COMMAND[@]}"
