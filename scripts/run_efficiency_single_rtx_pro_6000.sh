#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
usage() {
  cat <<'EOF'
Usage: run_efficiency_single_rtx_pro_6000.sh \
  LEDGER EVALUATION_REQUESTS EVALUATION_EXPECTED READINESS_REPORT \
  CHECKPOINT TRAINING_RESULT EVALUATION_RESULT SCORER

Runs one clean train -> checkpoint -> evaluation -> score repetition. This is
the internal worker used by the three-run physical campaign and performs real
training. Do not invoke it on a preparation-only host.
EOF
}

[[ $# -eq 8 ]] || { usage >&2; exit 2; }
LEDGER="$(realpath "${1}")"
EVALUATION_REQUESTS="$(realpath "${2}")"
EVALUATION_EXPECTED="$(realpath "${3}")"
READINESS_REPORT="$(realpath "${4}")"
CHECKPOINT="$(realpath -m "${5}")"
TRAINING_RESULT="$(realpath -m "${6}")"
EVALUATION_RESULT="$(realpath -m "${7}")"
SCORER="$(realpath "${8}")"

[[ -f "${LEDGER}" && -f "${EVALUATION_REQUESTS}" && \
   -f "${EVALUATION_EXPECTED}" && -f "${READINESS_REPORT}" && \
   -x "${SCORER}" ]] || { echo "single-run campaign inputs are missing" >&2; exit 2; }
[[ ! -e "${CHECKPOINT}" && ! -e "${TRAINING_RESULT}" && \
   ! -e "${EVALUATION_RESULT}" ]] || {
  echo "single-run campaign refuses existing outputs" >&2; exit 2
}

"${ROOT}/scripts/train_general_cuda_audited.sh" \
  "${LEDGER}" "${CHECKPOINT}" "${TRAINING_RESULT}" "${READINESS_REPORT}"
"${ROOT}/scripts/evaluate_general_cuda_audited.sh" \
  "${EVALUATION_REQUESTS}" "${EVALUATION_RESULT}" "${CHECKPOINT}" \
  "${TRAINING_RESULT}/training_artifact_manifest.tsv"
"${SCORER}" score-evaluation --expected "${EVALUATION_EXPECTED}" \
  --results "${EVALUATION_RESULT}" --output "${EVALUATION_RESULT}/scoring"
find "${EVALUATION_RESULT}" -type f ! -name complete_artifacts.sha256 -print0 |
  sort -z | xargs -0 sha256sum >"${EVALUATION_RESULT}/complete_artifacts.sha256"
