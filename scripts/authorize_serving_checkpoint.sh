#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CHECKPOINT="${1:?Usage: authorize_serving_checkpoint.sh CHECKPOINT TRAINING_MANIFEST [SOLSTICE_BIN]}"
MANIFEST="${2:?Usage: authorize_serving_checkpoint.sh CHECKPOINT TRAINING_MANIFEST [SOLSTICE_BIN]}"
BIN="${3:-}"

[[ -f "${CHECKPOINT}" ]] || { echo "Checkpoint not found: ${CHECKPOINT}" >&2; exit 2; }
[[ -f "${MANIFEST}" ]] || { echo "Training manifest not found: ${MANIFEST}" >&2; exit 2; }
CHECKPOINT="$(realpath "${CHECKPOINT}")"
MANIFEST="$(realpath "${MANIFEST}")"

"${ROOT}/scripts/create_training_artifact_manifest.sh" --verify "${MANIFEST}" >/dev/null

COUNT=0
AUTHORIZED_PATH=""
READINESS_COUNT=0
READINESS_PATH=""
while IFS=$'\t' read -r kind hash bytes path; do
  : "${hash}" "${bytes}"
  if [[ "${kind}" == checkpoint ]]; then
    COUNT=$((COUNT + 1))
    AUTHORIZED_PATH="${path}"
  elif [[ "${kind}" == readiness_report ]]; then
    READINESS_COUNT=$((READINESS_COUNT + 1))
    READINESS_PATH="${path}"
  fi
done <"${MANIFEST}"
((COUNT == 1)) || { echo "Training manifest must contain exactly one checkpoint" >&2; exit 1; }
((READINESS_COUNT == 1)) || { echo "Training manifest must contain exactly one readiness report" >&2; exit 1; }
[[ -f "${AUTHORIZED_PATH}" ]] || { echo "Authorized checkpoint path is missing" >&2; exit 1; }
AUTHORIZED_PATH="$(realpath "${AUTHORIZED_PATH}")"
[[ "${CHECKPOINT}" == "${AUTHORIZED_PATH}" ]] || {
  echo "Requested checkpoint is not the training-manifest checkpoint" >&2
  exit 1
}

if [[ -n "${BIN}" ]]; then
  [[ -x "${BIN}" ]] || { echo "Solstice binary is not executable: ${BIN}" >&2; exit 2; }
  [[ -f "${READINESS_PATH}" ]] || { echo "Manifest readiness report is missing" >&2; exit 1; }
  PROFILE_COUNT="$(grep -Ec '"profile"[[:space:]]*:[[:space:]]*"[^"]+"' "${READINESS_PATH}")"
  ((PROFILE_COUNT == 1)) || { echo "Readiness report must bind exactly one profile" >&2; exit 1; }
  PROFILE="$(sed -nE 's/.*"profile"[[:space:]]*:[[:space:]]*"([^"]+)".*/\1/p' "${READINESS_PATH}")"
  case "${PROFILE}" in
    general-h100|general-40g|general-v100-32g|general-v100-32g-text|general-v100-32g-500m|video-v100-32g|rtx-pro-6000-96g|general-rtx-pro-6000-96g|general-rtx-pro-6000-96g-text|video-rtx-pro-6000-96g) ;;
    *) echo "Readiness report binds an unsupported serving profile" >&2; exit 1 ;;
  esac
  "${BIN}" verify-checkpoint --checkpoint "${CHECKPOINT}" \
    --profile "${PROFILE}" --enforce-profile >/dev/null
fi
printf 'Authorized serving checkpoint: %s\n' "${CHECKPOINT}"
