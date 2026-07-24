#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BIN="${ROOT}/build/ubuntu-h100-cuda/solstice"
CHECKPOINT="${1:-${ROOT}/models/solstice_general_h100.rlfsp}"
TRAINING_MANIFEST="${2:-${ROOT}/results/codex_campaign/h100_training/training_artifact_manifest.tsv}"
RESULT_DIR="${RLF_SERVE_RESULTS:-${ROOT}/results/codex_campaign/h100_serving}"
[[ -x "${BIN}" ]] || { echo "Build the H100 target first." >&2; exit 1; }
[[ -f "${CHECKPOINT}" ]] || { echo "Checkpoint not found: ${CHECKPOINT}" >&2; exit 1; }
[[ -f "${TRAINING_MANIFEST}" ]] || { echo "Training manifest not found: ${TRAINING_MANIFEST}" >&2; exit 1; }
shift $(( $# >= 2 ? 2 : $# ))
mkdir -p "${RESULT_DIR}"
"${ROOT}/scripts/authorize_serving_checkpoint.sh" \
  "${CHECKPOINT}" "${TRAINING_MANIFEST}" "${BIN}" \
  | tee "${RESULT_DIR}/serving_authorization.txt"
{
  printf 'kind\tsha256\tpath\n'
  printf 'checkpoint\t%s\t%s\n' \
    "$(sha256sum -- "${CHECKPOINT}" | awk '{print $1}')" "$(realpath "${CHECKPOINT}")"
  printf 'training_manifest\t%s\t%s\n' \
    "$(sha256sum -- "${TRAINING_MANIFEST}" | awk '{print $1}')" "$(realpath "${TRAINING_MANIFEST}")"
} >"${RESULT_DIR}/serving_identity.tsv"
exec "${ROOT}/scripts/run_h100_with_vram_guard.sh" \
  --trace "${RESULT_DIR}/serving_vram.csv" \
  --summary "${RESULT_DIR}/serving_resource_summary.json" -- \
  "${BIN}" chat --checkpoint "${CHECKPOINT}" --profile general-h100 \
    --backend cuda --max-tokens 1024 --top-k 64 --temperature 0.75 "$@"
