#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORK="${ROOT}/results/training_data_layout_shell_test"
[[ "$(realpath -m "${WORK}")" == "$(realpath "${ROOT}")/results/training_data_layout_shell_test" ]]
rm -rf -- "${WORK}"
trap 'rm -rf -- "${WORK}"' EXIT

"${ROOT}/scripts/create_training_data_layout.sh" "${WORK}"
for profile in v100-multimodal v100-text pro6000-multimodal pro6000-text video-pro6000; do
  [[ -f "${WORK}/${profile}/ledger.tsv.template" ]]
done
[[ -f "${WORK}/v100-multimodal/shards/vision.tsv.template" ]]
[[ -f "${WORK}/pro6000-multimodal/shards/vision.tsv.template" ]]
[[ -f "${WORK}/video-pro6000/shards/video_frames.tsv.template" ]]
[[ -f "${WORK}/imagegen-language/ledger.tsv.template" ]]

printf 'sentinel\n' >"${WORK}/pro6000-text/ledger.tsv.template"
"${ROOT}/scripts/create_training_data_layout.sh" "${WORK}"
grep -qx sentinel "${WORK}/pro6000-text/ledger.tsv.template"

awk -F '\t' 'BEGIN { ok=1 } !/^#/ && NF && NF != 16 { print FILENAME ":" FNR ":" NF; ok=0 } END { exit !ok }' \
  "${ROOT}"/data_templates/*/ledger.tsv.template
awk -F '\t' 'BEGIN { ok=1 } !/^#/ && NF && NF != 3 { print FILENAME ":" FNR ":" NF; ok=0 } END { exit !ok }' \
  "${ROOT}"/data_templates/*/vision.tsv.template
awk -F '\t' 'BEGIN { ok=1 } !/^#/ && NF && NF != 7 { print FILENAME ":" FNR ":" NF; ok=0 } END { exit !ok }' \
  "${ROOT}"/data_templates/*/video_frames.tsv.template
