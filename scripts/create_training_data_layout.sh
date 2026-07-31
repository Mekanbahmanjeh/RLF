#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_ROOT="${1:-/data/rlf}"
mkdir -p "${DATA_ROOT}/v100-multimodal/shards" "${DATA_ROOT}/v100-multimodal/media/images" \
  "${DATA_ROOT}/v100-text/shards" \
  "${DATA_ROOT}/pro6000-multimodal/shards" "${DATA_ROOT}/pro6000-multimodal/media/images" \
  "${DATA_ROOT}/pro6000-text/shards" \
  "${DATA_ROOT}/video-pro6000/shards" "${DATA_ROOT}/video-pro6000/media/frames" \
  "${DATA_ROOT}/imagegen/train/images" \
  "${DATA_ROOT}/imagegen/development/images" \
  "${DATA_ROOT}/imagegen/evaluation/images" \
  "${DATA_ROOT}/imagegen-language/shards"
for profile in v100-multimodal v100-text pro6000-multimodal pro6000-text video-pro6000; do
  target="${DATA_ROOT}/${profile}/ledger.tsv.template"
  [[ -e "${target}" ]] || cp "${ROOT}/data_templates/${profile}/ledger.tsv.template" "${target}"
done
target="${DATA_ROOT}/v100-multimodal/shards/vision.tsv.template"
[[ -e "${target}" ]] || cp "${ROOT}/data_templates/v100-multimodal/vision.tsv.template" "${target}"
target="${DATA_ROOT}/pro6000-multimodal/shards/vision.tsv.template"
[[ -e "${target}" ]] || cp "${ROOT}/data_templates/pro6000-multimodal/vision.tsv.template" "${target}"
target="${DATA_ROOT}/video-pro6000/shards/video_frames.tsv.template"
[[ -e "${target}" ]] || cp "${ROOT}/data_templates/video-pro6000/video_frames.tsv.template" "${target}"
for mapping in \
  licenses.txt.template:licenses.txt.template \
  train_pairs.tsv.template:train/train_pairs.tsv.template \
  development_pairs.tsv.template:development/development_pairs.tsv.template \
  evaluation_pairs.tsv.template:evaluation/evaluation_pairs.tsv.template; do
  source_name="${mapping%%:*}"; target_name="${mapping#*:}"
  target="${DATA_ROOT}/imagegen/${target_name}"
  [[ -e "${target}" ]] || cp \
    "${ROOT}/data_templates/v100-imagegen-5k/${source_name}" "${target}"
done
target="${DATA_ROOT}/imagegen-language/ledger.tsv.template"
[[ -e "${target}" ]] || cp \
  "${ROOT}/data_templates/v100-imagegen-500m-text/ledger.tsv.template" \
  "${target}"
printf 'Created data layout under %s. Fill templates, compute hashes, then rename ledger.tsv.template to ledger.tsv.\n' "${DATA_ROOT}"
