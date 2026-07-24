#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BIN="${1:?solstice executable required}"
VALIDATE="${ROOT}/scripts/v100/validate_imagegen_5k_dataset.sh"
WORK="$(mktemp -d)"; trap 'rm -rf -- "${WORK}"' EXIT
DATA="${WORK}/data"
mkdir -p "${DATA}"/{train,development,evaluation}
printf 'CC0-1.0\n' >"${DATA}/licenses.txt"
MARKER='@neutral-gray128-target-size-v1'
MARKER_SHA="$(printf '%s' "${MARKER}" | sha256sum | awk '{print $1}')"
make_image() {
  local path="$1" red="$2" green="$3" blue="$4"
  {
    printf 'P6\n4 4\n255\n'
    for _ in $(seq 1 16); do
      printf "\\$(printf '%03o' "${red}")\\$(printf '%03o' "${green}")\\$(printf '%03o' "${blue}")"
    done
  } >"${path}"
}
make_image "${DATA}/train/a.ppm" 20 20 20
make_image "${DATA}/train/b.ppm" 70 20 20
make_image "${DATA}/development/c.ppm" 20 100 20
make_image "${DATA}/evaluation/d.ppm" 20 20 140
make_image "${DATA}/evaluation/e.ppm" 180 180 20
row() {
  local id="$1" image="$2" prompt="$3" tags="${4:-}"
  printf '%s\t%s\t%s\t%s\t%s\t%s\tfixture:%s\tCC0-1.0' \
    "${id}" "${MARKER}" "${MARKER_SHA}" "$(basename "${image}")" \
    "$(sha256sum -- "${image}" | awk '{print $1}')" "${prompt}" "${id}"
  [[ -z "${tags}" ]] || printf '\t%s' "${tags}"
  printf '\n'
}
row train-a "${DATA}/train/a.ppm" 'make charcoal' >"${DATA}/train/train_pairs.tsv"
row train-b "${DATA}/train/b.ppm" 'make crimson' >>"${DATA}/train/train_pairs.tsv"
row dev-c "${DATA}/development/c.ppm" 'make green' \
  >"${DATA}/development/development_pairs.tsv"
row eval-d "${DATA}/evaluation/d.ppm" 'please render blue' \
  'unseen_prompt,paraphrase,natural_image,multilingual' \
  >"${DATA}/evaluation/evaluation_pairs.tsv"
row eval-e "${DATA}/evaluation/e.ppm" 'make blue and then make yellow' \
  'unseen_prompt,composition,natural_image' \
  >>"${DATA}/evaluation/evaluation_pairs.tsv"
bash -n "${VALIDATE}"
RLF_IMAGEGEN_DATASET_CONTRACT_TEST_DOUBLES=1 bash "${VALIDATE}" \
  "${BIN}" "${DATA}" "${WORK}/audits" >"${WORK}/stdout.txt"
grep -Fqx 'dataset_contract_passed=true' "${WORK}/stdout.txt"
grep -q '"train_records": 2' "${WORK}/audits/dataset_contract.json"
grep -q '"test_doubles": true' "${WORK}/audits/dataset_contract.json"
grep -q '"natural_image_quality_measured": false' \
  "${WORK}/audits/dataset_contract.json"
grep -q '"paraphrase_evaluation_records": 1' \
  "${WORK}/audits/dataset_contract.json"
grep -q '"multilingual_evaluation_records": 1' \
  "${WORK}/audits/dataset_contract.json"
printf 'imagegen_5k_dataset_contract_test=pass\n'
