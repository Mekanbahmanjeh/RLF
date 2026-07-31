#!/usr/bin/env bash
set -euo pipefail

BIN="${1:?Usage: validate_imagegen_5k_dataset.sh SOLSTICE DATA_ROOT OUTPUT}"
DATA_ROOT="${2:?dataset root required}"
OUTPUT="${3:?output directory required}"
[[ -x "${BIN}" && -d "${DATA_ROOT}" && ! -e "${OUTPUT}" ]] || {
  echo "executable, dataset root, and new output path are required" >&2; exit 2;
}
DATA_ROOT="$(realpath "${DATA_ROOT}")"
TRAIN="${DATA_ROOT}/train/train_pairs.tsv"
DEVELOPMENT="${DATA_ROOT}/development/development_pairs.tsv"
EVALUATION="${DATA_ROOT}/evaluation/evaluation_pairs.tsv"
POLICY="${DATA_ROOT}/licenses.txt"
for file in "${TRAIN}" "${DEVELOPMENT}" "${EVALUATION}" "${POLICY}"; do
  [[ -f "${file}" && ! -L "${file}" ]] || {
    echo "required dataset file missing or unsafe: ${file}" >&2; exit 2;
  }
done

TEST_DOUBLES=false
TRAIN_EXPECTED=3500; DEVELOPMENT_EXPECTED=500; EVALUATION_EXPECTED=1000
MIN_UNSEEN=100; MIN_COMPOSITIONS=100; MIN_PARAPHRASES=100
MIN_NATURAL_IMAGES=500; MIN_MULTILINGUAL=100
if [[ "${RLF_IMAGEGEN_DATASET_CONTRACT_TEST_DOUBLES:-0}" == 1 ]]; then
  TEST_DOUBLES=true
  TRAIN_EXPECTED=2; DEVELOPMENT_EXPECTED=1; EVALUATION_EXPECTED=2
  MIN_UNSEEN=1; MIN_COMPOSITIONS=1; MIN_PARAPHRASES=1
  MIN_NATURAL_IMAGES=1; MIN_MULTILINGUAL=1
fi
count() { awk 'NF && $0 !~ /^#/ {n++} END {print n+0}' "$1"; }
TRAIN_COUNT="$(count "${TRAIN}")"
DEVELOPMENT_COUNT="$(count "${DEVELOPMENT}")"
EVALUATION_COUNT="$(count "${EVALUATION}")"
[[ "${TRAIN_COUNT}" == "${TRAIN_EXPECTED}" &&
   "${DEVELOPMENT_COUNT}" == "${DEVELOPMENT_EXPECTED}" &&
   "${EVALUATION_COUNT}" == "${EVALUATION_EXPECTED}" ]] || {
  echo "dataset must contain exact 3500/500/1000 split counts" >&2; exit 1;
}
awk -F'\t' 'NF && $0 !~ /^#/ && NF != 8 {exit 1}' "${TRAIN}" || {
  echo "training rows must have exactly eight fields" >&2; exit 1;
}
awk -F'\t' 'NF && $0 !~ /^#/ && NF != 8 && NF != 9 {exit 1}' \
  "${DEVELOPMENT}" || {
  echo "development rows must have eight or nine fields" >&2; exit 1;
}
awk -F'\t' '
  function valid_tags(value, parts,n,i,t,seen) {
    n=split(value,parts,",")
    if (n < 1) return 0
    for (i=1;i<=n;i++) {
      t=parts[i]
      if (t !~ /^(unseen_prompt|paraphrase|composition|natural_image|multilingual|spatial|attribute_binding)$/ || seen[t]++) return 0
    }
    return 1
  }
  NF && $0 !~ /^#/ && (NF != 9 || !valid_tags($9)) {exit 1}
' "${EVALUATION}" || {
  echo "evaluation rows require a valid ninth evaluation_tags field" >&2
  exit 1
}
awk -F'\t' '
  NF && $0 !~ /^#/ {
    if ((NF != 8 && NF != 9) || $1 == "" || ids[$1]++ || targets[$5]++) exit 1
  }
' "${TRAIN}" "${DEVELOPMENT}" "${EVALUATION}" || {
  echo "dataset IDs and target hashes must be globally unique" >&2
  exit 1
}
UNSEEN="$(awk -F'\t' '
  NR == FNR && NF >= 8 && $0 !~ /^#/ {seen[$6]=1; next}
  NF >= 8 && $0 !~ /^#/ && !($6 in seen) {n++}
  END {print n+0}
' "${TRAIN}" "${EVALUATION}")"
COMPOSITIONS="$(awk -F'\t' '
  NF == 9 && $0 !~ /^#/ && $9 ~ /(^|,)composition(,|$)/ {
    prompt=tolower($6)
    if (!(prompt ~ / and then / || prompt ~ / followed by / ||
        prompt ~ / then / || prompt ~ /;/)) exit 1
    n++
  }
  END {print n+0}
' "${EVALUATION}")"
tag_count() {
  awk -F'\t' -v tag="$1" '
    NF == 9 && $0 !~ /^#/ && $9 ~ ("(^|,)" tag "(,|$)") {n++}
    END {print n+0}
  ' "${EVALUATION}"
}
PARAPHRASES="$(tag_count paraphrase)"
NATURAL_IMAGES="$(tag_count natural_image)"
MULTILINGUAL="$(tag_count multilingual)"
((UNSEEN >= MIN_UNSEEN && COMPOSITIONS >= MIN_COMPOSITIONS &&
   PARAPHRASES >= MIN_PARAPHRASES && NATURAL_IMAGES >= MIN_NATURAL_IMAGES &&
   MULTILINGUAL >= MIN_MULTILINGUAL)) || {
  echo "evaluation lacks required unseen/composition/paraphrase/natural/multilingual coverage" >&2; exit 1;
}

PARENT="$(dirname "${OUTPUT}")"; mkdir -p "${PARENT}"
TEMP="$(mktemp -d "${PARENT}/.imagegen-5k-audit.XXXXXX")"
trap 'rm -rf -- "${TEMP}"' EXIT
run_audit() {
  "${BIN}" imagegen-audit-pairs --manifest "$1" \
    --evaluation-manifest "$2" --license-policy "${POLICY}" \
    --max-audit-records 5000 --near-duplicate-hamming 3 --output "$3" \
    >"$3.stdout"
}
run_audit "${TRAIN}" "${EVALUATION}" "${TEMP}/train-vs-evaluation"
run_audit "${TRAIN}" "${DEVELOPMENT}" "${TEMP}/train-vs-development"
run_audit "${DEVELOPMENT}" "${EVALUATION}" "${TEMP}/development-vs-evaluation"
run_audit "${EVALUATION}" "${TRAIN}" "${TEMP}/evaluation-vs-training"
cat >"${TEMP}/dataset_contract.json" <<EOF
{
  "schema": "rlf-imagegen-5k-dataset-contract-v1",
  "passed": true,
  "test_doubles": ${TEST_DOUBLES},
  "train_records": ${TRAIN_COUNT},
  "development_records": ${DEVELOPMENT_COUNT},
  "evaluation_records": ${EVALUATION_COUNT},
  "unseen_evaluation_prompts": ${UNSEEN},
  "composition_evaluation_records": ${COMPOSITIONS},
  "paraphrase_evaluation_records": ${PARAPHRASES},
  "natural_image_evaluation_records": ${NATURAL_IMAGES},
  "multilingual_evaluation_records": ${MULTILINGUAL},
  "train_manifest_sha256": "$(sha256sum -- "${TRAIN}" | awk '{print $1}')",
  "development_manifest_sha256": "$(sha256sum -- "${DEVELOPMENT}" | awk '{print $1}')",
  "evaluation_manifest_sha256": "$(sha256sum -- "${EVALUATION}" | awk '{print $1}')",
  "license_policy_sha256": "$(sha256sum -- "${POLICY}" | awk '{print $1}')",
  "all_media_pairwise_audited": true,
  "natural_image_quality_measured": false,
  "frontier_claim_authorized": false
}
EOF
mv -- "${TEMP}" "${OUTPUT}"; trap - EXIT
printf 'dataset_contract_passed=true\ntest_doubles=%s\nfrontier_claim_authorized=false\n' \
  "${TEST_DOUBLES}"
