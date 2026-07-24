#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VERIFIER="${ROOT}/scripts/verify_imagegen_data_audits.sh"
WORK="$(mktemp -d)"
trap 'rm -rf -- "${WORK}"' EXIT
mkdir -p "${WORK}/audits"
printf 'pair-1\tsource.ppm\t%s\ttarget.ppm\t%s\trotate\tfixture:one\tCC0-1.0\n' \
  "$(printf source | sha256sum | awk '{print $1}')" \
  "$(printf target | sha256sum | awk '{print $1}')" >"${WORK}/pairs.tsv"
PAIR_SHA="$(sha256sum -- "${WORK}/pairs.tsv" | awk '{print $1}')"
POLICY_SHA="$(printf license-policy | sha256sum | awk '{print $1}')"
EVAL_SHA="$(printf frozen-evaluation-manifest | sha256sum | awk '{print $1}')"

write_reports() {
  cat >"${WORK}/audits/data_audit.json" <<EOF
{"schema":"rlf-imagegen-data-audit-v1","passed":true,"test_doubles":false,"frontier_claim_authorized":false,"pair_manifest_sha256":"${PAIR_SHA}","records_audited":1,"media_hashes_verified":true,"provenance_complete":true}
EOF
  cat >"${WORK}/audits/license_report.json" <<EOF
{"schema":"rlf-imagegen-license-audit-v1","passed":true,"test_doubles":false,"frontier_claim_authorized":false,"pair_manifest_sha256":"${PAIR_SHA}","records_audited":1,"unresolved_license_records":0,"disallowed_license_records":0,"license_policy_sha256":"${POLICY_SHA}"}
EOF
  cat >"${WORK}/audits/exact_dedup_report.json" <<EOF
{"schema":"rlf-imagegen-exact-dedup-audit-v1","passed":true,"test_doubles":false,"frontier_claim_authorized":false,"pair_manifest_sha256":"${PAIR_SHA}","records_audited":1,"exact_duplicates_remaining":0}
EOF
  cat >"${WORK}/audits/near_dedup_report.json" <<EOF
{"schema":"rlf-imagegen-near-dedup-audit-v1","passed":true,"test_doubles":false,"frontier_claim_authorized":false,"pair_manifest_sha256":"${PAIR_SHA}","records_audited":1,"near_duplicates_remaining":0,"method":"sha256-prefix-lsh-v1"}
EOF
  cat >"${WORK}/audits/perceptual_dedup_report.json" <<EOF
{"schema":"rlf-imagegen-perceptual-dedup-audit-v1","passed":true,"test_doubles":false,"frontier_claim_authorized":false,"pair_manifest_sha256":"${PAIR_SHA}","records_audited":1,"perceptual_duplicates_remaining":0,"method":"phash-64-hamming-v1"}
EOF
  cat >"${WORK}/audits/contamination_report.json" <<EOF
{"schema":"rlf-imagegen-contamination-audit-v1","passed":true,"test_doubles":false,"frontier_claim_authorized":false,"pair_manifest_sha256":"${PAIR_SHA}","records_audited":1,"overlap_records":0,"evaluation_manifest_frozen":true,"benchmark_answers_present":false,"evaluation_manifest_sha256":"${EVAL_SHA}"}
EOF
}

bash -n "${VERIFIER}"
write_reports
bash "${VERIFIER}" "${WORK}/pairs.tsv" "${WORK}/audits" >"${WORK}/pass.txt"
grep -Fqx 'audit_validation=pass' "${WORK}/pass.txt"
grep -Fqx "pair_manifest_sha256=${PAIR_SHA}" "${WORK}/pass.txt"
grep -Fqx 'records_audited=1' "${WORK}/pass.txt"
grep -Fqx 'frontier_claim_authorized=false' "${WORK}/pass.txt"

# The report set cannot be replayed against modified training bytes.
printf '# post-audit mutation\n' >>"${WORK}/pairs.tsv"
set +e
bash "${VERIFIER}" "${WORK}/pairs.tsv" "${WORK}/audits" >/dev/null 2>"${WORK}/mutation.err"
MUTATION_STATUS=$?
set -e
[[ "${MUTATION_STATUS}" -eq 1 ]]
grep -q 'does not bind the exact pair manifest' "${WORK}/mutation.err"
sed -i '$d' "${WORK}/pairs.tsv"

# Duplicate security-critical keys and positive contamination both fail.
write_reports
sed -i 's/"passed":true/"passed":true,"passed":true/' "${WORK}/audits/data_audit.json"
set +e
bash "${VERIFIER}" "${WORK}/pairs.tsv" "${WORK}/audits" >/dev/null 2>"${WORK}/duplicate.err"
DUPLICATE_STATUS=$?
set -e
[[ "${DUPLICATE_STATUS}" -eq 1 ]]
grep -q 'exactly one passed=true' "${WORK}/duplicate.err"

write_reports
sed -i 's/"overlap_records":0/"overlap_records":1/' "${WORK}/audits/contamination_report.json"
set +e
bash "${VERIFIER}" "${WORK}/pairs.tsv" "${WORK}/audits" >/dev/null 2>"${WORK}/overlap.err"
OVERLAP_STATUS=$?
set -e
[[ "${OVERLAP_STATUS}" -eq 1 ]]
grep -q 'exactly one overlap_records=0' "${WORK}/overlap.err"

write_reports
rm -- "${WORK}/audits/license_report.json"
ln -s data_audit.json "${WORK}/audits/license_report.json"
set +e
bash "${VERIFIER}" "${WORK}/pairs.tsv" "${WORK}/audits" >/dev/null 2>"${WORK}/symlink.err"
SYMLINK_STATUS=$?
set -e
[[ "${SYMLINK_STATUS}" -eq 1 ]]
grep -q 'required imagegen audit report missing' "${WORK}/symlink.err"

printf 'imagegen_data_audits_shell_test=pass\nphysical_training_performed=false\nclaim_eligible=false\n'
