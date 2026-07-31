#!/usr/bin/env bash
set -euo pipefail

PAIR_MANIFEST="${1:?Usage: verify_imagegen_data_audits.sh PAIR_MANIFEST AUDIT_DIR}"
AUDIT_DIR="${2:?audit directory required}"
[[ -f "${PAIR_MANIFEST}" && ! -L "${PAIR_MANIFEST}" && -d "${AUDIT_DIR}" ]] || {
  echo "regular pair manifest and audit directory are required" >&2
  exit 2
}

json_string() {
  sed -nE "s/.*\"$2\"[[:space:]]*:[[:space:]]*\"([^\"]*)\".*/\1/p" "$1" | head -n 1
}
json_number() {
  sed -nE "s/.*\"$2\"[[:space:]]*:[[:space:]]*([0-9]+).*/\1/p" "$1" | head -n 1
}
require_literal_once() {
  local file="$1" key="$2" value="$3"
  local matches
  matches="$( (grep -Eo "\"${key}\"[[:space:]]*:[[:space:]]*${value}([,}[:space:]]|$)" \
    "${file}" || true) | wc -l | tr -d '[:space:]')"
  [[ "${matches}" -eq 1 ]] || {
    echo "${file} must contain exactly one ${key}=${value}" >&2
    exit 1
  }
}

PAIR_MANIFEST="$(realpath "${PAIR_MANIFEST}")"
AUDIT_DIR="$(realpath "${AUDIT_DIR}")"
PAIR_SHA="$(sha256sum -- "${PAIR_MANIFEST}" | awk '{print $1}')"
PAIR_RECORDS="$(awk 'NF && $0 !~ /^#/ { count += 1 } END { print count + 0 }' \
  "${PAIR_MANIFEST}")"
((PAIR_RECORDS > 0)) || { echo "imagegen pair manifest is empty" >&2; exit 1; }

for report in data_audit.json license_report.json exact_dedup_report.json \
  near_dedup_report.json perceptual_dedup_report.json contamination_report.json; do
  [[ -f "${AUDIT_DIR}/${report}" && ! -L "${AUDIT_DIR}/${report}" && \
     -s "${AUDIT_DIR}/${report}" ]] || {
    echo "required imagegen audit report missing: ${report}" >&2
    exit 1
  }
done

validate_bound_audit() {
  local file="$1" schema="$2"
  require_literal_once "${file}" schema "\"${schema}\""
  require_literal_once "${file}" passed true
  require_literal_once "${file}" test_doubles false
  require_literal_once "${file}" frontier_claim_authorized false
  [[ "$(json_string "${file}" pair_manifest_sha256)" == "${PAIR_SHA}" && \
     "$(json_number "${file}" records_audited)" == "${PAIR_RECORDS}" ]] || {
    echo "audit does not bind the exact pair manifest/record count: ${file}" >&2
    exit 1
  }
}

validate_bound_audit "${AUDIT_DIR}/data_audit.json" rlf-imagegen-data-audit-v1
require_literal_once "${AUDIT_DIR}/data_audit.json" media_hashes_verified true
require_literal_once "${AUDIT_DIR}/data_audit.json" provenance_complete true

validate_bound_audit "${AUDIT_DIR}/license_report.json" rlf-imagegen-license-audit-v1
require_literal_once "${AUDIT_DIR}/license_report.json" unresolved_license_records 0
require_literal_once "${AUDIT_DIR}/license_report.json" disallowed_license_records 0
LICENSE_POLICY_SHA="$(json_string "${AUDIT_DIR}/license_report.json" license_policy_sha256)"
[[ "${LICENSE_POLICY_SHA}" =~ ^[0-9a-f]{64}$ ]] || {
  echo "license audit lacks a hashed policy" >&2; exit 1;
}

validate_bound_audit "${AUDIT_DIR}/exact_dedup_report.json" rlf-imagegen-exact-dedup-audit-v1
require_literal_once "${AUDIT_DIR}/exact_dedup_report.json" exact_duplicates_remaining 0

validate_bound_audit "${AUDIT_DIR}/near_dedup_report.json" rlf-imagegen-near-dedup-audit-v1
require_literal_once "${AUDIT_DIR}/near_dedup_report.json" near_duplicates_remaining 0
NEAR_METHOD="$(json_string "${AUDIT_DIR}/near_dedup_report.json" method)"
[[ -n "${NEAR_METHOD}" ]] || { echo "near-dedup audit method is missing" >&2; exit 1; }

validate_bound_audit "${AUDIT_DIR}/perceptual_dedup_report.json" rlf-imagegen-perceptual-dedup-audit-v1
require_literal_once "${AUDIT_DIR}/perceptual_dedup_report.json" perceptual_duplicates_remaining 0
PERCEPTUAL_METHOD="$(json_string "${AUDIT_DIR}/perceptual_dedup_report.json" method)"
[[ -n "${PERCEPTUAL_METHOD}" ]] || {
  echo "perceptual-dedup audit method is missing" >&2; exit 1;
}

validate_bound_audit "${AUDIT_DIR}/contamination_report.json" rlf-imagegen-contamination-audit-v1
require_literal_once "${AUDIT_DIR}/contamination_report.json" overlap_records 0
require_literal_once "${AUDIT_DIR}/contamination_report.json" evaluation_manifest_frozen true
require_literal_once "${AUDIT_DIR}/contamination_report.json" benchmark_answers_present false
EVALUATION_MANIFEST_SHA="$(json_string "${AUDIT_DIR}/contamination_report.json" evaluation_manifest_sha256)"
[[ "${EVALUATION_MANIFEST_SHA}" =~ ^[0-9a-f]{64}$ ]] || {
  echo "contamination audit lacks a frozen evaluation-manifest hash" >&2; exit 1;
}

printf 'audit_validation=pass\n'
printf 'pair_manifest_sha256=%s\n' "${PAIR_SHA}"
printf 'records_audited=%s\n' "${PAIR_RECORDS}"
printf 'license_policy_sha256=%s\n' "${LICENSE_POLICY_SHA}"
printf 'evaluation_manifest_sha256=%s\n' "${EVALUATION_MANIFEST_SHA}"
printf 'near_dedup_method=%s\n' "${NEAR_METHOD}"
printf 'perceptual_dedup_method=%s\n' "${PERCEPTUAL_METHOD}"
printf 'frontier_claim_authorized=false\n'
