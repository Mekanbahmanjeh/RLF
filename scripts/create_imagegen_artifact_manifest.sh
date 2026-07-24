#!/usr/bin/env bash
set -euo pipefail

BUNDLE="${1:?Usage: create_imagegen_artifact_manifest.sh BUNDLE_DIR [MANIFEST]}"
MANIFEST="${2:-${BUNDLE}/artifact_manifest.tsv}"
BUNDLE="$(realpath "${BUNDLE}")"
MANIFEST="$(realpath -m "${MANIFEST}")"
[[ "$(dirname "${MANIFEST}")" == "${BUNDLE}" ]] || {
  echo "manifest must be directly inside the evidence bundle" >&2; exit 2;
}
SIDECAR="${MANIFEST}.sha256"
[[ ! -e "${MANIFEST}" && ! -e "${SIDECAR}" ]] || {
  echo "artifact manifest or sidecar already exists" >&2; exit 1;
}

kinds=(checkpoint ledger source_manifest data_audit license_report \
  exact_dedup_report near_dedup_report perceptual_dedup_report \
  contamination_report training_telemetry resource_summary raw_gpu_trace \
  environment checkpoint_inspection readiness_report \
  resume_equivalence_report)
paths=(model.rlfimg training_pairs.tsv source_manifest.tsv data_audit.json \
  license_report.json exact_dedup_report.json near_dedup_report.json \
  perceptual_dedup_report.json contamination_report.json \
  training_telemetry.txt resource_summary.json raw_gpu_trace.csv environment.txt \
  checkpoint_inspection.txt readiness.json resume_equivalence.json)

printf 'RLF_IMAGEGEN_ARTIFACT_MANIFEST\t1\n' >"${MANIFEST}"
for index in "${!kinds[@]}"; do
  relative="${paths[$index]}"; absolute="${BUNDLE}/${relative}"
  [[ -f "${absolute}" && ! -L "${absolute}" && -s "${absolute}" ]] || {
    echo "missing/non-regular/empty image-generation artifact: ${relative}" >&2
    rm -f -- "${MANIFEST}"
    exit 1
  }
  [[ "$(realpath "${absolute}")" == "${BUNDLE}/${relative}" ]] || {
    echo "noncanonical image-generation artifact: ${relative}" >&2
    rm -f -- "${MANIFEST}"
    exit 1
  }
  bytes="$(stat -c '%s' -- "${absolute}")"
  sha="$(sha256sum -- "${absolute}" | awk '{print $1}')"
  printf '%s\t%s\t%s\t%s\n' "${kinds[$index]}" "${relative}" \
    "${bytes}" "${sha}" >>"${MANIFEST}"
done
manifest_sha="$(sha256sum -- "${MANIFEST}" | awk '{print $1}')"
printf 'RLF_IMAGEGEN_MANIFEST_SHA256\t1\nsha256\t%s\n' \
  "${manifest_sha}" >"${SIDECAR}"
printf 'manifest=%s\nmanifest_sha256=%s\n' "${MANIFEST}" "${manifest_sha}"
