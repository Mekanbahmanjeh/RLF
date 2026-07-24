#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORK="${ROOT}/results/v100_1586h_campaign_shell_test"
[[ "$(realpath -m "${WORK}")" == "$(realpath "${ROOT}")/results/v100_1586h_campaign_shell_test" ]]
rm -rf -- "${WORK}"
mkdir -p "${WORK}/evidence" "${WORK}/manifests"
trap 'rm -rf -- "${WORK}" 2>/dev/null || true' EXIT

CONTROLLER="${ROOT}/scripts/v100/v100_1586h_controller.sh"
GATE="${ROOT}/scripts/v100/v100_scale_promotion_gate.sh"
RESUME_DRILL="${ROOT}/scripts/v100/verify_v100_resume_equivalence.sh"
IMAGEGEN_RESUME_DRILL="${ROOT}/scripts/v100/verify_imagegen_v100_resume_equivalence.sh"
BINDING=aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa
printf 'claim-ineligible source manifest fixture\n' >"${WORK}/evidence/source_manifest.tsv"
printf 'claim-ineligible checkpoint fixture\n' >"${WORK}/evidence/checkpoint.rlf"
printf 'claim-ineligible frozen split fixture\n' >"${WORK}/evidence/frozen_split.tsv"
printf '{"schema":"rlf-v100-resume-equivalence-v1","passed":false,"physical_training_performed":false,"test_doubles":true,"claim_eligible":false,"profile":"general-v100-32g-500m","campaign_binding_sha256":"%s","byte_identical":false}\n' "${BINDING}" >"${WORK}/evidence/resume_equivalence.json"
SOURCE_HASH="$(sha256sum "${WORK}/evidence/source_manifest.tsv" | awk '{print $1}')"
CHECKPOINT_HASH="$(sha256sum "${WORK}/evidence/checkpoint.rlf" | awk '{print $1}')"
FROZEN_HASH="$(sha256sum "${WORK}/evidence/frozen_split.tsv" | awk '{print $1}')"
RESUME_HASH="$(sha256sum "${WORK}/evidence/resume_equivalence.json" | awk '{print $1}')"

bash -n "${CONTROLLER}"
bash -n "${GATE}"
bash -n "${RESUME_DRILL}"
bash -n "${IMAGEGEN_RESUME_DRILL}"

# The physical drill cannot be launched directly or converted into synthetic
# evidence. This branch exits before reading a fake readiness report or
# starting the CUDA binary.
set +e
bash "${RESUME_DRILL}" --ledger "${WORK}/evidence/source_manifest.tsv" \
  --target-records 1 --readiness-report "${WORK}/evidence/source_manifest.tsv" \
  --output "${WORK}/must-not-drill" >/dev/null 2>"${WORK}/resume-drill.err"
RESUME_DRILL_STATUS=$?
set -e
[[ "${RESUME_DRILL_STATUS}" -eq 1 && ! -e "${WORK}/must-not-drill" ]]
grep -q 'inside the bound physical-throughput-probe stage' "${WORK}/resume-drill.err"

# Image-generation resume evidence is likewise impossible to mint directly or
# with claim-ineligible fixtures outside the controller-bound physical stage.
set +e
bash "${IMAGEGEN_RESUME_DRILL}" \
  --first-manifest "${WORK}/evidence/source_manifest.tsv" \
  --second-manifest "${WORK}/evidence/checkpoint.rlf" \
  --readiness-report "${WORK}/evidence/resume_equivalence.json" \
  --output "${WORK}/must-not-imagegen-drill" \
  >/dev/null 2>"${WORK}/imagegen-resume-drill.err"
IMAGEGEN_RESUME_DRILL_STATUS=$?
set -e
[[ "${IMAGEGEN_RESUME_DRILL_STATUS}" -eq 1 && \
   ! -e "${WORK}/must-not-imagegen-drill" ]]
grep -q 'inside the bound physical-throughput-probe stage' \
  "${WORK}/imagegen-resume-drill.err"

bash "${CONTROLLER}" --plan >"${WORK}/plan.txt"
grep -q '^profile=general-v100-32g-500m$' "${WORK}/plan.txt"
grep -q '^total_budget_seconds=5709600$' "${WORK}/plan.txt"
grep -q '^primary_target_records=200000000$' "${WORK}/plan.txt"
grep -q '^conditional_target_records=500000000$' "${WORK}/plan.txt"
grep -q '^training_authorized=false$' "${WORK}/plan.txt"
grep -Eq '^train-50m[[:space:]]+250[[:space:]]+50000000' "${WORK}/plan.txt"
grep -Eq '^train-200m-primary[[:space:]]+600[[:space:]]+200000000' "${WORK}/plan.txt"
grep -Eq '^train-500m-promoted[[:space:]]+300[[:space:]]+500000000' "${WORK}/plan.txt"
PLANNED_HOURS="$(awk '$1 ~ /^(hardware-preflight|cuda12-sm70-build-test|data-provenance-audit|physical-throughput-probe|train-50m|recovery-frozen-eval-50m|train-200m-primary|recovery-frozen-eval-200m|train-500m-promoted|final-external-evaluation|artifact-export)$/ { total += $2 } END { print total + 0 }' "${WORK}/plan.txt")"
[[ "${PLANNED_HOURS}" -eq 1586 ]]

bash "${CONTROLLER}" --start --state-dir "${WORK}/state" --binding "${BINDING}" >/dev/null
bash "${CONTROLLER}" --status --state-dir "${WORK}/state" >"${WORK}/status.txt"
grep -q '^total_budget_seconds=5709600$' "${WORK}/status.txt"
grep -q '^remaining_seconds=5709600$' "${WORK}/status.txt"

# A non-training stage runs only with the explicit no-training environment.
bash "${CONTROLLER}" --run-stage hardware-preflight \
  --state-dir "${WORK}/state" --binding "${BINDING}" -- \
  bash -c '[[ "$RLF_TRAINING_AUTHORIZED" == 0 && "$RLF_AUTHORIZED_RECORDS" == 0 ]]' \
  >"${WORK}/nontraining-stage.stdout"

# Exact training stages reject arbitrary commands even before evidence is
# considered. The controller, not the caller, selects the audited trainer.
set +e
bash "${CONTROLLER}" --run-stage train-200m-primary \
  --state-dir "${WORK}/state" --binding "${BINDING}" -- \
  bash -c 'touch "$1"' _ "${WORK}/must-not-run" \
  >"${WORK}/no-auth.stdout" 2>"${WORK}/no-auth.stderr"
NO_AUTH_STATUS=$?
set -e
[[ "${NO_AUTH_STATUS}" -eq 2 && ! -e "${WORK}/must-not-run" ]]
grep -q 'training stages do not accept an arbitrary command' "${WORK}/no-auth.stderr"

# Without an arbitrary command, an exact training stage still cannot start
# without the complete authorization bundle.
set +e
bash "${CONTROLLER}" --run-stage train-200m-primary \
  --state-dir "${WORK}/state" --binding "${BINDING}" \
  >"${WORK}/missing-auth.stdout" 2>"${WORK}/missing-auth.stderr"
MISSING_AUTH_STATUS=$?
set -e
[[ "${MISSING_AUTH_STATUS}" -eq 5 ]]
grep -q 'training stage requires evidence' "${WORK}/missing-auth.stderr"

# This bundle is intentionally and explicitly claim-ineligible. It exercises
# fail-closed hashing/binding without fabricating physical V100 evidence.
cat >"${WORK}/evidence/provenance.env" <<EOF
schema=rlf-v100-provenance-evidence-v1
campaign_binding_sha256=${BINDING}
claim_eligible=false
test_doubles=true
provenance_complete=true
dedup_complete=true
contamination_controls_passed=true
source_manifest_sha256=${SOURCE_HASH}
source_manifest_relative_path=source_manifest.tsv
audited_source_records=500000000
EOF
cat >"${WORK}/evidence/hardware.env" <<EOF
schema=rlf-v100-hardware-evidence-v1
campaign_binding_sha256=${BINDING}
physical_measurement=false
test_doubles=true
gpu_count=1
gpu_name=Tesla V100-SXM2-32GB
compute_capability=7.0
cuda_major=12
vram_total_bytes=34000000000
peak_vram_bytes=30000000000
EOF
cat >"${WORK}/evidence/stage.env" <<EOF
schema=rlf-v100-stage-evidence-v1
campaign_binding_sha256=${BINDING}
completed_stage_records=200000000
stage_completed=true
capacity_skips=0
completed_checkpoint_sha256=${CHECKPOINT_HASH}
EOF
cat >"${WORK}/evidence/throughput.env" <<EOF
schema=rlf-v100-throughput-evidence-v1
campaign_binding_sha256=${BINDING}
physical_measurement=false
test_doubles=true
measured_records=1000000
end_to_end_records_per_second=1000.0
measured_checkpoint_sha256=${CHECKPOINT_HASH}
EOF
cat >"${WORK}/evidence/checkpoint.env" <<EOF
schema=rlf-v100-checkpoint-evidence-v1
campaign_binding_sha256=${BINDING}
checkpoint_verified=true
resume_recovery_passed=true
resume_equivalence_report_sha256=${RESUME_HASH}
resume_equivalence_report_relative_path=resume_equivalence.json
checkpoint_sha256=${CHECKPOINT_HASH}
checkpoint_relative_path=checkpoint.rlf
checkpoint_profile=general-v100-32g-500m
checkpoint_training_records=200000000
resume_recovered_training_records=200000000
EOF
cat >"${WORK}/evidence/frozen_eval.env" <<EOF
schema=rlf-v100-frozen-eval-evidence-v1
campaign_binding_sha256=${BINDING}
contamination_audit_passed=true
evaluated_checkpoint_sha256=${CHECKPOINT_HASH}
frozen_split_sha256=${FROZEN_HASH}
frozen_split_relative_path=frozen_split.tsv
sample_count=1000
quality_before=0.50
quality_after=0.51
quality_direction=higher_is_better
EOF
cat >"${WORK}/evidence/resources.env" <<EOF
schema=rlf-v100-resource-evidence-v1
campaign_binding_sha256=${BINDING}
projection_target_records=500000000
projected_host_ram_bytes=100000000000
available_host_ram_bytes=200000000000
projected_disk_bytes=1000000000000
available_disk_bytes=2000000000000
projected_checkpoint_space_bytes=200000000000
available_checkpoint_space_bytes=400000000000
EOF

bash "${ROOT}/scripts/vast/artifact_manifest.sh" collect \
  --source "${WORK}/evidence" --manifest "${WORK}/manifests/evidence.tsv" >/dev/null

set +e
bash "${GATE}" authorize --campaign-state "${WORK}/state" \
  --evidence-dir "${WORK}/evidence" --manifest "${WORK}/manifests/evidence.tsv" \
  --target-records 500000000 --output "${WORK}/authorization.env" \
  >"${WORK}/gate.stdout" 2>"${WORK}/gate.stderr"
INELIGIBLE_STATUS=$?
set -e
[[ "${INELIGIBLE_STATUS}" -eq 1 ]]
grep -q 'claim_eligible must be true' "${WORK}/gate.stderr"
[[ ! -e "${WORK}/authorization.env" && ! -e "${WORK}/authorization.env.sha256" ]]

# A forged ticket cannot bypass re-verification of the hashed, ineligible
# evidence bundle, and the training command remains unexecuted.
cat >"${WORK}/forged-ticket.env" <<EOF
schema=rlf-v100-scale-authorization-v1
profile=general-v100-32g-500m
authorization=training_only_not_claim_evidence
frontier_claim_authorized=false
campaign_binding_sha256=${BINDING}
target_records=500000000
completed_stage_records=200000000
evidence_manifest_sha256=$(sha256sum "${WORK}/manifests/evidence.tsv" | awk '{print $1}')
source_manifest_sha256=${SOURCE_HASH}
checkpoint_sha256=${CHECKPOINT_HASH}
frozen_split_sha256=${FROZEN_HASH}
measured_end_to_end_records_per_second=1000.0
required_records_per_second_at_issue=1.0
available_training_seconds_at_issue=1080000
issued_epoch=1
EOF
printf '%s  %s\n' "$(sha256sum "${WORK}/forged-ticket.env" | awk '{print $1}')" \
  forged-ticket.env >"${WORK}/forged-ticket.env.sha256"
set +e
bash "${CONTROLLER}" --run-stage train-500m-promoted \
  --state-dir "${WORK}/state" --binding "${BINDING}" \
  --evidence-dir "${WORK}/evidence" --manifest "${WORK}/manifests/evidence.tsv" \
  --authorization "${WORK}/forged-ticket.env" \
  --ledger "${WORK}/evidence/source_manifest.tsv" \
  --checkpoint "${WORK}/evidence/checkpoint.rlf" \
  --result-dir "${WORK}/forged-result" \
  --readiness-report "${WORK}/evidence/source_manifest.tsv" \
  >"${WORK}/forged.stdout" 2>"${WORK}/forged.stderr"
FORGED_STATUS=$?
set -e
[[ "${FORGED_STATUS}" -eq 1 && ! -e "${WORK}/forged-result" ]]
grep -q 'claim_eligible must be true' "${WORK}/forged.stderr"

# Manifest integrity is checked before any evidence values are trusted.
printf '\n# tamper\n' >>"${WORK}/evidence/stage.env"
set +e
bash "${GATE}" authorize --campaign-state "${WORK}/state" \
  --evidence-dir "${WORK}/evidence" --manifest "${WORK}/manifests/evidence.tsv" \
  --target-records 500000000 --output "${WORK}/tampered-authorization.env" \
  >/dev/null 2>"${WORK}/tamper.stderr"
TAMPER_STATUS=$?
set -e
[[ "${TAMPER_STATUS}" -eq 1 ]]
grep -q 'artifact verification failed' "${WORK}/tamper.stderr"

printf 'v100_1586h_campaign_shell_test=pass\nclaim_eligible_fixture=false\nphysical_training_performed=false\n'
