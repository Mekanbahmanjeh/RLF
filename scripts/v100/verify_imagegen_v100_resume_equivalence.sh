#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: verify_imagegen_v100_resume_equivalence.sh \
  --first-manifest FILE --second-manifest FILE --readiness-report FILE \
  --output DIR [--gpu-index N] [--maximum-records N]

Run through either the isolated 40-hour image controller's resume-vram-probe
stage or the legacy 1,586-hour controller's physical-throughput-probe stage.
The two manifests must be distinct, immutable image-pair shards. This performs
the same physical CUDA updates in the same order along two paths:

  uninterrupted: bootstrap -> shard 1 -> shard 2
  resumed:       bootstrap -> shard 1 -> verify/reload -> shard 2

Every update is covered by the 30 GiB V100 guard. Success requires physical
device-local updates and byte-identical final checkpoints. The report is
resumability evidence only and never authorizes frontier, SOTA, or efficiency
claims.
EOF
}

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CAMPAIGN_PROFILE="${RLF_CAMPAIGN_PROFILE:-}"
IMAGE_PROFILE=imagegen-v100-32g
FIRST_MANIFEST=""
SECOND_MANIFEST=""
READINESS_REPORT=""
OUTPUT=""
GPU_INDEX=""
MAXIMUM_RECORDS=1000
while (($# > 0)); do
  case "$1" in
    --first-manifest) FIRST_MANIFEST="${2:?--first-manifest requires a file}"; shift 2 ;;
    --second-manifest) SECOND_MANIFEST="${2:?--second-manifest requires a file}"; shift 2 ;;
    --readiness-report) READINESS_REPORT="${2:?--readiness-report requires a file}"; shift 2 ;;
    --output) OUTPUT="${2:?--output requires a directory}"; shift 2 ;;
    --gpu-index) GPU_INDEX="${2:?--gpu-index requires an integer}"; shift 2 ;;
    --maximum-records) MAXIMUM_RECORDS="${2:?--maximum-records requires an integer}"; shift 2 ;;
    --help|-h) usage; exit 0 ;;
    *) echo "unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

[[ -f "${FIRST_MANIFEST}" && ! -L "${FIRST_MANIFEST}" && \
   -f "${SECOND_MANIFEST}" && ! -L "${SECOND_MANIFEST}" && \
   -f "${READINESS_REPORT}" && ! -L "${READINESS_REPORT}" ]] || {
  echo "two regular manifests and a regular readiness report are required" >&2
  exit 2
}
[[ -n "${OUTPUT}" && "${MAXIMUM_RECORDS}" =~ ^[1-9][0-9]*$ && \
   "${MAXIMUM_RECORDS}" -le 10000 ]] || {
  echo "--output and --maximum-records in 1..10000 are required" >&2
  exit 2
}
CAMPAIGN_SCHEMA=""; EXPECTED_STAGE=""; EXPECTED_AUTHORIZATION=""
case "${CAMPAIGN_PROFILE}" in
  general-v100-32g-500m)
    CAMPAIGN_SCHEMA=rlf-v100-1586h-campaign-v1
    EXPECTED_STAGE=physical-throughput-probe
    EXPECTED_AUTHORIZATION=0
    ;;
  imagegen-v100-32g)
    CAMPAIGN_SCHEMA=rlf-imagegen-v100-40h-campaign-v1
    EXPECTED_STAGE=resume-vram-probe
    EXPECTED_AUTHORIZATION=0
    ;;
  *)
    echo "imagegen resume drill must run inside the bound physical-throughput-probe stage or isolated resume-vram-probe stage" >&2
    exit 1
    ;;
esac
[[ "${RLF_CAMPAIGN_STAGE:-}" == "${EXPECTED_STAGE}" && \
   "${RLF_TRAINING_AUTHORIZED:-0}" == "${EXPECTED_AUTHORIZATION}" && \
   "${RLF_CAMPAIGN_BINDING_SHA256:-}" =~ ^[0-9a-f]{64}$ ]] || {
  echo "imagegen resume drill must run inside the bound physical-throughput-probe stage or isolated resume-vram-probe stage" >&2
  exit 1
}
[[ "${RLF_CAMPAIGN_STATE_DIR:-}" == /* && \
   -f "${RLF_CAMPAIGN_STATE_DIR}/campaign.meta" && \
   ! -L "${RLF_CAMPAIGN_STATE_DIR}/campaign.meta" ]] || {
  echo "regular absolute V100 campaign state is required" >&2
  exit 1
}
grep -Fqx "schema=${CAMPAIGN_SCHEMA}" \
  "${RLF_CAMPAIGN_STATE_DIR}/campaign.meta" || { echo "wrong campaign schema" >&2; exit 1; }
grep -Fqx "profile=${CAMPAIGN_PROFILE}" \
  "${RLF_CAMPAIGN_STATE_DIR}/campaign.meta" || { echo "campaign profile mismatch" >&2; exit 1; }
grep -Fqx "binding_sha256=${RLF_CAMPAIGN_BINDING_SHA256}" \
  "${RLF_CAMPAIGN_STATE_DIR}/campaign.meta" || { echo "campaign binding mismatch" >&2; exit 1; }

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

require_literal_once "${READINESS_REPORT}" schema '"rlf-imagegen-v100-readiness-v1"'
require_literal_once "${READINESS_REPORT}" ready true
require_literal_once "${READINESS_REPORT}" test_doubles false
require_literal_once "${READINESS_REPORT}" training_performed false
[[ "$(json_string "${READINESS_REPORT}" profile)" == "${IMAGE_PROFILE}" ]] || {
  echo "readiness profile mismatch" >&2; exit 1;
}

FIRST_MANIFEST="$(realpath "${FIRST_MANIFEST}")"
SECOND_MANIFEST="$(realpath "${SECOND_MANIFEST}")"
READINESS_REPORT="$(realpath "${READINESS_REPORT}")"
[[ "${FIRST_MANIFEST}" != "${SECOND_MANIFEST}" && \
   "$(basename "${FIRST_MANIFEST}")" != "$(basename "${SECOND_MANIFEST}")" ]] || {
  echo "resume drill manifests must have distinct paths and shard-ID basenames" >&2
  exit 1
}
FIRST_RECORDS="$(awk 'NF && $0 !~ /^#/ { count += 1 } END { print count + 0 }' "${FIRST_MANIFEST}")"
SECOND_RECORDS="$(awk 'NF && $0 !~ /^#/ { count += 1 } END { print count + 0 }' "${SECOND_MANIFEST}")"
TOTAL_RECORDS=$((FIRST_RECORDS + SECOND_RECORDS))
((FIRST_RECORDS > 0 && SECOND_RECORDS > 0 && TOTAL_RECORDS <= MAXIMUM_RECORDS)) || {
  echo "both imagegen shards must be nonempty and total at most ${MAXIMUM_RECORDS} records" >&2
  exit 1
}

BIN="${ROOT}/build/ubuntu-general-cuda-compat/solstice"
EXPECTED_BINARY_SHA256="$(json_string "${READINESS_REPORT}" solstice_binary_sha256)"
[[ -x "${BIN}" && "${EXPECTED_BINARY_SHA256}" =~ ^[0-9a-f]{64}$ && \
   "$(sha256sum -- "${BIN}" | awk '{print $1}')" == "${EXPECTED_BINARY_SHA256}" ]] || {
  echo "CUDA binary changed after imagegen readiness" >&2
  exit 1
}
EXPECTED_SOURCE_SHA256="$(json_string "${READINESS_REPORT}" source_manifest_sha256)"
[[ "${EXPECTED_SOURCE_SHA256}" =~ ^[0-9a-f]{64}$ ]] || {
  echo "readiness source-manifest identity is missing" >&2; exit 1;
}
SOURCE_CHECK_DIR="$(mktemp -d)"
trap 'rm -rf -- "${SOURCE_CHECK_DIR}"' EXIT
"${ROOT}/scripts/create_source_manifest.sh" "${SOURCE_CHECK_DIR}/source_manifest.tsv" >/dev/null
[[ "$(sha256sum -- "${SOURCE_CHECK_DIR}/source_manifest.tsv" | awk '{print $1}')" == \
   "${EXPECTED_SOURCE_SHA256}" ]] || {
  echo "source tree changed after imagegen readiness" >&2; exit 1;
}
READINESS_GPU_INDEX="$(json_number "${READINESS_REPORT}" gpu_index)"
[[ -n "${GPU_INDEX}" ]] || GPU_INDEX="${READINESS_GPU_INDEX}"
[[ "${GPU_INDEX}" =~ ^[0-9]+$ && "${GPU_INDEX}" == "${READINESS_GPU_INDEX}" ]] || {
  echo "GPU index differs from imagegen readiness" >&2; exit 1;
}
GPU_UUID="$(json_string "${READINESS_REPORT}" gpu_uuid)"
[[ -n "${GPU_UUID}" ]] || { echo "readiness GPU UUID is missing" >&2; exit 1; }

OUTPUT="$(realpath -m "${OUTPUT}")"
[[ ! -e "${OUTPUT}" ]] || { echo "imagegen resume-equivalence output already exists" >&2; exit 2; }
mkdir -p "${OUTPUT}"
rm -rf -- "${SOURCE_CHECK_DIR}"
trap - EXIT

UNINTERRUPTED="${OUTPUT}/uninterrupted.rlfimg"
RESUMED="${OUTPUT}/resumed.rlfimg"
SEED="${RLF_IMAGEGEN_SEED:-77}"
"${BIN}" imagegen-bootstrap --profile "${IMAGE_PROFILE}" \
  --checkpoint "${UNINTERRUPTED}" --seed "${SEED}" >/dev/null
"${BIN}" imagegen-bootstrap --profile "${IMAGE_PROFILE}" \
  --checkpoint "${RESUMED}" --seed "${SEED}" >/dev/null

run_guarded_shard() {
  local name="$1" checkpoint="$2" manifest="$3"
  "${ROOT}/scripts/run_general_cuda_with_vram_guard.sh" \
    --profile "${IMAGE_PROFILE}" --gpu-index "${GPU_INDEX}" \
    --expected-uuid "${GPU_UUID}" --trace "${OUTPUT}/${name}_vram.csv" \
    --summary "${OUTPUT}/${name}_resource.json" -- \
    "${BIN}" imagegen-train-manifest --profile "${IMAGE_PROFILE}" \
      --backend cuda --checkpoint "${checkpoint}" --manifest "${manifest}" \
      --max-train-shard-bytes 1073741824 \
    >"${OUTPUT}/${name}_telemetry.txt"
  grep -Fqx 'shard_already_completed=false' "${OUTPUT}/${name}_telemetry.txt" || {
    echo "${name} did not install a new imagegen shard" >&2; exit 1;
  }
  grep -Eq '^backend_device_local_update_calls=[1-9][0-9]*$' \
    "${OUTPUT}/${name}_telemetry.txt" || {
    echo "${name} did not execute physical CUDA local updates" >&2; exit 1;
  }
  require_literal_once "${OUTPUT}/${name}_resource.json" within_limit true
  require_literal_once "${OUTPUT}/${name}_resource.json" sampler_ok true
}

run_guarded_shard uninterrupted_shard1 "${UNINTERRUPTED}" "${FIRST_MANIFEST}"
run_guarded_shard uninterrupted_shard2 "${UNINTERRUPTED}" "${SECOND_MANIFEST}"
run_guarded_shard resumed_shard1 "${RESUMED}" "${FIRST_MANIFEST}"
"${BIN}" imagegen-verify --profile "${IMAGE_PROFILE}" --checkpoint "${RESUMED}" >/dev/null
"${BIN}" imagegen-inspect --checkpoint "${RESUMED}" \
  >"${OUTPUT}/controlled_stop_checkpoint_inspection.txt"
# The next invocation is a new process and reloads the verified checkpoint.
run_guarded_shard resumed_shard2 "${RESUMED}" "${SECOND_MANIFEST}"

"${BIN}" imagegen-verify --profile "${IMAGE_PROFILE}" --checkpoint "${UNINTERRUPTED}" >/dev/null
"${BIN}" imagegen-verify --profile "${IMAGE_PROFILE}" --checkpoint "${RESUMED}" >/dev/null
"${BIN}" imagegen-inspect --checkpoint "${UNINTERRUPTED}" >"${OUTPUT}/uninterrupted_inspection.txt"
"${BIN}" imagegen-inspect --checkpoint "${RESUMED}" >"${OUTPUT}/resumed_inspection.txt"
grep -Fqx "training_step=${TOTAL_RECORDS}" "${OUTPUT}/uninterrupted_inspection.txt" || {
  echo "uninterrupted imagegen record count mismatch" >&2; exit 1;
}
grep -Fqx "training_step=${TOTAL_RECORDS}" "${OUTPUT}/resumed_inspection.txt" || {
  echo "resumed imagegen record count mismatch" >&2; exit 1;
}
grep -Fqx 'completed_shards=2' "${OUTPUT}/uninterrupted_inspection.txt"
grep -Fqx 'completed_shards=2' "${OUTPUT}/resumed_inspection.txt"
UNINTERRUPTED_SHA256="$(sha256sum -- "${UNINTERRUPTED}" | awk '{print $1}')"
RESUMED_SHA256="$(sha256sum -- "${RESUMED}" | awk '{print $1}')"
[[ "${UNINTERRUPTED_SHA256}" == "${RESUMED_SHA256}" ]] || {
  echo "resumed imagegen checkpoint differs from uninterrupted checkpoint" >&2
  exit 1
}

READINESS_SHA256="$(sha256sum -- "${READINESS_REPORT}" | awk '{print $1}')"
FIRST_SHA256="$(sha256sum -- "${FIRST_MANIFEST}" | awk '{print $1}')"
SECOND_SHA256="$(sha256sum -- "${SECOND_MANIFEST}" | awk '{print $1}')"
TEMPORARY="$(mktemp "${OUTPUT}/.imagegen-resume-equivalence.XXXXXX")"
cat >"${TEMPORARY}" <<EOF
{
  "schema": "rlf-imagegen-v100-resume-equivalence-v1",
  "passed": true,
  "physical_training_performed": true,
  "physical_cuda_local_updates_verified": true,
  "test_doubles": false,
  "claim_eligible": false,
  "profile": "${IMAGE_PROFILE}",
  "campaign_profile": "${CAMPAIGN_PROFILE}",
  "campaign_binding_sha256": "${RLF_CAMPAIGN_BINDING_SHA256}",
  "readiness_report_sha256": "${READINESS_SHA256}",
  "source_manifest_sha256": "${EXPECTED_SOURCE_SHA256}",
  "solstice_binary_sha256": "${EXPECTED_BINARY_SHA256}",
  "gpu_uuid": "${GPU_UUID}",
  "first_manifest_sha256": "${FIRST_SHA256}",
  "second_manifest_sha256": "${SECOND_SHA256}",
  "first_manifest_records": ${FIRST_RECORDS},
  "second_manifest_records": ${SECOND_RECORDS},
  "total_training_records": ${TOTAL_RECORDS},
  "controlled_stop_after_shards": 1,
  "checkpoint_reloaded_in_new_process": true,
  "uninterrupted_checkpoint_sha256": "${UNINTERRUPTED_SHA256}",
  "resumed_checkpoint_sha256": "${RESUMED_SHA256}",
  "byte_identical": true,
  "frontier_claim_authorized": false,
  "state_of_art_claim_authorized": false,
  "efficiency_claim_authorized": false
}
EOF
mv -f -- "${TEMPORARY}" "${OUTPUT}/resume_equivalence.json"
(
  cd "${OUTPUT}"
  find . -maxdepth 1 -type f ! -name artifact_manifest.sha256 -printf '%f\0' | \
    sort -z | xargs -0 sha256sum >artifact_manifest.sha256
)
printf 'imagegen_resume_equivalence_passed=true\ncheckpoint_sha256=%s\nclaim_eligible=false\n' \
  "${RESUMED_SHA256}"
