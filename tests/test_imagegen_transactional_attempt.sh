#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORK="$(mktemp -d)"
trap 'rm -rf -- "${WORK}"' EXIT
mkdir -p "${WORK}/fake-root/scripts" "${WORK}/result" "${WORK}/failures"

cat >"${WORK}/fake-root/scripts/run_general_cuda_with_vram_guard.sh" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
TRACE=""; SUMMARY=""
while (($# > 0)); do
  case "$1" in
    --trace) TRACE="$2"; shift 2 ;;
    --summary) SUMMARY="$2"; shift 2 ;;
    --profile|--gpu-index|--expected-uuid) shift 2 ;;
    --) shift; break ;;
    *) exit 91 ;;
  esac
done
printf 'synthetic trace\n' >"${TRACE}"
printf '{"within_limit":true,"sampler_ok":true}\n' >"${SUMMARY}"
"$@"
EOF
chmod +x "${WORK}/fake-root/scripts/run_general_cuda_with_vram_guard.sh"

cat >"${WORK}/fake-solstice" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
COMMAND="$1"; shift
CHECKPOINT=""
while (($# > 0)); do
  case "$1" in
    --checkpoint) CHECKPOINT="$2"; shift 2 ;;
    *) shift ;;
  esac
done
case "${COMMAND}" in
  imagegen-train-manifest)
    printf 'mutated-by-synthetic-training\n' >"${CHECKPOINT}.tmp"
    mv -f -- "${CHECKPOINT}.tmp" "${CHECKPOINT}"
    printf 'backend_device_local_update_calls=1\n'
    ;;
  imagegen-verify)
    exit 9
    ;;
  *)
    exit 90
    ;;
esac
EOF
chmod +x "${WORK}/fake-solstice"

printf 'stable-checkpoint\n' >"${WORK}/model.rlfimg"
printf 'synthetic-manifest\n' >"${WORK}/pairs.tsv"
printf 'stale-finalization\n' >"${WORK}/result/artifact_manifest.tsv"
printf 'stale-sidecar\n' >"${WORK}/result/artifact_manifest.tsv.sha256"
STABLE_SHA="$(sha256sum -- "${WORK}/model.rlfimg" | awk '{print $1}')"

set +e
bash "${ROOT}/scripts/run_checkpoint_transaction.sh" \
  --checkpoint "${WORK}/model.rlfimg" --failure-root "${WORK}/failures" \
  --preserve "${WORK}/result/training_telemetry.txt" -- \
  bash "${ROOT}/scripts/run_imagegen_v100_training_attempt.sh" \
    "${WORK}/fake-root" "${WORK}/fake-solstice" "${WORK}/model.rlfimg" \
    "${WORK}/pairs.tsv" "${WORK}/result" 0 GPU-fixture 77 \
  >"${WORK}/stdout.txt" 2>"${WORK}/stderr.txt"
STATUS=$?
set -e

[[ "${STATUS}" -eq 9 ]]
[[ "$(sha256sum -- "${WORK}/model.rlfimg" | awk '{print $1}')" == "${STABLE_SHA}" ]]
[[ ! -e "${WORK}/result/artifact_manifest.tsv" && \
   ! -e "${WORK}/result/artifact_manifest.tsv.sha256" ]]
grep -q 'checkpoint transaction rolled back after exit 9' "${WORK}/stderr.txt"
grep -Rqx 'reason=command-failed' "${WORK}/failures"
grep -Rqx 'backend_device_local_update_calls=1' "${WORK}/failures"

printf 'imagegen_transactional_attempt_shell_test=pass\nphysical_training_performed=false\ntest_doubles=true\nclaim_eligible=false\n'
