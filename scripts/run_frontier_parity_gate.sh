#!/usr/bin/env bash
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BIN="${ROOT}/build/ubuntu-release/rlf_general_benchmark"
EVIDENCE="${1:-${ROOT}/benchmarks/general_frontier/evidence_template.tsv}"
OUTPUT="${2:-${ROOT}/results/general_frontier/frontier_parity_gate.json}"
TRAINING_MANIFEST="${3:-}"
[[ -x "${BIN}" ]] || { echo "CPU benchmark gate not built. Run the Ubuntu release build first." >&2; exit 1; }
mkdir -p "$(dirname "${OUTPUT}")"
ARGS=(--evidence "${EVIDENCE}" --output "${OUTPUT}")
if [[ -n "${TRAINING_MANIFEST}" ]]; then
  ARGS+=(--training-manifest "${TRAINING_MANIFEST}")
fi
"${BIN}" "${ARGS[@]}"
STATUS=$?
cat "${OUTPUT}"
exit ${STATUS}
