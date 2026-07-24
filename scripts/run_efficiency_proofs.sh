#!/usr/bin/env bash
set -euo pipefail

build_dir="${1:-build/ubuntu-release}"
output="${2:-results/efficiency/rlf_frontier_efficiency_proof_v2.json}"
mkdir -p "$(dirname "$output")"

"${build_dir}/rlf_efficiency_proof" --output "$output"
printf 'proof_report=%s\n' "$output"
