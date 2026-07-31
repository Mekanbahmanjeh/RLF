#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  local_rtx_pro_6000.sh --print-workflow [options]
  local_rtx_pro_6000.sh --search-now [options]
  local_rtx_pro_6000.sh --render-create OFFER_ID [options]

This local helper never accepts an API key and never creates, starts, stops, or
destroys a Vast.ai instance. --search-now performs only a read-only offer
search. --render-create prints a command for the operator to inspect and run.

Options:
  --disk-gib N       Container disk requested at creation (default: 2400 GB)
  --min-ram-gib N    Minimum advertised host RAM (default: 1120 GB)
  --min-disk-gib N   Minimum advertised offer disk (default: 2400 GB)
  --image IMAGE      Container image (default: official CUDA 13.3 devel image)
EOF
}

ACTION=print
OFFER_ID=""
# Vast advertises these fields in decimal GB, while the remote readiness gate
# intentionally measures binary GiB. The larger search defaults leave enough
# conversion and filesystem headroom to satisfy 1,024 GiB RAM / 2,048 GiB free
# disk after the container starts.
DISK_GIB=2400
MIN_RAM_GIB=1120
MIN_DISK_GIB=2400
IMAGE='nvidia/cuda:13.3.0-devel-ubuntu24.04'

while (($# > 0)); do
  case "$1" in
    --print-workflow) ACTION=print; shift ;;
    --search-now) ACTION=search; shift ;;
    --render-create)
      ACTION=render
      OFFER_ID="${2:?--render-create requires an offer ID}"
      shift 2
      ;;
    --disk-gib) DISK_GIB="${2:?--disk-gib requires an integer}"; shift 2 ;;
    --min-ram-gib) MIN_RAM_GIB="${2:?--min-ram-gib requires an integer}"; shift 2 ;;
    --min-disk-gib) MIN_DISK_GIB="${2:?--min-disk-gib requires an integer}"; shift 2 ;;
    --image) IMAGE="${2:?--image requires a value}"; shift 2 ;;
    --help|-h) usage; exit 0 ;;
    *) echo "unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

for value in "${DISK_GIB}" "${MIN_RAM_GIB}" "${MIN_DISK_GIB}"; do
  [[ "${value}" =~ ^[1-9][0-9]*$ ]] || {
    echo "disk and RAM values must be positive integers" >&2
    exit 2
  }
done
((DISK_GIB >= 2400 && MIN_DISK_GIB >= 2400 && MIN_RAM_GIB >= 1120)) || {
  echo "phase-one search may not lower the conversion-safe 2400 GB disk / 1120 GB RAM floors" >&2
  exit 2
}
[[ "${IMAGE}" != *[[:space:]]* ]] || {
  echo "container image may not contain whitespace" >&2
  exit 2
}

FILTER="verified=true rentable=true num_gpus=1 compute_cap>=1200 gpu_ram>=90 cuda_vers>=13.0 reliability>=0.99 duration>=5.5 disk_space>=${MIN_DISK_GIB} disk_bw>=1500 cpu_ram>=${MIN_RAM_GIB} direct_port_count>=1 static_ip=true"

print_search() {
  printf 'vastai search offers --type on-demand %q -o %q\n' "${FILTER}" 'dph+'
}

case "${ACTION}" in
  print)
    cat <<'EOF'
# Authenticate with the Vast CLI outside this repository. Never paste a key
# into chat, a shell script, a command transcript, or a repository file.
# Register the public half of a dedicated SSH key before creating an instance:
vastai create ssh-key ~/.ssh/id_ed25519.pub

# Read-only candidate search:
EOF
    print_search
    cat <<'EOF'

# Inspect the selected row and price/contract in the Vast UI. The search is a
# shortlist only; the remote preflight is authoritative for GPU identity,
# memory, compute capability, MIG state, RAM, and disk.

# Render (but do not execute) the create command:
./scripts/vast/local_rtx_pro_6000.sh --render-create OFFER_ID
EOF
    ;;
  search)
    command -v vastai >/dev/null 2>&1 || {
      echo "vastai CLI was not found; install it and authenticate per official Vast documentation" >&2
      exit 2
    }
    vastai search offers --type on-demand "${FILTER}" -o 'dph+'
    ;;
  render)
    [[ "${OFFER_ID}" =~ ^[1-9][0-9]*$ ]] || {
      echo "offer ID must be a positive integer" >&2
      exit 2
    }
    printf '# REVIEW THIS COMMAND; this helper does not execute it.\n'
    printf 'vastai create instance %q --image %q --disk %q --ssh --direct\n' \
      "${OFFER_ID}" "${IMAGE}" "${DISK_GIB}"
    printf 'vastai show instance INSTANCE_ID\n'
    printf 'vastai ssh-url INSTANCE_ID\n'
    ;;
esac
