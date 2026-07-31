#!/usr/bin/env bash
set -euo pipefail

if ! command -v apt-get >/dev/null 2>&1; then
  echo "This helper targets Ubuntu/Debian systems with apt-get." >&2
  exit 1
fi

SUDO=""
if [[ ${EUID} -ne 0 ]]; then
  if ! command -v sudo >/dev/null 2>&1; then
    echo "sudo is required to install packages." >&2
    exit 1
  fi
  SUDO="sudo"
fi

${SUDO} apt-get update
${SUDO} apt-get install -y \
  build-essential \
  cmake \
  ninja-build \
  pkg-config \
  libpng-dev \
  libjpeg-dev \
  git \
  ca-certificates

echo
echo "Base build dependencies are installed."
if command -v nvidia-smi >/dev/null 2>&1; then
  echo "NVIDIA driver detected:"
  nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
else
  echo "WARNING: nvidia-smi was not found. Install a current NVIDIA Linux driver first." >&2
fi

if command -v nvcc >/dev/null 2>&1; then
  echo "CUDA compiler detected:"
  nvcc --version | tail -n 4
else
  cat >&2 <<'MSG'
WARNING: nvcc was not found.
Install a current CUDA Toolkit from NVIDIA's Ubuntu repository, then reopen the
shell and verify `nvcc --version`. The RTX 3090 build preset compiles for
compute capability 8.6 (sm_86).
MSG
fi
