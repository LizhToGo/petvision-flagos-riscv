#!/usr/bin/env bash
# Source this on the K1 instance before running the harness.
# The shared libraries and the venv live in the private deployment directory;
# nothing here modifies the system Python or the system library path.
set -u
: "${PETVISION_K1:=$HOME/petvision-k1}"
export PETVISION_K1
export LD_LIBRARY_PATH="$PETVISION_K1/libs/root/usr/lib/riscv64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export GEMS_VENDOR=spacemit
export TRITON_CACHE_DIR="$PETVISION_K1/triton-cache"
export PYTHONPATH="$PETVISION_K1/app${PYTHONPATH:+:$PYTHONPATH}"
# shellcheck disable=SC1091
. "$PETVISION_K1/venv/bin/activate"
