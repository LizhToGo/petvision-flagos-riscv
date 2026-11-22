#!/usr/bin/env bash
# Run the PetVision / conv2d checks inside an extracted SpacemiT x64-emulation
# rootfs (no Docker daemon needed).  See README.md section 5.
#
# Prerequisites:
#   1. rootfs extracted with tools/fetch_oci_rootfs.py (default /var/tmp/sm-rootfs)
#   2. third_party/FlagGems checked out at third_party/FLAGGEMS_COMMIT.txt
#   3. the SpacemiT runtime (spine-rpc-server under qemu) already started inside
#      the image, exactly as the FlagTree spacemit3.6 CI does
#
# NOTE: the public image tag 202607-3.6-base ships torch/torchvision only.  Until
# FlagTree's third_party/spacemit/scripts/install_flagtree_plugin.sh has been run
# inside it, the first version check below fails with "No module named 'triton'"
# (this is what happened on 2026-11-22, see results/spacemit_emu_20261122/).
#
# Usage:
#   WEIGHTS=/path/to/best.pt ./scripts/run_in_spacemit_rootfs.sh
#
# Everything the script mounts is unmounted on exit; results are copied back to
# $OUT on the host.
set -euo pipefail

ROOTFS=${ROOTFS:-/var/tmp/sm-rootfs}
HERE=$(cd "$(dirname "$0")/.." && pwd)
OUT=${OUT:-$HERE/results/spacemit_emu}
WEIGHTS=${WEIGHTS:-}
WARMUP=${WARMUP:-1}
REPEAT=${REPEAT:-5}

[[ -d "$ROOTFS/usr/bin" ]] || { echo "no rootfs at $ROOTFS (see README 5.1)" >&2; exit 1; }
[[ -d "$HERE/third_party/FlagGems/src/flag_gems" ]] || {
  echo "missing FlagGems checkout; see README section 2" >&2; exit 1; }

mounts=()
cleanup() {
  for mount_point in "${mounts[@]}"; do
    sudo umount "$mount_point" 2>/dev/null || true
  done
}
trap cleanup EXIT

for fs in proc sys dev; do
  sudo mkdir -p "$ROOTFS/$fs"
  if ! mountpoint -q "$ROOTFS/$fs"; then
    sudo mount --bind "/$fs" "$ROOTFS/$fs"
    mounts+=("$ROOTFS/$fs")
  fi
done

# Code is copied in; results are copied back.  The checkpoint is bind-mounted
# read-only so the run cannot modify the course artifact.
sudo mkdir -p "$ROOTFS/root/petvision" "$ROOTFS/root/FlagGems"
sudo rsync -a --delete --exclude .git --exclude results --exclude third_party \
  "$HERE/" "$ROOTFS/root/petvision/"
sudo rsync -a --delete --exclude .git "$HERE/third_party/FlagGems/" "$ROOTFS/root/FlagGems/"

WEIGHT_ARG=()
if [[ -n "$WEIGHTS" ]]; then
  sudo mkdir -p "$ROOTFS/weights"
  sudo mount --bind "$WEIGHTS" "$ROOTFS/weights/best.pt"
  sudo mount -o remount,ro,bind "$ROOTFS/weights/best.pt"
  mounts+=("$ROOTFS/weights/best.pt")
  WEIGHT_ARG=(--weights /weights/best.pt)
fi

mkdir -p "$OUT"
sudo rm -rf "$ROOTFS/root/petvision/results"

# The rootfs must contain triton==3.6.0+spacemit.a5 and the RISC-V runtime.
# The public base tag used in the recorded attempt did not; FlagGems is used
# from the checkout through PYTHONPATH so the candidate patch is what runs.
run_inside() {
  sudo chroot "$ROOTFS" /usr/bin/env -i \
    HOME=/root \
    PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
    GEMS_VENDOR=spacemit \
    SPINE_TRITON_TARGET_ARCH=riscv64 \
    PYTHONPATH=/root/FlagGems/src \
    bash -lc "cd /root/petvision && $*"
}

run_inside "python3 -c 'import torch, triton; print(torch.__version__, triton.__version__)'"
run_inside "python3 conv2d_case.py --backend flag_gems --require-vendor spacemit \
  --warmup $WARMUP --repeat $REPEAT --out results/conv2d"
if [[ -n "$WEIGHTS" ]]; then
  run_inside "python3 petvision_forward.py --backend flag_gems --require-vendor spacemit \
    ${WEIGHT_ARG[*]} --warmup 1 --repeat 3 --out results/model"
fi

sudo rsync -a "$ROOTFS/root/petvision/results/" "$OUT/"
echo "results copied to $OUT"