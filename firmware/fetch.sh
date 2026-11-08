#!/bin/sh
# Fetch NVIDIA's signed GP102 firmware from linux-firmware into firmware/gp102.
# (linux-firmware uses symlinks between chips; cp -L dereferences them.)
# The VBIOS is card-specific: dump it with  pascal/dump_vbios.py  (writes firmware/gp102_vbios.rom).
set -e
here=$(cd "$(dirname "$0")" && pwd)
tmp=$(mktemp -d)
git clone -q --depth 1 --filter=blob:none --sparse \
  https://git.kernel.org/pub/scm/linux/kernel/git/firmware/linux-firmware.git "$tmp/lf"
(cd "$tmp/lf" && git sparse-checkout set nvidia/gp102 nvidia/gp100 nvidia/gp104 nvidia/gp107 nvidia/gp108)
mkdir -p "$here/gp102"
for d in acr gr sec2 nvdec; do mkdir -p "$here/gp102/$d"; cp -L "$tmp/lf/nvidia/gp102/$d/"* "$here/gp102/$d/"; done
cp "$here/gp102/acr/bl.bin" "$here/bl.bin"
rm -rf "$tmp"
(cd "$here" && shasum -a 256 -c SHA256SUMS 2>/dev/null | grep -v ': OK' || true)
echo "done; verify with: (cd firmware && shasum -a 256 -c SHA256SUMS)"
