#!/bin/bash
# Make modular 26.6.0 (MAX 26.6.0 / Mojo 1.1.0) generate working code for sm_61 (GTX 1080 Ti).
# Run INSIDE the MAX container, with this repo at /repo:
#     bash /repo/maxpatch/install.sh            # patch + rebuild + install
#     bash /repo/maxpatch/install.sh --revert   # restore every original package
#
# 1. std.mojoc: GTX 1080 Ti target "+ptx50,+sm_61" -> "+ptx63,+sm_61" (warp shuffles), in place.
# 2. kernel packages: apply patches/*.patch to a modular checkout at tag max/v26.6.0 and
#    `mojo precompile` each touched package over the shipped .mojoc (originals kept as .orig).
# 3. clear the MAX/Mojo kernel caches so nothing compiled for the old target is reused.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
SRC=${MODULAR_SRC:-/repo/upstream/modular}           # git clone --branch max/v26.6.0 modular/modular
LIB=$(python3 -c 'import importlib.util,os;print(os.path.join(list(importlib.util.find_spec("modular").submodule_search_locations)[0],"lib","mojo"))')
PKGS=$(cat "$HERE"/patches/*.patch | sed -n 's#^+++ b/max/kernels/src/\([^/]*\)/.*#\1#p' | sort -u)

if [ "${1:-}" = "--revert" ]; then
    for f in "$LIB"/*.mojoc.orig; do [ -e "$f" ] && cp -p "$f" "${f%.orig}" && echo "restored ${f%.orig}"; done
    rm -rf ~/.cache/modular/.max_cache ~/.cache/modular/.mojo_cache ~/.cache/modular/.mogg_cache
    exit 0
fi

python3 -c 'import zstandard' 2>/dev/null || pip install -q zstandard
python3 "$HERE/patch_std_target.py"

[ "$(git -C "$SRC" describe --tags --exact-match 2>/dev/null || true)" = "max/v26.6.0" ] \
    || { echo "error: $SRC is not a checkout of max/v26.6.0"; exit 1; }
git -C "$SRC" checkout -q -- max/kernels/src
for p in "$HERE"/patches/*.patch; do git -C "$SRC" apply "$p" && echo "applied $(basename "$p")"; done

mkdir -p /tmp/maxpatch
for pkg in $PKGS; do
    echo "precompiling $pkg ..."
    (cd "$SRC/max/kernels/src" && mojo precompile "$pkg" -o "/tmp/maxpatch/$pkg.mojoc" 2>&1 \
        | grep -E "error:" -A4 && exit 1 || true)
    [ -e "$LIB/$pkg.mojoc.orig" ] || cp -p "$LIB/$pkg.mojoc" "$LIB/$pkg.mojoc.orig"
    cp "/tmp/maxpatch/$pkg.mojoc" "$LIB/$pkg.mojoc"
    echo "installed $LIB/$pkg.mojoc"
done
rm -rf ~/.cache/modular/.max_cache ~/.cache/modular/.mojo_cache ~/.cache/modular/.mogg_cache
echo "maxpatch: done (packages: std $PKGS)"
