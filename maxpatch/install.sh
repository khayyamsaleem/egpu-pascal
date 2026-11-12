#!/bin/bash
# Make a modular (MAX + Mojo) install generate working code for sm_61 (GTX 1080 Ti).
# Run INSIDE the MAX container, with this repo at /repo:
#     bash /repo/maxpatch/install.sh            # patch + rebuild + install
#     bash /repo/maxpatch/install.sh --revert   # restore every original file
#
# Supported installs (patches/<version>/, source checkout and commit below):
#     26.6.0                   upstream/modular       tag  max/v26.6.0
#     26.8.0.dev2026110805     upstream/modular-main  2841b31631e7 (lockfile pin of that nightly)
#
# 1. patch_target.py: the GTX 1080 Ti target "+ptx50,+sm_61" -> "+ptx63,+sm_61" (warp shuffles),
#    byte-patched in whichever package holds it (std.mojoc in 26.6, max.mojoc in 26.8).
# 2. patches/<version>/*.patch applied to the source checkout; each touched package is rebuilt:
#      Mojo/stdlib/std/...        -> mojo precompile std   (over std.mojoc)
#      max/kernels/src/<pkg>/...  -> mojo precompile <pkg> (over <pkg>.mojoc)
#      max/python/max/<file>      -> copied over site-packages/max/<file>
#    Every replaced file is kept as <file>.orig.
# 3. kernels/w8 (the w8_linear custom op behind MAX's int8 path on NVIDIA, patch 0004) is copied
#    to site-packages/max/_maxpatch_kernels/w8 (removed by --revert).
# 4. clear the MAX/Mojo caches so nothing compiled for the old target or code is reused.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
VER=$(python3 -c 'from importlib.metadata import version; print(version("modular"))')
case "$VER" in
    26.6.0)               SRC=${MODULAR_SRC:-/repo/upstream/modular};      REF=max/v26.6.0 ;;
    26.8.0.dev2026110805) SRC=${MODULAR_SRC:-/repo/upstream/modular-main}; REF=2841b31631e7d31ae74f0ca20335ed1541cddec1 ;;
    *) echo "error: no patch set for modular $VER (have: $(ls "$HERE/patches" | tr '\n' ' '))"; exit 1 ;;
esac
PATCHES="$HERE/patches/$VER"
SITE=$(python3 -c 'import importlib.util,os;print(os.path.dirname(list(importlib.util.find_spec("modular").submodule_search_locations)[0]))')
LIB="$SITE/modular/lib/mojo"
clear_caches() { rm -rf ~/.cache/modular/.max_cache ~/.cache/modular/.mojo_cache ~/.cache/modular/.mogg_cache; }

if [ "${1:-}" = "--revert" ]; then
    for f in "$LIB"/*.mojoc.orig $(find "$SITE/max" -name '*.py.orig'); do
        [ -e "$f" ] && cp -p "$f" "${f%.orig}" && echo "restored ${f%.orig}"
    done
    rm -rf "$SITE/max/_maxpatch_kernels"
    clear_caches; exit 0
fi

python3 -c 'import zstandard' 2>/dev/null || pip install -q zstandard

[ "$(git -C "$SRC" rev-parse HEAD)" = "$(git -C "$SRC" rev-parse "$REF^{commit}")" ] \
    || { echo "error: $SRC is not at $REF (modular $VER)"; exit 1; }
FILES=$(cat "$PATCHES"/*.patch | sed -n 's#^+++ b/##p' | sort -u)
git -C "$SRC" checkout -q -- $FILES
for p in "$PATCHES"/*.patch; do git -C "$SRC" apply "$p" && echo "applied $(basename "$p")"; done

backup() { [ -e "$1.orig" ] || cp -p "$1" "$1.orig"; }
mkdir -p /tmp/maxpatch
precompile() {   # <dir to run in> <package> ; errors abort
    echo "precompiling $2 ..."
    if ! (cd "$1" && mojo precompile "$2" -o "/tmp/maxpatch/$2.mojoc" > "/tmp/maxpatch/$2.log" 2>&1); then
        grep -E "error" -A4 "/tmp/maxpatch/$2.log" | head -20; exit 1
    fi
    backup "$LIB/$2.mojoc"; cp "/tmp/maxpatch/$2.mojoc" "$LIB/$2.mojoc"; echo "installed $LIB/$2.mojoc"
}
echo "$FILES" | grep -q '^Mojo/stdlib/std/' && precompile "$SRC/Mojo/stdlib" std
for pkg in $(echo "$FILES" | sed -n 's#^max/kernels/src/\([^/]*\)/.*#\1#p' | sort -u); do
    precompile "$SRC/max/kernels/src" "$pkg"
done
for f in $(echo "$FILES" | sed -n 's#^max/python/max/##p'); do
    backup "$SITE/max/$f"; cp "$SRC/max/python/max/$f" "$SITE/max/$f"; echo "installed $SITE/max/$f"
done
rm -rf "$SITE/max/_maxpatch_kernels"; mkdir -p "$SITE/max/_maxpatch_kernels"
cp -r "$HERE/kernels/w8" "$SITE/max/_maxpatch_kernels/w8"; echo "installed $SITE/max/_maxpatch_kernels/w8"
python3 "$HERE/patch_target.py"     # last: a std rebuilt from source carries the +ptx50 original
clear_caches
echo "maxpatch: modular $VER done"
