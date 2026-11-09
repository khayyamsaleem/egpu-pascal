#!/bin/bash
# MODULAR_NVPTX_COMPILER_PATH wrapper: keep a copy of every PTX MAX/Mojo compiles, then run ptxas.
# PTX_CAPTURE_DIR (default /tmp/ptx-capture); PTXAS_REAL (default: the pip CUDA 12.4 ptxas).
REAL=${PTXAS_REAL:-/usr/local/lib/python3.12/site-packages/nvidia/cuda_nvcc/bin/ptxas}
DIR=${PTX_CAPTURE_DIR:-/tmp/ptx-capture}
mkdir -p "$DIR"
for a in "$@"; do
    case "$a" in *.ptx) cp "$a" "$DIR/$(date +%s%N)-$(basename "$a")" 2>/dev/null ;; esac
done
exec "$REAL" "$@"
