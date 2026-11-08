#!/bin/sh
# Build every kernels/*.ptx into kernels/<name>.sm_61.cubin with CUDA 12.4 ptxas (in Docker).
set -e
here=$(cd "$(dirname "$0")" && pwd)
img=egpu-ptxas:12.4
docker image inspect "$img" >/dev/null 2>&1 || docker build -q -t "$img" "$here"
for ptx in "$here"/*.ptx; do
  name=$(basename "$ptx" .ptx)
  docker run --rm -v "$here":/k "$img" ptxas -arch=sm_61 -v -o "$name.sm_61.cubin" "$name.ptx"
done
