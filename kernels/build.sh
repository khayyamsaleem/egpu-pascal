#!/bin/sh
# Build every kernels/*.ptx (CUDA 12.4 ptxas) and kernels/*.cu (CUDA 12.8 nvcc) into
# kernels/<name>.sm_61.cubin, in Docker.
set -e
here=$(cd "$(dirname "$0")" && pwd)
img=egpu-ptxas:12.4
docker image inspect "$img" >/dev/null 2>&1 || docker build -q -t "$img" "$here"
for ptx in "$here"/*.ptx; do
  name=$(basename "$ptx" .ptx)
  docker run --rm -v "$here":/k "$img" ptxas -arch=sm_61 -v -o "$name.sm_61.cubin" "$name.ptx"
done
nvimg=cuda-nvcc:12.8
docker image inspect "$nvimg" >/dev/null 2>&1 || docker build -q -t "$nvimg" -f "$here/Dockerfile.nvcc" "$here"
for cu in "$here"/*.cu; do
  [ -e "$cu" ] || continue
  name=$(basename "$cu" .cu)
  docker run --rm -v "$here":/k -w /k "$nvimg" nvcc -Wno-deprecated-gpu-targets -O3 -arch=sm_61 -cubin -Xptxas -v -o "$name.sm_61.cubin" "$name.cu"
done
