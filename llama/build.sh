#!/bin/sh
# Build llama.cpp's CUDA backend for the GTX 1080 Ti (sm_61), inside the `llamacpp` container
# (CUDA 12.8: the last toolkit series that compiles Pascal). Device code is left uncompressed
# (--no-compress) so shim/cudart can pick the sm_61 ELF out of each fatbin without a decompressor.
# No CUDA VMM, no CUDA graphs, no NCCL: the shim doesn't implement them.
set -e
SRC=${SRC:-/repo/upstream/llama.cpp}; OUT=${OUT:-/opt/llama-build}
cmake -S "$SRC" -B "$OUT" -DCMAKE_BUILD_TYPE=Release \
  -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=61-real \
  -DCMAKE_CUDA_FLAGS="--no-compress -Wno-deprecated-gpu-targets" \
  -DGGML_CUDA_NO_VMM=ON -DGGML_CUDA_GRAPHS=OFF -DGGML_CUDA_NCCL=OFF \
  -DGGML_NATIVE=OFF -DGGML_CPU_ARM_ARCH=armv8.2-a+dotprod+fp16 \
  -DLLAMA_OPENSSL=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF
cmake --build "$OUT" -j "${JOBS:-4}" --target llama-server llama-cli llama-bench
