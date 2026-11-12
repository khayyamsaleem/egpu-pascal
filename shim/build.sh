#!/bin/sh
# Build the libcuda.so.1 / libnvidia-ml.so.1 shims (Linux aarch64; run inside the MAX container).
set -e
cd "$(dirname "$0")"; B=${BUILD:-build}; mkdir -p $B
python3 src/gen_stubs.py src/cuda_symbols.txt src/cuda_shim.c $B/cuda_stubs.c CUresult 801
python3 src/gen_stubs.py src/nvml_symbols.txt src/nvml_shim.c $B/nvml_stubs.c int 3
gcc -shared -fPIC -O2 -Wall -Wno-unused -o $B/libcuda.so.1 src/cuda_shim.c $B/cuda_stubs.c -lpthread
gcc -shared -fPIC -O2 -Wall -Wno-unused -o $B/libnvidia-ml.so.1 src/nvml_shim.c $B/nvml_stubs.c
for l in libcuda.so.1 libnvidia-ml.so.1; do echo "$l: $(nm -D --defined-only $B/$l | grep -c ' T ') exported functions"; done
