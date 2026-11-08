#!/bin/sh
# Build the libcuda.so.1 / libnvidia-ml.so.1 shims (Linux aarch64; run inside the MAX container).
set -e
cd "$(dirname "$0")"; mkdir -p build
python3 src/gen_stubs.py src/cuda_symbols.txt src/cuda_shim.c build/cuda_stubs.c CUresult 801
python3 src/gen_stubs.py src/nvml_symbols.txt src/nvml_shim.c build/nvml_stubs.c int 3
gcc -shared -fPIC -O2 -Wall -Wno-unused -o build/libcuda.so.1 src/cuda_shim.c build/cuda_stubs.c -lpthread
gcc -shared -fPIC -O2 -Wall -Wno-unused -o build/libnvidia-ml.so.1 src/nvml_shim.c build/nvml_stubs.c
for l in libcuda.so.1 libnvidia-ml.so.1; do echo "$l: $(nm -D --defined-only build/$l | grep -c ' T ') exported functions"; done
