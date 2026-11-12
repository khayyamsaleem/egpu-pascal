#!/bin/sh
# Shim libraries for programs built with nvcc against the CUDA 12 runtime (llama.cpp), built in the
# `llamacpp` container (CUDA 12.8 headers):
#   libcuda.so.1 / libnvidia-ml.so.1   the driver shim (shim/src), rebuilt against this glibc
#   libcudart.so.12                    shim/cudart: the runtime API over our libcuda
#   libcublas.so.12, libcublasLt.so.12 shim/cublas: SIMT GEMMs / NOT_SUPPORTED stubs
# Symbol lists (cudart/cudart_symbols.txt, cublas/*_symbols.txt) are what libggml-cuda.so imports
# (`nm -D --undefined-only`); anything not implemented gets a logging stub.
set -e
cd "$(dirname "$0")"; B=${BUILD:-build-cuda12}; mkdir -p $B
BUILD=$B sh build.sh
python3 src/gen_stubs.py cudart/cudart_symbols.txt cudart/cudart_shim.c $B/cudart_stubs.c int 801 cudart_log_stub
gcc -shared -fPIC -O2 -Wall -Wno-unused -Wno-deprecated-declarations -I/usr/local/cuda/include \
    -o $B/libcudart.so.12 cudart/cudart_shim.c $B/cudart_stubs.c -L$B -l:libcuda.so.1 -lpthread \
    -Wl,-soname,libcudart.so.12 -Wl,--version-script=cudart/cudart.map
python3 src/gen_stubs.py cublas/cublas_symbols.txt cublas/cublas_shim.cu $B/cublas_stubs.c int 15 cublas_log_stub
nvcc -Wno-deprecated-gpu-targets -diag-suppress 20011 -O3 -arch=sm_61 --no-compress -shared -Xcompiler -fPIC -cudart shared \
    -o $B/libcublas.so.12 cublas/cublas_shim.cu $B/cublas_stubs.c \
    -Xlinker -soname=libcublas.so.12 -Xlinker --version-script=cublas/cublas.map
echo 'void cublaslt_log_stub(const char *n);' > $B/cublaslt_impl.c
python3 src/gen_stubs.py cublas/cublasLt_symbols.txt $B/cublaslt_impl.c $B/cublaslt_stubs.c int 15 cublaslt_log_stub
printf '#include <stdio.h>\nvoid cublaslt_log_stub(const char *n) { fprintf(stderr, "[cublasLt-shim] %%s -> NOT_SUPPORTED (stub)\\n", n); }\n' >> $B/cublaslt_stubs.c
gcc -shared -fPIC -O2 -o $B/libcublasLt.so.12 $B/cublaslt_stubs.c -Wl,-soname,libcublasLt.so.12 -Wl,--version-script=cublas/cublasLt.map
for l in libcudart.so.12 libcublas.so.12 libcublasLt.so.12; do echo "$l: $(nm -D --defined-only $B/$l | grep -c ' T ') exported functions"; done
