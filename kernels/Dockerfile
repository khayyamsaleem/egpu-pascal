# ptxas for sm_61 (Pascal).  CUDA 13 dropped Pascal; 12.4 is the pinned toolchain.
# NVIDIA's pip wheel ships a linux-aarch64 ptxas, so this runs natively under
# OrbStack / Docker on Apple Silicon.
FROM python:3.12-slim
RUN pip install --no-cache-dir nvidia-cuda-nvcc-cu12==12.4.131
ENV PATH=/usr/local/lib/python3.12/site-packages/nvidia/cuda_nvcc/bin:$PATH
WORKDIR /k
