# llama.cpp build/run image for the GTX 1080 Ti (sm_61): CUDA 12.8 toolkit (the last series that
# compiles Pascal) on linux-aarch64 (sbsa). At runtime the CUDA libraries come from shim/build-cuda12.
FROM ubuntu:22.04
ENV PATH=/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
RUN apt-get update && apt-get install -y --no-install-recommends wget ca-certificates && \
  wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/sbsa/cuda-keyring_1.1-1_all.deb && \
  dpkg -i cuda-keyring_1.1-1_all.deb && apt-get update && \
  apt-get install -y --no-install-recommends cuda-nvcc-12-8 cuda-cudart-dev-12-8 libcublas-dev-12-8 \
    cmake git g++ make curl python3 && rm -rf /var/lib/apt/lists/*
WORKDIR /opt
