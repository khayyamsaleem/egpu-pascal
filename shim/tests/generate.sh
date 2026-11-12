#!/bin/bash
# Text generation with MAX on the GTX 1080 Ti (or the CPU reference). Run inside the MAX container
# after maxpatch/install.sh, with pascal/shimd.py serving on the Mac.
#   bash /repo/shim/tests/generate.sh gpu "The capital of France is"
#   bash /repo/shim/tests/generate.sh cpu "The capital of France is"      # reference
#   Gemma 4 E2B (MAX nightly 26.8.0.dev2026110805 + maxpatch; E-series needs the graph-API arch):
#   MODEL=google/gemma-4-E2B-it ENCODING=bfloat16 MAX_GEMMA4_ACTIVATION_MIB=1024 \
#       bash /repo/shim/tests/generate.sh gpu "The capital of France is" --no-prefer-module-v3
# Greedy (--top-k 1) so GPU and CPU output can be compared token for token.
DEV=${1:-gpu}; PROMPT=${2:-"The capital of France is"}; shift 2
export LD_LIBRARY_PATH=/repo/shim/build SHIM_MODE=remote SHIM_LOG=${SHIM_LOG:-/tmp/shim.log}
export MODULAR_NVPTX_COMPILER_PATH=${MODULAR_NVPTX_COMPILER_PATH:-/usr/local/lib/python3.12/site-packages/nvidia/cuda_nvcc/bin/ptxas}
exec max generate --model ${MODEL:-HuggingFaceTB/SmolLM2-135M-Instruct} --quantization-encoding ${ENCODING:-float32} \
    --devices "$DEV" --max-length ${MAXLEN:-256} --max-batch-size 1 --max-new-tokens ${TOKENS:-64} --top-k 1 \
    --no-device-graph-capture --prompt "$PROMPT" "$@" 2>&1 | grep -v -e UserWarning -e warnings.warn -e click/core.py -e "for param in"
