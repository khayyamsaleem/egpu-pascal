# shim/: run Modular MAX / Mojo GPU code on the GTX 1080 Ti

MAX talks to NVIDIA GPUs only through `libcuda.so.1` + `libnvidia-ml.so.1` (it
`dlsym`s 118 `cu*` and 16 `nvml*` names). These shims replace both, so MAX
needs no patches:

```
Mojo / MAX (Linux aarch64 container)
  -> shim/build/libcuda.so.1, libnvidia-ml.so.1      SHIM_MODE=remote
  -> TCP host.docker.internal:7461
  -> pascal/shimd.py on the Mac                       nvpascal.Device + ptxas sm_61 JIT
  -> GTX 1080 Ti over TinyGPU
```

## Run

```sh
# Mac: bring the card up, then serve it
PY=~/.venvs/tinygpu/bin/python
$PY pascal/reset_engines.py && $PY pascal/gr.py
$PY pascal/shimd.py --vram-limit-gib 2          # MAX pools ~90% of what is reported

# container with MAX (python:3.12 + `pip install modular nvidia-cuda-nvcc-cu12==12.4.131`)
docker run -d --name maxlab -v ~/dev/egpu-pascal:/repo python:3.12 sleep infinity
docker exec maxlab sh -c 'cd /repo/shim && ./build.sh'
docker exec maxlab sh -c 'cd /repo/shim/tests &&
  export LD_LIBRARY_PATH=/repo/shim/build SHIM_MODE=remote SHIM_QUIET=1 \
         MODULAR_NVPTX_COMPILER_PATH=/usr/local/lib/python3.12/site-packages/nvidia/cuda_nvcc/bin/ptxas &&
  mojo build --target-accelerator=nvidia:61 -o /tmp/vadd vadd.mojo && /tmp/vadd'
```

`tests/vadd.mojo` prints `correct: 4096 of 4096 / RESULT: PASS` on the card. In sim
mode (no `SHIM_MODE`) it fails, because kernels are not executed there; that is the
negative control.

## Modes and knobs

| env | meaning |
|---|---|
| `SHIM_MODE=remote` | forward to `shimd` (default: discovery/sim mode, in-process fake GPU) |
| `SHIM_SERVER=host:port` | default `host.docker.internal:7461` |
| `SHIM_QUIET=1` | log only launches, module loads, stubs and errors |
| `SHIM_LOG=path` | log file instead of stderr |
| `SHIM_CAPTURE_DIR=dir` | save every module image (PTX/cubin) MAX loads |

## Status

Implemented: device/attributes (CC 6.1, 28 SMs), contexts, streams/events (synchronous),
alloc/free, H2D/D2H/D2D copies, memset 8/16/32, PTX/cubin modules (server JITs PTX
with CUDA 12.4 `ptxas -arch=sm_61`), `cuLaunchKernel`/`cuLaunchKernelEx` (parameter
layout from the cubin's `KPARAM_INFO`), function attributes. Stubbed (NOT_SUPPORTED):
CUDA graphs, VMM, multicast, tensor maps, peer access, host-mapped device pointers.
Run `max generate` with `--no-device-graph-capture` (no CUDA graphs).

Server side (`pascal/shimd.py` → `nvpascal`), modules are loaded like the CUDA driver
does: per-kernel local memory, module globals and constant banks (module-level and
per-kernel `.nv.constantN.<k>`) with `R_CUDA_32/64` relocations, a `vprintf` stub,
barrier counts, and an instruction-cache invalidate after uploads. Module space is
reclaimed when every client has disconnected. `SHIMD_DUMP=dir` on the server keeps
every module it loads.

## Tests (container, after `maxpatch/install.sh`)

| test | checks |
|---|---|
| `tests/vadd.mojo`, `tests/devinfo.mojo` | a Mojo GPU program end to end; device attributes |
| `tests/shfl.mojo` | warp shuffles (needs the PTX 6.3 target patch) |
| `tests/graph_add.py add\|matmul` | MAX graphs: fused elementwise, fp32 matmul vs numpy |
| `tests/ops_check.py` | 10 ops at SmolLM2 shapes vs numpy |
| `tests/exp_check.py` | transcendentals; `tanh` still doesn't compile for sm_61 |
| `tests/generate.sh gpu\|cpu "<prompt>"` | `max generate` SmolLM2-135M fp32, greedy; compare GPU vs CPU |

`tools/ptxas-capture.sh` as `MODULAR_NVPTX_COMPILER_PATH` keeps every PTX MAX compiles.
