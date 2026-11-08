HANDOFF (session 6, 2026-11-08) — supersedes HANDOFF-SESSION5.md and section 3/4 of HANDOFF-TO-OPUS.md
=====================================================================================================

TARGET (user): run models with Modular MAX on this 1080 Ti; user is OK patching modular/max locally.

RUNG STATUS
  0-3  DONE (unchanged).
  4    **DONE.**  GR channel context-switches through FECS; PGRAPH executes PASCAL_COMPUTE_B
       methods; an sm_61 kernel (ptxas 12.4) dispatched via QMD v02_01 writes a correct
       vector-add into host memory, with negative controls, 3/3 repeat.
  5    NOT STARTED.

RUN ORDER (warm card)
  ~/.venvs/tinygpu/bin/python pascal/reset_engines.py
  ~/.venvs/tinygpu/bin/python pascal/gr.py          # GR BRING-UP PASS
  ~/.venvs/tinygpu/bin/python pascal/compute.py     # RT-4 COMPUTE DISPATCH PASS
  (pascal/grchan.py = RT-4 GR channel only; pascal/fifo.py = RT-3 CE; pascal/ce_bw*.py = bandwidth)
  Cold card: devinit.py firmware/gp102_vbios.rom, fb.py first.
  Kernel toolchain: docker container `cuda12` (python:3.12-slim + nvidia-cuda-nvcc-cu12==12.4.131),
  ptxas at /usr/local/lib/python3.12/site-packages/nvidia/cuda_nvcc/bin/ptxas, ~/egpu-pascal/kernels -> /k.

WHAT FIXED RUNG 4 (do not re-learn)
  * Runlist 0 = GR + CE (engine 7, PMC bit 6) + CE1 (engine 8, bit 7).  Those GR copy engines must be
    out of reset or PBDMA0 raises INTR_0 bit 24 (ENG_RESET) at TSG load and the engine-0 ctxsw is never
    requested.  grchan.py / compute.py enable every engine on GR's runlist.
  * Compute SET_REPORT_SEMAPHORE_D: OPERATION_RELEASE = 0 (1 -> DATA_ERROR INVALID_ENUM).
  * sm_6x cbuf0 ABI: c[0][0x8] blockDim, c[0][0x14] gridDim, c[0][0x20] stack, params 0x140.
  * Session-5 misreadings corrected: 0x409014 = IRQMCLR (mask is 0x409018 = 0x8704);
    0x4098cc/0x40984c are write-only mailbox SET/CLEAR; PCCSR 0x800004 bits 24:27 = status,
    28 = busy, 1 = NEXT.  FECS DMEM/ICD are host-locked in LS mode; use TRACEPC (0x148/0x14c).

MEASURED CEILING
  VRAM at the 405 MHz idle clock (refpll path, MPLL off): CE traffic 24.5 GB/s (one CE),
  29.2 GB/s (two CEs) -> DRAM-bound.  Host->VRAM 0.69 GB/s.  Reclocking memory is unsolved on Pascal.

NEXT, IN ORDER
  1. DONE: SM clock 607 MHz, SM DRAM bandwidth 30.1 GB/s (pascal/bench.py); runtime pascal/nvpascal.py.
  2. Read-only reclock feasibility: VBIOS perf/memory-clock tables, PLL register protection.
  3. Driver as a service: alloc / copy / load-cubin / launch / sync API over pascal/*.py.
  4. MAX: Linux-aarch64 MAX in a container + a libcuda/NVML shim forwarding to (3); MAX compile via
     MODULAR_NVPTX_COMPILER_PATH=<CUDA 12 ptxas>, target table patch for sm_61; then SIMT fallbacks for
     matmul / attention / quant kernels (MAX built-in matmul needs sm_80).
