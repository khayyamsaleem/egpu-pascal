# Results: a GTX 1080 Ti (GP102) running compute kernels from a MacBook Air M1

Compiled 2026-11-08 from sessions 1–6; updated 2026-11-09 (session 7: first LLM). Every claim below was measured on the
card, with a negative control where one applies. The evidence trail is in
[`docs/evidence-log.md`](docs/evidence-log.md) and the raw output is in [`logs/`](logs/).

**Bottom line:** the card goes from cold power-on to running ptxas-compiled sm_61
kernels, with results landing in Mac host memory, entirely from macOS over
Thunderbolt and with no NVIDIA driver. **Modular MAX now runs an LLM on it:**
`max generate` with SmolLM2-135M-Instruct (fp32) produces text identical to
MAX-on-CPU over 64 greedy tokens, at **6.6 tokens/s** (§9). The card runs at boot
clocks: **607 MHz SM, 405 MHz memory, about 30 GB/s DRAM bandwidth**.

| | |
|---|---|
| Host | MacBook Air M1 (MacBookAir10,1), 16 GB, macOS 26.6.2 |
| Enclosure | Razer Core X V2 (Intel `8086:5786` bridge), TB3/USB4 port, PCIe Gen3 x4 |
| Card | EVGA GTX 1080 Ti, GP102 rev A1 (`10de:1b06`, subsystem `3842:6696`), 11 GB GDDR5X |
| Transport | tinygrad's TinyGPU DriverKit dext + `APLRemotePCIDevice` (tinygrad 0.14.0) |

---

## 1. The ladder

| Rung | Milestone | Status | Proof (script → log) |
|---|---|---|---|
| 0 | Transport, identification, VBIOS POST from macOS, all VRAM usable | **DONE** | `devinit.py`, `fb.py`, `vram_test.py` |
| 1 | A falcon runs microcode we wrote (DMEM/IMEM round trip) | **DONE** | `probes/phase0/rt0_*.py`, `rt1*.py` |
| 2 | ACR secure boot: WPR2 up, signed SEC2 RTOS bootstraps FECS + GPCCS | **DONE** | `acr.py` → `logs/acr-pass.log`, `sec2_rtos.py` → `logs/sec2-rtos.log`, `gr_ctxctl.py` |
| 3 | MMU v2 page tables + a GPFIFO channel; copy engine VRAM↔host memory | **DONE** | `mmu_test.py` (18/18) → `logs/mmu-test.log`, `fifo.py` → `logs/rt3-ce.log` |
| 4a | GR channel: FECS context switch, PGRAPH executes `PASCAL_COMPUTE_B` methods | **DONE** | `grchan.py` → `logs/s6-grchan-pass.log` |
| 4 | **Compute dispatch: an sm_61 vector-add writes correct results to host memory** | **DONE** | `compute.py` → `logs/s6-compute.log` (4059/4059, 3/3 runs) |
| 5a | Unmodified Mojo GPU programs and MAX graphs (elementwise, matmul) run via the libcuda/NVML shim | **DONE** | `shim/tests/vadd.mojo`, `shfl.mojo`, `graph_add.py`, `ops_check.py` (10/10) |
| 5 | **Model inference: `max generate` SmolLM2-135M-Instruct fp32, GPU text == CPU text (64 greedy tokens)** | **DONE** (first model) | `shim/tests/generate.sh` → `logs/s7-smollm2-*.log` |

Run order on a warm card: `reset_engines.py` → `gr.py` → `compute.py` (or
`bench.py`). On a cold card, run `devinit.py firmware/gp102_vbios.rom` then `fb.py` first.

## 2. What the bring-up chain is

Each step is a port of nouveau (and of Mesa for compute), driven over BAR0/BAR1
through the TinyGPU transport. Every wait is a poll; this transport has no
interrupts.

1. **POST from macOS** (`pascal/devinit.py`, `vbios.py`): run the VBIOS DEVINIT
   scripts from the host. Takes about 42 ms. The card comes up with GDDR5X trained
   at the boot clock.
2. **VRAM unlock** (`pascal/fb.py`): boot NVIDIA's signed NVDEC `scrubber.bin`
   once per power cycle to clear the VPR hole. All 11 GB are then usable.
3. **MMU v2** (`pascal/mmu.py`): 49-bit VA, 5-level tables, instance blocks,
   invalidates. Proven 18/18 on the card (a falcon DMA through our page tables,
   plus an MMU fault naming our instance on an unmapped VA).
4. **ACR secure boot** (`pascal/acr.py`, `acr_unload.py`): nouveau's gp102 ACR on
   SEC2 (at `0x87000` in the low aperture, not `0x840000`). WPR2 =
   `0x18040000–0x18080000`; the host is refused inside it. Repeatable, because
   the unload runs first.
5. **SEC2 RTOS → FECS/GPCCS** (`pascal/sec2_rtos.py`): `ACR_BOOTSTRAP_FALCON`
   returns error 0 for both. FECS answers methods (ctx image `0x146500`, zcull
   `0xf0c00`, pm `0x4c00`).
6. **GR init + golden context** (`pascal/gr.py`, `grctx.py`): 6 GPCs, 28 SMs
   (`tpc_nr = [4,4,5,5,5,5]`). The golden image is `0xa5e00` bytes.
7. **FIFO + copy engine** (`pascal/fifo.py`): runlist, TSG, RAMFC, USERD through
   physical BAR1. CE copies VRAM↔VRAM and VRAM↔host memory byte-exact.
8. **GR channel** (`pascal/grchan.py`): the golden image is patched per channel
   (mmio list, priv mappings of pagepool/bundle/attrib, `inst+0x210 = ctx VA | 4`).
   FECS context-switches the channel in (CURRENT_CTX = NEW_CTX = our instance).
9. **Compute** (`pascal/compute.py`, `nvpascal.py`): Mesa's compute state, a QMD
   v02_01 built from NVIDIA's `clc1c0qmd.h`, cbuf0 laid out per the sm_6x ABI,
   `SEND_PCAS_A/SEND_SIGNALING_PCAS_B`, then `WAIT_FOR_IDLE` + report semaphore.

## 3. Measured performance (boot clocks)

| Quantity | Measured | Spec / reference | How |
|---|---|---|---|
| SM clock | **607 MHz** | 1480 base / 1582 boost | `%clock64` vs `%globaltimer`, 10M and 200M cycles (`bench.py`) |
| FP32 peak (derived) | **~4.35 TFLOPS** | ~11.3 | 3584 cores × 2 × 0.607 GHz |
| Memory clock | **405 MHz** (refpll path, MPLL off) | 5505 (P0) | `0x1373f4 = 0x11`, `0x132000` bit 0 clear, refpll M=1 N=59 P=4 |
| DRAM bandwidth, SM kernel | **30.1 GB/s** | 484 GB/s | 16-B ld/st grid-stride copy, 64/256 MiB, 2 grid shapes (`bench.py`) |
| DRAM bandwidth, copy engines | 24.5 GB/s (1 CE), 29.2 GB/s (2 CEs) | — | `ce_bw.py`, `ce_bw2.py`; CE timestamps, so no TB latency |
| Host memory → VRAM (CE over TB) | 0.69 GB/s | link ~3 GB/s | 4 MiB, IOMMU 4 KiB pages |
| Launch + completion latency | **0.17 ms** (was 1.6 ms) end to end, host-observed | — | `test_modules.py` (5000 launches); §9 |
| LLM decode, SmolLM2-135M fp32 | **6.6 tokens/s**, TTFT 0.30 s (35-token prompt) | MAX-CPU in the container: 14 tok/s | 343 launches/token, 29 ms of it GPU time |

**What this means for models.** Token generation is bandwidth-bound. At about
30 GB/s, the card is below the M1 Air's own 68 GB/s. Prompt processing is
compute-bound, and there the card is ahead (~4.35 vs ~2.6 TFLOPS). So the payoff
for decode depends on a **memory reclock**, which no open driver has done on
Pascal (§7).

## 4. Blockers hit, and their root causes

| # | Symptom | Root cause | Fix |
|---|---|---|---|
| 1 | Upstream (`pascal-egpu`) concluded POST from the host was impossible ("circular dependency") | It isn't: the VBIOS DEVINIT scripts run fine from the host | `devinit.py` |
| 2 | VRAM above a hole unusable | VPR region locked at boot | signed NVDEC scrubber (`fb.py`) |
| 3 | SEC2 "missing" at `0x840000` | GP102 SEC2 lives at `0x87000` (low aperture); engines show PRI poison until enabled in PMC | correct base, PMC enable |
| 4 | ACR HS load hung / PRIV_VIOLATION | ctx DMA had to be virtual through a bound instance block | `ctx_dma = VIRT` + instance bind |
| 5 | ACR exit `0x25` | WPR below `0x18000000` | `WPR_ALLOC_ADDR = 0x18000000` |
| 6 | ACR exit `0x29` on every retry | **Self-inflicted.** `0x10e5f8[23:20]` is an ACR claim field (HW mutex) that a failed load leaves at 4 | run nouveau's gp102 unload on the PMU first (`acr_unload.py`) |
| 7 | PRI ring down (session 4) | a GR-init bug | enclosure power cycle; `card.py` now asserts BAR0 range |
| 8 | GR channel never loaded (`0x800004 = 0x11000001`) | PFIFO `ENG_FAULT` (`0x262c`) latched for PGRAPH | W1C before bind |
| 9 | **GR channel: PBDMA0 INTR_0 bit 24 at load; FECS never asked to switch** | **Runlist 0 also carries the GR copy engines CE/CE1 (engines 7/8, PMC bits 6/7), and both were held in reset.** Bit 24 is ENG_RESET (nvgpu's name, cleared by the GPU itself) | take every engine on GR's runlist out of reset (nvgpu enables all CEs at power-on) |
| 10 | PGRAPH DATA_ERROR INVALID_ENUM on the report semaphore | compute `SET_REPORT_SEMAPHORE_D`: OPERATION 1 isn't valid on `0xc1c0` (RELEASE = 0) | `D = 1 << 28` |
| 11 | clock probe read 0 ns | ptxas removed the side-effect-free spin loop | spin on `%clock64` itself |
| 12 | MAX: `Cannot select: intrinsic llvm.nvvm.shfl.sync.idx.i32` (every warp shuffle) | MAX 26.6's GTX 1080 Ti target is `+ptx50,+sm_61`; PTX 5.0 predates `shfl.sync` | `+ptx63` (byte patch in `std.mojoc`, `maxpatch/`) |
| 13 | MAX matmul: `Cannot select: ...ldmatrix...` | `comptime if use_tensor_core` instantiates the Ampere GEMM even when the runtime shape check skips it | linalg patch: no tensor-core block or cuBLAS fallback when `compute < 7.0` (`maxpatch/patches/0001`) |
| 14 | Second module loaded into a running server traps: warp ESR `0x9` ILLEGAL_INSTR_ENCODING | the SM instruction cache prefetches past the last kernel it ran; newly written code there is stale | `INVALIDATE_SHADER_CACHES` (0x021c) before the first launch after any code upload |
| 15 | MAX gather traps: ESR `0x0e` OOR_ADDR at `st l[$r1]` | no local memory: `FRAME_SIZE` lives in the module-level `.nv.info` (by symbol), QMD `LOCAL_MEMORY_LOW_SIZE` and `c[0][0x20]` (stack top) were 0 | per-kernel local size, Mesa-sized TLS (2 KiB/thread), cbuf0 driver ABI (`c[0][0x0/0x4/0x100/0x104]` windows) |
| 16 | same kernels: assert/printf path reads `c[0x4][...]` | module data (`.nv.global.init`) + constant bank 4 with `R_CUDA_32/64` relocations were never loaded | CUDA-style module loader: data sections, relocations, `vprintf` → `ret` stub, banks bound in the QMD |
| 17 | softmax: ESR `0x0b` ILLEGAL_INSTR_PARAM | QMD `BARRIER_COUNT = 0` with `bar.sync` in the code (count is `.text` `sh_flags >> 20`) | set it from the cubin; reproduced with 0 vs 1 |
| 18 | ~65th submission: MMU fault, client HOST, VA `0x204000` | 16 KiB pushbuffer never wrapped | 1 MiB pushbuffer that wraps once the channel has drained |
| 19 | LLM ran but output only special tokens; sigmoid = 0 | `rcp.rn.f32` reads literals from the **per-kernel** `.nv.constant2.<kernel>`, which wasn't bound | load/bind per-kernel banks; `test_modules.py`: 512/512 bit-exact rcp/div |
| 20 | `cuMemset` ≥ 4 KiB silently did nothing | `fill32(u64, u32, u64)` params packed with `struct.pack("<QIQ")` (no CUDA alignment): `n` read as 0 | `"<QI4xQ"` |

## 5. Beliefs that turned out wrong (don't re-learn these)

* "Pascal can't be POSTed from the host" (upstream): wrong, see #1.
* "`0x29` means PMU firmware is missing": wrong, it was our own leftover claim (#6).
* Session 5's reading of the rung-4 stall: wrong, because the registers were misread.
  * `0x409014` is falcon **IRQMCLR** (write-only), not the mask. The real mask
    `0x409018 = 0x8704` already enables the channel-switch interrupt. Writing
    `0xffffffff` there disabled every FECS interrupt for that boot, which tainted
    the experiments that followed.
  * `0x4098cc` / `0x40984c` are mailbox SET/CLEAR registers. They always read 0
    and say nothing about whether the handler ran.
  * PCCSR `0x800004`: bits 24:27 are STATUS (5 = ON_PBDMA, 7 = ON_ENG), bit 28 is
    BUSY, and bit 1 is NEXT (not "ACTIVE").
  * `0x409b04 = 0x8001160c` was left over from `bind_pointer`, not a pending
    request. A clean reproduction showed FECS was never asked at all.
* FECS DMEM and ICD are locked to the host in LS mode (`0xbadf548a` /
  `0xbadfbadf`). The trace-PC buffer (`0x148`/`0x14c`) still works and is the one
  view into FECS.
* MAX "plugs in a new accelerator through a device-driver connector": that
  generic seam isn't shipped yet. See §6.

## 6. MAX (target) feasibility findings

Researched 2026-11-08; links below.

* Qualcomm completed its acquisition of Modular on 2026-07-28. MAX's NVIDIA path
  is `DeviceContext` over **`libcuda.so.1` + `libnvidia-ml.so.1`**, and the
  documented library paths are Linux only. The "next-generation, more generic"
  DeviceContext for hardware vendors is described as in progress.
* Kernels are compiled ahead of time with a bundled CUDA-13 `libnvptxcompiler`,
  which has no Pascal. The escape hatch is **`MODULAR_NVPTX_COMPILER_PATH` =
  system `ptxas`** (our CUDA 12.4 one).
* Nightlies went "Turing and newer, driver ≥ 580" in Feb 2026. Pascal ran basic
  graphs on Sept 2025 nightlies, so the code paths aren't foreign. The
  architecture table is `gpu/host/info.mojo` (patchable locally).
* MAX's built-in matmul needs sm_80, and tensor-core/async paths need sm_80+.
  Pascal (no tensor cores, 1/64-rate fp16) needs SIMT fallbacks for matmul,
  attention and quantized-weight kernels.
* So the integration seam is a **libcuda/NVML shim** that forwards to this
  project's runtime. It reports driver ≥ 580 and compute capability 6.1. The
  likely topology is MAX for Linux-aarch64 in a container, forwarding over a
  socket to the macOS-side driver.

Sources: [forum: older NVIDIA GPUs](https://forum.modular.com/t/support-for-older-nvidia-gpus-and-drivers-in-mojo-and-max/2761) ·
[forum: GTX 1050 Pascal](https://forum.modular.com/t/compatability-for-gp107m-geforce-gtx-1050-mobile/1750) ·
[Modular × Qualcomm](https://www.modular.com/blog/modcon-qualcomm) ·
[acquisition](https://www.modular.com/blog/qualcomm-completes-acquisition-of-modular) ·
[MAX hardware compatibility](https://max.modular.com/gpu/hardware-compatibility/) ·
[Mojo requirements](https://mojolang.org/docs/requirements/)

## 7. Next steps

Done this phase: driver as a service (`shimd.py`), the libcuda/NVML shim, the MAX
patches (`maxpatch/`), and a first model. Next, in order of payoff:

1. **Launch overhead** (decode is 343 launches/token, 29 ms of GPU time, ~150 ms
   wall): asynchronous launches (no per-launch completion wait; sync only on
   copies and `cuStreamSynchronize`) plus fire-and-forget launch RPCs. The ceiling
   at boot clocks is ~30 tok/s for this model.
2. **Bigger models:** Llama-3.2-1B fp32 (~5 GB, fits in 11 GB; head depth 64, so naive
   MHA like SmolLM2), then quantized weights (q4_k/q4_0 GPU kernels on sm_61 are
   untested). Models with head depth 128 (Llama-3-8B, Qwen) route to the tensor-core FA2
   path and need `flash_attention_hw_supported` gated on `compute >= 8.0`.
3. **`tanh`** emits `tanh.approx.f32` (sm_75+), so GELU-tanh models (Gemma, GPT-2
   style) don't compile yet. That needs a std/kernels patch like `maxpatch/0001`.
4. **Faster SIMT matmul:** prefill uses `matmul_kernel_naive`; the tiled SIMT
   `matmul_kernel` in the same file is the obvious upgrade.
5. **Reclock feasibility** (read-only is done, §9): memory at 405 MHz caps decode
   bandwidth at 30 GB/s. The next test needs register writes, which need consent.

## 8. Toolchain and reproduction

* Python: `~/.venvs/tinygpu` (3.12, tinygrad 0.14.0).
* Firmware: `firmware/fetch.sh` (linux-firmware) + `pascal/dump_vbios.py`;
  checksums in `firmware/SHA256SUMS`.
* Kernels: `kernels/build.sh` (Docker image `egpu-ptxas:12.4`, ptxas
  `-arch=sm_61`). Inspect with `pascal/cubin.py` and `envydis -m gm107`.
* Offline test: `pascal/mmu.py --selftest`.

## 9. Phase 6 progress (rung 5): MAX on the card

**A kernel compiled by Mojo/MAX ran on the GTX 1080 Ti** (2026-11-08):
`shim/tests/vadd.mojo` → MAX's compiler (`mojo build --target-accelerator=nvidia:61`,
Mojo 1.1.0 / MAX 26.6.0, Linux-aarch64 container) → **`.target sm_61` PTX**
(`kernels/captured/mojo_vadd.ptx`) → CUDA 12.4 `ptxas` → `nvpascal` launch:
**4096/4096 correct floats, 34 µs** (`shim/tests/run_mojo_ptx.py`). For now the
bridge from the PTX to the launch is manual; stage 2 of the shim automates it.

What the discovery shim (`shim/`, stage 1) established:

* MAX `dlopen`s `libcuda.so.1` + `libnvidia-ml.so.1` and `dlsym`s **118 `cu*` + 16
  `nvml*` names** directly (no `cuGetProcAddress`). The shim exports all of them: 67
  implemented, 51 logging stubs (graphs, multicast, VMM, tensor maps, peer).
* With a GP102 presented as CC 6.1 / driver 580, `DeviceContext()` comes up as
  "NVIDIA GeForce GTX 1080 Ti". **Mojo 1.1's architecture table already lists
  sm_60/sm_61**, so no target patch is needed (`--target-accelerator=nvidia:61` at
  build time when no GPU is visible).
* MAX hands the driver **PTX** (`cuModuleLoadDataEx`), not a cubin, so the shim's
  server JITs it with our `ptxas`. Launches go through **`cuLaunchKernelEx`**.
* At startup MAX pre-allocates a pool of about 90% of reported free VRAM
  (`cuMemAlloc(9.9 GB)`) plus 1.28 GB of pinned host memory, and sub-allocates from
  them. So the real shim needs a large VA mapping (2 MB pages) and honest `cuMemGetInfo`.
* Parameter layout for packing `kernelParams` comes from the cubin's `KPARAM_INFO`
  (ordinal, offset, size = (w >> 18) & 0x3fff).

Read-only reclock findings (`pascal/vbios_perf.py`): VBIOS perf table 5.0 gives
memory **P8 405 / P5 810 / P2 5005 / P0 5505 MHz**, and P8's gpc2clk of 1215 MHz
matches the measured 607 MHz SM clock. So devinit leaves the card at P8. The rammap
bands run up to 7000 MHz. nouveau has no Pascal reclock (`ramgp100.c` is init-only),
and the nearest model is Kepler's GDDR5 `gk104_ram_calc_gddr5`. The decisive next
test needs register writes (needs consent; a mistake costs an enclosure power cycle).

**Shim stage 2 works: an unmodified Mojo GPU program runs on the card through
MAX's normal driver path.** `shim/tests/vadd.mojo`, built with `mojo build
--target-accelerator=nvidia:61`, goes `DeviceContext` → our `libcuda.so.1`
(container, `SHIM_MODE=remote`) → TCP → `pascal/shimd.py` (Mac) → `nvpascal` →
GP102. Result: **4096/4096 correct**. The negative control is the same binary in sim
mode, where kernels aren't executed: 1/4096. MAX pooled 1843 MiB of the 2 GiB
reported (mapped in 0.12 s), handed over PTX (JIT'd server-side with ptxas sm_61),
and the kernel ran in 8–14 µs. See [`shim/README.md`](shim/README.md).

**A MAX graph runs on the card.** `shim/tests/graph_add.py` (Python `max.graph` +
`max.engine.InferenceSession` on `Accelerator()`) compiles `a*2 + b` into one fused
elementwise kernel. MAX's graph compiler produced the **cubin** itself, through the
external ptxas. It ran on the GP102 with **max abs error 0 vs numpy**, repeatably (the
server releases each process's pool on disconnect).

**Former blocker: matmul (resolved in session 7, below).** `ops.matmul` failed to compile for sm_61 with `LLVM ERROR:
Cannot select: intrinsic llvm.nvvm.shfl.sync.idx.i32`. A 3-line warp-shuffle kernel
(`shim/tests/shfl.mojo`) reproduces it. Root cause, in MAX's source
(`max/mojo/max/_plugin/cuda/targets.mojo`): the GTX 1080 Ti target hardcodes
`features = "+ptx50,+sm_61"`. PTX 5.0 predates `shfl.sync` (PTX 6.0), which Pascal
hardware supports; Turing's target uses `+ptx63`, Ampere's `+ptx81`. There's no
plugin/override path: `TARGETS` is a fixed list in `_overlay.mojo`, and a second
`sm_61` entry is a "must match exactly one target" error. **Fix:** edit that features
line to `+ptx63,+sm_61` and rebuild the `max` Mojo library (bazel `mojo_library(name =
"max")` in `max/mojo/max/BUILD.bazel`) from a modular/modular checkout, with the
installed Mojo 1.1.0 compiler (the package header's build ID matches the compiler,
not a source commit). After that, expect sm_80-only paths inside matmul/attention to
need SIMT fallbacks.

### Session 7 (2026-11-09): matmul fixed, first LLM on the card

**MAX patches** (`maxpatch/`, applied with `docker exec maxlab bash /repo/maxpatch/install.sh`;
`--revert` restores stock):

1. `std.mojoc`: the GTX 1080 Ti target string `+ptx50,+sm_61` → `+ptx63,+sm_61`. The
   string sits in the package's zstd payload (string table, same length, no checksum).
   No bazel rebuild was needed after all; the target table is in `std`, not `max`.
2. `linalg`: `_matmul_gpu` skips the tensor-core multistage GEMM and cuBLAS fallback
   when `compute < 7.0`, so SIMT kernels run. Rebuilt with `mojo precompile` from the
   `max/v26.6.0` tag; an unmodified rebuild matches the shipped package within 0.03%.

Results: `shfl.mojo` (idx + xor butterfly) gets 32/32 lanes right, and 64×64 fp32
`ops.matmul` has max abs err 0 vs numpy.

**Runtime fixes the first model forced** (blockers 14–20 in §4). Each was reproduced
on the card before it was fixed. `nvpascal` now loads cubins the way the CUDA driver
does:

* per-kernel local memory, from the module-level `.nv.info`
* a 132 MiB TLS area (Mesa's 2 KiB/thread sizing)
* the sm_6x cbuf0 driver ABI, checked against ptxas output: shared and local
  generic windows at `c[0][0x0/0x100]` and `c[0][0x4/0x104]`, stack top at `c[0][0x20]`
* module data (`.nv.global*`, module and per-kernel `.nv.constantN`) with
  `R_CUDA_32/64` relocations, and `vprintf` → a `ret` stub
* barrier counts
* an instruction-cache invalidate after code uploads
* a wrapping pushbuffer

`pascal/test_modules.py` gates all of it:
* local and generic pointers, globals through relocated bank 4
* IEEE `rcp.rn`/`div.rn`: 512/512 bit-exact (bank 2)
* 5000-launch pushbuffer wrap

**First LLM:** `max generate --model HuggingFaceTB/SmolLM2-135M-Instruct
--quantization-encoding float32 --devices gpu --top-k 1` (`shim/tests/generate.sh`):

* "The capital of France is" → **"The capital of France is Paris."** on the GPU and
  on MAX-CPU.
* A 64-token story prompt gives **identical text** from GPU and CPU.
* `shim/tests/ops_check.py` passes 10/10, at the model's shapes against numpy:
  gather, prefill matmuls, decode gemv, RMSNorm, softmax, SiLU·mul and argmax.
* Kernels MAX runs for this model: naive SIMT matmul (prefill), `gemv_split_k`
  (decode), naive MHA (`mha_bmm0` / `softmax_temperature` / `mha_bmm1`), fused residual
  RMSNorm, and top-k sampling.

**Speed:** 1.05 → **6.6 tokens/s**, and TTFT 945 → 300 ms. Profiling one launch
showed two costs:
* `time.sleep(0.001)` costs ~1.5 ms on macOS, and the completion wait always slept
  once. It now spins first.
* QMD construction `eval`'d each bit-field. It now memoizes the field positions.

Launch latency went 1.6 → 0.17 ms. Per decode token: 343 launches, 29 ms GPU, about
150 ms wall, so launch overhead is still ~80% of the time (§7 item 1).

Known gaps:
* `tanh` doesn't compile for sm_61 (`tanh.approx.f32` is sm_75+).
* An assert firing in a kernel still traps the SM (needs a GR reset), since
  `vprintf` is a no-op stub.
* Module space is reclaimed only when every client disconnects.
