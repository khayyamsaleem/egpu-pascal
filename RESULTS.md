# Results: a GTX 1080 Ti (GP102) running compute kernels from a MacBook Air M1

Compiled 2026-11-08 from sessions 1–6; updated 2026-11-09 (session 7: first LLM). Every claim below was measured on the
card, with a negative control where one applies. The evidence trail is in
[`docs/evidence-log.md`](docs/evidence-log.md) and the raw output is in [`logs/`](logs/).

**Bottom line:** the card goes from cold power-on to running ptxas-compiled sm_61
kernels, with results landing in Mac host memory, entirely from macOS over
Thunderbolt and with no NVIDIA driver. **Modular MAX runs LLMs on it:** SmolLM2-135M
(fp32) gives text identical to MAX-on-CPU at **6.6 tokens/s**, and **Gemma 4 E2B**
(bf16, MAX nightly) answers "The capital of France is **Paris**." exactly as
transformers does, at **1.9 tokens/s** (§9). The card runs at boot
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
| 21 | **The Mac kernel-panicked and rebooted, twice** (`LLC Bus error ... addr=0x480400100 rd_ld` / `0x480703f6c ncwr`, panicked task `TinyGPU`) | the Mac slept during a run (`Clamshell Sleep`, then `Idle Sleep` with `pmset sleep 1`): the Thunderbolt link went down and the next MMIO on wake hit a missing device | `pascal/sleepguard.py`: every card-holding process keeps a `caffeinate -i -s` assertion, and every MMIO first checks `kern.waketime`/`kern.sleeptime` (4 us) and refuses after a sleep |
| 22 | Gemma 4 output varied run to run (from layer 1 on) | `Card.vram_write`/`vram_read` rounded unaligned addresses down and zero-padded odd lengths: a 2-byte-aligned bf16 copy landed shifted and clobbered up to 3 bytes past its end (fp32 workloads never hit it) | read-modify-write of the partial first/last word; random test vs a host model, `shim/tests/copy_check.py` |
| 23 | Gemma 4: per-layer inputs for layers ≥ 1 were zeros/garbage, deterministically | each launch's constant bank 0 was a 4 KiB slot bound with size 0x1000, but sm_6x allows 0x1100 bytes of params after the 0x140 driver area. MAX's multi-output fused kernels (one output pointer per layer, 3856-byte params) lost their last parameters: the later layers' output pointers read as 0 | 8 KiB slots bound at the cbuf's real size (+ a hard error past the slot), 8 KiB shim param blob |
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

### Session 8 (2026-11-12): Gemma 4 E2B on the card; the two reboots explained

**Gemma 4 E2B (bf16) runs on MAX nightly `26.8.0.dev2026110805`:**

* `max generate --model google/gemma-4-E2B-it --quantization-encoding bfloat16 --devices gpu --top-k 1 --no-prefer-module-v3`
  answers "The capital of France is **Paris**.", identical to transformers on CPU.
* A haiku prompt gives the same coherent text on two runs. A 64-token explanation of what a
  GPU does is fluent: 57 tokens, TTFT 2.2 s, 530 ms/token (1.9 tok/s).
* Logs: `logs/s8-gemma4-e2b-*.log`.

What it took. Each item was reproduced and isolated on the card, mostly by comparing per-layer
hidden states with a transformers reference.

* **MAX nightly, not 26.6.** The 26.6 release has no Gemma 4 E-series support.
  * On the nightly, the default ModuleV3 implementation lacks per-layer embeddings (PLE) and the
    double-wide MLP, so the run needs `--no-prefer-module-v3`.
  * The nightly needed two more shim entry points: `cuMemcpyBatchAsync` (12.8 ABI, unified
    addresses classified against the server's ranges) and `cuDeviceGetUuid[_v2]`.
  * `maxpatch` ports the target and matmul patches to the nightly.
  * It adds a std `tanh` fallback for pre-Turing GPUs; GELU-tanh and logit softcapping need it.
* **bf16 on Pascal works.** LLVM widens bf16 to fp32 on sm_61. `ops_check --dtype bfloat16`
  passes 12/12, and SmolLM2 bf16 is coherent.
* **Fitting 11 GB.** E2B is 5.1B params (9.3 GB of text weights in bf16), and 4.7 GB of that is
  the PLE table. Each token reads one 8960-wide row, so the table stays in host memory and the
  GPU gets the gathered rows, leaving 4.6 GB on the device. The memory planner and weight rules
  follow suit (`maxpatch/patches/26.8.*/0003`).
* **Two driver bugs, found because Gemma is the first bf16/large-kernel workload.**
  Blocker 22 was unaligned PRAMIN copies, which made runs non-deterministic. Blocker 23 was the
  4 KiB constant-bank slot, which truncated MAX's multi-output kernels' parameters. With both
  fixed, upstream's per-layer code runs unmodified, and the overlap scheduler works.
* **Ruled out on the way, each with a test:**
  * attention at depth 256/512: upstream's attention tests pass at E2B dims, sliding, global
    and KV sharing
  * RoPE: the applied cos/sin recovered from the kernel match the analytic values
  * MMU mapping of the 8.8 GiB pool: `pascal/test_bigmap.py`, 8990/8990 chunks
  * the host-PLE lookup mechanics: `shim/tests/host_embed_check.py`

**The reboots (blocker 21).** Both were kernel panics when the Mac woke while the driver held
the card. Lid close the first time, a 1-minute idle sleep the second (`pmset sleep 1`).
`pascal/sleepguard.py` now prevents the sleep and refuses MMIO after one.

Known gaps:
* Gemma 4 **E4B** needs 9.4 GB on the device even with host PLE, which doesn't fit.
* Decode is launch-bound: ~700 launches per token at ~0.4 ms each.
* `tanh` on 26.6 is still unpatched; only the nightly set has the std rebuild.

## 10. Memory reclock: 405 -> 1080 MHz (session 11, 2026-11-12)

**The GDDR5X runs at 1080 MHz instead of the 405 MHz boot clock: 87.3 GB/s read (was 31.1),
and Gemma 4 E4B decodes at 13.5 tok/s (was 6.83) with byte-identical output.** No open driver
had reclocked Pascal memory before; nouveau's `ramgp100.c` is init-only.

How (`pascal/mclk.py`, `pascal/vbinit.py`, evidence in `logs/mclk/`):

* GP102's FBPA broadcast block is Kepler's `0x10fXXX` moved to `0x9a0XXX` (per partition:
  `0x900000 + i * 0x4000`). Proof: VBIOS timing entry 0's words sit at `0x9a0290..` exactly
  where `gk104_ram_calc_gddr5` writes them, and nouveau's `nvkm_gddr5_calc` reproduces the
  live MR0 (`0x25`), MR1, MR3, MR8. Only steps whose formula reproduces the boot state were
  kept (MR7, `0x9a0824`, `0x9a0978`, `0x9a0ec4` do not, so they are left alone).
* The piece nouveau never needed: **per-mode FBPA register banks**, write bank = `0x9a065c`
  bits 7:4, active bank = bits 3:0 (`0x11 * rammap_11_11_0c`). Selecting bank 1 or 2 reads
  back reset values in `0x9006a0`/`0x9846a8`/`0x98467c`; devinit only programs bank 0. The
  VBIOS rammap header points at one init script per mode, run by `pascal/vbinit.py`.
* Those scripts gate FBPA3 behind condition `0x30` and FBPA4/5 behind `0x31`, and this VBIOS
  has **no legacy condition table** (BIT 'I' + 4 = 0; the PMU DEVINIT app evaluates them). With
  every condition false (nouveau's behaviour), FBPA3-5 trained to saturated delays (`0x1f`)
  and the GPU took an MMU PDE fault. Evaluating them from which partitions answer fixed it;
  script 0 under the same rule writes exactly what DEVINIT left in bank 0 on all six FBPAs.
* Sequence: bank program -> self-refresh in -> refpll post-divider / N -> band timings, MRs,
  `0x9a065c` -> self-refresh out -> retrain (`0x9a0910/914`, wait `0x900974` per FBPA). The
  refpll VCO stays 800..1620 MHz (VBIOS PLL type 0x0c), so everything here is "mode 1"; the
  MPLL (mode 2, >1620 MHz) is not attempted.
* FBPA0 is half-populated (1 GiB of 2): its absent half never finishes training (status bit 1,
  delay register `0x6a8` saturated) and is ignored; the pattern test is the judge.

| memory clock | read GB/s | write GB/s | VRAM test (9 GiB x 3 patterns) |
|---|---|---|---|
| 405 (boot, band 0) | 31.1 | 28.8 | pass |
| 810 (P5, band 1) | 64.0 | 52.2 | pass, also straight from a cold boot |
| **1080 (band 1, default)** | **87.3** | 66.0 | pass |
| 1134 (band 1) | 92.0 | 66.9 | pass + 100-pass soak + 30 s retention hold |
| 1188, 1296 (band 1) | — | — | train, but corrupt data (GPU MMU faults) |
| 1620 (band 2 config) | — | — | does not train; GPU work afterwards reset the card |

Jumps of more than ~54 MHz fail training (1620 -> 1296 did; 1242 -> 1296 did not), so
`mclk.py set` ramps through 810 and then 54 MHz steps. `pascal/memtest.py` is a full-VRAM GPU
pattern test with a bit-flip negative control (`--inject`); a PRAMIN spot check is **not**
evidence, because PRAMIN goes through the 2.75 MB L2.

`local-gpu` now reclocks to `--mclk` (default 1080) after bring-up, proves the VRAM with
`memtest.py`, and steps down to 810, then 405, on failure; shimd reports the live clock.

Gemma 4 E4B QAT Q4_0, llama.cpp, the 12-turn `bench_chat.py` script (logs/bench/):

| | 405 MHz (session 10) | 1080 MHz |
|---|---|---|
| wall | 304 s | **159 s** |
| decode | 6.83 tok/s | **13.5 tok/s** |
| prefill | 264 tok/s | 375 tok/s |
| TTFT median / max | 1.11 / 6.6 s | 0.7 / 5.0 s |
| output | — | **12/12 turns byte-identical** to the 405 MHz run |

Where decode time goes now (`NVPASCAL_PROF=1`, 258 tokens): ~64 ms/token of GPU time out of
~74 ms, so it is GPU-bound, not bandwidth-bound. Weight matvecs reach 55-68 GB/s of the 87
available (fused gate+up Q4_0 63 GB/s, lm_head Q6_K 8 ms/token at 68 GB/s), and ~900 launches
per token include many 13-20 us small kernels at a 607 MHz SM clock. The SM clock is the next
lever: the Kepler-style GPCPLL at `0x137000` reads `badf` on GP102 (the GPC clock is elsewhere,
likely NAFLL), and the NVVDD PWM (`0x20340` period 0xa0, `0x20344` duty 0x40) means a GPC
reclock may need a voltage change -- a different risk class from the memory clock.

## 11. Low-risk tok/s round (session 12, 2026-11-12)

Gemma 4 E4B, 12-turn `bench_chat.py`: **159 s -> 153 s, decode 13.5 -> 14.0 tok/s, prefill
375 -> 401 tok/s**, from llama.cpp options now on by default in `local-gpu`
(`LOCAL_GPU_PLAIN=1` turns them off):

| option | effect (measured) | output |
|---|---|---|
| `-bs` backend sampling | decode +1.5-2% (13.65 -> 13.9 tok/s) | identical |
| `--spec-type ngram-mod` | +2-3% on the bench; 15.8 vs 13.4 tok/s on text it has seen | 11/12 turns identical (batched verification rounds differently) |
| E4B embeddings on the card (`-ot per_layer_token_embd...=CUDA0`) | prefill +10% (pp512 764 -> 840), decode neutral; 2.7 GB VRAM | identical |

Measured and rejected:

* **1134 MHz memory** passes cold (36 C, 100 passes) but flips single bits warm (42 C: 17 words
  in 100 passes, top byte lane, bits 31/28/25). **1080 MHz passes 200 passes warm** and stays the
  default; `local-gpu`'s startup VRAM check is now 20 passes. Die temperature under sustained
  decode: 41-43 C (shimd op 11 TEMP; `local-gpu status` shows it).
* **Dropping the per-launch WAIT_FOR_IDLE**: with a pushbuffer semaphore, 4536/16384 pages wrong;
  with the QMD's own RELEASE0 semaphore, the semaphore went backwards (done 16 after launch 18) --
  grids from one channel are not serialized on GP102 without the WFI. Stream order without it
  needs dependent-QMD chaining (`DEPENDENT_QMD_*` fields). Ceiling ~5-7% of decode.
* CUDA graphs: llama.cpp disables them before Ampere.

Where the host time goes (`SHIMD_OPSTAT=1`): pure host cost ~13.6 us per launch; the GPU is
idle on the host only once per token (535 of 619k launches found an empty queue); ~30 syncs and
~12 host->device copies per token at the token boundary. Decode is GPU-bound (~86%).

Research (docs/reference/README.md): the GDDR5X datasheet puts the DRAM's own DDR-mode
(PLL-off) ceiling at f_CK 1375 MHz (5.5 Gb/s/pin) -- we run 540 MHz (2.2 Gb/s), so the 1188+ MHz
failures are controller-side. NVIDIA's nvgpu shows the Pascal GPC clock is an FLL programmed,
with the voltage PWM, by PMU firmware -- the hard path. The gpu-unlock repos unlock tensor-core
rates on Turing/Ampere/Ada and do not apply to GP102.
