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

## 12. VBIOS band 2 (>=1300 MHz) does not work on the refpll path (session 13, 2026-11-12)

Session 12's ranked task #1 was memory above 1134 MHz, "in the mode we are already in, by
configuring the next VBIOS band properly". It does not work, and this is what rules it out.

**What was tried.** `mclk.py set 1350 --live` from 1080 (a real ramp: 1134, 1188, 1242, 1296,
1350 -- the only attempt on record before this was a single `--jump` from 810 to 1620). The
refpll reaches 1350 (`mclk.py get` = 1350, coef 0x00013101) and band 2's bank script and timing
entry 2 are programmed (`9a065c` = 0x22 active bank 2, MR0 CL 20 / WL 5 / WR 24). But **training
never completes**: the `0x900974 & 0xf == 0` wait times out on all six FBPAs, status 0xa (bits
1|3) on every one, even though the trained delays are plausible (6a8/6ac mid-range, not the
saturated 0x1f of the 1620 attempt). `--soft` shows the run is otherwise complete.

**The first GPU work after it hangs.** `pascal/reset_engines.py` + `gr.py` still PASS (FECS fine,
no MMU fault), then `memtest.py` dies creating its device:
`submission 'compute state' did not complete: GP_GET=723497779 GP_PUT=1 chan=15000003
FIFO_INTR=20000000`. Identical signature to the 1620 attempt: the memory subsystem cannot serve
the fetch, not the engine.

**Ruled out, each by measurement:**

| candidate | why not |
|---|---|
| the ramp itself (1620 was a 810 MHz jump) | a 54 MHz-step ramp to 1350 fails identically |
| the 0x9a04xx band registers `gk104_ram_prog_0` writes | decoding this VBIOS with nouveau's `nvbios_rammapEp` v0x11 layout (`temp = rd32(entry + 0x09)`; field names encode byte+mask) shows **DEVINIT/the PMU already left band 2's values in all of them at boot**: 0x9a0400[4:0] = 0x1f = band 2's `0b_01f0`; 0x9a0420[0] = 1 = `0a_0400`; 0x9a0440[23:16] = 0xe4, [15:8] = 0xff = `0d`/`0f`; 0x9a0444[15:8] = 0x44 = `0e`; 0x9a0468[8:0] = 0x40 = `09_01ff`, [20:12] = 0x40 = `0a_03fe`. Bands 2-5 share bytes 0x08-0x10 in the rammap entry; band 0/1 differ, band 2 is what is live |
| the PHY trims (per-lane delay seeds, Vref, masks) | 10-combo sweep at 1350 (`retrain --patch=`, delays 0x00/0x08/0x0a/0x10/0x12/0x18, Vref 0x4d4d/0x5353/0x5a5a, band 1's masks): training completes in **none** (`logs/mclk/s13-sweep-*`, summary in `s13-trim-sweep-summary.txt`). The trim registers also do not read back what the script wrote (0x9846a8 = 0x12121212 after a 0x03030303 write; 0x9a070c reads 0) -- the PMU or the training engine owns them, so patching is a weak probe |

**Reading.** Band 2 is the mode-2 band: nouveau picks the FBPA bank from the rammap entry but
picks the *clock source* by frequency -- `ramgk104.c`: `ram->mode = (next->freq >
fuc->refpll.vco1.max_freq) ? 2 : 1`, i.e. mode 2 (MPLL at 0x132004) starts only above the refpll
VCO max of 1620 MHz, and mode 2 also flips `0x1373f4 |= 0x1100`, `0x1373f0 |= 2` and the
0x9a0830/0x9a0808/0x9a0200 bits `mclk.py` never writes.

**Mode 2 was then ported and it fails too -- at a clock that works in mode 1.** `mclk.py` now
implements the whole mode-2 path (nouveau's `gk104_pll_calc_hiclk`, `r1373f4_init`/`_fini`'s
mode-2 half, the mode-2 branches of `gk104_ram_calc_gddr5`, the `from == 2` way back down, and a
`src == 2` read-back; `--mode2` forces it). Its PLL math hits every target within 25 kHz. Then
`mclk.py set 1080 --live --mode2` -- band 1, the configuration that trains cleanly at 87.4 GB/s in
mode 1 -- programs as intended (refpll 0x072801 = 154.287 MHz as the reference, mempll 0x00000701,
`0x1373f4` = 0x12 = source mempll, read-back 1080.0 MHz) and **training does not complete on any
FBPA**, after which the first GPU work hangs exactly as band 2 does (`GP_GET` garbage, FIFO_INTR
0x20000000, PBDMA INTR bit 24). So the register-level mode-2 configuration is not what the DRAM
needs: either the mempll is not the memory clock source on GP102, or its input/enable/route
differs from nouveau's Kepler-era model -- note the read-back formula (`clk/gk104.c read_pll`
0x132000: `refpll_out * N / (M * P)`, no fractional term because `(u16)(0xf000 + 4096)` truncates
to 0) agrees with our intent, so it is not a measurement either.

The way back down is implemented and verified (`set 1080 --live` from mode 2 runs nouveau's
`from == 2 && mode != 2` branch, lands on the refpll and takes the 9 GiB test again: 87.4 GB/s,
0 bad words). Next thing to try, in order: (1) measure the *actual* memory clock instead of
trusting the PLL registers -- nvgpu's clock counters are the only host-readable clock source
(`trim_gpc_bcast_clk_cntr_ncgpcclk_cfg/cnt` 0x132924/0x132928, sys 0x1373b0/0x1373b4, ltc
0x1373c0/0x1373c4, source gpc2clk = 0x70000000), read twice with a known host delay, so a mode-2
attempt can be judged without running GPU work; (2) only then re-read `r1373f4_init`'s mempll
enable/input against `clk_gp106.c`/`clk_fll.c`/`clk_vin.c` (the GP106 code is the nearest
discrete-Pascal one nvgpu still has). Do not re-try the 0x9a04xx registers or the trim sweep.

**Operational, learned twice today:** after a band-2 excursion the startup check at 1080 fails
*immediately after a down-ramp* from 1350 (local-gpu stepped down to 810 on its own), and passes
again after cycling 810 -> 1080 (20 passes, 87.4 GB/s, 0 bad words). Recovery is
`mclk.py set 810 --live` then `set 1080 --live` -- not the down-ramp by itself. Also: `mclk.py
get` reports the *refpll coefficient*, which `devinit.py` + `fb.py` do not rewrite, so after a
repost it can name a clock the DRAM is no longer configured for; the VRAM pattern test is what
decides, never `get`.

New tooling (this session): `mclk.py --patch=ADDR:VAL[,ADDR:VAL...]` (extra writes after the
band script's trims, before training) and `--soft` (report incomplete training instead of
aborting), plus `logs/mclk/s13-trim-sweep.py`, the sweep driver.

**Model-side knob tried and rejected** (same 12-turn bench, 1080 MHz, session-12 defaults, greedy):

| config | wall | decode | prefill | note |
|---|---|---|---|---|
| session-12 defaults (re-measured) | 159 s | 13.4 tok/s (median turn 13.07) | 384 tok/s | `logs/bench/20261112-202727-*s13-baseline-1080` |
| `-ctk q8_0 -ctv q8_0` | 168 s | 12.6 tok/s (median turn 12.1) | 406 tok/s | `logs/bench/20261112-203047-*s13-kvq8-1080` |

KV quantisation is a loss here: the bench's context tops out at ~3.3k tokens, so KV traffic is a
small share of the per-token bytes while the dequantised attention path costs more. Same-config
runs also spread 13.07 / 13.52 / 13.78 median across sessions, so treat anything under ~4% as
noise. The default stays as session 12 left it; the 1080 MHz ceiling (87 GB/s against a 5.15 GB
per-token read) is what bounds decode, which is why task A is the memory PLL and not the knobs.

## 13. The memory clock is now measurable -- and it is no longer the constraint (session 15)

**The question this project was blocked on ("what clock does the DRAM actually get?", since the PLL
read-back is not a measurement) is now answered on the card, and the answer redirects the work.**

`pascal/mclk_measure.py` drives nvgpu's own trim counters -- `hw_trim_gp106.h` for the registers,
`gp106_get_rate_cntr` in `clk_gp106.c` for the method (both vendored under `docs/reference/files/`).
Each counter counts a selected clock for a fixed crystal-referenced window, so host timing cancels
out and the reading is a real frequency, not a register value:

| counter | cfg / cnt | source field |
|---|---|---|
| gpc2clk (SM) | 0x132924 / 0x132928 | 0x70000000 |
| sys2clk | 0x1373b0 / 0x1373b4 | 0x00000000 |
| xbar2clk | 0x1373c0 / 0x1373c4 | 0x20000000 |
| **dramdiv4_clk1** | **0x132128 / 0x13212c** | **0x30000000, scale 4** |

The DRAM one is nvgpu's `CTRL_CLK_DOMAIN_MCLK`: its source is the DRAM clock / 4, scaled by 4, so
it reads the memory clock itself. Measured (five samples each, spread 0.01%):

| driver asks for | dram counter f_CK | quoted (f_CK x 2) | read bandwidth (memtest) |
|---|---|---|---|
| 405 (boot) | — | — | 31.1 GB/s |
| 1080 | 537.1 - 537.4 | 1074 - 1075 | 87.3 GB/s |
| 1134 | 564.0 - 564.4 | 1128 - 1129 | 92.0 GB/s |
| 1350 **via mode 2** (training fails) | 593.3 | 1187 | — |

Four things follow:

1. **The register read-back was honest.** The counter puts 1080 at 537.1 f_CK = 1075 quoted, i.e.
   the PLL rounds to its nearest achievable point, which is exactly what a counter should show.
2. **The counter tracks the memory clock.** 1134/1080 = 1.05, and 537.1 x 1.05 = 564.0 -- measured.
   Any future reclock can now be judged without running GPU work, which is what section 12 needed.
3. **Mode 2 (the mempll at 0x132000) *does* drive the DRAM clock after all.** Asked for 1350 it
   achieves 593.3 f_CK (1187 quoted) -- up from 537 -- so the previous session's "0x132000 may not be
   a memory PLL at all on Pascal" is answered: it is, or at least it steers the memory clock. What
   fails is the *timing* side: band-2 training never completes, and the achieved clock is not the
   target. Recovery back to mode 1 lands at 537.4, verified healthy.
4. **1134 MHz buys nothing end to end, so the memory clock has stopped being the lever.** Same
   argument set, speculation off: 162 s / 13.5 tok/s at 1134 vs the recorded 159 s / 13.4 at 1080
   (inside the +-4% noise above), while the memory test itself does gain 87.3 -> 92.0 GB/s. Decode
   moves 69.5 GB/s of weights against a 92 GB/s achievable read = **75% of the bandwidth ceiling**,
   so the binding constraints are the 607 MHz SM clock and the ~900 launches per token. The same
   reasoning says band 2 (1350 -> 118 GB/s) would also buy nothing until the SM/launch path moves:
   task A is no longer the highest-value memory work, tasks B and C are.

**Calibration, and why these numbers are trustworthy.** The four counters agree with each other to
0.01% but neither paper reading of `noofipclks` puts gpc2clk at its known 607 MHz, so the scale is
anchored on the SM clock -- measured independently with `%clock64` in `bench.py` -- and every other
domain is a ratio against it. The anchor is then checkable: 537 f_CK x 4 transfers x 44 B = 94.5
GB/s ceiling vs 87.3 measured (92%) at 1080, and the same model predicts 35.6 / 71.3 GB/s at 405 /
810 against 31.1 / 64.0 measured. Four independent bandwidth points, one free constant.

**Two operational findings, both measured today.** n-gram speculation (`-bs --spec-type ngram-mod`,
on by default) changes the end-to-end shape completely -- 12 turns in 67 s / 42.8 tok/s overall with
it, 162 s / 13.5 without -- so end-to-end numbers are only comparable within one argument set, and
the recorded 159 s baselines are not comparable to a speculated run. And `set_mclk` used to settle
for the 810 fallback when the target failed verification after a clock excursion (costing 30% of the
bandwidth silently); it now cycles and retries the target once first, which is the recovery the
notes already described by hand.

## 14. Task B closed: the per-launch WAIT_FOR_IDLE cannot be dropped on GP102 (session 15)

Task B -- "drop the per-launch WAIT_FOR_IDLE, ~5-7% of decode" -- is answered, and the answer is no:
measured three ways, with the mechanism the handoff named now actually tried rather than assumed.

Today's launch path is `[QMD dispatch] + WAIT_FOR_IDLE (0x0110) + a pushbuffer semaphore`, one
doorbell per launch (`pascal/nvpascal.py` `launch_async`). WFI is what makes the completion signal
meaningful: it drains the engine before the semaphore fires. The untried half was the QMD's own
dependency fields, which v02_01 expresses as `DEPENDENT_QMD_SCHEDULE_ENABLE` + `DEPENDENT_QMD_TYPE=GRID`
(that version has no `DEPENDENT_QMD_POINTER`, unlike v01_07). New tooling: `pascal/test_depchain.py`
(a fill->copy dependency chain, modes off/none/dep/group/both, sampled verification, one mode per
process -- a bad mode wedges the FIFO) and `compute.qmd_chain`, which builds a QMD whose RELEASE0
semaphore is the completion signal.

| mode | measured |
|---|---|
| off (shipping: WFI + pushbuffer semaphore) | 0/48 outputs wrong, completion counter monotonic -- ordered, as it must be |
| none (no WFI, no dependency field, QMD RELEASE0 completion) | simple chain looks fine (0/48 wrong, 0 backwards), but on the real workload the **model load never completes** -- the release into host memory never arrives, so every wait blocks |
| dep / both (`DEPENDENT_QMD_SCHEDULE_ENABLE` + `TYPE=GRID`) | **deadlocks the channel at the second launch** (`GP_GET` stuck at 2 with `GP_PUT` at 258), reproducible; needs `reset_engines.py` + `gr.py` to recover |
| group (a distinct `QMD_GROUP_ID` per launch) | no effect -- identical to `none` |

What that pins down: a QMD RELEASE0 aimed at host memory (`sys_coh`) never signals, so the completion
has to come from the pushbuffer *after a drain* -- WFI stays; and the v02_01 dependency encoding does
not mean "wait for the previous grid" on GP102, it stops the scheduler. `v01_07`'s
`DEPENDENT_QMD_POINTER` is the only remaining candidate, and trying it means handing the hardware a
v01_07-layout QMD -- a different experiment, for whoever wants it. Until then the ~5-7% stays on the
table.

**Instrument caveat, stated so nobody over-reads the table:** the simple chain is *blind* to this
hazard -- it comes out correct with no ordering mechanism at all, so it can neither certify nor
refute a chaining change. Only the real workload decides, and there it hangs. The "0/48 wrong" row is
not evidence that chaining is safe.

The chain code stays in the tree for that future experiment, doubly gated (`NVPASCAL_CHAIN` **and**
`NVPASCAL_CHAIN_ALLOW_UNSAFE=1`) so no environment variable can break this stack by accident. That
leaves the SM clock (task C) as the only remaining decode lever.

## 15. SM clock: the harness, the rollback, and what the card actually allows (session 15)

Task C is not solved, but it now has a safe instrument, a rollback that has been *exercised*, and one
real measurement of what this card permits -- which is what was missing.

Why it is hard (reference notes, docs/reference/README.md): the GP102 GPC clock is FLL/ADC (NAFLL),
not a PLL -- the Kepler GPCPLL at 0x137000 reads `0xbadf` here -- and nvgpu, the only open reference,
never writes FLL registers from the host: it parses VBIOS tables (BIT 'C' FLL device table, entry 4 at
0x755c) and hands them to PMU firmware as board objects (`clk_fll.c`). There is no published host-side
FLL map to program.

Headroom, from the VBIOS's own perf table (read-only, `pascal/vbios_perf.py`): the card boots and stays
at **P8** -- memory 405 MHz, gpc2clk 1215 quoted = the measured **607 MHz** actual -- while the **P0**
row asks for 1594 quoted (~797 actual). So this lever is worth about **1.3x**, not the 2.4x that
comparing 607 with the 1480 MHz base clock suggests: the VBIOS quotes these clocks at 2x the rate the
hardware counts, exactly the convention the memory clock turned out to have (section 13).

New tool, `pascal/smclk.py` -- rollback first, because the only recovery from a bad clock write is a
power cycle of the enclosure, i.e. the user's hands:

| command | what it does |
|---|---|
| `--verify` | measures the real GPC clock on the card (the trim counter of section 13) -- a register read-back proves nothing on its own |
| `--scan` | read-only dump of the clock/voltage register families |
| `--snapshot F` / `--diff F` / `--restore F` | 413 registers as JSON + checksum; compare against the card; write back and verify every read-back |
| `--probe` | same-value writes: maps host-writability without changing any value |
| `--apply ADDR:VALUE` | one write, read-back verified, auto-restores that register if the read-back disagrees |
| `--test` | prints the two-test protocol: change present (measured), inference faster (same argument set) |

Measured, in order: baseline **GPC 607.0 MHz, DRAM f_CK 537.1**; snapshot 413 registers (checksum
`cf766bc1eaa0e470`); the scan found a PLL-shaped block at **0x137000-0x137100** mirroring the memory
PLL's layout (`0x137020 = 0x00030010` control, `0x137024 = 0x10814201` coefficient -> M=1, **N=66**,
P=1 on `mclk.py`'s decoder), i.e. a plausible SYS/GPC PLL; the probe showed all four of those registers
are **host-writable** (read-back exact); and a single-variable write of N 66 -> 67
(`0x10814201 -> 0x10814301`) read back correctly and **changed nothing** -- gpc2clk stayed at 44995
counts (607.00) before and after, with sys/xbar/dram unmoved too.

So that coefficient alone does not steer the GPC clock: it needs a strobe/enable the block's control
word has not been shown to have, or it drives a different domain. Rollback was then exercised for real
(`--restore` put `0x137024` back, `--diff` reported 0 registers differing, the clock measured 607.0
again), so the harness is proven in both directions.

Remaining candidates, each needing its own measured single-variable write with the snapshot armed: the
second copy `0x137044`, the control word `0x137020`'s strobe bits, and the 0x137300 block's
divider-shaped words (`0x137310 = 0x87101616`, `0x137330 = 0x87100634`). That is the same shape as the
memory-clock work, which took several sessions of measured attempts -- except a mistake here can wedge
the GPU. **No SM clock change is claimed, and nothing was left modified.**

### The sweep: 12 candidates, all writable, none effective

`pascal/smclk_sweep.py` drives the experiment properly: a same-value probe per register first (skip
anything that will not read back), then one variable at a time, all four counters read as the **median
of three** samples with a 0.5% threshold, and every candidate restored unless it actually moves a
clock. Evidence: `logs/smclk/s15-sweep.log`, pre-sweep snapshot checksum `cf766bc1eaa0e470`.

| candidate | result |
|---|---|
| both coefficient copies, N 66 -> 67 (`0x137024`, `0x137044`) | writable, no counter moved |
| control-word strobe bit (`0x137020` bit 0) | writable, no counter moved |
| re-latch: coefficient, then the control word again | writable, no counter moved |
| the second pair (`0x137040` / `0x137044`), both ideas | writable, no counter moved |
| coefficient N-1, and P+1 | writable, no counter moved |
| divider-shaped words `0x137310`, `0x137330` (two fields each) | writable, no counter moved |
| `0x137300`, low bit | writable, no counter moved |

gpc2clk sat at 44994-44997 counts (607.0 MHz) through every one of them, with sys/xbar/dram equally
unmoved. So that block is writable but does **not** own the GPC clock: it is a mirror, or it belongs to
a domain none of the four counters can see. With the boot path being the VBIOS's **PMU app**
(`pascal/devinit.py` loads it; there is no host init script for this clock) and nvgpu never writing FLL
registers from the host, the honest verdict is that **the SM clock is PMU-firmware territory on this
card**.

The remaining avenue is therefore the PMU rather than a register: nvgpu reaches clocks by parsing the
VBIOS FLL/VIN/volt tables into PMU *board objects* and then requesting a VFE (voltage/frequency) point
(`clk_fll.c`, `clk_vin.c`, `volt_dev.c`, `ctrl/ctrlclk.h`, `ctrl/ctrlvolt.h`). That is a project --
board-object plumbing against signed firmware whose command ABI would have to be read out of the
VBIOS's PMU app with envydis -- not a session step, and the power-cycle risk of a bad frequency/voltage
request is unchanged. Nothing is left modified: `smclk.py --diff` reports 0 registers differing from the
pre-sweep snapshot, and the card serves again at 1080 MHz memory / 607 MHz SM.


## 16. Session 16: a false benchmark number, MAX int8, vision, Hermes, and disconnect safety (2026-11-13)

### The "67 s / 42.8 tok/s with speculation" number was a degenerate loop

Section 0 of the handoff carried two results for the same arguments (`-bs --spec-type ngram-mod`):
153 s in session 12 and 67 s in session 15. The 67 s run
(`logs/bench/20261113-102439-gemma4-e4b.json`) is not a speedup: from turn 2 on, the reply is the
string "```" repeated until the 200-token budget ran out, which n-gram speculation predicts perfectly,
hence 47-69 tok/s. 11 turns, tool call not made. A fresh run with the same flags
(`logs/bench/20261113-114337-gemma4-e4b-s16-spec-on-repro.*`) gives **160 s, 13.4 tok/s decode, 377
tok/s prefill, 12 turns, tool call ok** -- the same as speculation off (162 s). The loop did not
reproduce; the bad run came four minutes before the 1134 MHz memory experiments were committed, and
1134 MHz is known to flip bits warm (section 11), but that link is a guess, not a measurement.
**Speculation buys nothing on this bench. 13.4-13.5 tok/s is the number.**

### Where decode time goes (per kernel)

`NVPASCAL_PROF=1`, `LOCAL_GPU_PLAIN=1`, 300 generated tokens, ~64 ms of GPU time per token (the
profile's `xcopy` row is the model upload, not decode):

| share of decode GPU time | kernel | per token |
|---|---|---|
| 31% | Q4_0 fused gate+up matvec | 42 x 471 us |
| 27% | Q4_0 matvec | 216 x 80 us |
| 13% | Q6_K output head | 1 x 8.07 ms |
| 6% | `quantize_q8_1` | 304 x 13 us |
| ~20% | ~700 small kernels (rms_norm, broadcast mul, flash-attn tiles, set_rows, gelu) | 12-20 us each |

So ~70% is weight matvecs and ~27% is fixed-cost small launches. Nothing cheap is hiding in there.

### llama.cpp op suite: closed

`test-backend-ops -b CUDA0`, full run: **20,290 cases OK, 0 failed, exit 0**
(`logs/s16/test-backend-ops.log.gz`). `SOLVE_TRI` passes and the shim logged no unimplemented-stub hit,
so session 9's exit-134 abort at 13,380 cases was the missing `cublasStrsmBatched`.

### MAX int8 weights (patch 0004): works, and it is a speed result too

The `no kernel registered for 'w8_linear'` blocker had two causes, both now in patch 0004:

1. `pipelines/lib/graph_specs.py`: a grouped compile (`GraphModule` of several graphs) only passes
   on the `custom_extensions` declared on each `GraphSpec`; kernel libraries a graph imported while
   tracing (`Graph._import_kernels`, or the process-wide overlay) are dropped. The patch unions each
   traced graph's `kernel_libraries_paths` into the compile.
2. `gemma4/layers/attention.py`: the per-device (sharded) copy of the attention layer took its dtype
   from `o_proj.weight.dtype`, which is int8 once the weights are quantized, so flash attention was
   asked for an int8 output against bf16 Q/K/V ("Q, K, V, output should have same dtype").

Gemma 4 E2B, `max generate`, greedy, 96 new tokens, 1080 MHz memory, same prompt, same session
(`logs/s16/max-e2b-cmp-int8_{0,1}.log`):

| | bf16 | int8 weights (`MAX_GEMMA4_INT8=1`) |
|---|---|---|
| decode | 9.8 tok/s | **13.4 tok/s** (1.36x) |
| prefill | 11.0 tok/s | **37.7 tok/s** (3.4x) |
| time to first token | 1.81 s | 0.53 s |
| text | -- | byte-identical to bf16 over all 97 tokens |

The shim log shows the two custom kernels loading (6- and 7-parameter `mojo_pkg_*` functions =
`w8_gemv` / `w8_gemm`), so this is not a silent bf16 fallback. Note the old "MAX E2B bf16 = 1.9
tok/s" figure dates from the 405 MHz memory clock; at 1080 MHz bf16 is 9.8.

Not done: Gemma 4 E4B on MAX. The memory planner sized it as bf16 (9.64 GiB > 9.0 available) and
refused before downloading; patch 0004 now subtracts the MLP weights' int8 saving (planner estimate
~7.8 GiB; the real figure is ~6 GiB of device weights). The run was stopped on purpose 2.7 GiB into
the 14.9 GiB download: int8 reads 8 bits per weight where llama.cpp's Q4_0 reads 4.5 for the same
model at 13.5 tok/s, so it cannot beat that. The planner hunk let the load proceed but has never
been exercised by a full E4B run.

### Gemma 4 E4B reads images

`local-gpu` now downloads the vision projector (`gemma-4-E4B-it-mmproj.gguf`, 0.95 GB, same repo)
and serves with `--mmproj` by default (`LOCAL_GPU_NO_VISION=1` leaves it out; it costs ~1 GiB of
VRAM). Test: two generated 640x480 PNGs with random text, plus a no-image control, same question:

| image | truth | answer |
|---|---|---|
| a | one red circle, "KESTREL 7391" | "KESTREL 7391", one circle, red |
| b | two blue squares, "OTTER 2058" | "OTTER 2058", 2 squares, blue |
| none | -- | "Please provide the image you are referring to." |

132 prompt tokens per image, 5-7 s per answer. The model is text-out only.

### Hermes Agent

Hermes rejects any model whose context is under 64,000 tokens, so `gemma4-e4b` now defaults to a
65,536-token context (2.9 GiB of VRAM still free with the projector loaded). With the provider
block from the README in `~/.hermes/config.yaml`, `hermes chat -m local` answers and makes tool
calls on the card (`echo pascal-$((6*7))` through the terminal tool -> "pascal-42"), and
`local-gpu / gemma4-e4b` appears in the model-picker payload the CLI, TUI and desktop app share
(`hermes_cli.inventory.build_models_payload`). Hermes' system prompt and tool definitions are
~13.9K tokens: the first turn of a cold server spends ~35 s on prefill, later turns reuse the cache.

### Start only with the card attached, stop when it goes

* `pascal/sleepguard.py: egpu_present()` asks the IO registry for an NVIDIA PCI device (~20 ms, no
  MMIO). `local-gpu serve` checks it before Docker, the containers or any download and exits 1 with
  "the eGPU is not connected ... Nothing was started."
* `SleepGuard` polls it every second in every card-holding process; once the card is gone, the next
  `check()` raises before the access, the same path a sleep takes (shimd exits without touching
  the card).
* `local-gpu watch` (spawned by `serve`) stops llama-server / max serve and the proxy and SIGKILLs
  shimd when the card leaves the registry, then exits.
* `local-gpu serve` starts Docker (OrbStack) itself if the daemon is down, e.g. after a reboot.

Tested with `PASCAL_SIMULATE_DISCONNECT_FILE` (the card reports absent while that file exists): the
gate refused to start and launched nothing; a simulated unplug while serving took everything down
within 10 s; `serve` afterwards was back in 26 s. **Not tested with a real unplug** -- an access
already in flight when the cable is pulled is outside what a one-second poll can prevent.

Spin-up, measured: 41 s from "Docker quit, driver stopped, card warm" to a healthy endpoint; the
model load is 31 s of that. A cold card adds the devinit + fb + reclock + gr path.

---

## 17. Session 17: the card as a Hermes agent -- what a multi-turn test showed, and the two levers that measured (2026-11-13)

Hermes has been able to answer on the card since session 16; this section is about whether it can
*work* there, and what a launch should look like. Method: the judge is always the card's own
server log (`slot print_timing` in `/tmp/llama-server.log` inside the `llamacpp` container), never
Hermes' reported `tokens.input` -- that field is a running session total and reads about double
the request. Hermes itself was driven with `--format stream-json` and `--resume <session-id>`,
so each turn of a multi-turn run leaves a JSONL trail with its own tool calls and timings.

### The fixed prompt is 14K tokens, and ~10.4K of it is tool schemas

A default request carries a 13,413-char system prompt and 37,453 chars of JSON schema for 20 tools.
Measured on the card by launching with different `-t` sets (each one a fresh tool set, so a real
full prefill):

| launch | tools | prompt tokens | prefill |
|---|---|---|---|
| default | 20 | 13,841 | 24.6 s (562 tok/s) |
| `-t file,terminal,code_execution,web` | 11 | 6,868 | 10.6 s (646 tok/s) |
| `-t debugging` | 10 | 6,146 | -- |
| `-t file,terminal,skills` | 8 | 8,316 | -- |
| `-t file,terminal` | 8 | 5,701 | -- |
| `--safe-mode` | 8 | 4,060 | -- |

The heaviest single schemas are `manage_connections` (1.1K tokens), `skill_manage` (0.93K),
`delegate_task` (0.87K), `browser_exec` (0.85K). There is no per-model toolset in Hermes config
(`platform_toolsets` is per surface), so this is a launch flag -- hence `bin/hermes-local`, which
checks the endpoint and then runs `hermes chat -m local -t file,terminal,code_execution,web`.
Measured end to end through that wrapper: 6,868-token prompt,
10.6 s of prompt processing, 17 s wall for a one-word answer. **Half the fixed cost, for one flag.**

The same table can be produced without the GPU at all: point Hermes at a mock OpenAI endpoint and
read the POST body (script in the `gpu-falcon-re` skill, `scripts/measure_prompt_composition.py`).

### Multi-turn: it works, and the cost is latency, not competence

Seven turns on `gemma4-e4b`, one resumed session, 8 tool calls, every turn checked against ground
truth:

| # | task | wall | result |
|---|---|---|---|
| 1 | `uname -a` via the terminal tool, quote the kernel string | 48.0 s | exact |
| 2 | write *that* string to `kernel.txt` (cross-turn memory) | 16.6 s | correct file, correct bytes |
| 3 | read it back and quote it | 13.7 s | verbatim |
| 4 | run python to print sum(1..100) | 36.3 s | 5050 -- via neither tool it tried (see below) |
| 5 | "what was the first shell command I asked for?" | 5.7 s | `uname -a` |
| 6 | which handoff file has the most lines? | 145.8 s | `HANDOFF-TO-OPUS.md`, 361 |
| 7 | summarize the session in three bullets | 19.9 s | right content, bullets ran together |

286 s total, ~41 s per turn. The model chooses sensible tools, holds state across turns, and
recovers when one fails: turn 4 tried `execute_code`, was refused, tried `python3 -c`, was
refused, and answered correctly anyway; turn 6 went `search_files` -> refused `execute_code` ->
`find | xargs wc -l`. Both refusals are a property of one-shot mode, not of the model -- with
`-q` there is nobody to approve `execute_code` or a `-c`-flagged shell command, so each costs a
wasted turn. Interactively (the PTY session run first) the same work runs with approvals and no
refusals. The slowest turn was 146 s, so the default 1800 s request budget needs no tuning.

Turn economics, from the server log: prompt-cache reuse is real (slot chosen by LCP similarity
0.98-0.998), so after the first turn only 5-200 new tokens are prefilled (60-340 tok/s) and the
turn is decode-bound at 11.5-11.9 tok/s plain, up to ~19.8 tok/s when n-gram speculation drafts
(acceptance 0.33-0.83, mean accepted run 22-54 tokens).

### As a subagent

A real `delegate_task` child, with the delegation route pinned at the card
(`delegation.base_url` + `api_key` + `model` + `api_mode` -- the provider names the example config
lists are not the only route): it read a 127-line handoff and returned line count, first heading
and an open task, **3/3 correct**, in 103.6 s over 2 API calls. On-card execution is not taken on
trust: the log shows a 10,915-token prompt prefilled in 18.4 s (592 tok/s) then 288 tokens at
11.9 tok/s, a 3,088-token follow-up at 433 tok/s, 610 tokens at 11.6 tok/s. Children get a ~10.9K
prompt of their own. The pin was reverted afterwards: it routes *every* child, and children carry
the bulk of a session's tokens.

### Two levers that measured as nothing (do not re-try these)

* **Temperature.** Hermes sends none, so llama-server's 1.0 / top_k 64 / top_p 0.95 applies. The
  chat harness's finding (0.0 burns ~139 tokens where 1.0 burns ~270 for the same answer,
  `localgpu/chat.py`) does **not** transfer to the Hermes route. Same prompt, same toolset,
  generated tokens on the answer turn: 1.0 -> 619 (73 s) / 427 (58 s) / 810 (88 s); 0.2 -> 796
  (83 s) / 1509 (147 s) / 954 (116 s); 0.0 -> 1387 (138 s). Run-to-run variance of 427-1509 tokens
  for one identical prompt swamps any sampling effect. Left at the default.
* **`max_tokens` as a latency guard.** A 900-token cap did not shorten the turn: Hermes saw the
  truncation and continued the answer in a *second* request (900 tokens, then a 664-token prompt
  for 334 more), 128 s, same total. Reverted.

### One lever that is free, and taken

Hermes fires a second call per session to name it (`json_schema` response format,
`reasoning_effort: none`, ~275-token prompt, ~20 tokens out). `auxiliary.title_generation.
model_upgrade_enabled: false` keeps the instant derived title and drops that round trip --
verified by counting requests for a one-shot: 3 -> 1. On a 12 tok/s card that is a free second or
three per session.

**Verdict.** Viable as a primary agent for tool-using work on this project, at ~41 s per turn and
~11.5 tok/s; viable as a subagent via an explicit delegation pin, at the same rate. Use
`bin/hermes-local` rather than bare `hermes chat -m local` -- it is the same thing with half the
prompt -- and prefer an interactive session over `-q` when the work needs `execute_code` or
`python3 -c`.

---

## 18. The same model on both runtimes: llama.cpp vs MAX on Gemma 4 E2B (session 17, 2026-11-13)

Every earlier cross-runtime comparison changed the model as well as the runtime, so the runtime was
never isolated. Gemma 4 E2B exists in both worlds -- Google ships a QAT Q4_0 GGUF
(`google/gemma-4-E2B-it-qat-q4_0-gguf`, 3.35 GB) and MAX loads the HF checkpoint
(`google/gemma-4-E2B-it`, bf16) -- so it is the one model where the quantization changes while the
model is held fixed. New model-table entries: `gemma4-e2b` (llama), `gemma4-e2b-max-int8`, and the
pre-existing `gemma4-e2b-max`.

Protocol: one client (curl through the Ollama/OpenAI proxy on 127.0.0.1:11434, so nothing in the
test knows which backend is up), one body -- a single user turn, temperature 0, max_tokens 96,
streamed -- two runs per backend. Prefill is the same shape with a 629-token prompt and 8 output
tokens. Every backend produced **exactly 96 tokens**, so the token budget is held fixed as well.
llama.cpp ran with `LOCAL_GPU_PLAIN=1` (no n-gram speculation) because MAX has no equivalent.
Decode is 96 / (wall - time-to-first-token), client-side, because MAX reports no per-request
timings; llama.cpp's own reported decode agrees to within 2% (17.98 vs 17.6 client-side).

| Gemma 4 E2B, greedy, same prompt | decode tok/s | prefill tok/s | TTFT (31-token prompt) |
|---|---|---|---|
| llama.cpp, QAT Q4_0 (~2.8 GB/token) | **17.98** | **1156** | 0.03 s |
| MAX, int8 W8A16 (patch 0004, ~4.8 GB/token) | 13.0 / 13.2 | 231 | 0.37 s |
| MAX, bf16 (~9.6 GB/token) | 9.6 / 9.6 | 12.9 | 2.0 s |

What it settles:

* **llama.cpp wins on the same model, but by less than the byte count predicts.** 17.98 against
  13.1 tok/s is 1.38x, where 4.5 bits against 8 bits predicted about 1.8x. Against MAX bf16 it is
  1.87x. So the earlier "~1.8x expected" was the right direction and the wrong size.
* **MAX is not the slower engine per byte -- it moves MORE of them.** At these rates llama.cpp
  shifts ~50 GB/s of weights, MAX int8 ~63 GB/s and MAX bf16 ~92 GB/s, i.e. MAX bf16 decode sits
  exactly on this card's measured ceiling (~92 GB/s, section 13), int8 leaves headroom, and
  llama.cpp's 50 GB/s says its E2B decode is launch-bound rather than bandwidth-bound. llama.cpp
  wins because the QAT GGUF reads ~40% fewer bytes per token, not because the runtime is better at
  this card. That is the honest correction to "llama.cpp always beats MAX": it wins on every
  configuration measured, for a quantization reason.
* **Prefill is where the gap is real**: 5x against MAX int8, 90x against MAX bf16. The int8 patch
  buys 18x on MAX's own prefill (12.9 -> 231 tok/s) and 1.36x on decode (9.6 -> 13.1), which is the
  strongest argument on record for keeping patch 0004.
* Text is **not** byte-identical across runtimes: llama.cpp answers in prose, MAX in markdown with
  a heading (`## How a Hash Table Handles Collisions`). Each runtime renders its own chat template,
  which is a confound on both speed and text -- within MAX, int8 and bf16 openings are identical,
  as section 16 found.

**Still not measured:** E4B on MAX (the 14.9 GiB download), and llama.cpp at the *same* 8 bits as
MAX int8 (a Q8_0 GGUF) -- that is the pair that would separate runtime from quantization
completely. On the evidence above the defensible claim is "llama.cpp is faster in every
configuration measured, and the reason is the quantization", not "llama.cpp is always faster".

**Hygiene note.** The first request after a change of prompt shape absorbs a KV-cache rebuild that
is visible only client-side: 18 s and 35 s of TTFT were seen while the server's own `prompt_ms`
stayed at 272 ms. Warm up once and discard that request, and do not measure while a model download
is saturating the container.

### The other axis: E4B against E2B, same runtime and harness

Same protocol as the table above -- llama.cpp, `LOCAL_GPU_PLAIN=1`, context 8192, same client,
same 96-token budget -- so only the model changes:

| Gemma 4 on llama.cpp, Q4_0 | weights per token | decode tok/s | prefill tok/s |
|---|---|---|---|
| E4B (the default, ~7.5B params) | 5.15 GB | 12.29 | 524 |
| E2B (~5B params) | 3.35 GB | **17.98** | **1156** |

E2B is 1.46x faster on decode and 2.2x on prefill, and the byte ratio (5.15 / 3.35 = 1.54)
predicts the decode gap almost exactly: both models move 60-63 GB/s of weights, so on this card
decode speed tracks weight bytes. The larger model is therefore not the faster one -- E4B is the
default because it is the more capable model (tool calling, vision), not the quicker one.

Two caveats on that table. E4B's `-ot` embedding-offload flag is dropped by `LOCAL_GPU_PLAIN`, so
its prefill here is below the pp512 840 measured with the flag on. And E4B's *serving* config (64K
context, n-gram speculation on) measured 13.3 tok/s on this same prompt, 8% above the plain number
above, while the 12-turn chat bench showed speculation buying nothing at all (section 16) -- this
answer's repeated structure is what a draft model feeds on.

### E2B as a Hermes agent: the same eight turns -- and it is now the default

A real E4B Hermes session (eight turns: what is installed, what is in /Applications, the attached
GPU's specs and how the host drives it, "use the docker containers to figure that out", research
tinygrad/TinyGPU "and dont ssh to cherryblossom", summarize the patches in that repo, is the VRAM
in use, does the VRAM hold the model) was replayed on E2B -- same cwd, same Hermes, one resumed
session, `-m local`, same prompts verbatim:

| # | E4B | E2B |
|---|---|---|
| 1 | brew + npm + pip, then invented `fd-cli`, `fd-list`, `golang` | 2 calls; every brew package and app it named is real, but no npm/pip sweep |
| 2 | `find ... *.app`, listed 30 of the 34 entries | no tool call, reused turn 1 (the more complete list) |
| 3 | `system_profilers` (a typo), `nvidia-smi`, **no answer from tools** -- answered from memory and named the 3090 as the attached card | loaded the `gpu-falcon-re` skill and answered GP102 / 607 MHz / shimd correctly |
| 4 | `docker ps`, no final answer | `docker ps` + `docker inspect llamacpp`, answered (image, mounts, port 8080) |
| 5 | two web searches, no ssh | one web search, no ssh |
| 6 | read five real modules, grounded | search + web; claimed the project **bypasses DriverKit** -- it uses it (wrong, and it contradicts its own turn 5) |
| 7 | "yes", VMM + test state | hedged: "implies that the VRAM is being used" |
| 8 | "yes ... via tinygrad" (the LLMs run on llama.cpp/MAX) | "yes, llama.cpp and Modular MAX" -- correct |

Wall clock, prompt to final answer, from the transcript timestamps: **E4B 833 s over the eight
turns, E2B 397 s** -- 2.1x, and most of that gap is E4B's thinking channel. E2B also served the
same 64K context with 7.0 GiB of VRAM free against E4B's 2.9.

So E2B does the same work, and it is the better *tool chooser* on a question the workspace already
answers (turn 3) and the more accurate summarizer of long tool output (turn 1); E4B is still the
one to trust when the answer requires reading source (turn 6). Both invented details in turn 1 --
E4B more.

**E2B is therefore `DEFAULT` in `localgpu/local_gpu.py` and what the `local` alias resolves to in
`~/.hermes/config.yaml`**, with E4B one flag away (`local-gpu serve -m gemma4-e4b`, alias
`local-e4b`). The serve default and the alias have to move together: the proxy rewrites every
request's model name to whatever is being served, so `-m local-e4b` against a server started on
E2B silently answers from E2B. E2B's context entry is now 65536 (the Hermes floor), so reproducing
the comparison table above needs `--ctx 8192` on both sides.
