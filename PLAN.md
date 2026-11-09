# Making the GTX 1080 Ti (GP102) do inference from this MacBook Air

> **Status 2026-11-09 (session 7): rungs 0–5 DONE.** MAX runs SmolLM2-135M on the card
> through the libcuda/NVML shim + `maxpatch/` (GPU text == CPU text, 6.6 tok/s; see
> [RESULTS.md](RESULTS.md) §9). Next: launch overhead, bigger models (§7). Boot clocks:
> 607 MHz SM / 405 MHz memory.


Plan of record, 2026-11-08. Host: MacBook Air M1, macOS 26.6.2. Enclosure: Razer
Core X V2 (Intel `8086:5786` bridge) on a TB3/USB4 port, link trained PCIe Gen3
x4. Transport: tinygrad's TinyGPU DriverKit extension + remote-PCI server.

## 0. The ladder — what "as close as possible" means

Every rung is independently verifiable and independently useful. The goal is to
climb as high as the hardware and the transport allow, and to know exactly where
we stopped.

| Rung | Milestone | Verification |
|---|---|---|
| 0 | Transport + identification | **DONE 2026-11-08** |
| 1 | GPU's Falcon runs code we control | mailbox echo |
| 2 | ACR secure boot | WPR2 established, FECS/GPCCS booted |
| 3 | MMU + FIFO channel | no-op pushbuffer advances channel get pointer |
| 4 | Compute dispatch | vector-add kernel writes correct result to sysmem |
| 5 | Inference on the card | `llama-bench` / tinygrad LLM numbers over TB |

## 1. Three routes, ranked by time-to-inference

| Route | What | Time | Confidence | Ceiling |
|---|---|---|---|---|
| **A** | macOS-native: Pascal bring-up on the TinyGPU transport (this document) | 3–6 months focused | ~40–60% to rung 4; ~20–30% to rung 5 | unknown; likely boot-clock-limited |
| **B** | Linux on the M1 Air (Asahi/m1n1) + nouveau + NVK + llama.cpp Vulkan | weeks–months | low | boot clocks, no CUDA |
| **C** | Card in a PCIe host, Mac as client (LAN Ollama/llama.cpp) | hours–days | ~certain | full card |

Route B is not the shortcut it looks like: Apple Silicon **USB4/Thunderbolt in
Linux still lacks PCIe tunneling** in upstream terms, the community enablement
effort for eGPU on the M1 platform (`dracinn/Apple-Silicon-eGPU`) explicitly
excludes NVIDIA, and even on success nouveau **cannot reclock GP10x** — Pascal
runs at boot clocks forever without signed PMU firmware (Collabora/Mesa).
NVK does support Pascal (conformant Vulkan 1.4 since Mesa 25.1), so if the
platform layer ever lands, this route becomes cheap. Today it is not.

Route C is the only route with a guaranteed inference outcome, and it shows what
the card is actually worth: on a normal PCIe slot a 1080 Ti does **pp2048 875
t/s with llama.cpp CUDA vs 141 t/s Vulkan** (CUDA needs driver ≥ 570 for the
sm_61 PTX in the prebuilt CUDA 12.8 binaries; CUDA 13 dropped Pascal).

## 2. Measured basis for Route A (2026-11-08, this machine)

Everything below was read from the card through the TinyGPU transport, read-only.

- BAR0 16 MB / BAR1 256 MB / BAR3 32 MB mapped; tunnel Gen3 x4; **bus mastering
  already enabled** (PBUS config mirror at BAR0+0x088000 reads `0x1b0610de`,
  `0x00100007` → command = IO|MEM|MASTER).
- Identity: `PMC_BOOT_0 = 0x132000a1` → GP102 rev A1; `PMC_BOOT_42` → arch `0x13`.
- **PMU Falcon @0x10a000 is fully decoded and alive** (all 9 registers readable,
  `IDLESTATE = 0x4`). The ACR/FWSEC chain runs on this engine — it is reachable.
- SEC2 @0x840000, FECS @0x409000, GPCCS @0x41a000 read **poison** → those GM-era
  offsets are wrong for GP102. Register addresses must come from a generated
  gp102 database, not from older-generation docs.
- **VBIOS is visible** at BAR0+0x300000 (`0x55AA` signature); PCI ROM BAR assigned
  (`config 0x30 = 0x81000000`).
- `linux-firmware` ships the signed blobs: `nvidia/gp102/{acr,gr,sec2}/` —
  `acr/bl.bin`, `acr/ucode_load.bin`, `acr/ucode_unload.bin`, `gr/fecs_*`,
  `gr/gpccs_*`, `gr/sw_*.bin`, `sec2/*`.
- `0xBADF****` / `0xBAD0****` are NVIDIA PRI poison ("no target for that
  address"), so "decoded value vs poison" is a reliable signal for map hunting.
- **FLR trap:** `nvdev.py` reads a Turing-era WPR2 register at `0x1fa828`; on
  Pascal that is poison, which is nonzero, so the runtime concludes "WPR2 is up"
  and issues a PCIe function-level reset *before* it ever reaches the
  architecture lookup. Passing that gate means a reset on an Intel-bridge
  enclosure, where a non-returning FLR costs a power cycle.
- VRAM backing is **not yet proven**: BAR1 offset 0 returns a repeating
  `0xBAD0ACxx` pattern. Either stale VRAM content or an unmapped-aperture
  sentinel — must be disambiguated before anything relies on VRAM.

## 3. Phases

### Phase 0 — reconnaissance (mostly done; 3 items, hours)
1. VRAM-backing test: write/readback a scratch dword at BAR1+0x100000.
2. FLR survival test: deliberately reset once, see whether the card returns.
   Costs an enclosure power cycle if it does not.
3. Register database: generate a gp102 map from envytools' regdb (tinygrad's own
   generator is fed from NVIDIA's published headers, which stop at Turing — so it
   cannot supply Pascal). Deliverable: `gp102_regs.py`.
4. Dump + parse the VBIOS (straps, DEVINIT scripts, ACR entry points).

### Phase 1 — transport hardening (days)
- Measure RPC latency; build batched `read_block`/`write_block` (AMC writes
  thousands of registers; one RPC per dword is fine for probing, not for init).
- Confirm `MAP_SYSMEM_FD` semantics for instance blocks and page directories.
- **No interrupts exist on this transport** → every wait is a poll with a
  watchdog: falcon halt, FIFO progress, semaphore release.

### Phase 2 — Falcon bring-up (1–3 weeks) — first real driver code
- Standard falcon sequence on the PMU engine: reset (`FALCON_CPUCTL`), load
  IMEM/DMEM by DMA, set `BOOTVEC`, start, poll `HALT`, read `MAILBOX0`.
- **Gate: our own ucode runs and echoes a known value in MAILBOX0.** This proves
  DMA + IMEM + start/halt + mailbox end-to-end, which is the whole shape of the
  ACR problem in miniature.

### Phase 3 — ACR secure boot (3–8 weeks) — **GO/NO-GO**
- Port nouveau's `gm200`/`gp102` secboot path (`acr_r352`): WPR headers, LSB
  headers, HS/LS handoff, ACR unload, wait for FECS/GPCCS boot.
- **Gate: WPR2 established and FECS/GPCCS report booted.** Nothing downstream
  matters if this fails; this is the coin-flip phase.
- Risks: nouveau's handshake assumes interrupts in places (mitigate with
  watchdog+retry); the Intel-bridge enclosure historically retrains the link
  under sustained VRAM traffic (Ada needed a GSP registry pin). Mitigation if it
  bites: an ASM2464PD-class USB4 adapter (~$130–300) — the bridge family
  tinygrad's own NVIDIA path was validated on.

### Phase 4 — MMU + FIFO (2–4 weeks)
- Pascal MMU v2 page tables (`gp100-mmu-format.pdf`), instance blocks, RAMIN,
  PFIFO/PBDMA channel, RAMFC/RAMHT/RAMUSERD.
- Gate: no-op pushbuffer advances the channel get pointer.

### Phase 5 — Compute dispatch (1–3 weeks)
- `PASCAL_COMPUTE_B` (0xC1C0) class, **QMD v01_07**, constant buffers, semaphore
  release. Gate: **vector-add kernel writes a correct result to system memory.**

### Phase 6 — Inference stack (1–3 months, a second project)
- *Framework path:* a GSP-less Pascal device in tinygrad's NV runtime (maps from
  Phase 0.3, our init in place of `NV_GSP`, QMD v01_07 emit, MMU v2). Tractable,
  but the payoff is tinygrad's LLM runner only.
- *Real path, option 1 — CUDA shim:* a userspace `libtinynv`-equivalent without
  GSP plus cudart/cuBLAS compatibility shims — the macuda design, retargeted — so
  **unmodified llama.cpp** runs on the card. Largest single chunk of work in the
  plan; the only route to *llama.cpp specifically*.
- *Real path, option 2 — a MAX backend (CHOSEN 2026-11-08; llama.cpp is
  explicitly not required):* Qualcomm
  completed its acquisition of Modular (July 2026) and Modular has since released
  the remaining MAX source — graph compiler, runtime and hardware backends —
  under the MAX License, explicitly as the starting point for bringing MAX up on
  new accelerators.  Its "device driver connector" is the same seam NVIDIA, AMD
  and Apple backends use, so our transport + MMU + FIFO/GPFIFO stack can plug in
  beneath it instead of us writing a CUDA compatibility layer.  Mojo kernels
  lower through LLVM, so the Pascal target is PTX assembled with CUDA 12.x
  `ptxas sm_61` (CUDA 13 dropped Pascal).  Trade-off to accept consciously: MAX
  serves models through `max serve`, **not** llama.cpp — pick this option if the
  goal is inference on the card rather than llama.cpp's binary.  Licence notes:
  free commercial use and modification, but no using MAX as input to reimplement
  it, modified source only distributed as contributions back, and hosted
  commercial inference on MAX must carry the "Modular Accelerated" wordmark.
- Initial work items either way: QMD upload + launch over the channel, kernel
  binaries for sm_61, memory management over our MMU, and a device-selection
  shim so the runtime finds the card.
- SASS: CUDA 12.x `ptxas` targets `sm_61`; verify NAK coverage for sm_61 if we
  want to avoid Docker.


### Expectations to set now
Even at rung 5, this card over this transport will likely run at **boot clocks**
(no PMU firmware → no reclocking on any open path), so treat throughput as
unknown-but-modest until rung 4 measures it. The transport itself is not the
bottleneck — on the same path an Ampere card used 1–2 % of its memory bandwidth.

## 4. Immediate next actions

1. Phase 0.3 — generate the gp102 register database (unblocks everything).
2. Phase 0.1/0.2 — VRAM-backing test, then one deliberate FLR test.
3. Phase 0.4 — VBIOS dump + DEVINIT parse.
4. Fork `TheTom/pascal-egpu` into `~/egpu-pascal/` as the scaffold; port the
   existing read-only probes in; express rungs 0–1 as a runnable test suite.
5. Decide on Route C (card in a PCIe host) as the guaranteed inference result,
   and whether to buy an ASM2464PD adapter as enclosure insurance.

## 5. Likelihood, plainly

| Outcome | Likelihood | Time |
|---|---|---|
| Solid rung 0 + register map + VBIOS parse + repeatable probe suite | very likely | days |
| Rung 1 — Falcon runs our code | plausible | weeks |
| Rung 2 — ACR boots GR firmware | coin flip | weeks–months |
| Rungs 3–4 — channel + a kernel executes | uncertain | months |
| Rung 5 — inference on the card from macOS | unlikely but real | months |

No one has run a kernel on a Pascal card over this transport. Rung 4 would be a
first. Route C gives inference on this GPU this week; Route A is the project.
