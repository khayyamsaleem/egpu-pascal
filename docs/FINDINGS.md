# GTX 1080 Ti over TinyGPU on a MacBook Air M1 — findings, 2026-11-08

Host: MacBook Air M1 (MacBookAir10,1), 16 GB, macOS 26.6.2 (25G83), arm64
Enclosure: Razer Core X V2 — Intel Barlow Ridge bridge (`8086:5786`), TB5
enclosure on a TB3/USB4 host port; tunnel trained **PCIe Gen3 x4** (40 Gb/s USB4)
Card: NVIDIA GTX 1080 Ti — GP102, `10de:1b06`, subsystem `3842:6696` (EVGA)
Driver: `org.tinygrad.tinygpu.driver2` 1.0.0/3 (tiny corp signed, teamID 9YG3G8543N), activated + enabled

## What works

| Check | Result |
|---|---|
| Enclosure enumerated | ✅ Razer Core X V2, USB4 mode, 40 Gb/s |
| GPU enumerated | ✅ `display@0`, class `030000` (VGA controller) |
| Driver binds the card | ✅ `tinygpu: opened device ven=0x10de dev=0x1b06` |
| PCIe link | ✅ 8.0 GT/s x4 (Gen3 x4), capability Gen3 x16 |
| BAR0 MMIO (16 MB) | ✅ mapped, registers read |
| BAR1 VRAM aperture (256 MB) | ⚠️ readable, but content is a repeating `0xBAD0ACxx` pattern — backing not yet proven (open item) |
| BAR3 (32 MB, RAMIN window) | ✅ mapped |
| Chip identification | ✅ `PMC_BOOT_0 = 0x132000a1` → GP102 rev A1 |
| Arch decode | ✅ `PMC_BOOT_42 = 0x132a1000` → architecture `0x13` = Pascal |
| GPU clock alive | ✅ PTIMER (`0x009400`/`0x009410`): advanced 80,645,056 ns over an ~80 ms window |

This matches the furthest documented Pascal-over-TinyGPU state (the four
register-level milestones of the `pascal-egpu` project), reproduced on GP102
through an Intel-bridge enclosure rather than their GP106 + ASM2464-style path.

## Where it stops

1. `runtime/support/nv/nvdev.py`:
   `chip_name = {0x17:'GA1', 0x19:'AD1', 0x1b:'GB2'}[chip_details['architecture']]`
   → **KeyError: 0x13** (Pascal is not in the map).
2. *Before* that line, the same file reads `NV_PFB_PRI_MMU_WPR2_ADDR_HI` from the
   `dev_fb/tu102` map (offset `0x1fa828`). On Pascal that address is undecoded
   and returns **`0xBADF1100`** (NVIDIA's PRI poison = "no target for that
   address"), which is nonzero, so the runtime concludes "WPR2 is up" and issues a
   **PCIe function-level reset**. A FLR that does not return behind an Intel-bridge
   enclosure is cleared only by power-cycling the enclosure.
   → Patching the architecture map is *not* a safe way to see the next failure.
3. Past the lookup the runtime is unconditionally GSP-based: `NV_GSP` is
   constructed, `fmc_boot` requires arch ≥ 0x1a, register maps are
   `tu102`/`ga102`/`gh100`. Pascal has no GSP and no FMC — it needs the older
   Falcon-PMU + ACR secure-boot + FIFO/QMD-v01 init sequence. That is the
   unfinished multi-month `pascal-egpu` plan (its phase 5, Falcon PMU init, has
   not been started), not a configuration change.
4. The CUDA-shim route (macuda/cuBLAS) boots GSP firmware too, so it hits the
   same wall.

Conclusion: **no compute for Pascal over this transport yet.** Register-level
access is the current ceiling for anyone, not just this machine.

## Environment built for this

- venv: `~/.venvs/tinygpu` (python 3.12.13) with `tinygrad 0.14.0` + numpy
- probes: `~/egpu-pascal/pascal_probe{,2,3}.py` — read-only, no writes, no reset
- run with: `APL_REMOTE_SOCK=/tmp/tinygpu.sock ~/.venvs/tinygpu/bin/python ~/egpu-pascal/pascal_probe2.py`
- the server is auto-spawned by `APLRemotePCIDevice`: `TinyGPU.app/Contents/MacOS/TinyGPU server /tmp/tinygpu.sock`

## The sane paths to actual compute on this desktop

1. Swap in the RTX 3090 (Ampere, supported) — same enclosure, driver already
   installed on this machine. tinygrad-only ≈ 2.3 tok/s on an 8B Q4; the
   macuda/CUDA-shim route is what makes it useful (unmodified llama.cpp /
   stable-diffusion.cpp CUDA backends).
2. Put the 1080 Ti in a machine with a real PCIe slot; use it over the LAN.
3. Keep it as a science project against `TheTom/pascal-egpu` (PLAN.md, 8 phases).

---

## Session 2 (2026-11-08): writes work, engines are gated, FECS found

See `PHASE0-PROGRESS.md`. Headlines:

- **BAR0 writes work** through the transport (proven by a `PMC_ENABLE` write/readback).
- **Poison at an engine window = engine disabled**, not a wrong address. Enabling
  `PGRAPH` (bit 12) brought **FECS @0x409000** into MMIO space
  (HWCFG=0x20202060, IDLESTATE=0x4). PMU @0x10a000 was already live
  (HWCFG=0x400e0100).
- `PMC_ENABLE` out of reset = `0x40002020`; bits 16/28/29 refuse to set.
- **SEC2 @0x840000 remains poison** with PSEC enabled — a problem, because
  nouveau's GP102 ACR *load* runs on SEC2. Prime suspect is the "corrected"
  0x87000 base in the low aperture, or a PRAMIN-window access path.
- Consequence: Phase 2 (run our own microcode on a falcon) can begin on the PMU
  falcon now; locating SEC2 is the new top Phase 0 item.

---

## Session 3 (2026-11-08): MMU v2 page tables built and proven; the ACR runs

See `MMU-AND-ACR.md` for the full record. Headlines:

- **`pascal/mmu.py` is complete** — a port of nouveau's `vmmgp100.c` /
  `vmmgf100.c` / `vmm.c` for the 4 KiB page descriptor: 49-bit VA
  (12+9+8+9+9+2), PD0 as 16-byte dual entries with the 4 KiB table in the high
  half, PTE/PDE aperture encodings (PTE VRAM = 0, PDE VRAM = 1), the instance
  block header, the MMU invalidate sequence, and `gm200_flcn_bind_inst`.
- **RT-2 PASSES on the card** (`pascal/mmu_test.py`, 18/18): a falcon xfer reads
  256 B from a *virtual* address into DMEM and writes it back to another, both
  byte-identical; the negative control (an unmapped VA) produces an MMU fault
  whose record names **our instance block** — `client=SEC reason=PDE
  addr=0x800000 inst=0x10001000`.
- **The ACR hang is fixed.** With `ctx_dma = VIRT` and the instance block bound,
  the HS bootloader runs to completion in 2 ms with no MMU fault (was: hung for
  2 s with SEC / PHYSICAL / PRIV_VIOLATION), and its application code now
  executes. Two of the app's own exit codes are decoded from a fuc6
  disassembly: `0x25` = "WPR below 0x18000000" (fixed by moving
  `WPR_ALLOC_ADDR` to `0x18000000`) and `0x29` = the current stop:
  `0x10e5f8` bits[23:20] must be 0 and they are 4.
- `pascal/falcon_peek.py` is new and generally useful: it reads (and tries to
  write) *any* MMIO register from the falcon's side through SEC2's external port
  — the path the ACR app itself uses, and one the host is refused. That is how
  `0x10e5f8` was shown to read the same for both masters and to be writable by
  neither.


---

## Session 4 (2026-11-08): ACR secure boot PASSES

See `MMU-AND-ACR.md` (session 4 section). Headlines:

- **The 0x29 stop was self-inflicted.** `0x10e5f8[23:20]` is an ACR *claim*
  field (HW mutex #2) that the HS load sets to 4 (GP102) as its second step,
  before the WPR range check. The first VMM run claimed it and then failed 0x25,
  so every later run failed 0x29. It is not PMU-firmware state.
- **`pascal/acr_unload.py`**: nouveau's gp102 unload on the PMU; mbox0 = 0x1d,
  claim 4 → 0, WPR2 torn down. Makes the ACR retryable without a power cycle.
- **ACR load PASS**: mbox0 = 0, WPR2 = 0x18040000-0x18080000 as requested, and the
  host is refused inside the WPR (0xbad0acXX + REGION_VIOL fault). Repeatable.
- Next: the SEC2 RTOS (LS) + cmdq path to bootstrap FECS/GPCCS. On GP102 the HS
  ACR does not boot FECS/GPCCS itself (nouveau `gp102_acr` has no
  `bootstrap_falcons`).
- **SEC2 RTOS boots and bootstraps FECS + GPCCS** (`pascal/sec2_rtos.py`): init
  message after 1 ms, `ACR_BOOTSTRAP_FALCON` replies `error_code 0` for both,
  both go to LS mode.
- **FECS + GPCCS run** (`pascal/gr_ctxctl.py`): FECS ready in 1 ms, answers host
  methods (ctx image 0x146500, zcull 0xf0c00, pm 0x4c00). PLAN rung 2 gate met.
  Next: `gf100_gr_init`, golden context, FIFO channel (RT-3).
- **RT-3 PASSES** (`pascal/fifo.py`): a GPFIFO channel on CE2's runlist fetches
  and retires our pushbuffer; host semaphore + copy-engine copies VRAM→VRAM,
  host memory→VRAM and VRAM→host memory, all byte-exact, repeatable.
