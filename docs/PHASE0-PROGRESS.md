# Phase 0 progress — 2026-11-08 (session 2)

## New capabilities proven on this hardware

1. **BAR0 writes work through the TinyGPU transport.** Wrote `PMC_ENABLE` and read
   back an identical value. Route A is no longer read-only — the whole bring-up
   depends on this and it is now confirmed.
2. **"PRI poison" at an engine window means the engine is switched off.** This is
   documented in envytools' PMC page ("Most engines disappear from MMIO space and
   reset to default state when disabled") and it is the general method for
   mapping this card: enable the block, then its registers appear.
   Proof: enabling `PGRAPH` (`PMC_ENABLE` bit 12) turned FECS @0x409000 from
   `0xbadf1201` poison into a live falcon.
3. **Two live falcons located:**

   | Falcon | Base | HWCFG | IDLESTATE | Notes |
   |---|---|---|---|---|
   | PMU | 0x10a000 | 0x400e0100 | 0x4 | visible out of reset |
   | FECS | 0x409000 | 0x20202060 | 0x4 | only after PGRAPH is enabled |

4. **PMC_ENABLE quirks.** Out of reset it reads `0x40002020` — only PRING (5),
   PDAEMON (13), PDISPLAY (30) are set; PFIFO, PGRAPH, PSEC, PTIMER, PBFB,
   PCOPY, PCOUNTER, PFFB, PVDEC, PVENC are all off. Writing every documented
   engine bit gives a readback of `0x5024f120` instead of the written
   `0x7035f120`: **bits 16 (PTIMER), 28 (PCOUNTER) and 29 (PFFB) refuse to set**
   (presumably always-on or owned by another block).

## RESOLVED: the falcons are in the LOW aperture, not at 0x840000

- nouveau's GP102 ACR **load** runs on **SEC2**
  (`gp102_acr_load_fwif` → `NVKM_ACR_HSF_SEC2`); only the *unload* runs on PMU.
- SEC2 @0x840000 returns `0xbadf1100` = "no target for that address" even with
  PSEC (bit 14) enabled, and a 256-page scan of the whole 0x800000–0x900000
  aperture finds no falcon there at all.
- A hobbyist GV100 project reports the same poison for SEC2 through plain BAR0
  and attributes it to those falcons being reached through **PRAMIN / falcon DMA
  windows** rather than direct BAR0 MMIO; it also lists a *corrected* SEC2 base
  of **0x87000** in the low aperture.
- **Found.** A full 4 KB-page scan of the low aperture turns up falcon-shaped
  windows at **0x84000** and **0x87000** — the docs' `0x840000` was almost
  certainly a hex-place error (`0x84000 << 4`). Nothing exists in the
  0x800000–0x900000 aperture at all. (The scan also rejects pages that mirror a
  single value — probe6's hits at 0x61a000/0x61b000 were that kind of artifact.)
- Gating, measured one PMC_ENABLE bit at a time:

  | Window | HWCFG | IDLESTATE | Gate |
  |---|---|---|---|
  | PMU 0x10a000 | 0x400e0100 | 0x4 | always on |
  | **0x87000** | 0x20420100 | 0x4 | **always on** (live at power-on) |
  | 0x84000 | 0x2040c028 | 0x4 | only with the full engine set — one of PRING/PVDEC/PTIMER/PVENC/PBFB/PCOPY2/PCOUNTER/PFFB/PDISPLAY |
  | FECS 0x409000 | 0x20202060 | 0x4 | PGRAPH (bit 12) |
  | 0x8b000 | 0x00a600a6 | 0x01000100 | always readable but synthetic (mirrored values) — not a falcon |
  | GPCCS 0x41a000 | poison 0xbadf3000 | — | **never appears**, even with every engine bit set |

- **SEC2 is therefore 0x87000** — the GV100 project's "corrected" base, and the
  only always-on falcon besides the PMU. For GP102 the ACR *load* runs on SEC2,
  so this was the blocker for Phase 3; it is now located and readable.
- **GPCCS is not at the gf100-era 0x41a000 on GP102** and does not appear in
  0x400000–0x440000 at all, even with all engines enabled. It presumably needs
  the GR-internal access path / secure unlock (`gp102_flcn_reset_eng`) that the
  real ACR flow performs. Deferred, not blocking: the ACR bootstraps FECS first
  and SEC2 drives GPCCS.

## Falcon programming facts taken from nouveau (for Phase 2)

- **PIO load path (gm200 falcon, applies to gp102):**
  - IMEM write init: `wr32(0x180 + port*0x10, (sec ? BIT(28) : 0) | BIT(24) | imem_base)`
  - DMEM write init: `wr32(0x1c0 + port*0x8, BIT(24) | dmem_base)`
  - DMEM read init: `wr32(0x1c0 + port*0x8, BIT(25) | dmem_base)`
  - DMEM data port: `0x1c4 + port*0x8`
  - TRACEPC scratch: SCTL `0x240`, TIDX `0x148`, IP `0x14c` (falcon exception PC)
- **gp102 PMU falcon hal:** `reset_eng = gp102_flcn_reset_eng`,
  `debug = 0xc08`, `cmdq = {0x4a0, 0x4b0, 4}`, `msgq = {0x4c8, 0x4cc, 0}`,
  `bind_inst = gm200_pmu_flcn_bind_inst`, `reset_wait_mem_scrubbing`.
- **ACR firmware:** `nvidia/gp102/acr/{bl.bin,ucode_load.bin}` for load,
  `{unload_bl.bin,ucode_unload.bin}` for unload.
- **ACR WPR layout** (`gp102_acr_wpr_layout`): 11 × `wpr_header_v1` (MAX_LSF),
  align 0x100; +0x100 shared sub-WPR headers; then per-LSF: LSB header
  (256-aligned), image (4096-aligned), bootloader data (256-aligned).
  `wpr_alloc`: `ALIGN(wpr_size, 0x40000) << 1`, split into `shadow_start`,
  `wpr_start`, `wpr_end`.
- **WPR region properties** (`gp102_acr_load_setup`): `region_id = 1`,
  `no_regions = 2`, `read_mask = 0xf`, `write_mask = 0xc`, `client_mask = 0x2`,
  addresses are `>> 8`.
- **SEC2 ACR bootstrap is message-based**, not register pokes:
  `NV_SEC2_ACR_CMD_BOOTSTRAP_FALCON` with `FLAGS_RESET_YES` sent over the SEC2
  command queue with a 1 s timeout — so the ACR path needs a working cmdq, which
  in turn needs the instance block / MMU work scheduled for Phase 4.

## Consequence for the plan

Phase 2 (run our own code on a falcon) can start on the **PMU** falcon today —
it is visible, writing to it is proven possible, and nouveau's PIO load sequence
plus the reset/enable hal are known. SEC2 must be located before Phase 3 (ACR),
so it moves to the top of the Phase 0 list.
