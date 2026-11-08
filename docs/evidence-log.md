# Pascal MMU v2 page tables + ACR status — 2026-11-08 (session 3)

Resumes from the point where session 2 stopped: the ACR HS bootloader hung on its
first DMA (SEC / PHYSICAL / PRIV_VIOLATION) and the fix was identified as
"build Pascal MMU v2 page tables and bind SEC2 to them", but the page-table module
was only a 27-line stub.

## What was built

| File | What it is |
|---|---|
| `pascal/mmu.py` | MMU v2 (GP100-class) page tables, ported from nouveau `subdev/mmu/{vmmgp100.c,vmmgf100.c}` + `vmm.c` (level walk) + `subdev/acr/base.c` (instance block) + `falcon/gm200.c` (`bind_inst`) |
| `pascal/mmu_test.py` | the hardware round trip (RT-2), 18 checks |
| `pascal/acr.py` | now drives the MMU path: HS image mapped at a VA, instance block bound, `ctx_dma = VIRT`; `--phys` reproduces the old physical-DMA attempt |
| `pascal/falcon_peek.py` | read/write any MMIO register **from the falcon's side** via SEC2's external port — the same path the ACR app uses |
| `pascal/probe_ext_port.py`, `probe_10e000.py`, `probe_pm_block.py`, `probe_fb_size.py` | the probes that produced the tables below |

`mmu.py` implements exactly the 4 KiB-page descriptor `gp100_vmm_desc_12`: a
49-bit VA split `[2][9][9][8][9][12]`, PD0 as 16-byte *dual* entries with the
4 KiB page table in the **high** 8 bytes (`nvkm_vmm_ref_hwpt` puts it in
`pt[desc[lvl-1].type == SPT]` = index 1, which `gp100_vmm_pd0_pde` writes as
`data[1]`), PTE/PDE aperture encodings (PTE VRAM = 0, PDE VRAM = 1), the
instance block (`0x200` = PD base | VER2 | 64 KiB, `0x208` = VA limit − 1), the
`0x100cb8`/`0x100cec`/`0x100cbc` invalidate sequence, and `gm200_flcn_bind_inst`.
It also carries a software walker (`VMM.walk`, which re-reads the tables *out of
VRAM*) and a GPU-free self test (`python3 pascal/mmu.py --selftest`).

## RT-2 PASSES — the card translates our page tables

`pascal/mmu_test.py`, 18/18:

| Check | Result |
|---|---|
| tables written to VRAM read back as a translate-able tree | ✅ (6 tables, PD0 dual entry low half 0 / high half the PT) |
| instance block `0x200` = `pd3 \| VER2 \| 64 KiB`, `0x208` = `0x1ffffffffffff` | ✅ |
| MMU invalidate completes | ✅ 0.1 ms |
| SEC2 binds to the instance block (bind status → 5 → 0) | ✅ `0x054=40010001` |
| **read round trip**: falcon xfer loads 256 B from a mapped *VA* into DMEM | ✅ byte-identical |
| **write round trip**: falcon xfer stores DMEM to a mapped *VA*, VRAM changes | ✅ byte-identical |
| **negative control**: same xfer from an unmapped VA | ✅ no data, and an MMU fault `unit 18 client=SEC reason=PDE addr=0x800000 inst=0x10001000` |

The negative control is the important one: the fault record carries *our instance
block address*, so the translation genuinely went through the tables we wrote.

## The ACR: the hang is gone, and two firmware conditions are decoded

`pascal/acr.py` before → after (identical setup, only the MMU path changed):

```
before: ACR HS: stopped=False after 2000 ms mbox0=cafebeef  pc=0000fd75   + SEC/PHYSICAL/PRIV_VIOLATION fault
after : ACR HS: stopped=True  after    2 ms mbox0=00000025  pc=00001336   + no MMU fault at all
```

The bootloader now runs to completion and its *application* executes (PC 0x1336 is
inside the app's IMEM, not the bootloader's). What is left is the app's own
environment checks, each one now decodable because `envydis -m fuc -V fuc6`
disassembles the app and the exit code lands in mbox0 (`SCRATCH0`):

* **`mbox0 = 0x25`, fixed.** The app's range check at imem `0x13d4`/`0x1444`
  returns 0x25 when the WPR region start is `<= 0x17ffffff`; it then refuses to
  program WPR2 (`0x100cd8`) from the descriptor, which is what
  `gm200_acr_wpr_check` checks afterwards. Fix: `WPR_ALLOC_ADDR = 0x18000000`
  (was `0x02000000`). Result: mbox0 0x25 → 0x29.
* **`mbox0 = 0x29`, current.** The step at imem `0x1600` reads `0x10e5f8` and
  errors unless bits[23:20] are zero, before programming the field to 2:

  ```
  0x1616  mov $r10 0x10e5f8 ; read32
  0x1620  extr $r9 $r10 0x14:0x17     ; bits 20-23
  0x1623  bra  e 0x162c               ; ok only if zero
  0x1626  mov  $r1 0x29               ; else fail  <-- what we hit
  0x1636  mov  $r0 0xff0fffff ; and   ; clears bits 20-23
  0x163e  or   $r0 $r9                ; stores the argument (2)
  ```
* **`mbox0 = 0x29` was briefly `0x25` with the WPR at 32 MB, and 0x1b/0x23/0x24
  are also produced by that step** — the same routine returns 0x1b if the
  read-back after the write differs.

### `0x10e5f8` (the current blocker) — measured facts

| Fact | How |
|---|---|
| it reads `0x80400000`; bits[23:20] = 4 | host BAR0 read, and **falcon-side read through the ext port** (controls 0x100ce0 → 0x2c8 and 0x12006c → 0xb both MATCH the host, so the port's addressing is trusted) |
| the host cannot write it | write, read back unchanged |
| the **falcon** cannot write it either | `falcon_peek.poke()` → read back unchanged |
| the block at `0x10e000` *is* decoded and FB-related | 9 distinct values across a sweep; `0x10e5f0 = 0x2c8` = the same value PFB `0x100ce0` reports as the FB size; `0x10e5f4 = 0x1e00`, `0x10e5fc = 0x68200dd3`, `0x10e600 = 0xcf` |
| novaue never touches `0x10e000` | grep of the whole nvkm tree |
| the neighbouring PDAEMON block (`0x10a000`) reads `0x10a5f8 = 0` | its documented reset value, per `rnndb/pm/pdaemon.xml` |

`0x10e5f8` is therefore real, read-only *status* in an FB-ish block, and the app
requires it at its reset value. Two candidates remain, and they are separable:

1. the field is card state that an earlier init stage should have left at 0 —
   most likely the **PMU firmware**, which this project has never loaded.
   Supporting evidence: nouveau's `gp100_ltc_init` is an empty function whose only
   content is `/*XXX: PMU LS call to setup tagram address*/`, and the LTC size
   register the app's own VRAM-size helper reads (`0x17e278`, used at imem
   `0x1393` as `(value & 0x3ffffff) * ltc_count << 11`) reads **0** on this card
   where a real 11 GB / 11-LTC card would have `0x80000`;
2. the register is only writable through a path/lock we have not found, so the
   ACR app would also fail its read-back (error 0x1b) on this card.

Distinguishing them needs either the PMU firmware running, or a card known to pass
this step. It is *not* an MMU problem any more: the app's own DMA works, with no
faults, and it is doing MMIO to a block that has never been initialised.

## Reproduce

```
~/.venvs/tinygpu/bin/python pascal/mmu.py --selftest        # encodings, no GPU
~/.venvs/tinygpu/bin/python pascal/mmu_test.py              # RT-2 on the card (18/18)
~/.venvs/tinygpu/bin/python pascal/acr.py                   # the ACR with the MMU path
~/.venvs/tinygpu/bin/python pascal/acr.py --phys            # the old physical-DMA path
~/.venvs/tinygpu/bin/python pascal/falcon_peek.py           # falcon-side register reads (+poke)
~/.venvs/tinygpu/bin/python pascal/probe_10e000.py          # the 0x10e000 / 0x10a000 sweeps
```

After any enclosure power cycle the card is cold: run `pascal/devinit.py
firmware/gp102_vbios.rom` first, then `pascal/fb.py` (as before).

## Where it stops

* RT-2 is **done**: page tables + instance block + binding, proven by round trip
  and by a fault at an unmapped VA.
* The ACR's original blocker is **fixed**; the ACR app now runs. It stops at its
  own step 0x29 on `0x10e5f8`, plus the LTC size register reading 0 — both
  consistent with the missing PMU firmware stage.
* Next rungs, in order: (a) decide the `0x10e5f8` question — load the PMU
  firmware (`subdev/pmu/gp102.c` on disk; note the PMU LSF is bootstrapped *by*
  the ACR on gp102, so this may need the "unload"/`FLR`-free re-init order
  nouveau uses), or find the write path; (b) then the WPR/LSF bootstrap
  (FECS + GPCCS are non-lazy in our WPR, so the ACR app drives them); (c) RT-3
  (FIFO/QMD pushbuffer).

---

# Session 4 (2026-11-08): the ACR load PASSES; 0x29 was self-inflicted

## What 0x10e5f8 actually is

Not PMU state and not a POST leftover. Disassembling the three HS images that
reference it (`acr/ucode_load.bin`, `acr/ucode_unload.bin`, `nvdec/scrubber.bin`)
shows bits[23:20] are an **ACR claim field** shared by the load and the unload,
guarded by falcon hardware mutex #2 (mutex acquire = falcon `0xa48`, mutex
registers at falcon `0xa50 + 4*n`; routine 0x14c2 / release 0x1481):

| image | routine | what it does to 0x10e5f8[23:20] |
|---|---|---|
| load (SEC2), main step 2 (`0x220` → `0x15fd`) | value = chip map in `0x166f` from `PMC_BOOT_42` (0xa00) bits[28:20]: **0x132 (GP102) → 4**, 0x134 → 4, 0x136 → 5, 0x137/0x138 → 2, else 0 | requires 0, else **0x29**; writes 4; read-back mismatch → 0x1b |
| unload (PMU), main step 2 (`0x1b7` → `0x322`) | same chip map | requires == 4, else 0x29; clears to 0 |
| scrubber (NVDEC), `0xaa4` | — | only ORs bit 31 (the `0x8...` in `0x80400000`) |

The load's main (`imem 0x100`) runs: chip check (`0x16af`, err 0xf) →
**claim (`0x220`)** → `0x12e0` bit 0 check (err 0x21) → … → WPR setup (`0xffe`,
where the 0x25 range check lives). So the **first** VMM run (`acr-vmm.log`, WPR at
32 MB) claimed the card, then failed 0x25. Every run after it failed 0x29
because the claim was still 4. The 0x25 fix (WPR above 0x18000000) was real; the
"0x25 → 0x29 progression" was a misreading.

## The fix: run the ACR unload, like nouveau's `nvkm_acr_reload`

`pascal/acr_unload.py` — nouveau's `gp102_acr_unload_fwif`:
`acr/unload_bl.bin` (start_tag 0x100 → boot 0x10000) + `acr/ucode_unload.bin` on
the **PMU** (0x10a000, `gp102_pmu_flcn`, debug fuse 0xc08), image mapped through a
VMM, PMU bound with **its own** sequence (`gm200_pmu_flcn_bind_inst`: FBIF
`0xe00..0xe10` = 4/0/4/5/6, `0x090` bit16, `0x480` = inst; status via `0x200 =
0x30e` then `0x20c` bits[14:12]), success = **mbox0 == 0x1d**.

```
ACR unload: stopped=True after 1 ms  mbox0=0000001d (want 0000001d)
0x10e5f8 80400000 -> 80000000 (claim 4 -> 0), no MMU faults, temp 37 C unchanged
```

Note: this stops the VBIOS PRE_OS app on the PMU (it was sleeping at pc 0xa4d).
Fan PWM is a hardware register (`0x10eb10/0x10eb14`, 800/1000 = 80%) and stays put.

## ACR load: PASS

```
ACR HS: stopped=True after 3 ms  mbox0=00000000
WPR2 regs: start=0x18040000 limit=0x18080000  (expected 0x18040000-0x18080000)
host read inside WPR @0x18040000 -> bad0ac1d (REFUSED) + fault PHYSICAL/HOST_CPU/REGION_VIOL
ACR RESULT: PASS
```

Round-trip evidence: (1) mbox0 = 0 is the app's own success code; (2) WPR2
(`0x100cd4` idx 2/3) is programmed to exactly our descriptor's region, and it is
`0xfffe0000/0x20000` (off) after an unload; (3) **negative control**: the WPR
range read back our bytes before the load, and after it the host read is refused
with `0xbad0acXX` data and a `REGION_VIOL` fault naming that address.

Repeatability: unload → load run twice back to back (`scratchpad cycle`),
identical results each time: claim 4→0→4, WPR2 off→on, WPR readable→refused.
`pascal/acr.py` now does `nvkm_acr_reload` itself (unload first if the claim is
set; `--no-unload` to skip).

## Where it stops now

FECS and GPCCS stay halted, not in LS mode (`sctl = 0x3000`, no LSMODE bit).
That is expected: nouveau's `gp102_acr` has **no `bootstrap_falcons`** (gm200 has
FECS|GPCCS), so on GP102 the HS ACR does not boot them itself. nouveau puts the
**SEC2 RTOS** (`sec2/{sig,image,desc}[-1].bin`, LSF id 7) in the WPR non-lazy,
with FECS/GPCCS lazy. The HS ACR leaves SEC2 in LS mode with the RTOS loaded; the
host starts SEC2 (`nvkm_sec2_init`: `0x014 = ~0`, start), waits for the RTOS
init message on its msgq (`0xa30/0xa34`, EMEM-backed, `emem_addr = 0x01000000`),
then sends `NV_SEC2_ACR_CMD_BOOTSTRAP_FALCON` (flags RESET_YES) for FECS and
GPCCS over the cmdq (`0xa00/0xa04`). Then `gf100_gr_init_ctxctl_ext` starts them.
That is the next rung.

## Session 4, continued: SEC2 RTOS → FECS/GPCCS bootstrapped → FECS running

`pascal/sec2_rtos.py` (log: `sec2-rtos.log`), nouveau's gp102 layout and protocol:

* WPR = FECS (lazy), GPCCS (lazy), **SEC2 RTOS v1** (`sec2/{sig,image,desc}-1.bin`,
  non-lazy; BLD = `flcn_bl_dmem_desc_v2`, `ctx_dma = FALCON_SEC2_DMAIDX_UCODE (6)`,
  `argc = 1`, `argv = EMEM base 0x01000000`). `acr.LSFW_SEC2` + `acr_load(sec2_rtos=1)`.
* HS ACR: PASS; it leaves SEC2 halted **in LS mode** (`sctl 0x3000 → 0x7021`,
  `cpuctl 0x50` = halted + alias, bootvec `0xfd00` = the RTOS BL).
* `nvkm_sec2_init`: `0x014 = ~0`, start via the CPUCTL alias (`0x130 = 2`).
  The RTOS posts its init message after **1 ms**: unit 0x01, 32 B, cmdq index 0
  @ EMEM `0x1000000` + 0x80, msgq index 0 @ `0x1000080` + 0x80 (queue
  registers `0xa00/0xa04`, `0xa30/0xa34`; EMEM PIO `0xac0/0xac4`).
* `NV_SEC2_ACR_CMD_BOOTSTRAP_FALCON` (unit **0x08**, 16 B, flags RESET_YES):
  FECS → reply `error_code 0, falcon_id 2` (1 ms); GPCCS → `error_code 0,
  falcon_id 3` (5 ms). Both falcons go `sctl 0x3000 → 0x7021` (LS mode), FECS
  bootvec `0x5c00` = its BL. No MMU faults. Wrapped in `0x260 = 0 / 1`
  (`nvkm_mc_unk260`) as nouveau does.

`pascal/gr_ctxctl.py` (log: `gr-ctxctl.log`), the tail of `gf100_gr_init_ctxctl_ext`:
`0x409800 = 0x41a10c = 0x40910c = 0`, start GPCCS then FECS (alias). FECS sets
`0x409800` bit 0 after **1 ms**; both falcons then sleep (`cpuctl 0x60`) waiting
for work. FECS answers host methods (`0x409500` data / `0x409504` method, host
zeroes `0x409800` first):

| method | meaning | answer |
|---|---|---|
| 0x21 | watchdog = 0x7fffffff | (no reply expected) |
| 0x10 | main ctx image size | **0x146500** |
| 0x16 | zcull image size | **0xf0c00** |
| 0x25 | pm image size | **0x4c00** |

This is PLAN.md rung 2's gate ("WPR2 established and FECS/GPCCS report booted").
Note `gf100_gr_init` (the GR register init nouveau runs *before* ctxctl:
sw_nonctx, GPC MMU, zcull, ROP/LTC counts, exceptions, ZBC) has **not** been run;
the sizes above come from FECS on an otherwise un-initialised PGRAPH, so treat the
zcull number in particular as provisional until it is.

Next rungs: `gf100_gr_init` for gp102 (+ `sw_nonctx`/`sw_ctx`/`sw_bundle_init`/
`sw_method_init` from `firmware/gp102/gr/`), golden context, then RT-3
(FIFO/PBDMA channel, runlist, USERD, a no-op pushbuffer).

## Session 4, continued: RT-3 PASSES — GPFIFO channel + copy engine, VRAM and host memory

`pascal/fifo.py` (log: `rt3-ce.log`), a port of nouveau gk104/gp100 FIFO with
formats from NVIDIA's class headers (`tools/nvclass/clc06f.h`, `clc1b5.h`):

* topology: CE2 = engine 5, **runlist 5**, PBDMA2 (`0x2398` mask 0x26), PMC
  reset bit 21 (was off; enabled). 4 PBDMAs, 4096 channels.
* USERD: BAR1 is unbound (`0x1704 = 0`) = physical, so BAR1 offset == VRAM
  address (PRAMIN-write/BAR1-read checked). USERD at VRAM `0x03000000`,
  `0x2254 = 0x10000000 | 0x3000`; GP_PUT is written **through BAR1** (snooped).
* channel 1: 4 KiB instance block = RAMFC (`gk104_chan_ramfc_write` values) +
  our VMM joined at +0x200; runlist 5 = TSG header + channel entry; bind
  `0x800000`, enable `0x800004 |= 0x400`, commit `0x2270/0x2274`.
* pushbuffer: host SEMAPHORE A–D release, `SET_OBJECT(PASCAL_DMA_COPY_B 0xc1b5)`,
  CE 1-D pitch copy + CE semaphore, `LAUNCH_DMA = 0x18e`.

| check | result |
|---|---|
| PBDMA2 loaded our RAMFC (USERD 0x3000200, GP base 0x100000, 512 entries, chid 1) | ✅ |
| GP_GET 0 → 1, PB get = 0x200058 (all 22 words) | ✅ 1 ms |
| host semaphore (PBDMA executes methods) | ✅ 0x5e4a0001 |
| CE copy VRAM → VRAM 4 KiB + CE semaphore | ✅ byte-exact |
| 2nd submission: CE **host memory → VRAM** (IOVA 0x4000, `sys_coh` PTE) | ✅ byte-exact |
| 2nd submission: CE **VRAM → host memory** (read back on the Mac) | ✅ byte-exact |
| negative controls (dst prefilled, semaphores 0, pages differ before) | ✅ |
| faults / FIFO interrupts | none |

Repeatability: re-running without teardown fails with FIFO **BIND_ERROR 0x01
(BIND_NOT_UNBOUND)** and a stuck runlist update; `Channel.teardown()` (disable,
TSG preempt `0x2634`, empty runlist, unbind) runs automatically when the chid is
already bound, and the rerun passes.

Next: GR. `gf100_gr_init` (gp102), golden context, a GR channel with
`PASCAL_COMPUTE_B (0xc1c0)`, QMD, and a kernel (SASS for sm_61).

## Session 4: INCIDENT — PRI ring taken down by a GR-init bug (needs a power cycle)

What happened: `pascal/gr.py` first version read the TPC count as
`rd(GPC_UNIT(g, 0x2608))`, which returns `0x0004000N`. nouveau stores it in a
`u8` (→ 4 or 5); the port did not mask it, so `tpc_nr = 262148`. The TPC
exception loops in `gf100_gr_init` then wrote fixed values (0xffffffff,
0xc0000000, 0x00dffffe, 0x105) at a 0x800 stride across all MMIO from 0x504000
upward: nonexistent TPC units, display, the PRAMIN window (aimed at the WPR, so
refused: REGION_VIOL write fault), PCCSR (64 garbage channels bound, FIFO
BIND_ERROR), and beyond. The out-of-range indices (>16 MB) did not raise.

Result: the **PRI ring is down**. Every ring station, including the ring master
(0x120000), returns `0xbad00100`; PBUS_INTR = PRIVRING|TIMEOUT. Host-side blocks
still work (PMC 0x0/0x200, PBUS 0x1xxx, PTIMER 0x9400, PFIFO 0x2xxx, PCCSR
0x800000), so the link and transport are fine.

Tried (pascal/recover.py + scratch):
* PCCSR cleanup: 64 stray channels disabled+unbound — worked.
* PBUS/FIFO status clear — worked.
* PMC_ENABLE bit 5 = PRIV_RING: confirmed it controls the ring (cleared →
  `0xbad0da00`, set → `0xbad00100`).
* nvgpu ring enable (0x12004c = 4, 0x122204 = 2) and the VBIOS's own ring
  bring-up script (gp102_vbios.rom @0x8f19: 0x122204=1, 0x120060=0,
  0x12004c=3, wait cond 0x0d `(0x12004c & 0x3f)==0`, 0x120048=0x503b4b49,
  0x12004c=1, 0x122204=2) after a bit-5 reset: the host's writes to 0x12004c /
  0x122204 themselves time out (PBUS fault record 0x8012004d/f). Not recovered.

Recovery: **power-cycle the enclosure**, then `devinit.py ROM`, `fb.py`, and the
chain. Fixes so it cannot recur:
* `card.py`: `rd`/`wr` assert the address is inside BAR0 (16 MB) and aligned.
* `gr.py`: `tpc_nr & 0xff` like nouveau, and asserts gpc_nr ≤ 6, each tpc_nr in
  1..5, rop_nr ≤ 16 before any loop writes.
`gf100_gr_init` itself is therefore still unvalidated on hardware.

## Session 4, after the power cycle: recovery + GR init VALIDATED

Enclosure power-cycled → card cold (PMC_ENABLE 0x40002020, ring master answers,
35 °C). `devinit.py` (42 ms) + `fb.py` (scrubber ok, 11 GB ok) as usual.
Baseline the old handoff asked for (A1): after POST + scrubber, `0x10e5f8 =
0x80000000` → claim **0**; the 4 seen before really was our own ACR runs.

`pascal/gr.py` (fixed) — full chain in nouveau order, log `gr-init.log`:
ACR + RTOS PASS → GR oneinit: 6 GPCs, TPCs [4,4,5,5,5,5] = **28 SMs** (1080 Ti =
3584 cores ✓), PPC mask 7 each, rop_nr 0xb → `gf100_gr_init` (312 sw_nonctx
writes, idle) → PGRAPH intr `0x400100 = 0` with all exceptions enabled →
bootstrap FECS/GPCCS error 0 → FECS ready 1 ms, methods answer (sizes
unchanged: they are firmware constants) → no faults. Readback spot-check: 5/7
exact; `0x500910` gains bit 30 (hw), `0x409c24` reads 0x1fffff (FECS rewrites
its exception mask on start, which happens after GR init).

Next: golden context (ctxgf100.c gf100_grctx_generate + gp102 overrides).

## Session 4: golden GR context generated (`pascal/grctx.py`, log `grctx.log`)

Port of `gf100_grctx_generate` (firmware path) + gp102_grctx overrides: global
buffers (pagepool 0x20000, bundle CB 0x3000, attrib CB 0x5e7400) mapped priv into
a VMM, ctx pointer at inst+0x210, FECS `bind_pointer` (method 3) → ok; main:
518 sw_ctx writes, pagepool/bundle/attrib setup, unkn, floorsweep (sm_id, tpc
counts, rop mapping, dist skip, tpc mask, smid config), 960 sw_bundle_init ICMDs
(0 timeouts), methods for classes 0xc197 (1511), 0x902d (17), 0xc1c0 (8);
PGRAPH idle, intr 0; FECS `wfi_golden_save` (method 9) → ok. 2.5 s. No faults.

Image (`golden_ctx_gp102.bin`): buffer prefilled with 0xdeadbeef; FECS wrote
0x0–0xa4b00 and nothing past it. Header word 1 = **0xa5e00** (saved length),
word 9 = 0xf0c00 (= zcull size from method 0x16), word 12 = 0x4c00 (= pm size,
method 0x25). Structure: hub section, then **6 segments of 0x9630 bytes = one per
GPC**. Note FECS's method-0x10 size read 0x146500 right after start and
0x275600 later; it is an allocation bound, not the image length.

---

# Session 5 (2026-11-08): the GR channel runs, but PGRAPH's context switch never completes

RT-4's first half, on hardware: `pascal/grchan.py`, a `PASCAL_COMPUTE_B` channel on
**runlist 0** built from the golden image.  It is the piece `fifo.py` (CE) did not
need: `gf100_gr_chan_new` + `gf100_gr_chan_bind` + `gk104_ectx_bind`:

* golden image (`golden_ctx_gp102.bin`) copied into a per-channel VRAM ctx buffer,
  header patched as `gf100_gr_chan_bind` does in the firmware path: 0xf4=0, 0xf8=0,
  0x10 = mmio_nr/2, 0x14/0x18 = the mmio list VA, 0x1c=1, 0x20=0x28=0x2c=0;
* the **mmio list** (`grctx.chan_buffers` with `mmio=True`, 145 pairs) written to a
  4 KiB VRAM page mapped **priv** into the channel's own VMM;
* pagepool / bundle CB / attrib CB mapped **priv** at the same VAs the golden image
  was generated against (0x1000000 / 0x1020000 / 0x1030000), so the image's embedded
  values agree with the mmio list;
* `inst+0x210 = ctx VA | 4` (`gk104_ectx_bind`), then the usual FIFO bind +
  `0x800004 |= runl<<16` + enable + runlist commit.

| check | result |
|---|---|
| channel binds on runlist 0, runlist 0 commit completes | ✅ `0x800000=8001160c`, `0x2284=2` |
| PBDMA0 loads the RAMFC (its 0x40000 register block mirrors it: 0x30000200 userd, 0x100000 GP base, chid 1) | ✅ |
| GP entry fetched and retired; **host semaphore executes** (PBDMA-side, subchannel 0) | ✅ `0x51e4a001`, GP_GET 0→1 |
| **class method executes** (SET_OBJECT 0xc1c0 on subchannel 4 → PGRAPH writes a semaphore) | ❌ never runs |
| PGRAPH exception / MMU fault / HCE error | none: 0x400100=0, 0x259c=0, PBDMA intr_1=0 |

So the GR channel's GPFIFO path works; what is missing is the **channel switch to
PGRAPH**, which the PBDMA waits for before it will hand a class method to the
engine.  Evidence of the stall: PBDMA0 `INTR_0` (`0x40108`) latches **bit 24**
(`0x01000000`, not in nouveau's name table and *not* maskable via `0x4010c`), and
`FIFO_INTR` (`0x2100`) bit `0x20000000` stays pending — it cannot be cleared by
writing it back, because the PBDMA holds the condition.  That bit also stops
*other* runlists (a later CE run then fails with `HCE_ILLEGAL_MTHD` on PBDMA2),
i.e. the whole FIFO scheduler wedges.

## The FECS side: the ctxsw handler spins at imem 0x11d9

While the switch is stuck, `0x409b00` **and** `0x409b04` read `0x8001160c` (bit 31
"next channel valid" + our channel's `inst>>12`), and FECS's PC sits in the loop

```
0x11b4  ld $r15 D[$r1]              ; r1 = 0x870   (DMEM)
0x11b6  ld $r9  D[$r0]              ; r0 = 0x86c
0x11b8  cmp $r15 $r9
0x11ba  bra e   0x11d3
0x11be  lcall   0xc5a
0x11d3  iord    $r9 I[0x4800+$r2*4] ; host 0x409120
0x11d6  and     $r9 0x2
0x11d9  bra ne  0x11b4              ; SPIN while 0x409120 bit 1 is set
0x11dd  xdwait
0x11df  clear   $r15
0x11e5  iowr    I[0x28800] $r15     ; host 0x409a20 = 0
0x11f0  iowr    I[0x21300] $r15     ; host 0x40984c = 0x200   (0x200 @ 0x11e8)
```

(i = `firmware/gp102/gr/fecs_inst.bin`, `envydis -m fuc -V fuc6 -i -b 0x0`; the
falcon I-space maps to host offsets as **host = I >> 6**, checked against the two
known anchors: SCRATCH0 `I[0x1000]` = base+0x40 and the external port
`I[0x1c000..0x1c300]` = base+0x700..0x70c.)

`0x409120` reads `0x00010012` in this state (bit 1 **set**, bit 4, bit 16) and the
**host cannot clear it** — writes of 0 or 1 to bit 1, and to bits 0-3, are ignored,
so the ack has to come from the engine/FIFO side.  While FECS is in this loop it
answers **no** methods at all (method 0x10 read 0 for 2 s until the engines were
reset): a stuck ctxsw takes the FECS's whole method interface with it.

## Recovery without a power cycle (new)

* **FIFO/PBDMA wedge** — reset PFIFO: `mask(0x200, 1<<8, 0)`, then
  `mask(0x200, 1<<8, 1<<8)`, then `fifo.Fifo(c).init()`.  Clears `0x2100`,
  `0x25a0`, `0x40108` and any stale channel binding; RT-3/CE passes again after it.
* **FECS stuck in the ctxsw loop** — reset GR as well (PMC bit 12) and re-run
  `pascal/gr.py` (ACR unload/load → SEC2 RTOS → GR init → ctxctl).  FECS then
  answers 0x146500/0xf0c00/0x4c00 again.
* The order used at the end of session 5, verified: reset bits 8+12 → `gr.py`
  (PASS) → `fifo.py` (RT-3 + sysmem PASS).

## Other facts learned

* FECS **method 0x10 is a variable allocation bound**: 0x146500 (fresh bring-up),
  0x275600 and 0x433000 seen in other runs (`gr->size` in nouveau = the value
  discovered at ctxctl time; the real image length is header word 1 = 0xa5e00).
  `grctx.py`'s assert was relaxed to `< 0x800000`.
* `fecs_bind_pointer` (method 3) only completes inside `gf100_grctx_generate`'s
  prologue (FECS reset 0x409614=0x70 → mask 0x700 → FE_PWR_MODE_FORCE_ON/AUTO →
  SCC RAM init 0x40802c=1).  Called later, from a FECS that has seen a failed
  switch, it times out forever (0x409800 keeps the stale method answer).  So it is
  **not** a step to call for an ordinary channel; nouveau calls it only while
  generating the golden context, with the *first channel's* instance block.
  `grchan.py --gen` (regenerate the golden image against this channel's inst, i.e.
  nouveau's real order) is written but has never got past that bind_pointer.
* `gf100_gr_fecs_ctrl_ctxsw(0x39)` (start FECS ctxsw) returns ok and changes
  nothing about the stall.
* `pascal/fifo.py` gained two hooks used by `grchan.py`: `Channel.pre_bind()` and
  `Channel.alloc_map(..., priv=)`.

## Session 5, continued: the channel never becomes ACTIVE, and what 0x409120 really is

Two corrections and one new rule, all from hardware:

* **`0x409120` is the falcon `XFER_STATUS`, not an engine ack.** rnndb's falcon
  group (`tools/envytools/rnndb/falcon.xml`) defines offset 0x120 as
  `XFER_STATUS`: bit 1 `PENDING`, bits 4:5 `UNK4`, bits 16:18 `STORES_PENDING`,
  bits 24:26 `LOADS_PENDING`. The value seen in the stuck state, `0x00010012`,
  is one outstanding **store** plus `PENDING` — FECS is waiting for its own FBMEM
  DMA to retire, i.e. for a translation through the instance's page tables.
  When that DMA can't complete, FECS spins at imem 0x11b4-0x11d9 and answers no
  methods. The fault record when it does break out names the *instance*:
  `unit 0 (eng0) inst=0x11001000 addr=0x2000000 hub=1 client=FECS reason=PDE`
  — a ctx VA walked with a *different* channel's page directory.
* **So FECS's MMU binding is per-instance and only `bind_pointer` (FECS method 3)
  moves it.** `ctx_chan` (FECS method 1, `0x409840=0x80000000` then 0x409500 =
  0x80000000|inst>>12, 0x409504 = 1) times out on GP102 whether or not the
  bound instance maps the ctx VA, and never rebinds the PD. Sending the
  `xfers` for a channel whose instance is not the bound one is exactly the PDE
  fault above. `pascal/grchan.py --ctx-chan` implements it for the record.
* **Channel state (`0x800004`) semantics** (rnndb `PFIFO_CHAN.STATE`, GK104+):
  bit 0 ENABLED, bit 10 ENABLE_TRIGGER, bit 11 DISABLE_TRIGGER, bits 16:19
  ENGINE, bits 24:26 and 28 read-only status. Two live signatures:
  - `0x00000003` right after bind (bit 1 *set*): the PBDMA has loaded the
    channel; a kick then executes subchannel-0 host methods (seen once);
  - `0x11000001` (bits 24, 28): the PBDMA refuses to load it; GP_GET never
    moves and no FIFO/PBDMA interrupt is raised.
  bit 1 is therefore "channel ACTIVE on its PBDMA", and **it only goes set when
  FECS was already bound to that same instance** (`bind_pointer` run for this
  channel, as `grctx.generate()` does in its prologue).

### The compute class is on subchannel 1, not 4

`tools/nvclass/clc1c0.h` is the class, and Mesa's `nv_push.h` maps
`SUBC_NVC1C0 = 1` (subchannel 4 is `NVC1B5`, the copy engine). `grchan.py` was
sending SET_OBJECT on 4 — the copy engine then rejects the class (that is why
the PBDMA latched `INTR_0` bit 24 with `intr_1 = 0`). Now fixed to 1.

### FECS method 0x10 is a per-boot watermark, not a size

0x146500 on a fresh FECS, then 0x275600, 0x433000, 0x95b000 after successive
generations in one boot — each `gf100_grctx_generate` advances it. Never use it
as the channel ctx size and never generate more than once per FECS boot;
`grctx.generate(..., size=)` now takes an explicit bound (the real image length
is still header word 1 = 0xa5e00).

### Where it stops (updated)

The GR channel binds, FECS can be bound to it, and the PBDMA-side host method
has executed. What has **not** happened yet: `0x800004` bit 1 set at bind time
in the current state, and any PGRAPH class method. Best sequence found so far,
in order, on a freshly reset FIFO:

1. `reset_engines.py` (PMC bits 8 + 12) then `pascal/gr.py` (PASS);
2. `recover_fifo.py` (PFIFO reset + init) so no stale channel state survives;
3. `pascal/grchan.py --gen` (generation *is* the bind_pointer step; it must run
   against `ch.inst` and `ch.vmm`, after `vmm.join`), then the kick.

Next thing to explain: why bit 1 does not set in that sequence while one early
run (pre-`--gen`, `chan=0x00000003`) had it set and executed host methods. The
difference to chase is the `0x800004` bit-24/28 latches at bind time.

### Isolation result: PFIFO's ENG_FAULT latch was blocking the channel (solved)

`0x00262c` (`ENG_FAULT`, write-1-to-clear, bit N = engine N) read **0x00000001**
— PGRAPH latched as faulted by PFIFO. With that set, every GR channel binds as
`0x800004 = 0x11000001` and the PBDMA never loads it. Clearing bit 0 and
re-binding the *same* channel gives:

```
chan 0x800004 = 00000003          <- ACTIVE immediately
after kick: GP_GET 0->1, host semaphore 51e4a001 OK
```

So the runlist-0/PGRAPH path works once the fault latch is clear; nothing else
in the channel/instance/context code had to change. Reproduced twice.

### Two bind outcomes, and what precedes each (new)

With `ENG_FAULT` clear, the GR channel's bind lands in one of two states:

| state at bind | meaning |
|---|---|
| `0x800004 = 0x00000003` | clean: PBDMA runs it, `GP_GET` advances, subchannel-0 host methods execute |
| `0x800004 = 0x15000003` / `0x15000001` | ACTIVE but bits 24/26/28 latched at bind: `0x40108 = 0x01000000` is already set, `GP_GET` stays 0, a later kick is ignored |

Nothing in the channel setup differs between them; what differs is the FIFO's
prior state. The clean case came from a quiet FIFO where only `ENG_FAULT` was
cleared. Two neighbours were measured:

* `recover_fifo.py` (PFIFO reset + init) *then* bind → dirty (`0x15000003`), and
  `FIFO_INTR = 0x20000000` is pending before the kick even happens;
* a CE run started on that dirty FIFO fails the same way (`chan 0x15000001`,
  `PBDMA2 GP_GET=0`, `FIFO_INTR = 0x20000100`, PB position 0x200178) — so it is a
  FIFO-wide dirty state, not a GR-specific one. (An earlier note claiming "CE then
  GR hangs the commit" was wrong: that CE run had already failed, exit 1.)

So: reset, clear `ENG_FAULT`, bind, and do not reuse a FIFO that just ran
another engine.

### CE vs GR, same traced bind: the divergence is the engine, not the channel

Identical code, identical RAMFC content, freshly reset FIFO for each. Every step
is byte-identical until the runlist commit:

```
CE  (chid 2, runlist 5): [after runlist commit] chan=00000003 fifo_intr=00000000 pbdma0 intr=00000000 0x40100=10011111
GR  (chid 1, runlist 0): [after runlist commit] chan=15000003 fifo_intr=20000000 pbdma0 intr=01000000 0x40100=00000000
```

So on engine 0 the PBDMA performs a real context switch (it clears its status
word `0x40100`, rnndb `UNK100` "some sort of status") and raises event 24 when
that switch cannot complete; on engine 5 it simply activates the channel. Event
24 is therefore the "engine 0 ctxsw did not finish" report, not a channel
malformation — consistent with FECS being the only thing that can finish it.

Re-clearing the event and re-committing does not help: bits 24, `0x2100`
`0x20000000` and state `0x15000003` return on every attempt, and the retries
wedge the runlist counter (`0x2284 = 0x00100002`). That retry loop has been
removed from `setup()`.

Next experiment: after a failed GR commit, walk candidate engine-ready/ack writes
(`0x002630` ENG_FREEZE, `0x00262c` ENG_FAULT, `0x400500`, `0x409808` bit 19,
`0x409800` high bits) one at a time and watch `0x40108` bit 24, `0x40100` and
`PBDMA0 ib_get` for the one that lets the PBDMA resume. That turns "find the ack"
into a bounded search instead of a guess.

**The FECS switch handler never runs (2026-11-08, from fecs_inst.bin).**
Disassembling the handler settles what the handshake is. Entry (`0x10e9`):
`I[0x23300]` → `0x4098cc = 0x200`, then an ack code `0x8000000N` → `I[0x28800]`
= `0x409a20`. It then pumps the context through the falcon's XD DMA
(`xdld`/`xdst` with `$xdbase`), spins on the XFER_STATUS *pending counts*
(`0x11b4`-`0x11d9`, DMEM 0x86c/0x870), and exits with `0x409a20 = 0` and
`I[0x21300]` → `0x40984c = 0x200`.

At our stall **all three of those registers read 0** (`0x409a20 = 0x4098cc =
0x40984c = 0`), i.e. the handler is never entered: FECS is not told about the
switch request that the PBDMA queued in `0x409b04` (`0x8001160c`). So the
failure is interrupt/handler delivery into the FECS, not the PBDMA's own logic.

Interrupt mask is not the lever: `0x409014` reads 0 and **ignores a write of
0xffffffff** (stays 0) while `0x40901c` accepts `0x40` — consistent with an
ACR/LS-mode falcon that owns its own interrupt delivery.

Next thing to try: the FECS *doorbell*. Nouveau's firmware path has a
"start/resume context switching" control (FECS control method 0x39, already
wired as `--ctxsw-start`). It was tried before, but only while the channel was
in the failed `0x11000001` state and `ENG_FAULT` was latched. Retry it now on a
channel that is ACTIVE with event 24 set, and watch `0x409a20`: non-zero there
means the handler finally ran, and the PBDMA should then resume.

**Doorbell result (2026-11-08): negative.** `--ctxsw-start-after` fires method
0x39 at the live ACTIVE channel; FECS accepts it (`ok`), but `0x409a20`,
`0x4098cc` and `0x40984c` all stay 0 and the PBDMA stays parked
(`intr = 0x01000000`, `ib_get = 0`). So 0x39 is not the trigger for servicing a
queued switch.  Note the host-side view is asymmetric: `0x409b00` bit 31
("switch in progress") is **clear** while `0x409b04` bit 31 (request valid)
is set — a request is queued that FECS never picks up.  Next candidates for who
is supposed to deliver it: the PFIFO engine/preempt path (runlist entry or TSG
preempt flag), or a PBDMA write to a FECS mailbox that our nominal bind path
never issues.


**Host→PGRAPH injection is not a working control path either.** Injecting a report semaphore for class 0xc1c0 over `0x404488/0x40448c` — both
after `setup()` and *inside* `grctx.generate()` right after the `sw_method_init`
pack (new `--gen-inject` hook, `grctx.generate(inject=…)`) — leaves `sem2 = 0`
with `PGRAPH intr = 0`, no exception and no method left in `0x400704`. The class
bind itself is accepted (`0x404204 = 0xc1c0`), so the ctrl-word encoding is
right, but PGRAPH does not execute the semaphore method: with no FIFO-driven
context switch it has nothing current to apply methods to. This kills the
"drive PGRAPH from the host as a fallback" idea for anything stateful; the
engine really does need the channel switch to complete.


**Result (2026-11-08): no host-side ack exists.** Thirteen candidates —
ENG_FREEZE, ENG_FAULT W1C, `0x400500 |= 0x10000`, `0x409808 |= bit19`,
`0x409800 = 0/1`, TSG preempt, runlist recommit, `0x800004` trigger again,
`0x002300 = 0`, `0x002628 |= 1`, `0x400704` clear, `0x409c20 = 1` — each followed
by a kick, left `0x40108` bit 24 set, `ib_get = 0` and `gp_get = 0`. Only
`0x40100` moved (0x80000002 → 0x80000008 after the TSG preempt). Conclusion: the
PBDMA is waiting on something only the engine side can produce, so the fix is
not a host write. Also structural: `SUBFIFO_ENG_MASK` (0x2390 series) puts engine
0 on PBDMA0/1 only, so there is no alternate PBDMA to route GR through.



`0x404200 + subc*4` (what `gf100_gr_intr` reads) reads
`sc0=c197 sc1=c1c0 sc2=a140 sc3=902d sc4..7=5040` after a *failed* run. Those are
`gf100_grctx_sw_method_init`'s bindings from golden generation — `sc1 = c1c0`
is **not** evidence that our `SET_OBJECT` reached PGRAPH. PGRAPH also shows
`intr=0`, `0x400704 = 0x80000000`, `0x400708 = 0` at the stall, i.e. it received
nothing from our pushbuffer.

At that stall: `PBDMA0 ib_get=0`, `pb_pos=0`, `0x40100 = 0x80000002`,
`0x40108 = 0x01000000`, `0x40148 = 0`; FECS idle and healthy
(`0x409800 = 0x275600`, `xfer = 0x10`, `0x409b00 = 0x0001160c`,
`0x409b04 = 0x8001160c`, `0x409808` load bit 19 = 0).
 With the channel ACTIVE the PBDMA
executes subchannel-0 host methods but stops at SET_OBJECT on subchannel 1 —
`PBDMA0 INTR_0` latches bit 24 (`0x01000000`, undocumented in both nouveau's
name table and rnndb's `gf100_psubfifo_intr`), `intr_1 = 0`, PGRAPH sees nothing
(`0x400100 = 0`, no method in `0x400704`). PSUBFIFO progress confirms the stall:
`IB_PUT = 1` but `IB_GET = 0`, `DMA_GET = 0`, and `0x40100 = 0x80000008`.

FECS is *not* stuck at that point: PC 0x518d (idle loop), `XFER_STATUS = 0x10`
(no pending), method 0x10 answers 0x275600, and `0x409b00 = 0x0001160c` /
`0x409b04 = 0x8001160c` — the switch request was consumed (next == current), yet
the PBDMA never resumes. Next: find the ack the PBDMA wants after the engine
side has the instance (PFIFO `ENG_STATE_0` = 0x80 for engine 0 vs 0x10013001 for
the CE's engine 5; `0x40100` bit 3), i.e. whether PGRAPH must be told the
channel is loaded, or that latch is just "switch in progress" and needs the
subchannel-1 class bound through a different method path.


Earlier test (negative): added the scheduler-reset half of `teardown()` (disable + TSG preempt + empty
runlist commit) to `Channel.setup()` unconditionally, so a fresh chid is inserted
the same way a re-bound one is. Result: **no change** — the GR channel still
binds as `0x11000001` and never becomes ACTIVE.

But the same code path on the copy engine still passes RT-3, and its channel
reaches `0x800004 = 0x00000003` (ACTIVE) immediately on runlist 5. So the
channel/instance/context construction is fine; what differs is **runlist 0 /
PGRAPH**: PFIFO will not load a channel for engine 0 even though the instance is
bound and FECS is bound to that instance. That is where the next session should
look — the PFIFO engine table for engine 0 (`0x00263c`/`0x2640`/`0x2644` read
0x5/0x80/0x1 here), runlist-0 blocking (`0x2630`), and whether PGRAPH must be
registered to the runlist by an explicit engine-table write before a ctxsw
channel can be loaded.




# Session 6 (2026-11-08): plan review against the MAX target; memory runs at 405 MHz

## VRAM bandwidth at boot clocks (`pascal/ce_bw.py`, `pascal/ce_bw2.py`, log `ce-bw.log`)

Copy engine, physical addressing, GPU-side timing from four-word CE semaphore
timestamps (no Thunderbolt latency in the number); 4/4 spot-check probes byte-exact
after a 1 GiB copy; warm-up copies byte-exact.

| test | result |
|---|---|
| CE2 VRAM->VRAM, 16 MiB .. 1 GiB | 24.5 GB/s traffic (read+write), flat across sizes |
| CE3 alone | 24.7 GB/s |
| CE2 + CE3 concurrently, 1 GiB each | **29.2 GB/s aggregate** -> DRAM-bound, not CE-bound |
| sysmem -> VRAM (CE reads host memory over TB), 4 MiB | 0.69 GB/s |

Clock registers agree: `0x1373f4 = 0x11` (mem clock source = refpll path, gk104
`read_mem` case 1), MPLL `0x132000 = 0x18010000` (bit 0 clear = **MPLL off**),
refpll `0x132024 = 0x00043b01` (M=1 N=59 P=4), `0x132030 >> 16 = 0x1000` (fN) ->
27 MHz * 60 / 4 = **405 MHz**, the 1080 Ti's idle (P8) memory clock (P0 = 5505).
Theoretical at 405 MHz: 484 GB/s * 405/5505 = ~35.6 GB/s; measured 29.2.

Consequence for the target: LLM decode is bandwidth-bound, so at this clock the
card (~29 GB/s) is below the M1 Air's own unified memory (68 GB/s). Getting real
speed needs a memory reclock (MPLL + GDDR5X training), which no open driver does
on Pascal. Feasibility of that is now a first-class question.

## Corrections to the session-5 rung-4 evidence (read from the card + fecs_inst.bin)

* `0x409014` is falcon **IRQMCLR** (write-only), not the mask. The real mask
  `0x409018` reads **0x8704** on a fresh FECS boot (bits 2, 8, 9, 10, 15 —
  identical to nouveau's hub.fuc), so the channel-switch interrupt (ext bit 8) is
  enabled. The session-5 write of 0xffffffff to 0x409014 *disabled every FECS
  interrupt* for the rest of that FECS boot; negatives measured after it in the
  same boot (incl. the method-0x39 doorbell) are tainted. `0x40901c` is IRQDEST;
  writing 0x40 routed SWGEN0 to the host.
* `0x4098cc` / `0x40984c` are CTXSW_MAILBOX_SET(3) / _CLEAR(3) (nvgpu:
  set = 0x4098c0+4i, clear = 0x409840+4i) — write-only, always read 0. The
  mailbox itself is `0x40980c`. "All three breadcrumbs read 0" does not prove the
  switch handler never ran; only `0x409a20` and `0x40980c` are meaningful.
* The spin at imem 0x11b4-0x11d9 reads DMEM 0x86c/0x870, which are the head/tail
  of firmware **queue B** (enqueue routine at 0xbe): it is a generic
  wait-for-XD-DMA loop that services queue B meanwhile, not the switch handler.
* `0x800004` (PCCSR channel) per nvgpu hw_ccsr: bits 24:27 STATUS, bit 28 BUSY,
  bit 1 NEXT (not "ACTIVE"). `0x15000003` = STATUS 5 ON_PBDMA + busy (PBDMA
  holds the channel, engine never took it); `0x11000001` = STATUS 1 PENDING;
  `0x00000003` = IDLE (+NEXT, enabled).
* FECS firmware interrupt path (iv0 = 0x33a -> 0x2bb): IRQSTAT bit 2 (method) ->
  dispatcher 0x1c8; ext bit 8 (CHSW, routed by init `I[0x10100] = 0x2003`) ->
  enqueue 0x10000 into **queue A** (entries DMEM 0x7e4, ptrs 0x824/0x828); ext
  bits 9/15 -> queue B; ext bit 10 -> queue C (0x874, ptrs 0x8b4/0x8b8). The main
  loop drains them; DMEM flags 0x750 / 0x758 gate parts of it (set by FECS
  method handlers at 0x4d6c / 0x4df7) — candidates for a missing init method.

Next rung-4 step: one clean reproduction with no register pokes, then a read-only
snapshot at the stall: FECS IRQSTAT/IRQMASK, PC, DMEM queue A/B/C pointers and
flags 0x750/0x758 (if host DMEM reads work in LS mode), PFIFO engine status
`0x2640`, `0x409b00/0x409b04`. That separates "IRQ never arrives" / "arrives but
firmware gated" / "processed but DMA fails".

## Session 6: clean rung-4 reproduction (no probe flags; log `s6-grchan-clean.log`)

Sequence: `reset_engines.py` -> `gr.py` (PASS, FECS 0x146500) -> `grchan.py`
(stored golden image, ENG_FAULT clear, no --gen / no pokes), with the read-only
`pascal/fecs_snap.py` snapshot before the channel, after the runlist commit and
after the kick.

* **FECS is never asked to switch.** IRQSTAT = 0, NEW_CTX `0x409b04` = 0,
  CURRENT_CTX = 0, and the trace-PC buffer (`0x148`/`0x14c`) is identical before
  and after (idle at 0x518d). Session 5's `0x409b04 = 0x8001160c` was left over
  from bind_pointer/--gen, not a request from the scheduler.
* **PFIFO never started an engine-0 context load**: `0x2640` (engine 0 status)
  = `0x00000080` (ctx_status INVALID, not busy) throughout; CE engines show their
  TSGs (e5 `0x10011001`, e6 `0x10021002`).
* **PBDMA0 has the channel as a valid context** (`0x3080 = 0x10013001`,
  `0x40120 = 0x2001`, PCCSR status 5 ON_PBDMA + busy) and raises INTR_0 bit 24 at
  load time, before any method (`0x400c0/0x400c4 = 0`, ib_get = pb_pos = 0).
* Clearing it the way nouveau's `gf100_runq_intr` does (`0x400c0 = 0x80600008`,
  `0x40108 = 0x01000000`, then `0x2100 = 0x20000000`) does **not** clear bit 24;
  `0x40100` moves `0x80000002 -> 0x80008008`, nothing advances. Bit 24 is outside
  nouveau's fatal mask `0xc67fe000`.
* Host DMEM (`0x1c0/0x1c4` -> `0xbadf548a`) and ICD (`0xbadfbadf`) are locked
  while FECS is in LS mode; the trace-PC buffer is the only FECS-internal view.

So the rung-4 blocker is between PBDMA0 and PFIFO's engine-0 scheduling, not FECS.

Card left: GR channel chid 1 stalled on runlist 0 (state above). Recover with
`reset_engines.py` then `gr.py` (and `fifo.py` to re-verify RT-3) before the next run.

## Session 6: RT-4 GR CHANNEL PASSES — the blocker was the GR copy engines held in reset

**Root cause.** Runlist 0 carries three engines on GP102: GR (engine 0), CE
(engine 7, PMC reset bit 6) and CE1 (engine 8, bit 7). Our bring-up only ever
enabled the engines it used, so `PMC_ENABLE = 0x4060f121` had bits 6/7 clear —
both GR copy engines in reset (their PFIFO status words `0x2640+7*8 = 06da0202`,
`+8*8 = 00100001` were never-initialised latches). When the scheduler loads a TSG
on runlist 0, PBDMA0 raises INTR_0 **bit 24 = ENG_RESET** (nvgpu name, "cleared by
the GPU" — which is why W1C never cleared it) and the engine-0 context switch is
never requested. nvgpu enables every CE at power-on (`gk20a_init_ce_support`).

**Fix:** set the PMC reset bit of every engine on GR's runlist before binding
(`grchan.py` now does it from the device table). Second bug found the moment
PGRAPH started receiving methods: compute-class SET_REPORT_SEMAPHORE_D used
OPERATION = 1, which is not an enum value on 0xc1c0 (RELEASE = 0) -> PGRAPH
DATA_ERROR INVALID_ENUM (0x400100 bit 20, 0x400110 = 5). Now `1 << 28`.

**Result (`s6-grchan-pass.log`), fresh reset -> gr.py -> grchan.py:**

| check | value |
|---|---|
| bind / commit | PCCSR `0x00000003` (IDLE), no PBDMA interrupt, FIFO_INTR 0 |
| host semaphore (PBDMA) | `51e4a001` OK, GP_GET 0->1, all 12 words consumed |
| **FECS context switch** | CURRENT_CTX = NEW_CTX = `0x8001160c`; trace PCs 0x4e66 / 0x47d3 / 0x392d / 0x38fa (switch code); engine 0 status `0x10013001` (TSG 1 VALID) |
| **PGRAPH class method** | report semaphore `c0de0007` written by class 0xc1c0, 1 ms after kick |
| negative control | same method at unmapped VA 0x8000000: target untouched, MMU fault `inst=0x1160c000 client=FE reason=PDE` |

**RT-4 (GR CHANNEL): PASS.** Next: compute dispatch — QMD v02_01 + compute
state + an sm_61 kernel writing a vector-add result to sysmem.

Note: the negative control leaves an MMU fault and FIFO_INTR 0x10000000 latched;
run `reset_engines.py` + `gr.py` before the next GR run.

## Session 6: **RT-4 COMPUTE DISPATCH PASSES** — an sm_61 kernel runs on the 1080 Ti (`pascal/compute.py`, `s6-compute.log`)

Kernel: `kernels/vadd.ptx` -> `ptxas -arch=sm_61` (CUDA 12.4, container `cuda12`,
pip `nvidia-cuda-nvcc-cu12==12.4.131` on OrbStack arm64) -> `kernels/vadd.sm_61.cubin`
(256 B SASS, 8 regs, params at c[0][0x140]). `pascal/cubin.py` reads it; SASS
verified with `envydis -m gm107`.

Launch path (all on the RT-4 GR channel):
* compute state = Mesa `nve4_screen_compute_setup` (< GV100): SET_SHADER_LOCAL_MEMORY_A/B
  (TLS 32 x 0x20000), NON_THROTTLED/THROTTLED A/B/C, local window 0xff<<24, shared
  window 0xfe<<24, SET_PROGRAM_REGION_A/B, SET_SPA_VERSION 0x400;
* QMD v02_01 built from `clc1c0qmd.h` (values as Mesa `gp100_compute_setup_launch_desc`,
  plus the five INVALIDATE_* bits; QMD_VERSION fields left 0 as Mesa does);
* cbuf0 per the sm_6x ABI read off the SASS: c[0][0x8..] blockDim, c[0][0x14..]
  gridDim, c[0][0x20] stack, params at 0x140;
* SEND_PCAS_A = qmd >> 8, SEND_SIGNALING_PCAS_B = 3, WAIT_FOR_IDLE, report semaphore.

| check | result |
|---|---|
| vadd, 32 x 128 threads, n = 4059, a/b in VRAM, **c in host memory** | **4059 / 4059 correct** |
| negative control: c prefilled 0xdeadbeef, tail [n, 4096) | untouched |
| completion semaphore after WAIT_FOR_IDLE | `c0de0008`, 1 ms after kick |
| PGRAPH / SM / MMU errors | none (0x400100 = 0, SM ESR 0, no fault) |
| repeat (re-bind over the previous channel, no reset) | 3 / 3 PASS |

**Rung 4 (PLAN.md gate: "vector-add kernel writes a correct result to system
memory") is met.**  Run order on a warm card: `reset_engines.py` -> `gr.py` -> `compute.py`.

## Session 6: runtime + measured compute ceiling (`pascal/nvpascal.py`, `pascal/bench.py`, `s6-bench.log`)

`nvpascal.Device`: one GR channel with persistent compute state; `alloc` (VRAM, from
1 GiB up, clear of WPR2), `alloc_host` (IOMMU sys_coh), `load` (all kernels of an sm_61
cubin into a 1 MiB code region), `launch` (QMD/cbuf ring of 64, four-word report-semaphore
timestamps around PCAS -> GPU ns). `mmu.VMM.map` now writes one page table (<= 512 PTEs)
per transfer instead of one PTE (selftest PASS; identical PTE values).

| measurement | result |
|---|---|
| vadd 65536 x u32 via runtime, out in host memory | 0 wrong |
| **SM clock** (`%clock64` vs `%globaltimer`, 10M and 200M cycles) | **607 MHz** (spec 1480 base / 1582 boost) |
| **SM-side DRAM bandwidth** (16-B ld/st grid-stride copy, 64/256 MiB, 224 or 896 CTAs) | **30.1 GB/s** traffic, flat; spot checks byte-exact |
| copy engines (earlier) | 24.5 (one) / 29.2 (two) GB/s |

So at boot clocks: memory 405 MHz -> ~30 GB/s achievable (DRAM-bound, both SM and CE);
cores 607 MHz -> 3584 x 2 x 0.607 = ~4.35 TFLOPS FP32 peak. Versus the M1 Air's own GPU
(~2.6 TFLOPS, 68 GB/s): the eGPU is ahead on compute-bound work (prefill) and behind on
bandwidth-bound work (decode) until memory is reclocked.
