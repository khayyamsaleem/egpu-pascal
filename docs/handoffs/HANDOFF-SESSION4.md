HANDOFF (session 4, 2026-11-08) — supersedes HANDOFF-TO-OPUS.md sections 2, 4, 5, 7
==========================================================================================

Read: this file, then MMU-AND-ACR.md (session 4 sections at the end), then the code.
HANDOFF-TO-OPUS.md is still right about environment/tooling/gotchas (its sections 1, 3, 6, 8).

PRI-RING INCIDENT: RESOLVED by a power cycle; gr.py (fixed) now PASSES. History:
A GR-init bug (unmasked TPC count) flooded MMIO; see the INCIDENT section at the
end of MMU-AND-ACR.md. Fixed in card.py (BAR0 range assert) and gr.py (u8 mask +
topology asserts). After the power cycle: devinit.py ROM, fb.py, then gr.py.

WHERE IT STANDS (all on hardware, all repeatable)
-------------------------------------------------
  transport, POST (devinit.py), 11 GB VRAM (fb.py)            done earlier
  RT-0 / RT-1 / RT-2 (falcon PIO, our ucode, MMU v2)           done earlier
  ACR HS load                PASS   pascal/acr.py            (auto-unloads first)
  ACR HS unload (PMU)        PASS   pascal/acr_unload.py     mbox0 0x1d, claim 4 -> 0
  SEC2 RTOS + bootstrap      PASS   pascal/sec2_rtos.py      FECS/GPCCS -> LS mode
  FECS/GPCCS running         PASS   pascal/gr_ctxctl.py      FECS answers methods
  RT-3 GPFIFO + copy engine  PASS   pascal/fifo.py           VRAM<->VRAM, host<->VRAM

  GR init + FECS/GPCCS   PASS   pascal/gr.py             28 SMs, PGRAPH intr 0
  NOT done:, golden context, GR channel,
  PASCAL_COMPUTE_B, QMD, a kernel (RT-4), anything llama.cpp.

RUN ORDER on a warm, POSTed card (after a power cycle: devinit.py ROM, then fb.py)
--------------------------------------------------------------------------------
  ~/.venvs/tinygpu/bin/python pascal/sec2_rtos.py     # unload if needed, ACR+RTOS, bootstrap FECS/GPCCS
  ~/.venvs/tinygpu/bin/python pascal/gr_ctxctl.py     # start FECS/GPCCS, query ctx sizes
  ~/.venvs/tinygpu/bin/python pascal/fifo.py          # RT-3 on CE2 (independent of GR)

KEY FACTS LEARNED THIS SESSION (don't re-derive)
------------------------------------------------
* 0x10e5f8[23:20] = ACR claim field (HW mutex #2). Load requires 0 and writes the
  chip value (GP102 -> 4, from PMC_BOOT_42 0xa00 bits[28:20]) as its SECOND step,
  before the WPR checks; unload requires 4 and clears it. A failed load leaves 4 ->
  every later load fails 0x29. Fix = run the unload (PMU). Old "PMU firmware
  missing" hypothesis was wrong.
* PMU HS binding differs from SEC2: FBIF at 0xe00..0xe10 (4/0/4/5/6), 0x090 bit16,
  instance at 0x480; bind status via 0x200 = 0x30e then 0x20c[14:12].
* nouveau gp102_acr has no bootstrap_falcons: FECS/GPCCS must go through the SEC2
  RTOS (sec2/*-1.bin, BLD flcn_bl_dmem_desc_v2, ctx_dma 6, argv = EMEM 0x01000000).
  RTOS queues: cmdq 0xa00/0xa04, msgq 0xa30/0xa34 (stride 8), in EMEM
  (PIO 0xac0/0xac4); NV_SEC2_UNIT_ACR = 0x08; init msg 32 B.
* After the HS ACR the host cannot read the WPR (0xbad0acXX + REGION_VIOL fault
  PHYSICAL/HOST_CPU). acr.py clears that expected fault.
* FIFO: CE2 = runlist 5 / PBDMA2 / PMC bit 21. GR = runlist 0 (PBDMA0/1).
  BAR1 is physical (0x1704 = 0) -> USERD in low VRAM (0x03000000), 0x2254 points
  at it, GP_PUT written through BAR1. Rebinding a bound chid -> BIND_ERROR 0x01;
  fifo.Channel.teardown() handles it.
* Sysmem DMA works: tinygrad alloc_sysmem IOVAs mapped with PTE aperture sys_coh.
* Running the unload kills the VBIOS PRE_OS app on the PMU. Fan PWM stays 80%
  (0x10eb14 = 800/1000), temp ~37 C. Thermal sensor: 0x20460 (gp100_temp_get).

TOOLS ADDED
-----------
  tools/nvkm/fetched/   nvfw/sec2.h, falcon/qmgr.h, sec2/base.c, falcon/fw.c (upstream)
  tools/nvclass/        NVIDIA class headers: clc06f (GPFIFO), clc0b5/clc1b5 (CE),
                        clc0c0/clc1c0 (compute A/B). QMD header (clc0c0qmd.h) NOT
                        fetched yet: it is in NVIDIA/open-gpu-doc classes/compute/.

NEXT, IN ORDER (RT-4)
---------------------
1. pascal/gr.py: port gf100_gr_oneinit (gpc/tpc/ppc counts, gm200 sm_id + tiles)
   and gf100_gr_init with gp102's func table (engine/gr/gp102.c:99): init_gpc_mmu
   (gm200), sw_nonctx netlist (firmware/gp102/gr/sw_nonctx.bin, addr/value pairs),
   wait_idle, vsc_stream_master (gk104), zcull (gf117), num_active_ltcs (gm200),
   rop_active_fbps (gp100), swdx_pes_mask (gp102), fs, exceptions, ZBC.
   Then ctxctl: re-bootstrap FECS/GPCCS through the RTOS (RESET_YES) AFTER GR
   init, start them (gr_ctxctl.py) -- nouveau's order is GR init, then ctxctl.
2. golden context: ctxgf100.c gf100_grctx_generate (+ gm200/gp100/gp102
   overrides: pagepool, bundle CB, attrib CB), sw_ctx / sw_bundle_init /
   sw_method_init netlists, FECS bind_pointer + wfi_golden_save.
3. GR channel on runlist 0 (fifo.Channel works; add ctx pointer at inst+0x210 =
   ctx VA | 4, per gk104_ectx_bind), SET_OBJECT PASCAL_COMPUTE_B 0xc1c0.
4. compute launch: Mesa nve4_compute.c is the reference (QMD v02_01 for Pascal).
   Kernel: hand-assemble with envytools `envyas -m gm107` (Maxwell encoding ==
   sm_6x for the basic ops); a real compiler later = CUDA 12 ptxas sm_61 in a
   linux/amd64 container (OrbStack is installed but not running).
5. RT-4 = C = ReLU(A@B + bias) vs numpy (ROUNDTRIP-PLAN.md).

SESSION 4 END STATE (after the power cycle)
-------------------------------------------
  GR init + FECS/GPCCS    PASS  pascal/gr.py      (28 SMs, PGRAPH intr 0)
  golden GR context       PASS  pascal/grctx.py   image golden_ctx_gp102.bin: hdr[1]=0xa5e00 saved
                                                  length, hub + 6 x 0x9630 per-GPC segments
  Card was left warm: ACR/RTOS/FECS up, golden ctx generated in VMM heap 0x11000000.
  The FIFO channel from fifo.py was NOT re-run after the power cycle.

RT-4 NOTES (researched, not built)
----------------------------------
* Compute class PASCAL_COMPUTE_B methods (tools/nvclass/clc1c0.h, bare-hex format):
  SET_OBJECT 0x0, SET_SHADER_SHARED_MEMORY_WINDOW 0x214, INVALIDATE_SHADER_CACHES 0x21c,
  SEND_PCAS_A 0x2b4 (= QMD VA >> 8), SEND_SIGNALING_PCAS_B 0x2bc (= 3: invalidate|schedule),
  SET_SHADER_LOCAL_MEMORY_NON_THROTTLED_A/B/C 0x2e4.., THROTTLED 0x2f0.., SET_SPA_VERSION 0x310
  (Mesa writes 0x400 for NVF0+), SET_SHADER_LOCAL_MEMORY_WINDOW 0x77c (Mesa 0xff<<24),
  SET_SHADER_LOCAL_MEMORY_A/B 0x790, SET_PROGRAM_REGION_A/B 0x1608, INVALIDATE_SHADER_CACHES_NO_WFI
  0x1698, SET_REPORT_SEMAPHORE_A-D 0x1b00 (D: op 1:0 RELEASE=0, STRUCTURE_SIZE bit 28).
* Setup sequence = Mesa tools/mesa/nve4_compute.c nve4_screen_compute_setup (lines 43-178);
  launch = gp100_compute_setup_launch_desc (636) + nve4_launch_grid (785).
* QMD V02_01: tools/nvclass/clc0c0qmd.h (open-gpu-doc). Fields are "MW(hi:lo)", arrays
  "MW((640+(i)*1):...)", enums "_NAME value" -> write a small parser. Set what Mesa sets plus
  QMD_VERSION / QMD_MAJOR_VERSION (check Mesa nve4_compute_alloc_launch_desc for values),
  CONSTANT_BUFFER_ADDR_LOWER/UPPER/SIZE_SHIFTED4/VALID(0) for kernel params.
* GR channel still to build: fifo.Channel on runlist 0 (PBDMA0), GR ctx buffer = golden image +
  header patches from gf100_gr_chan_bind (0xf4=0,0xf8=0,0x10=mmio_nr/2,0x14/0x18=mmio list VA,
  0x1c=1,0x20=0,0x28=0,0x2c=0), inst+0x210 = ctx VA | 4 (gk104_ectx_bind), mmio list =
  grctx.GrCtx.mmio_list() written to a VRAM page mapped priv.
* SASS: envytools `envyas -m gm107` HAS Maxwell sched-control support (tabsched, schedpos 0x20,
  "sched (st ..)" syntax) -> hand-assemble vector add (S2R tid, IADD.CC/IADD.X address from c[0],
  LDG.E, FADD, STG.E, EXIT); round-trip through envydis before running.
