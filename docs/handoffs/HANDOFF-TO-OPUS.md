HANDOFF — GTX 1080 Ti (GP102) bring-up over TinyGPU on a MacBook Air M1
Rewritten 2026-11-08, end of session 5.  This file replaces the old session-3
handoff entirely (its ACR sections are long since resolved; nothing here is
stale-by-supersession).

Read first, in this order (do not re-derive any of it):
  ~/egpu-pascal/HANDOFF-SESSION5.md   current rung-4 state, the blocker, corrections
  ~/egpu-pascal/MMU-AND-ACR.md        the evidence trail: traces, disassembly, negatives
  ~/egpu-pascal/PLAN.md               the 8-phase plan; Phase 6 now targets MAX
  ~/egpu-pascal/FINDINGS.md           session 1/2/3 history (older, still valid)
Then the code: pascal/fifo.py, grchan.py, grctx.py, gr.py, mmu.py, card.py.

GOAL (user's, standing): run LLM inference on this 1080 Ti over the Thunderbolt /
USB4 enclosure from this Mac.  llama.cpp is NO LONGER required: Phase 6 was
decided (2026-11-08) to build a backend for Modular's MAX (Qualcomm), whose
graph compiler, runtime and hardware backends are public under the MAX License
and whose device-driver connector is designed to take a new accelerator.
The user wants persistence and "know exactly where we stopped" over optimism.

=====================================================================
1. ENVIRONMENT AND HOW TO RUN ANYTHING
=====================================================================
Host: MacBook Air M1, macOS 26.6.2.  Enclosure: Razer Core X V2 (Intel bridge) on
a TB3/USB4 port; link trains PCIe Gen3 x4.  Card: GTX 1080 Ti, GP102 10de:1b06.
Transport: tinygrad's TinyGPU DriverKit dext (installed, enabled) + its server
socket /tmp/tinygpu.sock.  python = ~/.venvs/tinygpu/bin/python (3.12.13).

  ~/.venvs/tinygpu/bin/python pascal/mmu.py --selftest   # page-table encodings, no GPU
  ~/.venvs/tinygpu/bin/python pascal/mmu_test.py         # MMU on-card selftest (expect 18/18)
  ~/.venvs/tinygpu/bin/python pascal/gr.py               # ACR + SEC2 RTOS + GR bring-up (expect PASS)
  ~/.venvs/tinygpu/bin/python pascal/fifo.py             # RT-3: CE channel + semaphores (expect PASS)
  ~/.venvs/tinygpu/bin/python pascal/grctx.py            # golden context generation (expect PASS)
  ~/.venvs/tinygpu/bin/python pascal/grchan.py [--gen]   # RT-4: GR compute channel (FAILS at rung 4)

  PASCAL_TRACE=1 adds per-step PBDMA/PFIFO snapshots inside fifo.Channel.setup().

Recovery (copied out of the 24h-pruned scratch dir in session 5 — keep them):
  pascal/reset_engines.py    PMC_ENABLE bits 8 (PFIFO) + 12 (GR) reset
  pascal/recover_fifo.py     PFIFO reset + Fifo.init(), clears a wedged scheduler

After any enclosure power cycle the card is cold:
  pascal/devinit.py firmware/gp102_vbios.rom   then   pascal/fb.py
before anything else (fb.py also runs the signed nvdec/scrubber.bin once per
power cycle to unlock the VPR).

Reference material already local (no network needed): tools/nvkm/ (nouveau, flat
names), tools/envytools/ with tools/envytools/build/envydis/envydis, firmware/
gp102/{acr,gr,sec2,nvdec}, firmware/gp102_vbios.rom.

=====================================================================
2. WHERE THE WORK STANDS
=====================================================================
Rung ladder (PLAN.md section 0): 0 transport+ID, 1 falcon runs our code,
2 ACR secure boot, 3 MMU + FIFO channel, 4 compute dispatch, 5 inference.

  Rung 0  DONE.  Transport, enumeration, BAR0/BAR1/PRAMIN, VBIOS POST from macOS
          (devinit.py), all 11 GB VRAM (the VPR hole fixed by fb.py).
  Rung 1  DONE.  Our own unsigned microcode runs on the falcon(s); DMEM PIO RT.
  Rung 2  DONE.  ACR secure boot: WPR2 established, NVIDIA signed blobs run,
          FECS and GPCCS bootstrapped.  (PLAN called this a coin flip.)
  Rung 3  DONE AND EXCEEDED.  MMU v2 + instance blocks (18/18); PFIFO/PBDMA
          channels; RAMFC/RAMUSERD; GPFIFO fetch/retire; semaphores; CE copies
          VRAM <-> sysmem both directions with negative controls (RT-3 + RT-3
          SYSMEM PASS, reproduced many times).
  Rung 4  IN PROGRESS — see section 3.  No PGRAPH class method has executed yet.
  Rung 5  NOT STARTED.  Phase 6 target decided: a MAX backend (PLAN.md).

=====================================================================
3. THE CURRENT BLOCKER (rung 4), EXACTLY
=====================================================================
Everything below is measured on this card, not inferred.

Symptom.  A PASCAL_COMPUTE_B (0xc1c0) channel binds on runlist 0, but as soon as
the runlist commit is processed the PBDMA raises event 24 (`0x40108` bit 24 =
0x01000000), FIFO_INTR (0x2100) latches 0x20000000, `0x800004` = 0x15000003
(ENABLED + ACTIVE + bits 24/26/28), ib_get/pb_pos stay 0 and a later kick is
ignored.  No HCE code (intr1 = 0), no PGRAPH exception, no MMU fault.

It is engine-0-specific.  The identical traced Channel.setup() on the copy engine
(chid 2, runlist 5) lands `0x800004 = 0x00000003` with no event at all, on a
freshly reset FIFO; every step before the commit is byte-identical between the
two (PASCAL_TRACE=1).  SUBFIFO_ENG_MASK (0x2390 series) ties engine 0 to PBDMA0/1
only, so there is no alternate PBDMA.

Necessary precondition, proven.  ENG_FAULT (`0x00262c`, write-1-to-clear, bit N =
engine N) must be clear.  With PGRAPH latched (bit 0 set) the channel binds as
`0x800004 = 0x11000001` and the PBDMA never loads it.  grchan.py clears it
automatically before binding; always read this register before blaming a channel.

Where the request dies.  FECS's switch handler (fecs_inst.bin, entry at imem
0x10e9) writes breadcrumbs to host registers:
    entry:  0x4098cc = 0x200, then an ack code 0x8000000N -> 0x409a20
    (then it pumps the context through the falcon's XD DMA — xdld/xdst with
     $xdbase — spinning on XFER_STATUS pending counts at imem 0x11b4-0x11d9)
    exit:   0x409a20 = 0, 0x200 -> 0x40984c
At our stall ALL THREE read 0: the handler is never entered.  Meanwhile
`0x409b04` bit 31 (request valid) is set while `0x409b00` bit 31 (switch in
progress) is clear — a request is queued that FECS never picks up.

Every host-writable lever has been tried and excluded (each followed by a kick,
watching 0x40108 / 0x40100 / ib_get / gp_get):
  - 13 candidate acks: ENG_FREEZE, ENG_FAULT W1C, 0x400500 |= 0x10000,
    0x409808 |= bit19, 0x409800 = 0/1, TSG preempt, runlist recommit, 0x800004
    trigger, 0x002300 = 0, 0x002628 |= 1, 0x400704 clear, 0x409c20 = 1.
  - FECS control method 0x39 ("start/resume context switching") fired at the live
    ACTIVE channel — accepted, handler still not entered.
  - irqmsk (0x409014) is not host-writable in this mode: 0xffffffff reads back 0
    (0x40901c does accept 0x40).  An ACR/LS-mode falcon owns its own delivery.
  - Host -> PGRAPH method injection (0x404488/0x40448c, class in the ctrl word)
    binds the class (`0x404204 = 0xc1c0`) but will not execute a stateful method
    (report semaphore), either after setup() or inside generation (--gen-inject):
    without the switch there is no current context to apply methods to.

=====================================================================
4. NEXT STEPS, RANKED
=====================================================================
1. Find who is supposed to DELIVER the switch request to FECS.  Candidates: the
   PFIFO engine/preempt path (does runlist 0 need the TSG entry flagged, or the
   channel inserted so the scheduler requests a switch rather than the request
   appearing only at bind?), or a PBDMA write into a FECS mailbox that our
   hand-built bind sequence never issues.  Read nvkm's engine/fifo/*.c around
   gk104_runl_commit and the engn paths with exactly this question in mind — the
   bind itself is now known-good.
2. Reproduce the one clean GR load (chan = 0x00000003 with subchannel-0 host
   methods actually executing).  It happened once; the difference is still
   unknown and is the strongest evidence rung 4 is reachable.
3. Only after 1–2: QMD v02_01 + compute setup (Mesa tools/mesa/nve4_compute.c) +
   a `ptxas sm_61` kernel (CUDA 12.x — CUDA 13 dropped Pascal) writing a
   vector-add result to sysmem.  That is the rung-4 gate.
4. Phase 6, MAX: start from the device-driver connector seam; Mojo kernels must
   lower to PTX assembled with ptxas sm_61.  Ceiling stays boot clocks (no PMU
   firmware -> no reclocking on any open path).

=====================================================================
5. STANDING CORRECTIONS AND GOTCHAS (do not re-learn these)
=====================================================================
* ENG_FAULT (0x262c bit 0) blocks all engine-0 channel loads — clear it first.
* Generate the golden context AT MOST ONCE per FECS boot.  FECS method 0x10 is an
  allocation watermark, not a size: 0x146500 fresh, then 0x275600 / 0x433000 /
  0x95b000 after successive generations.  Pass an explicit size to
  grctx.generate(size=...); the real image length is header word 1 (0xa5e00).
* An instance is not an address space until its PD is joined into the instance
  block (vmm.join(inst)) — grctx.generate does this before bind_pointer.
* FECS's MMU binding is per-instance and only bind_pointer (FECS method 3) moves
  it; ctx_chan (method 1) times out on GP102 and rebinds nothing.  A PDE fault
  raised while FECS works names the instance whose page directory it used.
* Host->PGRAPH method encoding is `ctrl | (addr << 14)` with the FULL 16-bit
  method address (field is bits 14..29).  It was wrong until session 5; fixed in
  gr.py:methods() and grctx.mthd().
* 0x404200+subc*4 class registers are set by golden generation (sw_method_init):
  sc1 = c1c0 there is NOT proof that your own SET_OBJECT landed.
* Compute class subchannel is 1 (SUBC_NVC1C0); 4 is the copy engine (NVC1B5).
* A failed GR run wedges the scheduler FIFO-wide: a following CE run fails too.
  Always reset PFIFO (pascal/reset_engines.py, pascal/recover_fifo.py) between
  engines.
* Do not re-add a commit retry loop: re-clearing event 24 and re-committing
  re-raises it deterministically and wedges the runlist counter (0x2284 bit 20).
* This tree is NOT a git repository.  Consider committing pascal/ + docs (tools/
  ignored or submoduled) before large edits.
* Ground rule from earlier sessions, still binding: every claim is a round trip
  with a negative control, and print the raw register value the judgement was
  made from.  Never trust a "poison vs decoded" read without a control register.

=====================================================================
6. FILE MAP
=====================================================================
  pascal/card.py          BAR0/PRAMIN/BAR1 access: rd/wr/vram_read/vram_write
  pascal/mmu.py           MMU v2: VMM, instance blocks, VramHeap, walker (~470 lines)
  pascal/mmu_test.py      on-card MMU selftest (18/18)
  pascal/acr.py           ACR secure boot (WPR2, signed blobs)
  pascal/gr.py            GR bring-up: FECS/GPCCS, netlists, methods()
  pascal/gr_ctxctl.py     FECS method plumbing (bind_pointer, wfi_golden_save, ...)
  pascal/grctx.py         golden context generation: generate(size=, inject=)
  pascal/fifo.py          Fifo, Channel (setup/teardown/push/kick), PBDMA init,
                          PASCAL_TRACE per-step trace, ramfc_dump()
  pascal/grchan.py        GRChannel + probes: --gen --gen-inject --ctx-chan
                          --ack-search --irq-unmask --ctxsw-start(-after) --runq1
  pascal/fb.py            VRAM/fb topology, VPR/scrubber unlock
  pascal/devinit.py       VBIOS POST
  pascal/faults.py        MMU fault decode/clear
  pascal/reset_engines.py, pascal/recover_fifo.py    recovery recipes

Evidence/notes: HANDOFF-SESSION5.md, MMU-AND-ACR.md (long evidence trail),
PLAN.md (phases; Phase 6 = MAX), FINDINGS.md, ROUNDTRIP-PLAN.md.

Card state as handed over: warm and idle — reset_engines + gr.py (GR BRING-UP
PASS) + fifo.py (RT-3 + RT-3 SYSMEM PASS) were the last operations; FIFO_INTR 0,
no MMU faults, ENG_FAULT clear, runlist idle.  Nothing running in the background.
