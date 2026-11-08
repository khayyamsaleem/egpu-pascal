HANDOFF (session 5, 2026-11-08) — supersedes HANDOFF-SESSION4.md's "NEXT, IN ORDER"
=================================================================================

Read: this file, then the session 5 section at the end of MMU-AND-ACR.md, then the code.
HANDOFF-TO-OPUS.md is still right about environment/tooling/gotchas (its 1, 3, 6, 8).

STATE OF THE CARD RIGHT NOW (left warm, verified in this order)
-------------------------------------------------------------
  reset PMC_ENABLE bits 8 (PFIFO) + 12 (GR)
  ~/.venvs/tinygpu/bin/python pascal/gr.py     -> GR BRING-UP PASS (FECS 0x146500/0xf0c00/0x4c00)
  ~/.venvs/tinygpu/bin/python pascal/fifo.py   -> RT-3 + RT-3 SYSMEM PASS (CE2, runlist 5)
  GR channel (pascal/grchan.py) NOT currently bound; the FIFO is clean.

RUN ORDER on a warm card (no power cycle needed any more)
---------------------------------------------------------
  ~/.venvs/tinygpu/bin/python pascal/gr.py         # ACR unload/load, SEC2 RTOS, GR init, ctxctl
  ~/.venvs/tinygpu/bin/python pascal/fifo.py       # RT-3 on CE2
  ~/.venvs/tinygpu/bin/python pascal/grchan.py     # RT-4: GR channel on runlist 0
After a *cold* card (enclosure power cycle): devinit.py firmware/gp102_vbios.rom, fb.py, then the above.

WHAT SESSION 5 ADDED
--------------------
  pascal/grchan.py     GR = PASCAL_COMPUTE_B (0xc1c0) channel on runlist 0:
                       golden image + gf100_gr_chan_bind header patches, the mmio
                       list, priv mappings of pagepool/bundle/attrib, inst+0x210 =
                       ctx VA | 4.  Host-semaphore method executes (PBDMA verified);
                       the class method never reaches PGRAPH.
  fifo.py hooks        Channel.pre_bind(), Channel.alloc_map(..., priv=)
  fit recovery         PMC bit 8 reset + Fifo.init() un-wedges a stuck FIFO/PBDMA
                       (PMC bit 12 + gr.py for a stuck FECS) -- no power cycle
  grctx.py             FECS ctx-size assert relaxed to < 0x800000

THE BLOCKER (updated after the second round on this rung)
---------------------------------------------------------
Corrected facts:
* `0x409120` is falcon `XFER_STATUS` (bit1 PENDING, 16:18 STORES_PENDING) — FECS
  waits on its own FBMEM DMA, not an engine ack. A PDE fault while it does names
  the instance whose page directory it used; only `bind_pointer` (FECS method 3,
  inside gf100_grctx_generate's prologue) rebinds that. `ctx_chan` (method 1)
  times out on GP102 and rebinds nothing.
* Channel state `0x800004`: bit0 ENABLED, bit10 ENABLE_TRIGGER, 16:19 ENGINE.
  `0x00000003` = PBDMA loaded the channel (host methods execute);
  `0x11000001` (bits 24,28) = PBDMA refuses to load, GP_GET never moves, and no
  interrupt is raised. Bit 1 sets only when FECS was already bound to that same
  instance before the channel is enabled.
* Pascal compute is subchannel 1 (`SUBC_NVC1C0`); 4 is the copy engine. grchan.py
  now uses 1.
* FECS method 0x10 is a per-boot watermark (0x146500 -> 0x275600 -> 0x433000 ->
  0x95b000); generate at most once per boot, pass `grctx.generate(size=...)`.

NEXT, IN ORDER
--------------
1. Bounded engine-ack search after a failed GR commit: try, one at a time,
   `0x002630` ENG_FREEZE, `0x00262c` ENG_FAULT, `0x400500`, `0x409808` bit 19,
   `0x409800` high bits — watching `0x40108` bit 24, `0x40100` and PBDMA0
   `ib_get` for the write that lets the PBDMA resume (`0x40100` is the PBDMA
   status word the switch clears; rnndb calls it `UNK100`).
2. If no ack is found: switch to the software context-switch path
   (`0x409840` / `0x409500 | inst>>12` / `0x40904`) so the FIFO does not have to
   finish an engine-0 switch at all.
3. Side-quest only if 1–2 stall: host→PGRAPH method injection over
   `0x404488/0x40448c` (encoding now correct, `0x404204 = 0xc1c0` proves the
   class bind lands).  Useful to validate QMD/method encodings, useless as a
   delivery mechanism — it cannot carry an inference workload.
4. Then rung 4 proper: QMD v02_01 + compute setup (Mesa nve4_compute.c) + a
   `ptxas sm_61` vector-add kernel that writes its result to sysmem.

Standing corrections (this session)
-----------------------------------
* `ENG_FAULT` (`0x00262c`, W1C, bit0 = PGRAPH) must be clear or no GR channel
  will ever load: it binds as `0x800004 = 0x11000001` and the PBDMA never starts
  it.  Clearing it is necessary and proven.
* Trace (`PASCAL_TRACE=1`) shows every step up to the runlist commit is clean;
  event 24 is raised *by the load itself* on engine 0 only.  CE on runlist 5
  takes the identical path and lands `0x00000003` with no event — so the
  remaining blocker is the engine-0 context switch, not the channel.
* Host→PGRAPH method encoding was wrong (`(addr & 0xffff0000) >> 14` then
  `<< 14`); it is `ctrl | (addr << 14)` with the full 16-bit method address.
  Fixed in `gr.py:methods()` / `grctx.mthd()`.
* The commit retry loop was removed from `fifo.Channel.setup()`: re-clearing and
  re-committing re-raises event 24 deterministically and wedges the runlist
  counter (`0x2284` bit 20).
* Not a git repository — no version control on this tree.

