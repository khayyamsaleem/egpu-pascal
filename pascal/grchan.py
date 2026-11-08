"""RT-4 step 1: a GR channel on runlist 0, executing PASCAL_COMPUTE_B methods.

fifo.py proved a FIFO channel + a *firmware-free* engine (a copy engine).  GR is
different in one respect only: PGRAPH runs out of a per-channel *context*, so the
channel needs the pieces nouveau's gf100_gr_chan_new / gf100_gr_chan_bind build:

  * the golden context image (`golden_ctx_gp102.bin`, gr->data, gr->size bytes)
    copied into a per-channel buffer and patched (gf100_gr_chan_bind, firmware
    path): 0xf4=0, 0xf8=0, 0x10 = mmio_nr/2, 0x14/0x18 = mmio list VA,
    0x1c=1, 0x20=0, 0x28=0, 0x2c=0;
  * the "mmio list": 32-bit (addr, data) pairs the HUB falcon replays on the
    first load of this context, holding the per-channel addresses of the
    pagepool / bundle CB / attrib CB (grctx.chan_buffers with mmio=True);
  * those three global buffers mapped privileged into the channel's own VMM;
  * the context pointer, inst+0x210 = ctx VA | 4 (gk104_ectx_bind).

Then the channel is bound on runlist 0 (PBDMA0/1) exactly as for CE, and the
round trip is:

  1. a *host* semaphore release (subchannel 0)      -> PBDMA is fetching/decoding
  2. SET_OBJECT PASCAL_COMPUTE_B (subchannel 1)     -> the class is accepted
  3. SET_REPORT_SEMAPHORE A-D (0x1b00, one word)    -> **PGRAPH** writes VRAM
     through the channel's page tables.  This is the first instruction the GPU
     decoder hands to the graphics/compute engine, i.e. the first thing that
     proves the context above is real.

Negative control: the same GR method pointed at an unmapped VA must leave the
semaphore alone and raise an MMU fault naming this channel's instance block.
"""
import os, struct, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from card import Card
import mmu as nvmmu
import fifo as fifomod
import gr as grmod
import grctx as grctxmod
import fecs_snap

HERE = os.path.dirname(os.path.abspath(__file__))
GOLDEN = os.path.join(HERE, "..", "golden_ctx_gp102.bin")

PASCAL_COMPUTE_B = 0xc1c0
GR_RUNLIST = 0
SUBC_CLASS = 1                  # Mesa nv_push.h: SUBC_NVC1C0 (Pascal compute)
SUBC_HOST = 0                   # NV01_SUBCHAN: engine-independent host methods
GLOBALS_VA = 0x01000000         # == the VAs used to generate the golden image
CTX_VA = 0x02000000
MMIO_VA = 0x02100000


def ctx_image(mmio_nr, mmio_va=MMIO_VA, path=GOLDEN, img=None):
    """gf100_gr_chan_bind: golden image + the firmware-path header patches."""
    img = bytearray(open(path, "rb").read() if img is None else img)

    def w(off, v):
        struct.pack_into("<I", img, off, v & 0xffffffff)

    w(0xf4, 0)
    w(0xf8, 0)
    w(0x10, mmio_nr // 2)                    # number of (addr, data) pairs
    w(0x14, mmio_va & 0xffffffff)
    w(0x18, mmio_va >> 32)
    w(0x1c, 1)
    w(0x20, 0)
    w(0x28, 0)
    w(0x2c, 0)
    return bytes(img)


class GRChannel(fifomod.Channel):
    """fifo.Channel + the PGRAPH context (gf100_gr_chan_new + gk104_ectx_bind)."""

    def __init__(self, fifo, chid, gc, runl=GR_RUNLIST, runq=0):
        super().__init__(fifo, chid, runl, gc.heap, runq=runq)
        c = self.c
        self.gc = gc
        # gf100_gr_chan_new: map the global buffers priv into this channel's VMM
        gc.map_globals(self.vmm, GLOBALS_VA)
        # ...and build this channel's mmio list from their VAs
        self.mmio = gc.mmio_list()
        self.mmio_pa = self.alloc_map(MMIO_VA, 0x1000, priv=True)
        words = [x for pair in self.mmio for x in pair]
        c.vram_write(self.mmio_pa, struct.pack("<%dI" % len(words), *words))
        # the per-channel context image
        img = ctx_image(len(self.mmio))
        self.ctx_size = len(img)
        self.ctx_pa = self.alloc_map(CTX_VA, nvmmu.align(len(img), 0x1000), priv=True)
        self.ctx_alloc = nvmmu.align(len(img), 0x1000)
        c.vram_write(self.ctx_pa, img)
        print("GR ctx: image %#x bytes (%d mmio pairs), ctx pa=%#x va=%#x, mmio pa=%#x va=%#x"
              % (self.ctx_size, len(self.mmio), self.ctx_pa, CTX_VA, self.mmio_pa, MMIO_VA))
        print("  globals: %s" % {k: hex(v) for k, v in gc.va.items()})
        print("  ctx header: %s" % " ".join("%08x" % x for x in struct.unpack("<8I", img[:32])))
        for va, pa, size, aper in ((CTX_VA, self.ctx_pa, self.ctx_size, "vram"),):
            got = self.vmm.walk(va + 0x1000)
            print("  walk ctx va %#x -> %s" % (va + 0x1000, got))

    def set_image(self, img):
        """(Re)write this channel's context buffer from a golden image + patches.

        Used when the golden context is regenerated against *this* channel's
        instance block (nouveau generates it inside gf100_gr_chan_new, so FECS's
        "current instance" is a live channel rather than a throwaway one).
        """
        data = ctx_image(len(self.mmio), img=img)
        need = nvmmu.align(len(data), 0x1000)
        if need > self.ctx_alloc:                 # a regenerated image can be bigger
            self.ctx_pa = self.vmm.heap.alloc(need, 0x1000)
            self.vmm.map(CTX_VA, self.ctx_pa, need, priv=True, flush=False)
            self.ctx_alloc = need
        self.ctx_size = len(data)
        self.c.vram_write(self.ctx_pa, data)
        self.c.vram_write(self.ctx_pa + self.ctx_size, bytes(nvmmu.align(self.ctx_size, 0x1000) - self.ctx_size))
        return data

    def pre_bind(self):
        """gk104_ectx_bind (inst+0x210 = ctx VA | 4) + the runlist id in 0x800004."""
        c = self.c
        c.vram_w32(self.inst + 0x210, (CTX_VA & 0xffffffff) | 4)
        c.vram_w32(self.inst + 0x214, CTX_VA >> 32)
        c.mask(0x800004 + self.chid * 8, 0x000f0000, self.runl << 16)
        c.wr(0x002140, 0x7fffffff)                       # gk104_fifo_init


def sem_release(sem_va, payload):
    """PGRAPH report semaphore: A=addr hi, B=addr lo, C=payload, D=1 word."""
    # D: OPERATION_RELEASE = 0 (clc1c0.h; 1 is invalid -> PGRAPH DATA_ERROR
    # INVALID_ENUM), STRUCTURE_SIZE_ONE_WORD = bit 28
    return fifomod.mthd(SUBC_CLASS, 0x1b00, sem_va >> 32, sem_va & 0xffffffff,
                        payload, (1 << 28))


def fecs_dump(c, tag="FECS"):
    """FECS registers that matter for a PGRAPH channel switch (firmware path).

    0x409b00 = the instance FECS is switching to/from (gf100_gr_ctxsw_inst),
    0x409808 = its status word (bit 19 = ctxsw load, gf100_gr_chsw_load),
    0x409800/0x409804 = method mbox, 0x40981c = its trace/debug word.
    """
    d = dict(regs=[(a, c.rd(a)) for a in (0x409800, 0x409804, 0x409808, 0x409b00, 0x409b04, 0x40981c, 0x409c18,
                                          0x400100, 0x400110, 0x400704, 0x400708, 0x409604)])
    print("  %s: " % tag + " ".join("%06x=%08x" % (a, v) for a, v in d["regs"]))
    print("        0x409b00 ctxsw inst = %#x  0x409808 load(bit19)=%d"
          % ((c.rd(0x409b00) & 0x0fffffff) << 12, bool(c.rd(0x409808) & 0x00080000)))
    return d


def fecs_ctrl_ctxsw(c, mthd):
    """gf100_gr_fecs_ctrl_ctxsw: 0x39 = start/resume FECS context switching."""
    c.wr(0x409804, 0xffffffff); c.wr(0x409800, 0); c.wr(0x409500, 0xffffffff); c.wr(0x409504, mthd)
    ok = c.wait(lambda: c.rd(0x409804) in (1, 2), 2.0)
    st = c.rd(0x409804)
    return ("error" if st == 2 else "ok" if st == 1 else "timeout") if ok else "timeout"


def fecs_ctx_chan(c, inst):
    """Make a channel current on PGRAPH the way the *software* ctx path does
    (ctxgf100.c:1504-1516): 0x409840 = 0x80000000, then FECS FIFO data =
    0x80000000 | inst>>12, FIFO cmd = 1 (ctx_chan), wait 0x409800 bit 31.
    """
    c.wr(0x409840, 0x80000000)
    c.wr(0x409500, 0x80000000 | (inst >> 12))
    c.wr(0x409504, 0x00000001)
    c.wait(lambda: c.rd(0x409800) & 0x80000000, 2.0)
    return bool(c.rd(0x409800) & 0x80000000)


def gr_ready(c):
    """LS mode + running (grctx.fecs_running) and PGRAPH in fifo mode."""
    sctl = c.rd(0x409240)
    return bool(sctl & 1) and not (c.rd(0x409100) & 0x10) and \
        (c.rd(0x400500) & 0x00010001) == 0x00010001


def main():
    import faults
    c = Card()
    if not os.path.exists(GOLDEN):
        print("missing %s -- run pascal/grctx.py first" % GOLDEN); return 1

    cold = "--cold" in sys.argv
    if cold:
        import subprocess
        for s in ("devinit.py", "fb.py"):
            r = subprocess.run([sys.executable, os.path.join(HERE, s)], capture_output=True, text=True)
            print("%s: %s" % (s, (r.stdout or r.stderr).strip().splitlines()[-1] if (r.stdout or r.stderr).strip() else "?"))

    if not gr_ready(c):
        print("GR/FECS not up (sctl=%08x cpuctl=%08x 0x400500=%08x) -> full bring-up"
              % (c.rd(0x409240), c.rd(0x409100), c.rd(0x400500)))
        out = grmod.bring_up(c, verbose=True)
        if out is None or not out[4] or not all(out[4].values()):
            print("bring-up failed"); return 1
    else:
        print("GR up and warm: FECS sctl=%08x 0x400500=%08x 0x409800=%08x"
              % (c.rd(0x409240), c.rd(0x400500), c.rd(0x409800)))
    g = grmod.GR(c).oneinit(verbose=True)
    if "--reinit" in sys.argv:
        g.init()
    if "--ctxsw-start" in sys.argv:
        print("FECS ctrl_ctxsw(0x39, start/resume): %s" % fecs_ctrl_ctxsw(c, 0x39))
    fecs_dump(c, "FECS before channel")
    fecs_snap.snap(c, "before channel")

    heap = nvmmu.VramHeap(base=0x11000000)
    f = fifomod.Fifo(c); f.init()
    devs = {d["name"]: d for d in __import__("fb").top_parse(c)}
    grdev = devs["GR"]
    if grdev.get("reset") is not None:
        c.mask(0x200, 1 << grdev["reset"], 1 << grdev["reset"])
    # Every engine that shares GR's runlist must be out of reset, or PBDMA0 raises
    # INTR_0 bit 24 (ENG_RESET in nvgpu's naming) when it loads the TSG and the
    # engine-0 context switch is never requested (session 6).  On GP102 runlist 0
    # carries GR + the two GR copy engines (CE engine 7 / reset bit 6, CE1 engine 8 /
    # bit 7).  nvgpu enables all CEs at power-on (gk20a_init_ce_support).
    for d in devs.values():
        if d["runlist"] == grdev["runlist"] and d.get("reset") is not None:
            c.mask(0x200, 1 << d["reset"], 1 << d["reset"])
    print("GR top: engine %s runlist %s reset %s -> PMC_ENABLE %08x"
          % (grdev.get("engine"), grdev.get("runlist"), grdev.get("reset"), c.rd(0x200)))

    gc = grctxmod.GrCtx(c, g, heap)
    runq = 1 if "--runq1" in sys.argv else 0
    ch = GRChannel(f, chid=1, gc=gc, runq=runq)
    print("channel runq %d (PBDMA%d)" % (runq, runq))
    sem0_va, sem1_va, sem2_va = 0x00600000, 0x00601000, 0x00602000
    sem0_pa = ch.alloc_map(sem0_va, 0x1000)
    sem1_pa = ch.alloc_map(sem1_va, 0x1000)
    sem2_pa = ch.alloc_map(sem2_va, 0x1000)
    ch.vmm.flush()
    if "--gen" in sys.argv:
        # nouveau generates the golden context inside gf100_gr_chan_new, against
        # the *first channel's* instance, so FECS's "current instance" is a live
        # channel.  Ours came from a throwaway instance in a past run.
        # An instance is not a usable address space until the PD is joined into
        # its instance block (gc.generate() does this *before* bind_pointer).
        # 0x409120 bit 1 in the failure state is XFER_STATUS.PENDING, i.e. the
        # falcon is waiting on a DMA through the channel page tables.
        ch.vmm.join(ch.inst)
        t0 = time.time()
        def _inject():
            # Host->PGRAPH method injection while PGRAPH is in the same receptive
            # state generation itself uses.  Report semaphore on the compute
            # class: if VRAM gets 0xc0de0007, PGRAPH executes our class methods
            # and the QMD/launch work can ride the same seam.
            c.vram_w32(sem2_pa, 0)
            gc.mthd({PASCAL_COMPUTE_B: [(0x1b00, sem2_va >> 32), (0x1b04, sem2_va & 0xffffffff),
                                        (0x1b08, 0xc0de0007), (0x1b0c, 0x1 | (1 << 28))]})
            ok = c.wait(lambda: c.vram_r32(sem2_pa) == 0xc0de0007, 0.5)
            print("   INJECT during generation -> sem2=%08x %s  PGRAPH intr=%08x 0x400704=%08x subc1=%04x"
                  % (c.vram_r32(sem2_pa), "OK" if ok else "not written", c.rd(0x400100),
                     c.rd(0x400704), c.rd(0x404204) & 0xffff))
        img = gc.generate(ch.inst, ch.vmm, 0x03000000, verbose=True, size=0x200000,
                          inject=(_inject if "--gen-inject" in sys.argv else None))
        if img is None:
            print("golden regeneration FAILED"); return 1
        ch.set_image(img)
        print("golden ctx regenerated against this channel's inst %#x in %.1f s" % (ch.inst, time.time() - t0))
        fecs_dump(c, "FECS after regeneration")
    print("channel: inst=%#x pd3=%#x gp=%#x/%#x pb=%#x/%#x"
          % (ch.inst, ch.vmm.pd3, ch.gp_va, ch.gp_pa, ch.pb_va, ch.pb_pa))
    for name, va in (("ctx", CTX_VA), ("mmio", MMIO_VA), ("pagepool", gc.va["pagepool"]),
                     ("bundle", gc.va["bundle"]), ("attrib", gc.va["attrib"]),
                     ("pb", ch.pb_va), ("gp", ch.gp_va), ("sem", sem2_va)):
        got = ch.vmm.walk(va)
        print("  walk %-9s %#010x -> %s" % (name, va, "UNMAPPED" if got is None else "%s pa=%#x" % got))
    if "--irq-unmask" in sys.argv:
        # fecs_inst.bin's switch handler is interrupt-entered; at the stall the
        # handler's own scratch registers (0x409a20/0x4098cc/0x40984c) are all 0
        # and irqmsk (0x409014) is 0 -- i.e. FECS may never be told about the
        # request.  Try unmasking before the bind.
        print("FECS irqmsk before: %08x irqdest=%08x" % (c.rd(0x409014), c.rd(0x40901c)))
        c.wr(0x409014, 0xffffffff)
        c.wr(0x40901c, 0x00000040)
        print("FECS irqmsk after:  %08x irqdest=%08x" % (c.rd(0x409014), c.rd(0x40901c)))
    if c.rd(0x262c) & 1:
        # ENG_FAULT (0x262c, W1C, bit0 = PGRAPH): while PFIFO has PGRAPH latched
        # as faulted, no channel for engine 0 will ever be loaded -- it binds as
        # 0x800004 = 0x11000001 and the PBDMA never starts it.
        c.wr(0x262c, 0x00000001)
        print("ENG_FAULT had PGRAPH latched -> cleared (0x262c=%08x)" % c.rd(0x262c))
    if not ch.setup():
        faults.dump(c); return 1
    fecs_snap.snap(c, "after runlist commit", chid=ch.chid)

    def stall_dump(tag):
        print("-- %s" % tag)
        print("   PGRAPH subchannel classes 0x404200+i*4: " +
              " ".join("sc%d=%04x" % (i, c.rd(0x404200 + i * 4) & 0xffff) for i in range(8)))
        print("   PGRAPH intr=%08x exc=%08x 0x400704=%08x 0x400708=%08x 0x4041c4=%08x"
              % (c.rd(0x400100), c.rd(0x400110), c.rd(0x400704), c.rd(0x400708), c.rd(0x4041c4)))
        print("   PBDMA0 ib_get=%08x pb_pos=%08x 0x40100=%08x intr=%08x intr1=%08x ch=%08x"
              % (c.rd(0x40014), c.rd(0x40054), c.rd(0x40100), c.rd(0x40108), c.rd(0x40148), c.rd(0x40120)))
        print("   FECS 0x409808=%08x 0x409800=%08x 0x409b00=%08x 0x409b04=%08x xfer=%08x"
              % (c.rd(0x409808), c.rd(0x409800), c.rd(0x409b00), c.rd(0x409b04), c.rd(0x409120)))
        # switch-handler handshake (from fecs_inst.bin disassembly):
        #   entry: I[0x23300]=0x200 -> 0x4098cc ; ack code 0x8000000N -> I[0x28800]=0x409a20
        #   exit:  0x409a20 = 0 ; I[0x21300]=0x200 -> 0x40984c
        print("   FECS handshake 0x409a20=%08x 0x4098cc=%08x 0x40984c=%08x irqmsk(0x014)=%08x irqdest(0x01c)=%08x"
              % (c.rd(0x409a20), c.rd(0x4098cc), c.rd(0x40984c), c.rd(0x409014), c.rd(0x40901c)))
        print("   chan 0x800004=%08x gp_get=%08x eng_fault=%08x eng_state0=%08x"
              % (c.rd(0x800004 + ch.chid * 8), f.userd_r32(ch.chid, 0x88), c.rd(0x262c), c.rd(0x2640)))
    if "--ctxsw-start-after" in sys.argv:
        # Doorbell test: fire FECS control method 0x39 once the channel IS ACTIVE
        # with event 24 set (the state we only reach after clearing ENG_FAULT).
        # Success = 0x409a20 goes non-zero (handler entered) and the PBDMA resumes.
        print("FECS method 0x39 (start/resume ctxsw) fired at the live channel: %s"
              % fecs_ctrl_ctxsw(c, 0x39))
        time.sleep(0.05)
        print("   after doorbell: 0x409a20=%08x 0x4098cc=%08x 0x40984c=%08x 0x409b00=%08x 0x409b04=%08x"
              " pbdma0 intr=%08x ib_get=%08x chan=%08x"
              % (c.rd(0x409a20), c.rd(0x4098cc), c.rd(0x40984c), c.rd(0x409b00), c.rd(0x409b04),
                 c.rd(0x40108), c.rd(0x40014), c.rd(0x800004 + ch.chid * 8)))
    if "--ctx-chan" in sys.argv:
        okc = fecs_ctx_chan(c, ch.inst)
        print("FECS ctx_chan(inst %#x): %s  0x409800=%08x 0x409b00=%08x 0x409b04=%08x chan 0x800004=%08x"
              % (ch.inst, "ok" if okc else "TIMEOUT", c.rd(0x409800), c.rd(0x409b00), c.rd(0x409b04),
                 c.rd(0x800004 + ch.chid * 8)))

    if "--mmio-mthd" in sys.argv:
        # gf100_gr_mthd (host -> PGRAPH 0x404488/0x40448c), no PBDMA involved.
        # The class travels in the ctrl word, so this also stands in for
        # SET_OBJECT: if the report semaphore lands, PGRAPH is executing methods
        # for class 0xc1c0 and we have a control path that bypasses the FIFO.
        c.vram_w32(sem2_pa, 0)
        gc.mthd({PASCAL_COMPUTE_B: [(0x0000, PASCAL_COMPUTE_B),
                                    (0x1b00, sem2_va >> 32),
                                    (0x1b04, sem2_va & 0xffffffff),
                                    (0x1b08, 0xc0de0007),
                                    (0x1b0c, 0x1 | (1 << 28))]})
        okm = c.wait(lambda: c.vram_r32(sem2_pa) == 0xc0de0007, 1.0)
        print("MMIO method injection -> sem2=%08x (%s) PGRAPH intr=%08x 0x400704=%08x 0x404200+4=%08x"
              % (c.vram_r32(sem2_pa), "OK" if okm else "not written", c.rd(0x400100),
                 c.rd(0x400704), c.rd(0x404204)))

    # ---- the channel runs: host semaphore, SET_OBJECT, PGRAPH semaphore
    c.vram_w32(sem0_pa, 0); c.vram_w32(sem1_pa, 0); c.vram_w32(sem2_pa, 0)
    before = (c.vram_r32(sem0_pa), c.vram_r32(sem1_pa), c.vram_r32(sem2_pa))
    print("\nnegative control before kick: sem0=%08x sem1=%08x sem2=%08x  PGRAPH intr=%08x"
          % (before + (c.rd(0x400100),)))
    pb = []
    pb += fifomod.mthd(SUBC_HOST, 0x0010, sem0_va >> 32, sem0_va & 0xffffffff, 0x51e4a001, 0x2 | (1 << 24))
    pb += fifomod.mthd(SUBC_CLASS, 0x0000, PASCAL_COMPUTE_B)          # SET_OBJECT
    pb += sem_release(sem2_va, 0xc0de0007)                            # PGRAPH writes VRAM
    ch.push(pb)
    print("pushbuffer %d words: %s" % (len(pb), ch.status()))
    t0 = time.time()
    ch.kick()
    ok_host = c.wait(lambda: c.vram_r32(sem0_pa) == 0x51e4a001, 2.0)
    ok_gr = c.wait(lambda: c.vram_r32(sem2_pa) == 0xc0de0007, 2.0)
    dt = (time.time() - t0) * 1000
    print("after kick (%.0f ms): %s" % (dt, ch.status()))
    print("  host semaphore (PBDMA decoded subchannel 0): %08x %s" % (c.vram_r32(sem0_pa), "OK" if ok_host else "not released"))
    print("  PGRAPH semaphore (class %#x wrote VRAM):     %08x %s" % (PASCAL_COMPUTE_B, c.vram_r32(sem2_pa), "OK" if ok_gr else "not released"))
    print("  PBDMA0: %s" % fifomod.pbdma_dump(c, 0))
    print("  PGRAPH: intr 0x400100=%08x 0x400700=%08x 0x40060c=%08x"
          % (c.rd(0x400100), c.rd(0x400700), c.rd(0x40060c)))
    faults.dump(c)
    fecs_snap.snap(c, "after kick", chid=ch.chid)
    stall_dump("stall state")
    fecs_dump(c, "FECS after failure")
    rt4 = ok_host and ok_gr
    if not rt4 and "--ack-search" in sys.argv:
        # Bounded search for the write that lets PBDMA0 resume after the failed
        # engine-0 switch.  Success = GP_GET advances or the host semaphore lands.
        print("\nengine-ack search (baseline intr=%08x 0x40100=%08x ib_get=%08x chan=%08x)"
              % (c.rd(0x40108), c.rd(0x40100), c.rd(0x40014), c.rd(0x800004 + ch.chid * 8)))
        cands = [("ENG_FREEZE=0", 0x002630, 0),
                 ("ENG_FAULT W1C", 0x00262c, 1),
                 ("PGRAPH 0x400500 |= 0x10000", 0x400500, c.rd(0x400500) | 0x00010000),
                 ("0x409808 |= bit19", 0x409808, c.rd(0x409808) | (1 << 19)),
                 ("0x409800 = 0", 0x409800, 0),
                 ("0x409800 = 1", 0x409800, 1),
                 ("preempt TSG (kick chid)", 0x002634, 0x01000000 | ch.chid),
                 ("runlist recommit", None, None),
                 ("0x800004 trigger", 0x800004 + ch.chid * 8, 0x00000400),
                 ("0x002300 = 0", 0x002300, 0),
                 ("0x002628 |= 1", 0x002628, c.rd(0x002628) | 1),
                 ("0x400704 clear", 0x400704, 0),
                 ("0x409c20 = 1", 0x409c20, 1)]
        for name, a, v in cands:
            if a is None:
                c.wr(0x002270, (0 << 28) | (ch.runlist_pa >> 12)); c.wr(0x002274, (ch.runl << 20) | 2)
            else:
                c.wr(a, v)
            time.sleep(0.01)
            c.vram_w32(sem0_pa, 0)
            ch.kick()
            time.sleep(0.05)
            gp = f.userd_r32(ch.chid, 0x88)
            print("  %-26s intr=%08x 0x40100=%08x ib_get=%08x gp_get=%d sem0=%08x %s"
                  % (name, c.rd(0x40108), c.rd(0x40100), c.rd(0x40014), gp,
                     c.vram_r32(sem0_pa), "RESUMED" if (gp or c.vram_r32(sem0_pa)) else ""))
            if gp or c.vram_r32(sem0_pa) == 0x51e4a001:
                print("  >>> engine-ack search: %s woke the PBDMA" % name)
                break
    if not rt4:
        print("RT-4 (GR CHANNEL) RESULT: FAIL"); return 1

    # ---- negative control: same method, semaphore VA unmapped in this VMM
    NEG_VA = 0x08000000                       # not mapped in this channel's VMM
    c.vram_w32(sem2_pa, 0x11111111)
    pb = fifomod.mthd(SUBC_CLASS, 0x0000, PASCAL_COMPUTE_B) + sem_release(NEG_VA, 0xdeadbeef)
    ch.push(pb); ch.kick()
    time.sleep(0.3)
    neg_ok = c.vram_r32(sem2_pa) == 0x11111111
    print("\nnegative control (GR semaphore at unmapped VA %#x): sem2=%08x (%s), GP_GET=%d"
          % (NEG_VA, c.vram_r32(sem2_pa), "untouched" if neg_ok else "CHANGED", f.userd_r32(ch.chid, fifomod.U_GPGET)))
    faults.dump(c, clear=True)
    print("RT-4 (GR CHANNEL) RESULT:", "PASS" if (rt4 and neg_ok) else "PASS (gr channel), negative control inconclusive")
    return 0


if __name__ == "__main__":
    sys.exit(main())
