"""RT-3: a GPFIFO channel the GPU fetches and retires -- port of nouveau gk104/gp100 FIFO.

Sources (tools/nvkm/engine/fifo): gp100.c (runlist entry, fifo func), gk104.c
(chan bind/start, RAMFC, runlist commit, fifo init, PBDMA init), gk110.c (TSG
entry), gk208.c / gf100.c (PBDMA init), gm107.c (chan func: bind = bind_inst).
Class formats from NVIDIA's published headers (tools/nvclass): clc06f.h
(PASCAL_CHANNEL_GPFIFO_A: method headers, GP entries, USERD control block),
clc1b5.h (PASCAL_DMA_COPY_B).

USERD: pre-Volta, the host updates GP_PUT by writing USERD *through BAR1*;
PFIFO is told where USERD lives in BAR1 via 0x2254.  BAR1 is unbound on this
card (0x1704 = 0), i.e. physical: BAR1 offset == VRAM address (checked by a
PRAMIN-write / BAR1-read round trip), so USERD sits in low VRAM.

Round trip (CE channel; Pascal copy engines need no firmware):
  host semaphore release  (PBDMA executes methods)   -> sem0 == 0x5e4a0001
  CE copy src -> dst, then CE semaphore release      -> dst == src, sem1 == 0xc0de0002
  negative control: dst prefilled with a pattern != src; both semaphores 0 before.
"""
import os, struct, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from card import Card
import mmu as nvmmu

PASCAL_DMA_COPY_B = 0xc1b5
USERD_BASE = 0x03000000          # VRAM == BAR1 offset (BAR1 physical); 0x200 per channel
USERD_SIZE = 0x200
U_PUT, U_GET, U_GPGET, U_GPPUT = 0x40, 0x44, 0x88, 0x8c


def mthd(subc, addr, *data):
    """NVC06F_DMA_SEC_OP_INC_METHOD header + data."""
    return [0x20000000 | (len(data) << 16) | (subc << 13) | (addr >> 2)] + list(data)


class Fifo:
    def __init__(self, c):
        self.c = c
        self.bar1 = c.dev.map_bar(1, fmt="I")

    def init(self, verbose=True):
        c = self.c
        c.mask(0x002200, 0x00000001, 0x00000001)                   # gf100_fifo_init: PFIFO CTRL.ENABLE
        c.wr(0x002628, 0x00000001)                                 # gf100_fifo_init: ENGINE_INTR_EN
        c.wr(0x002254, 0x10000000 | (USERD_BASE >> 12))           # gk104_fifo_init
        nr = c.rd(0x002004) & 0xff                                 # gm200_fifo_runq_nr
        for i in range(nr):                                        # gk208_runq_init
            b = i * 0x2000
            c.mask(0x04013c + b, 0x10000100, 0)
            c.wr(0x040108 + b, 0xffffffff); c.wr(0x04010c + b, 0xfffffeff)
            c.wr(0x040148 + b, 0xffffffff); c.wr(0x04014c + b, 0xffffffff)
            c.wr(0x04012c + b, 0x000f4240)
        c.wr(0x000204, (1 << nr) - 1)                              # gk104_fifo_init_pbdmas
        c.mask(0x002a04, 0xbfffffff, 0xbfffffff)
        c.wr(0x002100, 0xffffffff)
        if verbose:
            print("FIFO init: %d PBDMAs, 0x2200=%08x 0x2254=%08x, 0x204=%08x"
                  % (nr, c.rd(0x2200), c.rd(0x2254), c.rd(0x204)))

    def userd_w32(self, chid, off, v): self.c.guard.check(); self.bar1[(USERD_BASE + chid * USERD_SIZE + off) // 4] = v
    def userd_r32(self, chid, off): self.c.guard.check(); return self.bar1[(USERD_BASE + chid * USERD_SIZE + off) // 4]


class Channel:
    def __init__(self, fifo, chid, runl, heap, runq=0, gp_entries=512, pb_size=0x100000):
        self.f, self.c, self.chid, self.runl, self.runq = fifo, fifo.c, chid, runl, runq
        c = self.c
        self.heap = heap
        self.vmm = nvmmu.VMM(c, heap)
        self.inst = nvmmu.inst_alloc(c, heap)
        self.gp_entries = gp_entries
        self.gp_va, self.pb_va = 0x00100000, 0x00200000
        self.gp_pa = self.alloc_map(self.gp_va, gp_entries * 8)
        self.pb_size = pb_size
        self.pb_pa = self.alloc_map(self.pb_va, pb_size)
        self.gp_put = 0
        self.pb_off = 0

    def alloc_map(self, va, size, fill=0, priv=False):
        size = nvmmu.align(size, 0x1000)
        pa = self.heap.alloc(size, 0x1000)
        self.c.vram_write(pa, bytes([fill]) * size if fill else bytes(size))
        self.vmm.map(va, pa, size, priv=priv, flush=False)
        return pa

    def pre_bind(self):
        """Hook for engine-specific channel state (see grchan.GRChannel)."""

    def teardown(self, verbose=True):
        """gk104_chan_stop + gk110_cgrp_preempt + empty runlist + gk104_chan_unbind.

        Needed before re-binding a chid that a previous run left bound: binding a
        bound channel raises FIFO BIND_ERROR 0x01 (BIND_NOT_UNBOUND) and the
        runlist update never completes."""
        c, chid = self.c, self.chid
        was = c.rd(0x800000 + chid * 8), c.rd(0x800004 + chid * 8)
        c.mask(0x800004 + chid * 8, 0x00000800, 0x00000800)          # ENABLE_CLR
        c.wr(0x002634, 0x01000000 | chid)                            # preempt TSG (tsgid == chid here)
        pre = c.wait(lambda: not (c.rd(0x002634) & 0x00100000), 1.0)
        empty = self.heap.alloc(0x1000, 0x1000)
        c.wr(0x002270, empty >> 12); c.wr(0x002274, (self.runl << 20) | 0)
        rl = c.wait(lambda: not (c.rd(0x002284 + self.runl * 8) & 0x00100000), 1.0)
        c.wr(0x800000 + chid * 8, 0)
        if c.rd(0x2100) & 1:
            c.wr(0x2100, 1)
        if verbose:
            print("teardown chid %d: was %08x/%08x  preempt %s  empty runlist %s  now %08x/%08x  FIFO_INTR=%08x"
                  % (chid, was[0], was[1], "ok" if pre else "TIMEOUT", "ok" if rl else "PENDING",
                     c.rd(0x800000 + chid * 8), c.rd(0x800004 + chid * 8), c.rd(0x2100)))
        return pre and rl

    def _t(self, tag):
        """Per-step PBDMA/PFIFO snapshot, enabled with PASCAL_TRACE=1."""
        if not os.environ.get("PASCAL_TRACE"):
            return
        c, chid = self.c, self.chid
        print("   [%-20s] chan=%08x fifo_intr=%08x pbdma0 intr=%08x intr1=%08x 0x40100=%08x"
              " ib_get=%08x pb_pos=%08x eng_fault=%08x"
              % (tag, c.rd(0x800004 + chid * 8), c.rd(0x2100), c.rd(0x40108), c.rd(0x40148),
                 c.rd(0x40100), c.rd(0x40014), c.rd(0x40054), c.rd(0x262c)))

    def ramfc_dump(self):
        c, chid = self.c, self.chid
        offs = (0x08, 0x0c, 0x10, 0x30, 0x48, 0x4c, 0x84, 0x94, 0x9c, 0xac, 0xb8, 0xe4, 0xe8, 0xf8, 0xfc)
        print("  RAMFC ch%d: " % chid + " ".join("%02x=%08x" % (o, c.vram_r32(self.inst + o)) for o in offs))

    def setup(self, verbose=True):
        c, chid = self.c, self.chid
        self._t("entry")
        if c.rd(0x800000 + chid * 8) & 0x80000000:
            self.teardown(verbose)
            self._t("after teardown")
        # gk104_chan_stop + an empty runlist commit: the only GR run that ever
        # reached ACTIVE (0x800004 bit 1) had a teardown in front of it, so do
        # the scheduler-reset half unconditionally before inserting the entry.
        c.mask(0x800004 + chid * 8, 0x00000800, 0x00000800)
        self._t("after disable")
        c.wr(0x002634, 0x01000000 | chid)
        c.wait(lambda: not (c.rd(0x002634) & 0x00100000), 1.0)
        empty = self.heap.alloc(0x1000, 0x1000)
        c.wr(0x002270, empty >> 12); c.wr(0x002274, (self.runl << 20) | 0)
        c.wait(lambda: not (c.rd(0x002284 + self.runl * 8) & 0x00100000), 1.0)
        self._t("after tsg+empty commit")
        self.vmm.join(self.inst)
        self.vmm.flush()
        self._t("after vmm join")
        for off in (0x40, 0x44, 0x48, 0x4c, 0x50, 0x58, 0x5c, 0x60, 0x88, 0x8c):   # gf100_chan_userd_clear
            self.f.userd_w32(chid, off, 0)
        userd = USERD_BASE + chid * USERD_SIZE
        limit2 = (self.gp_entries).bit_length() - 1                               # ilog2(length / 8)
        ramfc = {0x08: userd & 0xffffffff, 0x0c: userd >> 32, 0x10: 0x0000face, 0x30: 0xfffff902,
                 0x48: self.gp_va & 0xffffffff, 0x4c: (self.gp_va >> 32) | (limit2 << 16),
                 0x84: 0x20400000, 0x94: 0x30000000 | 0xfff, 0x9c: 0x00000100, 0xac: 0x0000001f,
                 0xe4: 0x00000020, 0xe8: chid, 0xb8: 0xf8000000, 0xf8: 0x10003080, 0xfc: 0x10000010}
        for off, v in ramfc.items():
            c.vram_w32(self.inst + off, v)
        self._t("after ramfc+userd")
        # runlist: TSG header (gk110_runl_insert_cgrp) + channel (gp100_runl_insert_chan)
        self.runlist_pa = self.heap.alloc(0x1000, 0x1000)
        tsg = chid
        rl = struct.pack("<IIII", (1 << 26) | (128 << 18) | (3 << 14) | 0x00002000 | tsg, 0,
                         chid | (self.runq << 14), self.inst >> 12)
        c.vram_write(self.runlist_pa, rl)
        # gk104_chan_bind_inst + gk104_chan_start
        self.pre_bind()
        self._t("after pre_bind")
        c.wr(0x800000 + chid * 8, 0x80000000 | (self.inst >> 12))
        self._t("after bind")
        c.mask(0x800004 + chid * 8, 0x00000400, 0x00000400)
        self._t("after enable trigger")
        # gk104_runl_commit
        c.wr(0x002270, (0 << 28) | (self.runlist_pa >> 12))
        c.wr(0x002274, (self.runl << 20) | 2)
        done = c.wait(lambda: not (c.rd(0x002284 + self.runl * 8) & 0x00100000), 1.0)
        self._t("after runlist commit")
        if verbose:
            print("channel %d: inst=%#x pd3=%#x userd=%#x gpfifo va=%#x pa=%#x pb va=%#x pa=%#x runlist@%#x"
                  % (chid, self.inst, self.vmm.pd3, userd, self.gp_va, self.gp_pa, self.pb_va, self.pb_pa, self.runlist_pa))
            print("  runlist %d commit: %s  0x2284+%d*8=%08x  chan 0x800000=%08x 0x800004=%08x"
                  % (self.runl, "done" if done else "STILL PENDING", self.runl, c.rd(0x2284 + self.runl * 8),
                     c.rd(0x800000 + chid * 8), c.rd(0x800004 + chid * 8)))
        return done

    def push(self, words):
        """Append words to the pushbuffer and queue one GP entry for them."""
        data = struct.pack("<%dI" % len(words), *words)
        assert len(data) <= self.pb_size, "push of %d bytes exceeds the pushbuffer" % len(data)
        if self.pb_off + len(data) > self.pb_size:       # wrap: only once the host has drained it
            gp_get = lambda: self.f.userd_r32(self.chid, U_GPGET)
            if not self.c.wait(lambda: gp_get() == self.gp_put, 5.0):
                raise RuntimeError("pushbuffer full and channel not idle: " + self.status())
            self.pb_off = 0
        va = self.pb_va + self.pb_off
        self.c.vram_write(self.pb_pa + self.pb_off, data)
        self.pb_off += nvmmu.align(len(data), 0x100)
        self.gp_entry(va, len(data))

    gp_host = None          # memoryview of the GPFIFO when it lives in host memory (nvpascal)

    def gp_entry(self, va, nbytes):
        """Queue one GP entry for nbytes of pushbuffer at va (not kicked)."""
        e = struct.pack("<II", va & 0xfffffffc, ((va >> 32) & 0xff) | ((nbytes >> 2) << 10))
        if self.gp_host is not None:
            self.gp_host[self.gp_put * 8:self.gp_put * 8 + 8] = e
        else:
            self.c.vram_write(self.gp_pa + self.gp_put * 8, e)
        self.gp_put = (self.gp_put + 1) % self.gp_entries

    def kick(self):
        self.f.userd_w32(self.chid, U_GPPUT, self.gp_put)

    def status(self):
        c = self.c
        return "GP_GET=%d GP_PUT=%d chan=%08x FIFO_INTR=%08x" % (
            self.f.userd_r32(self.chid, U_GPGET), self.f.userd_r32(self.chid, U_GPPUT),
            c.rd(0x800004 + self.chid * 8), c.rd(0x2100))


def pbdma_dump(c, i):
    b = 0x40000 + i * 0x2000
    return " ".join("%03x=%08x" % (o, c.rd(b + o)) for o in (0x08, 0x0c, 0x14, 0x48, 0x4c, 0x54, 0x84, 0xe8, 0x108, 0x148))


if __name__ == "__main__":
    import fb, faults
    c = Card()
    devs = {d["name"]: d for d in fb.top_parse(c)}
    ce = devs["CE2"]
    runl = ce["runlist"]
    pbdma = next(i for i in range(4) if c.rd(0x2390 + 4 * i) & (1 << runl))
    print("CE2: engine %d runlist %d reset bit %d -> PBDMA%d" % (ce["engine"], runl, ce["reset"], pbdma))
    c.mask(0x200, 1 << ce["reset"], 1 << ce["reset"])
    print("PMC_ENABLE %08x" % c.rd(0x200))
    faults.dump(c, clear=True)

    f = Fifo(c); f.init()
    heap = nvmmu.VramHeap(base=0x10200000)
    ch = Channel(f, chid=1, runl=runl, heap=heap)
    N = 0x1000
    src_va, dst_va, sem_va = 0x00400000, 0x00500000, 0x00600000
    src_pa = ch.alloc_map(src_va, N)
    dst_pa = ch.alloc_map(dst_va, N)
    sem_pa = ch.alloc_map(sem_va, 0x1000)
    pattern = bytes((i * 37 + 11) & 0xff for i in range(N))
    c.vram_write(src_pa, pattern)
    c.vram_write(dst_pa, b"\xad\xde\xad\xde" * (N // 4))
    ch.vmm.flush()
    before = (c.vram_read(dst_pa, N) == pattern, c.vram_r32(sem_pa), c.vram_r32(sem_pa + 0x10))
    print("negative control before kick: dst==src %s, sem0=%08x, sem1=%08x" % before)
    if not ch.setup():
        faults.dump(c)

    S = 4   # subchannel
    pb = []
    pb += mthd(0, 0x0010, sem_va >> 32, sem_va & 0xffffffff, 0x5e4a0001, 0x2 | (1 << 24))   # host SEMAPHORE A-D, release 4 B
    pb += mthd(S, 0x0000, PASCAL_DMA_COPY_B)                                                # SET_OBJECT
    pb += mthd(S, 0x0400, src_va >> 32, src_va & 0xffffffff, dst_va >> 32, dst_va & 0xffffffff, N, N, N, 1)
    pb += mthd(S, 0x0240, (sem_va + 0x10) >> 32, (sem_va + 0x10) & 0xffffffff, 0xc0de0002)  # SET_SEMAPHORE_A/B/PAYLOAD
    pb += mthd(S, 0x0300, 0x2 | 0x4 | (1 << 3) | 0x80 | 0x100)                              # LAUNCH_DMA
    ch.push(pb)
    print("pushbuffer: %d words; before kick: %s | PBDMA%d %s" % (len(pb), ch.status(), pbdma, pbdma_dump(c, pbdma)))
    t0 = time.time()
    ch.kick()
    ok_sem0 = c.wait(lambda: c.vram_r32(sem_pa) == 0x5e4a0001, 2.0)
    ok_sem1 = c.wait(lambda: c.vram_r32(sem_pa + 0x10) == 0xc0de0002, 2.0)
    dt = (time.time() - t0) * 1000
    got = c.vram_read(dst_pa, N)
    print("after kick (%.0f ms): %s | PBDMA%d %s" % (dt, ch.status(), pbdma, pbdma_dump(c, pbdma)))
    print("  host semaphore: %08x (%s)" % (c.vram_r32(sem_pa), "RELEASED" if ok_sem0 else "not released"))
    print("  CE semaphore:   %08x (%s)" % (c.vram_r32(sem_pa + 0x10), "RELEASED" if ok_sem1 else "not released"))
    print("  dst == src: %s (first words %s)" % (got == pattern, got[:16].hex()))
    faults.dump(c)
    rt3 = ok_sem0 and ok_sem1 and got == pattern
    print("RT-3 RESULT:", "PASS" if rt3 else "FAIL")
    if not rt3:
        sys.exit(1)

    # ---- second submission on the same channel: host memory <-> VRAM through the IOMMU
    aper = "sys_ncoh" if "--ncoh" in sys.argv else "sys_coh"
    smem, iovas = c.dev.alloc_sysmem(0x2000)
    up_va, down_va = 0x00700000, 0x00701000
    ch.vmm.map(up_va, iovas[0], 0x1000, aper=aper, flush=False)
    ch.vmm.map(down_va, iovas[1], 0x1000, aper=aper, flush=False)
    ch.vmm.flush()
    host_pat = bytes((i * 101 + 7) & 0xff for i in range(0x1000))
    vram_pat = bytes((i * 13 + 200) & 0xff for i in range(0x1000))
    smem[0:0x1000] = host_pat
    smem[0x1000:0x2000] = b"\x5a" * 0x1000
    up_dst_pa = ch.alloc_map(0x00800000, 0x1000, fill=0xa5)          # VRAM, prefilled
    c.vram_write(src_pa, vram_pat)
    ch.vmm.flush()
    print("\nsysmem (%s) pages at IOVA %#x / %#x" % (aper, iovas[0], iovas[1]))
    print("negative control: VRAM dst==host pattern %s, host page1==VRAM pattern %s"
          % (c.vram_read(up_dst_pa, 0x1000) == host_pat, bytes(smem[0x1000:0x2000]) == vram_pat))
    pb = []
    pb += mthd(S, 0x0400, up_va >> 32, up_va & 0xffffffff, 0, 0x00800000, 0x1000, 0x1000, 0x1000, 1)
    pb += mthd(S, 0x0240, 0, sem_va + 0x20, 0xc0de0003)
    pb += mthd(S, 0x0300, 0x2 | 0x4 | (1 << 3) | 0x80 | 0x100)
    pb += mthd(S, 0x0400, src_va >> 32, src_va & 0xffffffff, down_va >> 32, down_va & 0xffffffff, 0x1000, 0x1000, 0x1000, 1)
    pb += mthd(S, 0x0240, 0, sem_va + 0x30, 0xc0de0004)
    pb += mthd(S, 0x0300, 0x2 | 0x4 | (1 << 3) | 0x80 | 0x100)
    ch.push(pb); t0 = time.time(); ch.kick()
    ok3 = c.wait(lambda: c.vram_r32(sem_pa + 0x20) == 0xc0de0003, 2.0)
    ok4 = c.wait(lambda: c.vram_r32(sem_pa + 0x30) == 0xc0de0004, 2.0)
    dt = (time.time() - t0) * 1000
    up_ok = c.vram_read(up_dst_pa, 0x1000) == host_pat
    down_ok = bytes(smem[0x1000:0x2000]) == vram_pat
    print("after kick 2 (%.0f ms): %s" % (dt, ch.status()))
    print("  host -> VRAM (CE reads sysmem):  sem %s, VRAM dst == host pattern: %s" % (ok3, up_ok))
    print("  VRAM -> host (CE writes sysmem): sem %s, host page == VRAM pattern: %s  (%s)"
          % (ok4, down_ok, bytes(smem[0x1000:0x1010]).hex()))
    faults.dump(c)
    print("RT-3 SYSMEM RESULT:", "PASS" if (ok3 and ok4 and up_ok and down_ok) else "FAIL")
