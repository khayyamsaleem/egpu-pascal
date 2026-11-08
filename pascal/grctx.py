"""Golden GR context for GP102 -- port of nouveau gf100_grctx_generate (gr->firmware path)
with gp102_grctx's overrides (engine/gr/ctxgp102.c) and gf100_gr_oneinit's global buffers.

During generation nouveau's chan->mmio is NULL, so every gf100_grctx_patch_wr32
is a direct register write; per-channel "mmio lists" (the same writes, applied by
FECS on first context load) are built later for each real channel (mmio_list()).

Negative control: the context buffer is prefilled with 0xdeadbeef; after
WFI_GOLDEN_SAVE, FECS must have overwritten it with a real image.
"""
import os, struct, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from card import Card
import gr as grmod
from gr import GPC_UNIT, TPC_UNIT, PPC_UNIT
import mmu as nvmmu

CB_RESERVED = 0x80000
BUNDLE_SIZE, BUNDLE_MIN_GPM_FIFO_DEPTH, BUNDLE_TOKEN_LIMIT = 0x3000, 0x180, 0x900
PAGEPOOL_SIZE = 0x20000
ATTRIB_NR, ALPHA_NR_MAX, ALPHA_NR, GFXP_NR = 0x320, 0xc00, 0x800, 0xba8
PPC_NR, FUNC_GPC_NR, FUNC_TPC_NR = 3, 6, 5


def popcount(x): return bin(x).count("1")


class Patch:
    """gf100_grctx_patch_wr32: direct write (mmio=None) or append to an mmio list."""
    def __init__(self, c, mmio=False):
        self.c, self.list = c, ([] if mmio else None)
    def wr(self, a, d):
        if self.list is None: self.c.wr(a, d)
        else: self.list.append((a, d & 0xffffffff))


class GrCtx:
    def __init__(self, c, g, heap):
        self.c, self.g, self.heap = c, g, heap
        g.ppc_tpc_nr = [[popcount(m) for m in row] for row in g.ppc_tpc_mask]
        nz = [n for row in g.ppc_tpc_nr for n in row if n]
        g.ppc_tpc_min, g.ppc_tpc_max = min(nz), max(nz)
        self.attrib_cb_size = self._attrib_cb_size()
        # gf100_gr_oneinit: global context buffers (VRAM)
        self.pagepool_pa = heap.alloc(PAGEPOOL_SIZE, 0x1000)
        self.bundle_pa = heap.alloc(BUNDLE_SIZE, 0x1000)
        self.attrib_pa = heap.alloc(nvmmu.align(self.attrib_cb_size, 0x1000), 0x1000)
        self.size = None

    def _attrib_cb_size(self):   # gp102_grctx_generate_attrib_cb_size
        g = self.g
        size = ALPHA_NR_MAX * g.tpc_total
        for _ in range(g.gpc_nr):
            size += GFXP_NR * PPC_NR * g.ppc_tpc_max
        return ((size * 0x20) + 127) & ~127

    # -------------------------------------------------- per-channel buffer setup (patch_wr32)
    def map_globals(self, vmm, va_base):
        """gf100_gr_chan_new: map pagepool / bundle CB / attrib CB (priv) into a channel VMM."""
        self.va = {}
        va = va_base
        for name, pa, size in (("pagepool", self.pagepool_pa, PAGEPOOL_SIZE), ("bundle", self.bundle_pa, BUNDLE_SIZE),
                               ("attrib", self.attrib_pa, nvmmu.align(self.attrib_cb_size, 0x1000))):
            vmm.map(va, pa, nvmmu.align(size, 0x1000), priv=True, flush=False)
            self.va[name] = va
            va = nvmmu.align(va + size, 0x10000)
        return va

    def chan_buffers(self, p):
        g, va = self.g, self.va
        # gp100_grctx_generate_pagepool
        p.wr(0x40800c, va["pagepool"] >> 8); p.wr(0x408010, 0x8007d800)
        p.wr(0x419004, va["pagepool"] >> 8); p.wr(0x419008, 0)
        # gm107_grctx_generate_bundle
        a, size = va["bundle"], BUNDLE_SIZE
        state_limit = min(BUNDLE_MIN_GPM_FIFO_DEPTH, size // 0x20)
        p.wr(0x408004, a >> 8); p.wr(0x408008, 0x80000000 | (size >> 8))
        p.wr(0x418e24, a >> 8); p.wr(0x418e28, 0x80000000 | (size >> 8))
        p.wr(0x4064c8, (state_limit << 16) | BUNDLE_TOKEN_LIMIT)
        # gp100_grctx_generate_attrib_cb (-> gm107 -> gf100)
        a, size = va["attrib"], self.attrib_cb_size
        p.wr(0x418810, 0x80000000 | a >> 12); p.wr(0x419848, 0x10000000 | a >> 12)
        p.wr(0x419c2c, 0x10000000 | a >> 12)
        p.wr(0x419b00, a >> 12); p.wr(0x419b04, 0x80000000 | size >> 7)
        # gp102_grctx_generate_attrib
        alpha, attrib, gfxp = ALPHA_NR, ATTRIB_NR, GFXP_NR
        ao, bo, n = 0, ALPHA_NR_MAX * g.tpc_total, 0
        p.wr(0x405830, attrib); p.wr(0x40585c, alpha); p.wr(0x4064c4, ((alpha // 4) << 16) | 0xffff)
        for gpc in range(g.gpc_nr):
            for ppc in range(PPC_NR):
                as_ = alpha * g.ppc_tpc_nr[gpc][ppc]
                bs, gs = attrib * g.ppc_tpc_max, gfxp * g.ppc_tpc_max
                u, o, pp = 0x418ea0 + n * 4, PPC_UNIT(gpc, ppc, 0), GPC_UNIT(gpc, 0xc44 + ppc * 4)
                n += 1
                if not (g.ppc_mask[gpc] & (1 << ppc)):
                    continue
                p.wr(o + 0xc0, gs); p.wr(pp, bs); p.wr(o + 0xf4, bo); p.wr(o + 0xf0, bs)
                bo += gs
                p.wr(o + 0xe4, as_); p.wr(o + 0xf8, ao)
                ao += ALPHA_NR_MAX * g.ppc_tpc_nr[gpc][ppc]
                p.wr(u, bs)
        p.wr(0x4181e4, 0x100); p.wr(0x41befc, 0x100)

    def mmio_list(self):
        p = Patch(self.c, mmio=True); self.chan_buffers(p); return p.list

    # -------------------------------------------------- floorsweep (direct)
    def floorsweep(self):
        c, g = self.c, self.g
        for sm, (gpc, tpc) in enumerate(g.sm):                   # gm107_grctx_generate_sm_id
            c.wr(TPC_UNIT(gpc, tpc, 0x698), sm); c.wr(GPC_UNIT(gpc, 0x0c10 + tpc * 4), sm)
            c.wr(TPC_UNIT(gpc, tpc, 0x088), sm)
        for pd, ds in ((False, True), (True, False)):            # gf100_gr_init_num_tpc_per_gpc
            gpc = 0
            for i in range(4):
                data = 0
                for j in range(8):
                    if gpc >= g.gpc_nr: break
                    data |= g.tpc_nr[gpc] << (j * 4); gpc += 1
                if pd: c.wr(0x406028 + i * 4, data)
                if ds: c.wr(0x405870 + i * 4, data)
        # gf117_grctx_generate_rop_mapping
        data, data2 = [0] * 6, [0, 0]
        for i in range(32):
            data[i // 6] |= (g.tile[i] & 7) << ((i % 6) * 5)
        shift, ntpcv = 0, g.tpc_total
        while not (ntpcv & (1 << 4)):
            ntpcv = (ntpcv << 1) & 0xff; shift += 1
        data2[0] = (ntpcv << 16) | (shift << 21) | (((1 << 5) % ntpcv) << 24)
        for i in range(1, 7):
            data2[1] |= ((1 << (i + 5)) % ntpcv) << ((i - 1) * 5)
        c.wr(0x418bb8, (g.tpc_total << 8) | g.screen_tile_row_offset)
        for i in range(6): c.wr(0x418b08 + i * 4, data[i])
        c.wr(0x41bfd0, (g.tpc_total << 8) | g.screen_tile_row_offset | data2[0])
        c.wr(0x41bfe4, data2[1])
        for i in range(6): c.wr(0x41bf00 + i * 4, data[i])
        c.wr(0x4078bc, (g.tpc_total << 8) | g.screen_tile_row_offset)
        for i in range(6): c.wr(0x40780c + i * 4, data[i])
        # gm200_grctx_generate_dist_skip_table
        db = bytearray(32)
        for gpc in range(g.gpc_nr):
            for ppc in range(PPC_NR):
                tpcs, m = g.ppc_tpc_nr[gpc][ppc], g.ppc_tpc_mask[gpc][ppc] & 0xff
                while tpcs > g.ppc_tpc_min:
                    m &= m - 1; tpcs -= 1
                m ^= g.ppc_tpc_mask[gpc][ppc] & 0xff
                db[gpc] |= m
        for i, v in enumerate(struct.unpack("<8I", bytes(db))): c.wr(0x4064d0 + i * 4, v)
        c.wr(0x406500, 0)                                           # gm200_grctx_generate_r406500
        c.wr(0x405b00, (g.tpc_total << 8) | g.gpc_nr)              # gk104_grctx_generate_gpc_tpc_nr
        tmp = 0                                                     # gm200_grctx_generate_tpc_mask
        for i in range(g.gpc_nr): tmp |= ((1 << g.tpc_nr[i]) - 1) << (i * FUNC_TPC_NR)
        c.wr(0x4041c4, tmp)
        dist, gpcs = [0] * 64, [0] * 16                             # gp100_grctx_generate_smid_config
        for sm, (gpc, tpc) in enumerate(g.sm):
            dist[sm // 4] |= ((gpc << 4) | tpc) << ((sm % 4) * 8)
            gpcs[gpc + FUNC_GPC_NR * (tpc // 4)] |= sm << ((tpc % 4) * 8)
        for i in range(-(-g.tpc_total // 4)): c.wr(0x405b60 + i * 4, dist[i])
        for i in range(16): c.wr(0x405ba0 + i * 4, gpcs[i])

    # -------------------------------------------------- icmd / mthd
    def icmd(self, entries):
        """gf100_gr_icmd (32-bit packs)."""
        c = self.c
        c.wr(0x400208, 0x80000000)
        data, first, slow = None, True, 0
        for a, d in entries:
            if first or d != data:
                c.wr(0x400204, d); data = d; first = False
            c.wr(0x400200, a)
            if (a & 0xffff) == 0xe100:
                self.g.wait_idle()
            t0 = time.time()
            while c.rd(0x400700) & 4:
                if time.time() - t0 > 2.0: slow += 1; break
        c.wr(0x400208, 0)
        return slow

    def mthd(self, packs):
        """gf100_gr_mthd: ctrl = 0x80000000 | class, then 0x404488 = ctrl | (addr << 14)
        where addr is the *full* 16-bit method address (nouveau writes `addr << 14`
        straight from the pack; the field is bits 14..29)."""
        c = self.c
        data, first = None, True
        for cls, entries in packs.items():
            ctrl = 0x80000000 | cls
            for addr, d in entries:
                if first or d != data:
                    c.wr(0x40448c, d); data = d; first = False
                c.wr(0x404488, ctrl | ((addr & 0xffff) << 14))

    # -------------------------------------------------- the generator
    def generate(self, inst, vmm, ctx_va, verbose=True, size=None, inject=None):
        """gf100_grctx_generate with gr->firmware = true.  Returns the golden image bytes.

        `size` overrides the FECS "image size" answer.  That answer is not a
        constant: a fresh FECS reports 0x146500, and it GROWS every time a
        golden context is generated (0x275600, 0x433000, 0x95b000 seen in one
        session) -- so it is an allocation watermark, not the image length.
        The real saved length is header word 1 (0xa5e00).  Always generate no
        more than once per FECS boot.
        """
        c, g = self.c, self.g
        if size is None:
            size = fecs_size(c)
        # Keep only a sanity bound: see the note above.
        assert size and size < 0x800000, "FECS ctx size %r" % size
        self.size = size
        c.wr(0x404170, 0x12); c.wait(lambda: not (c.rd(0x404170) & 0x10), 2.0)     # FE_PWR_MODE_FORCE_ON
        c.wr(0x409614, 0x70); time.sleep(0.0001)                                     # gf100_gr_fecs_reset
        c.mask(0x409614, 0x700, 0x700); time.sleep(0.0001); c.rd(0x409614)
        c.wr(0x404170, 0x10); c.wait(lambda: not (c.rd(0x404170) & 0x10), 2.0)     # FE_PWR_MODE_AUTO
        c.wr(0x40802c, 1)                                                            # SCC RAM init
        dsize = nvmmu.align(CB_RESERVED + size, 0x1000)
        data_pa = self.heap.alloc(dsize, 0x1000)
        c.vram_write(data_pa, bytes(CB_RESERVED))
        c.vram_write(data_pa + CB_RESERVED, b"\xef\xbe\xad\xde" * (size // 4))      # negative control
        vmm.map(ctx_va, data_pa, dsize, flush=True)
        c.vram_w32(inst + 0x210, ((ctx_va + CB_RESERVED) & 0xffffffff) | 4)
        c.vram_w32(inst + 0x214, (ctx_va + CB_RESERVED) >> 32)
        addr = inst >> 12
        st = fecs_bind_pointer(c, 0x80000000 | addr)
        if verbose: print("FECS bind_pointer(inst %#x): %s  0x409800=%08x" % (inst, st, c.rd(0x409800)))
        if st != "ok": return None
        for off, v in ((0x1c, 1), (0x20, 0), (0x28, 0), (0x2c, 0)):                 # as nouveau: data, not ctx
            c.vram_w32(data_pa + off, v)
        # ---- gf100_grctx_generate_main
        c.wr(0x260, 0)
        swctx = grmod.netlist("sw_ctx", aiv=True)
        for a, d in swctx: c.wr(a, d)
        idle1 = g.wait_idle()
        idle_timeout = c.mask(0x404154, 0xffffffff, 0)
        self.chan_buffers(Patch(c))
        for a, m, v in ((0x418c6c, 1, 1), (0x41980c, 0x10, 0x10), (0x41be08, 4, 4),   # gk104_grctx_generate_unkn
                        (0x4064c0, 0x80000000, 0x80000000), (0x405800, 0x08000000, 0x08000000), (0x419c00, 8, 8)):
            c.mask(a, m, v)
        self.floorsweep()
        idle2 = g.wait_idle()
        bundle = grmod.netlist("sw_bundle_init")
        slow = self.icmd(bundle)
        self.icmd([])                                    # sw_veid_bundle_init: NULL pack (enable/disable only)
        c.wr(0x404154, idle_timeout)
        meth = grmod.methods()
        self.mthd(meth)
        if inject is not None:
            inject()                                     # host->PGRAPH method injection hook
        c.wr(0x260, 1)
        c.mask(0x419a3c, 0x14, 0)                        # gm200_grctx_generate_r419a3c
        c.mask(0x408840, 3, 0)                           # gp102_grctx_generate_r408840
        idle3 = g.wait_idle()
        if verbose:
            print("grctx main: sw_ctx %d, bundle %d (icmd timeouts %d), methods %s; idle %s/%s/%s; PGRAPH intr 0x400100=%08x"
                  % (len(swctx), len(bundle), slow, {hex(k): len(v) for k, v in meth.items()}, idle1, idle2, idle3, c.rd(0x400100)))
        st = fecs_wfi_golden_save(c, 0x80000000 | addr)
        c.mask(0x409b00, 0x80000000, 0)
        if verbose: print("FECS wfi_golden_save: %s  0x409800=%08x" % (st, c.rd(0x409800)))
        img = c.vram_read(data_pa + CB_RESERVED, size) if st == "ok" else None
        c.vram_w32(inst + 0x210, 0); c.vram_w32(inst + 0x214, 0)
        return img


def fecs_size(c):
    import gr_ctxctl
    return gr_ctxctl.fecs_mthd(c, 0x10)


def fecs_bind_pointer(c, inst):
    c.mask(0x409800, 0x30, 0); c.wr(0x409500, inst); c.wr(0x409504, 3)
    t0 = time.time()
    while time.time() - t0 < 2.0:
        s = c.rd(0x409800)
        if s & 0x20: return "error"
        if s & 0x10: return "ok"
    return "timeout"


def fecs_wfi_golden_save(c, inst):
    c.mask(0x409800, 3, 0); c.wr(0x409500, inst); c.wr(0x409504, 9)
    t0 = time.time()
    while time.time() - t0 < 2.0:
        s = c.rd(0x409800)
        if s & 2: return "error"
        if s & 1: return "ok"
    return "timeout"


def fecs_running(c):
    """LS mode (sctl bit 0) and not halted."""
    return bool(c.rd(0x409240) & 1) and not (c.rd(0x409100) & 0x10)


if __name__ == "__main__":
    import faults
    c = Card()
    if not fecs_running(c):
        print("FECS not running -> gr.bring_up() first")
        out = grmod.bring_up(c, verbose=False)
        assert out and out[4], "GR bring-up failed"
    g = grmod.GR(c).oneinit(verbose=False)
    heap = nvmmu.VramHeap(base=0x11000000)
    vmm = nvmmu.VMM(c, heap)
    inst = nvmmu.inst_alloc(c, heap)
    gc = GrCtx(c, g, heap)
    gc.map_globals(vmm, 0x01000000)
    vmm.join(inst); vmm.flush()
    print("globals: pagepool pa=%#x bundle pa=%#x attrib pa=%#x (+%#x)  va=%s"
          % (gc.pagepool_pa, gc.bundle_pa, gc.attrib_pa, gc.attrib_cb_size, {k: hex(v) for k, v in gc.va.items()}))
    faults.dump(c, clear=True)
    t0 = time.time()
    img = gc.generate(inst, vmm, 0x10000000)
    print("generation took %.1f s" % (time.time() - t0))
    faults.dump(c)
    if img:
        words = struct.unpack("<%dI" % (len(img) // 4), img)
        left = sum(1 for w in words if w == 0xdeadbeef)
        nz = sum(1 for w in words if w)
        print("golden image: %d bytes, %d words, %d still 0xdeadbeef (prefill), %d nonzero" % (len(img), len(words), left, nz))
        print("  header: %s" % " ".join("%08x" % w for w in words[:16]))
        open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "golden_ctx_gp102.bin"), "wb").write(img)
        # FECS's method-0x10 size is an allocation bound; the saved image length is
        # header word 1 (0xa5e00 on this card: hub section + one 0x9630-byte segment
        # per GPC).  Pass = that range written (prefill < 5%, gaps are padding), and
        # nothing beyond it, and the header's zcull/pm sizes match FECS's answers.
        n = words[1] // 4
        pre_in = sum(1 for x in words[:n] if x == 0xdeadbeef)
        wr_out = sum(1 for x in words[n:] if x != 0xdeadbeef)
        print("  image length (hdr[1]) %#x: prefill inside %d/%d, written beyond %d; hdr zcull %#x pm %#x"
              % (words[1], pre_in, n, wr_out, words[9], words[12]))
        ok = 0 < n <= len(words) and pre_in < n // 20 and wr_out == 0
        print("GOLDEN CONTEXT RESULT:", "PASS" if ok else "FAIL")
    else:
        print("GOLDEN CONTEXT RESULT: FAIL")
