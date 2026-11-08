"""PGRAPH bring-up for GP102: nouveau gf100_gr_oneinit + gf100_gr_init with gp102_gr's
function table (engine/gr/gp102.c:99), then ctxctl (FECS/GPCCS via the SEC2 RTOS).

Not ported (not needed for compute, noted so nobody assumes otherwise):
  ZBC clear tables (gf100_gr_zbc_init: render clear values, need LTC ZBC regs),
  clock gating packs, PMU power-gating (nvkm_pmu_pgob is a no-op without PMU fw).

Order (nouveau): GR init with FECS/GPCCS NOT running, then gf100_gr_init_ctxctl_ext
(ACR bootstrap through the SEC2 RTOS, 0x260 = 0/1, start GPCCS then FECS, wait
0x409800 bit 0, watchdog, size discovery).
"""
import os, struct, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from card import Card
import acr

FW = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "firmware", "gp102", "gr")
def ROP_UNIT(u, r): return 0x410000 + u * 0x400 + r
def GPC_BCAST(r): return 0x418000 + r
def GPC_UNIT(t, r): return 0x500000 + t * 0x8000 + r
def PPC_UNIT(t, m, r): return 0x503000 + t * 0x8000 + m * 0x200 + r
def TPC_UNIT(t, m, r): return 0x504000 + t * 0x8000 + m * 0x800 + r

FUNC_PPC_NR = 3          # gp102_gr.ppc_nr
FUNC_GPC_NR, FUNC_TPC_NR = 6, 5


def netlist(name, aiv=False):
    """gk20a_gr_av_to_init / gk20a_gr_aiv_to_init: (addr, data) entries."""
    b = open(os.path.join(FW, name + ".bin"), "rb").read()
    if aiv:
        return [(a, d) for a, _i, d in struct.iter_unpack("<III", b)]
    return list(struct.iter_unpack("<II", b))


def methods(name="sw_method_init"):
    """gk20a_gr_av_to_method: {class: [(method_addr, data)]}."""
    out = {}
    for a, d in struct.iter_unpack("<II", open(os.path.join(FW, name + ".bin"), "rb").read()):
        out.setdefault(a & 0xffff, []).append((a >> 16, d))
    return out


class GR:
    def __init__(self, c):
        self.c = c

    # ---------------------------------------------------------------- oneinit
    def oneinit(self, verbose=True):
        c = self.c
        self.rop_nr = c.rd(0x12006c)                         # gm200_gr_rops
        self.gpc_nr = c.rd(0x409604) & 0x1f
        self.tpc_nr, self.ppc_mask, self.ppc_tpc_mask = [], [], []
        for i in range(self.gpc_nr):
            self.tpc_nr.append(c.rd(GPC_UNIT(i, 0x2608)) & 0xff)    # nouveau stores it in a u8 (reg reads 0x0004000N)
            m, tm = 0, []
            for j in range(FUNC_PPC_NR):
                v = c.rd(GPC_UNIT(i, 0x0c30 + j * 4)); tm.append(v)
                if v: m |= 1 << j
            self.ppc_mask.append(m); self.ppc_tpc_mask.append(tm)
        self.tpc_max = max(self.tpc_nr)
        self.tpc_total = sum(self.tpc_nr)
        # Hard sanity limits BEFORE anything loops over these counts: a bad read once
        # drove the TPC exception loops 262148 times per GPC across all of MMIO and
        # took down the PRI ring (session 4).  gp102: <= 6 GPCs, <= 5 TPCs each.
        assert 1 <= self.gpc_nr <= FUNC_GPC_NR, "gpc_nr %r" % self.gpc_nr
        assert all(1 <= t <= FUNC_TPC_NR for t in self.tpc_nr), "tpc_nr %r" % self.tpc_nr
        assert 1 <= self.rop_nr <= 16, "rop_nr %r" % self.rop_nr
        self.oneinit_tiles()
        self.sm = [(g, t) for t in range(self.tpc_max) for g in range(self.gpc_nr) if t < self.tpc_nr[g]]
        if verbose:
            print("GR: rop_nr(0x12006c)=%#x gpc_nr=%d tpc_nr=%s (total %d, max %d) ppc_mask=%s"
                  % (self.rop_nr, self.gpc_nr, self.tpc_nr, self.tpc_total, self.tpc_max, self.ppc_mask))
            print("    ppc_tpc_mask=%s" % self.ppc_tpc_mask)
            print("    screen_tile_row_offset=%d tiles=%s" % (self.screen_tile_row_offset, self.tile[:self.tpc_total]))
            print("    sm_nr=%d" % len(self.sm))
        return self

    def oneinit_tiles(self):
        """gm200_gr_oneinit_tiles -> gf100_gr_oneinit_tiles (no hardcoded map for this config)."""
        assert not (self.gpc_nr == 6 and self.tpc_total == 24), "gm200 hardcoded 6/24 map not ported"
        assert not (self.gpc_nr in (2, 4)), "gm200 hardcoded maps not ported"
        primes = (3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61)
        t = self.tpc_total
        fixed = {15: 6, 14: 5, 13: 2, 11: 7, 10: 6, 7: 1, 5: 1, 3: 2, 2: 1, 1: 1}
        if t in fixed:
            self.screen_tile_row_offset = fixed[t]
        else:
            self.screen_tile_row_offset = 3
            for p in primes:
                if t % p:
                    self.screen_tile_row_offset = p; break
        gpc_map = list(range(self.gpc_nr))
        done = False
        while not done:                                      # the same bubble sort, same tie order
            done = True
            for i in range(self.gpc_nr - 1):
                if self.tpc_nr[gpc_map[i + 1]] > self.tpc_nr[gpc_map[i]]:
                    gpc_map[i], gpc_map[i + 1] = gpc_map[i + 1], gpc_map[i]; done = False
        mul = 2 if (self.gpc_nr * self.tpc_max) & 1 else 1
        denom = self.gpc_nr * self.tpc_max * mul
        frac = [self.tpc_nr[gpc_map[i]] * self.gpc_nr * mul for i in range(self.gpc_nr)]
        run = [frac[i] + (i * self.tpc_max * mul - denom // 2) for i in range(self.gpc_nr)]
        self.tile = []
        while len(self.tile) < t:
            for j in range(self.gpc_nr):
                if run[j] * 2 >= denom:
                    if len(self.tile) < t: self.tile.append(gpc_map[j])
                    run[j] += frac[j] - denom
                else:
                    run[j] += frac[j]
        self.tile += [0xff] * (256 - len(self.tile))

    # ---------------------------------------------------------------- init
    def wait_idle(self, timeout=2.0):
        c = self.c
        t0 = time.time()
        while time.time() - t0 < timeout:
            c.rd(0x400700)
            if not (c.rd(0x200) & 0x1000) or not (c.rd(0x40060c) & 1):
                return True
        return False

    def init(self, verbose=True):
        """gf100_gr_init with gp102_gr's function table."""
        c = self.c
        c.mask(0x400500, 0x00010001, 0)
        # gm200_gr_init_gpc_mmu
        c.wr(0x418880, c.rd(0x100c80) & 0xf0001fff)
        c.wr(0x418890, 0); c.wr(0x418894, 0)
        c.wr(0x4188b4, c.rd(0x100cc8)); c.wr(0x4188b8, c.rd(0x100ccc)); c.wr(0x4188b0, c.rd(0x100cc4))
        nonctx = netlist("sw_nonctx")
        for a, d in nonctx:
            c.wr(a, d)
        idle = self.wait_idle()
        # gk104_gr_init_vsc_stream_master
        c.wr(GPC_UNIT(0, 0x3018), 1)
        # gf117_gr_init_zcull
        magic = -(-0x00800000 // self.tpc_total)
        tile_nr = FUNC_GPC_NR * FUNC_TPC_NR
        bank = [0] * 32
        for i in range(0, tile_nr, 8):
            data = 0
            for j in range(8):
                if i + j >= self.tpc_total: break
                g = self.tile[i + j]; data |= bank[g] << (j * 4); bank[g] += 1
            c.wr(GPC_BCAST(0x0980 + (i // 8) * 4), data)
        for g in range(self.gpc_nr):
            c.wr(GPC_UNIT(g, 0x0914), (self.screen_tile_row_offset << 8) | self.tpc_nr[g])
            c.wr(GPC_UNIT(g, 0x0910), 0x00040000 | self.tpc_total)
            c.wr(GPC_UNIT(g, 0x0918), magic)
        c.wr(GPC_BCAST(0x3fd4), magic)
        # gm200_gr_init_num_active_ltcs
        c.wr(GPC_BCAST(0x08ac), c.rd(0x100800)); c.wr(GPC_BCAST(0x033c), c.rd(0x100804))
        # gp100_gr_init_rop_active_fbps
        fbp = c.rd(0x12006c) & 0xf
        c.mask(0x408850, 0xf, fbp); c.mask(0x408958, 0xf, fbp)
        # gp102_gr_init_swdx_pes_mask
        mask = 0
        for g in range(self.gpc_nr):
            mask |= (c.rd(GPC_UNIT(g, 0x0c50)) & 0xf) << (g * 4)
        c.wr(0x4181d0, mask)
        c.wr(0x400500, 0x00010001)
        c.wr(0x400100, 0xffffffff); c.wr(0x40013c, 0xffffffff); c.wr(0x400124, 0x00000002)
        c.wr(0x409c24, 0x000e0002)                           # gp100_gr_init_fecs_exceptions
        c.wr(0x405848, 0xc0000000); c.mask(0x40584c, 1, 1)   # gm200_gr_init_ds_hww_esr_2
        c.wr(0x404000, 0xc0000000); c.wr(0x404600, 0xc0000000); c.wr(0x408030, 0xc0000000)
        c.wr(0x406018, 0xc0000000); c.wr(0x404490, 0xc0000000)
        c.wr(0x407020, 0x40000000)                           # gk104_gr_init_sked_hww_esr
        c.wr(0x405840, 0xc0000000); c.wr(0x405844, 0x00ffffff)
        c.mask(0x419cc0, 8, 8)                               # gf100_gr_init_419cc0
        for g in range(self.gpc_nr):
            for t in range(self.tpc_nr[g]):
                c.wr(TPC_UNIT(g, t, 0x48c), 0xc0000000)
        for g in range(self.gpc_nr):                         # gk104_gr_init_ppc_exceptions
            for p in range(FUNC_PPC_NR):
                if self.ppc_mask[g] & (1 << p):
                    c.wr(PPC_UNIT(g, p, 0x038), 0xc0000000)
        for g in range(self.gpc_nr):
            c.wr(GPC_UNIT(g, 0x0420), 0xc0000000); c.wr(GPC_UNIT(g, 0x0900), 0xc0000000)
            c.wr(GPC_UNIT(g, 0x1028), 0xc0000000); c.wr(GPC_UNIT(g, 0x0824), 0xc0000000)
            for t in range(self.tpc_nr[g]):
                c.wr(TPC_UNIT(g, t, 0x508), 0xffffffff); c.wr(TPC_UNIT(g, t, 0x50c), 0xffffffff)
                c.wr(TPC_UNIT(g, t, 0x224), 0xc0000000)      # gf100_gr_init_tex_hww_esr
                c.wr(TPC_UNIT(g, t, 0x084), 0xc0000000)
                c.wr(TPC_UNIT(g, t, 0x430), 0xc0000000)      # gm107_gr_init_504430
                c.wr(TPC_UNIT(g, t, 0x644), 0x00dffffe)      # gp100_gr_init_shader_exceptions
                c.wr(TPC_UNIT(g, t, 0x64c), 0x00000105)
            c.wr(GPC_UNIT(g, 0x2c90), 0xffffffff); c.wr(GPC_UNIT(g, 0x2c94), 0xffffffff)
        for r in range(self.rop_nr):                         # gf100_gr_init_rop_exceptions
            c.wr(ROP_UNIT(r, 0x144), 0x40000000); c.wr(ROP_UNIT(r, 0x070), 0x40000000)
            c.wr(ROP_UNIT(r, 0x204), 0xffffffff); c.wr(ROP_UNIT(r, 0x208), 0xffffffff)
        c.wr(0x400108, 0xffffffff); c.wr(0x400138, 0xffffffff)
        c.wr(0x400118, 0xffffffff); c.wr(0x400130, 0xffffffff)
        c.wr(0x40011c, 0xffffffff); c.wr(0x400134, 0xffffffff)   # gf100_gr_init_exception2
        if verbose:
            print("GR init: sw_nonctx %d writes, idle after nonctx=%s, swdx_pes_mask=%#x, zcull magic=%#x"
                  % (len(nonctx), idle, mask, magic))
            print("  PGRAPH status 0x400700=%08x 0x40060c=%08x  intr 0x400100=%08x  FECS 0x409c18=%08x"
                  % (c.rd(0x400700), c.rd(0x40060c), c.rd(0x400100), c.rd(0x409c18)))
        return idle


def bring_up(c, verbose=True):
    """Full chain in nouveau's order.  Returns (gr, sec2, cmdq, msgq, sizes)."""
    import fb, faults, acr_unload, sec2_rtos, gr_ctxctl
    mem, iovas = c.dev.alloc_sysmem(0x1000, contiguous=True)
    fb.fb_init(c, iovas[0])
    if fb.vpr_scrub_required(c):
        nv = next(d for d in fb.top_parse(c) if d["name"].startswith("NVDEC"))
        print("VPR scrub:", fb.vpr_scrub(c, nv["addr"], nv["reset"])[:2])
    c.mask(0x200, 1 << 12, 1 << 12)
    faults.dump(c, clear=True)
    sec2 = acr.Falcon(c, acr.SEC2, pmc_bit=14)
    if sec2_rtos.rtos_running(sec2):
        sec2_rtos.rtos_unload(c, sec2)
    if acr.acr_claim(c):
        u = acr_unload.acr_unload(c, verbose=False)
        print("ACR unload: ok=%s mbox0=%#x claim=%d" % (u["ok"], u["mbox0"], acr.acr_claim(c)))
    r = acr.acr_load(c, sec2_rtos=1, verbose=False)
    print("ACR load + SEC2 RTOS LSF:", "PASS" if r["ok"] else "FAIL mbox0=%08x" % r["mbox0"])
    if not r["ok"]:
        return None
    q = sec2_rtos.start_rtos(c, sec2, verbose=False)
    if q is None:
        print("SEC2 RTOS: no init message"); return None
    cmdq, msgq = q
    print("SEC2 RTOS up")
    gr = GR(c).oneinit(verbose)
    gr.init(verbose)
    c.wr(0x260, 0)
    res = [sec2_rtos.bootstrap_falcon(c, sec2, cmdq, msgq, acr.LSF_FECS, 0, verbose=False),
           sec2_rtos.bootstrap_falcon(c, sec2, cmdq, msgq, acr.LSF_GPCCS, 1, verbose=False)]
    c.wr(0x260, 1)
    print("bootstrap FECS/GPCCS: %s" % [None if x is None else hex(x["error"]) for x in res])
    sizes = gr_ctxctl.ctxctl_start(c, verbose=verbose)
    faults.dump(c)
    return gr, sec2, cmdq, msgq, sizes


if __name__ == "__main__":
    c = Card()
    out = bring_up(c)
    ok = out is not None and out[4] is not None and all(out[4].values())
    print("GR BRING-UP RESULT:", "PASS" if ok else "FAIL", "" if not ok else out[4])
