"""Memory clock (MCLK) on the GP102: inspect, and reclock along the refpll path.

Register sequence follows nouveau's gk104 GDDR5 code (nvkm/subdev/fb/ramgk104.c,
vendored at tools/nvkm/subdev/fb/), which GM107/GM200 reuse; nouveau has no Pascal
reclock (ramgp100.c is init-only), so every write here is checked against what the
card reads back, and `status` is the negative control before and after.

  status              read-only: PLL limits, rammap/ramcfg/timing for this strap, live registers
  vbios               read-only: the VBIOS tables only (no card needed)
  get                 the current memory clock in MHz (0 if not on the refpll path)
  set <MHz> [--live]  reclock along the refpll path (<= 1620 MHz); without --live, print the
                      writes only. The card must be idle: no shimd, no client. Ramps through
                      810 and then 54 MHz steps (big jumps fail training); --jump disables that.
  retrain <MHz>       re-program the active band's bank + re-run the training tail at the current
                      clock (no clock change): the cheap way to test PHY settings.

  --patch=ADDR:VAL[,ADDR:VAL...]  extra register writes after the band script's trims, before
                      training (hex).  --soft: report incomplete training instead of aborting.
                      Both exist for PHY trim sweeps: whether a DRAM trains at a given clock is
                      board-specific and the VBIOS only ships NVIDIA's reference trims.

Measured on this card (logs/mclk/, pascal/memtest.py, 9 GiB x 3 patterns unless noted):
  405 (boot)  31.1 GB/s read   810  64.0   1080  87.3 (200-pass soak at 42 C: clean)
  1134  92.0: clean cold (36 C, 100 passes) but 17 single-bit flips warm (42 C), top byte lane
  1188, 1296: train but corrupt data -> card faults.  1620 (band 2 config): does not train.
  Band 2 (>=1300) never completes training on the refpll path, at any clock tried (1350, 1620),
  with or without a ramp: the DRAM delays do train but FBPA status bits 1|3 stay set, and the
  first GPU work after it hangs with GP_GET garbage (PBDMA INTR 0x20000000).  Its bank script's
  trims (0x03 lanes / Vref 0x5353) and the band-2 timing (CL 20) are the difference from band 1.

Usage: ~/.venvs/tinygpu/bin/python pascal/mclk.py status
"""
import os
import struct
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from vbios import VBIOS

ROM = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "firmware", "gp102_vbios.rom")
REFCLK_KHZ = 27000


# ----------------------------------------------------------------------------- VBIOS
def pll_table(v):
    """nouveau bios/pll.c: BIT 'C' -> PLL limits table; returns {type: (reg, entry_ptr, ver)}."""
    C = v.bit("C")
    data = v.rd32(C["offset"]) if C["version"] == 2 else v.rd16(C["offset"] + 8)
    ver, hdr, ln, cnt = (v.rd8(data + i) for i in range(4))
    out = {}
    for i in range(cnt):
        e = data + hdr + i * ln
        typ = v.rd8(e)
        if typ == 0:
            continue
        ptr = v.rd16(e + 1)
        reg = v.rd32(e + 3) if ver < 0x50 else 0
        out[typ] = dict(reg=reg, ptr=ptr, raw=v.raw(e, ln).hex())
    return ver, out


def pll_limits(v, ptr):
    """nouveau nvbios_pll_parse, table version 0x40 layout (all MHz)."""
    r16, r8 = v.rd16, v.rd8
    # (the reference clock is in the type entry, +7/+9, not here)
    return dict(vco_min=r16(ptr + 0), vco_max=r16(ptr + 2), in_min=r16(ptr + 4), in_max=r16(ptr + 6),
                m=(r8(ptr + 8), r8(ptr + 9)), n=(r8(ptr + 10), r8(ptr + 11)),
                p=(r8(ptr + 12), r8(ptr + 13)), raw=v.raw(ptr, 20).hex())


def ramcfg_index(v, strap_reg):
    """nouveau nvbios_ramcfg_index: strap = (0x101000 & 0x3c) >> 2, translated by BIT 'M'."""
    strap = (strap_reg & 0x3c) >> 2
    M = v.bit("M")
    xlat = 0
    if M:
        if M["version"] == 1 and M["length"] >= 5:
            xlat = v.rd16(M["offset"] + 3)
        if M["version"] == 2 and M["length"] >= 3:
            xlat = v.rd16(M["offset"] + 1)
    return (v.rd8(xlat + strap) if xlat else strap), strap, xlat


def rammap(v):
    P = v.bit("P")
    ptr = v.rd32(P["offset"] + 4)
    ver, hdr, ln, ssz, snr, cnt = (v.rd8(ptr + i) for i in range(6))
    bands = []
    for i in range(cnt):
        e = ptr + hdr + i * (ln + ssz * snr)
        subs = [v.raw(e + ln + j * ssz, ssz) for j in range(snr)]
        bands.append(dict(min=v.rd16(e), max=v.rd16(e + 2), raw=v.raw(e, ln), subs=subs))
    return dict(ver=ver, hdr=hdr, len=ln, ssz=ssz, snr=snr, cnt=cnt, ptr=ptr, bands=bands)


def timing_table(v):
    """BIT 'P' + 8 -> timing table (nouveau bios/timing.c; ver 0x20: hdr, cnt, len, snr, ssz)."""
    P = v.bit("P")
    ptr = v.rd32(P["offset"] + 8)
    ver = v.rd8(ptr)
    if ver == 0x20:     # nouveau nvbios_timingTe: hdr@1 len@2 ssz@3 snr@4 cnt@5
        hdr, ln, ssz, snr, cnt = (v.rd8(ptr + i) for i in range(1, 6))
    else:
        hdr, ln, cnt, snr, ssz = v.rd8(ptr + 1), v.rd8(ptr + 2), v.rd8(ptr + 3), 0, 0
    ents = [v.raw(ptr + hdr + i * (ln + snr * ssz), ln) for i in range(cnt)]
    return dict(ver=ver, hdr=hdr, len=ln, cnt=cnt, ptr=ptr, ents=ents)


def show_vbios(v, idx=None):
    ver, plls = pll_table(v)
    print("PLL limits table ver %#x" % ver)
    for typ, p in sorted(plls.items()):
        print("  type %#04x reg %#08x entry @%#x  %s" % (typ, p["reg"], p["ptr"], p["raw"]))
        if p["ptr"] and ver < 0x50:
            print("        %s" % pll_limits(v, p["ptr"]))
    rm = rammap(v)
    print("rammap ver %#x hdr %d len %d sub %dx%d" % (rm["ver"], rm["hdr"], rm["len"], rm["snr"], rm["ssz"]))
    for i, b in enumerate(rm["bands"]):
        print("  band %d %4d..%4d  entry %s" % (i, b["min"], b["max"], b["raw"].hex()))
        for j, s in enumerate(b["subs"]):
            mark = " <== this card" if j == idx else ""
            print("      strap %d: %s%s" % (j, s.hex(), mark))
    tt = timing_table(v)
    print("timing table ver %#x hdr %d len %d count %d @%#x" % (tt["ver"], tt["hdr"], tt["len"], tt["cnt"], tt["ptr"]))
    for i, e in enumerate(tt["ents"]):
        print("  timing %2d: %s" % (i, e.hex()))


# ----------------------------------------------------------------------------- card
# On Pascal the FBPA broadcast block nouveau's Kepler code calls 0x10fXXX lives at 0x9a0XXX
# (verified: VBIOS timing entry 0 words sit at 0x9a0290.. exactly where gk104 writes them),
# and per-partition copies at 0x900000 + i * 0x4000 (gk104: 0x110000 + i * 0x1000).
FB = 0x9a0000
REG_GROUPS = [
    ("mclk pll/ctl", 0x132000, 0x132060),
    ("clk 1373xx", 0x137300, 0x137400),
    ("fbpa 200", FB + 0x200, FB + 0x320),
    ("fbpa 330 (MR)", FB + 0x330, FB + 0x358),
    ("fbpa 600", FB + 0x600, FB + 0x700),
    ("fbpa 800", FB + 0x800, FB + 0x840),
    ("fbpa 900", FB + 0x900, FB + 0x920),
    ("fbpa 970", FB + 0x970, FB + 0x980),
    ("fbpa 090", FB + 0x090, FB + 0x094),
]


def snapshot(c):
    out = {}
    for _, a0, a1 in REG_GROUPS:
        for a in range(a0, a1, 4):
            out[a] = c.rd(a)
    return out


def mclk_khz(regs):
    """gk104 clk read_mem: 0x1373f4 & 3: 1 = refpll path, 2 = mpll path."""
    src = regs[0x1373f4] & 3
    r = regs[0x132024]
    m, n, p = r & 0xff, (r >> 8) & 0xff, (r >> 16) & 0x3f
    fn = regs[0x132030] >> 16
    # nouveau gk104 read_pll: sclk * N + ((u16)(fN + 4096) * sclk >> 13)
    ref = REFCLK_KHZ * (n + ((fn + 4096) & 0xffff) / 8192.0) / m if m else 0
    out = dict(src=src, refpll=(m, n, p, fn), refpll_vco_khz=ref, refpll_out_khz=ref / p if p else 0)
    mp = regs[0x132004]
    out["mpll"] = (mp & 0xff, (mp >> 8) & 0xff, ((mp >> 28) & 0x7) + 1, bool(regs[0x132000] & 1))
    return out


def live_mclk_mhz(c):
    """Current memory clock in MHz from the registers (refpll path only; 0 if not on it)."""
    r = {a: c.rd(a) for a in (0x1373f4, 0x132024, 0x132030, 0x132004, 0x132000)}
    m = mclk_khz(r)
    return round(m["refpll_out_khz"] / 1000) if m["src"] == 1 else 0


def dump_regs(regs, prev=None):
    for name, a0, a1 in REG_GROUPS:
        line = []
        for a in range(a0, a1, 4):
            if (regs[a] == 0 or regs[a] >> 16 == 0xbadf) and (prev is None or prev.get(a, 0) == regs[a]):
                continue
            d = "" if prev is None or prev.get(a) == regs[a] else "  (was %08x)" % prev[a]
            line.append("    %06x = %08x%s" % (a, regs[a], d))
        if line:
            print("  %s" % name)
            print("\n".join(line))


# ----------------------------------------------------------------------------- reclock
def bits(b, m):
    sh = (m & -m).bit_length() - 1
    return (b & m) >> sh


def ramcfg_fields(rc):
    """nouveau nvbios_rammapSp, version 0x11."""
    return dict(timing=rc[0], r01_01=bits(rc[1], 0x01), r01_02=bits(rc[1], 0x02), r01_04=bits(rc[1], 0x04),
                r01_08=bits(rc[1], 0x08), r01_10=bits(rc[1], 0x10), DLLoff=bits(rc[1], 0x20),
                r01_40=bits(rc[1], 0x40), r01_80=bits(rc[1], 0x80),
                r02_03=bits(rc[2], 0x03), r02_04=bits(rc[2], 0x04), r02_08=bits(rc[2], 0x08),
                r02_10=bits(rc[2], 0x10), r02_40=bits(rc[2], 0x40), r02_80=bits(rc[2], 0x80),
                r03_0f=bits(rc[3], 0x0f), r03_30=bits(rc[3], 0x30), r03_c0=bits(rc[3], 0xc0), r03_f0=bits(rc[3], 0xf0),
                r04=rc[4], r06=rc[6],
                r07_02=bits(rc[7], 0x02), r07_04=bits(rc[7], 0x04), r07_08=bits(rc[7], 0x08),
                r07_10=bits(rc[7], 0x10), r07_40=bits(rc[7], 0x40), r07_80=bits(rc[7], 0x80),
                r08_01=bits(rc[8], 0x01), r08_02=bits(rc[8], 0x02), r08_04=bits(rc[8], 0x04),
                r08_08=bits(rc[8], 0x08), r08_10=bits(rc[8], 0x10), r08_20=bits(rc[8], 0x20), r09=rc[9])


def rammap_fields(e):
    """nouveau nvbios_rammapEp, version 0x11 (the fields the GDDR5 path uses)."""
    return dict(min=struct.unpack_from("<H", e, 0)[0], max=struct.unpack_from("<H", e, 2)[0],
                m08_01=bits(e[8], 0x01), m08_0c=bits(e[8], 0x0c), m08_10=bits(e[8], 0x10), m11_0c=bits(e[0x11], 0x0c))


def timing_fields(t):
    """nouveau nvbios_timingEp, version 0x20."""
    w = struct.unpack_from("<11I", t, 0)
    t2c = struct.unpack_from("<H", t, 0x2c)[0]
    t31 = struct.unpack_from("<H", t, 0x31)[0]
    return dict(w=w, t2e_03=bits(t[0x2e], 0x03), t2e_30=bits(t[0x2e], 0x30), t2e_c0=bits(t[0x2e], 0xc0),
                t2f_03=bits(t[0x2f], 0x03), t2c_003f=t2c & 0x3f, t2c_1fc0=(t2c & 0x1fc0) >> 6,
                t30_07=t[0x30] & 7, t30_f8=t[0x30] >> 3,
                WL=(w[1] & 0xf80) >> 7, CL=w[1] & 0x1f, WR=(w[2] & 0x7f0000) >> 16)


def target_config(v, idx, mhz):
    rm = rammap(v)
    band = next((i for i, b in enumerate(rm["bands"]) if b["min"] <= mhz <= b["max"]), None)
    if band is None:
        raise SystemExit("no rammap band covers %d MHz" % mhz)
    b = rm["bands"][band]
    rc = ramcfg_fields(b["subs"][idx])
    tm = timing_fields(timing_table(v)["ents"][rc["timing"]])
    return band, rammap_fields(b["raw"]), rc, tm


def refpll_coef(mhz, cur):
    """refpll coefficients for `mhz`. VCO = 27 MHz * (N + (fN + 4096) / 8192) / M (nouveau gk104
    read_pll), range 800..1620 MHz (VBIOS PLL type 0x0c); out = VCO / P. Keeps the running VCO
    when an integer P reaches the target, else P = smallest that puts the VCO in range, M = 1,
    integer N when possible. Returns (coef, fN)."""
    m, n, p, fn = cur
    vco = REFCLK_KHZ * (n + ((fn + 4096) & 0xffff) / 8192.0) / m / 1000
    P = vco / mhz
    if abs(P - round(P)) < 1e-6 and 1 <= round(P) <= 31:
        return (round(P) << 16) | (n << 8) | m, fn
    P = next(q for q in range(1, 32) if 800 <= mhz * q <= 1620)
    ntot = mhz * P / 27.0                     # N + (fN + 4096) / 8192, M = 1
    N = int(ntot) - 1
    fN = (round((ntot - N) * 8192) - 4096) & 0xffff
    return (P << 16) | (N << 8) | 1, fN


class Seq:
    """Every register access of a reclock goes through here: printed, and only issued when live."""

    def __init__(self, c, live):
        self.c, self.live, self.n = c, live, 0
        self.shadow = {}

    def rd(self, a):
        if not self.live and a in self.shadow:
            return self.shadow[a]
        return self.c.rd(a)

    def wr(self, a, d, why=""):
        self.n += 1
        print("    wr %06x <- %08x  %s" % (a, d, why))
        self.shadow[a] = d
        if self.live:
            self.c.wr(a, d)

    def mask(self, a, m, d, why=""):
        """nouveau ram_mask semantics: (old & ~m) | d (d is not masked)."""
        o = self.rd(a)
        n = (o & ~m) | d
        if n != o:
            self.wr(a, n, why + "  (was %08x)" % o)
        return o

    def wait(self, a, m, d, timeout=0.5, why=""):
        print("    wait %06x & %08x == %08x  %s" % (a, m, d, why))
        if not self.live:
            return True
        if not self.c.wait(lambda: (self.c.rd(a) & m) == d, timeout, 0.0005):
            raise RuntimeError("timeout waiting for %06x & %08x == %08x (is %08x)" % (a, m, d, self.c.rd(a)))
        return True


def train(s, fbpas, m, d, why, soft=False):
    s.mask(FB + 0x910, m, d, why)
    s.mask(FB + 0x914, m, d, why)
    if d & 0x80000000:
        bad = []
        full = max(s.c.rd(0x90020c + i * 0x4000) for i in fbpas)
        for i in fbpas:
            # A half-populated FBPA (GP102 on the 1080 Ti: FBPA0, 1 GiB of 2) never completes
            # training on its absent half: status bit 1 stays set and that half's delay register
            # (0x6a8) stays saturated, while DEVINIT's boot training leaves it untouched. Accept
            # exactly that bit there; the VRAM pattern test is the judge of the present half.
            half = s.c.rd(0x90020c + i * 0x4000) < full
            m = 0xd if half else 0xf
            try:
                s.wait(0x900974 + i * 0x4000, m, 0, timeout=1.0,
                       why="FBPA%d training done%s" % (i, " (half partition: bit 1 ignored)" if half else ""))
            except RuntimeError:
                bad.append(i)
        if s.live:
            for i in fbpas:
                b = 0x900000 + i * 0x4000
                print("      FBPA%d status %x  trained delays 6a8=%08x 6ac=%08x" % (i, s.c.rd(b + 0x974) & 0xf, s.c.rd(b + 0x6a8), s.c.rd(b + 0x6ac)))
        if bad:
            msg = "training did not complete on FBPA %s" % bad
            if not soft:
                raise RuntimeError(msg)
            print("      INCOMPLETE: %s (--soft, continuing)" % msg)


def cli_patches(argv):
    """--patch=ADDR:VAL[,ADDR:VAL...] (hex): extra register writes applied after the bank script's
    trims and before training/retraining.  Exists for PHY trim sweeps: which per-lane delay/Vref
    values the DRAM trains cleanly with is board-specific, and the VBIOS only ships NVIDIA's."""
    out = []
    for a in argv:
        if a.startswith("--patch="):
            for kv in a[len("--patch="):].split(","):
                k, _, v = kv.partition(":")
                out.append((int(k, 16), int(v, 16)))
    return out


def apply_patches(s, patches):
    for a, val in patches:
        s.wr(a, val, "patch (after the bank script)")



def cond_overrides(fbpas):
    """The mode scripts gate FBPA3 behind condition 0x30 and FBPA4/5 behind 0x31; the VBIOS has
    no legacy condition table, and DEVINIT evidently evaluated both true on this card (bank 0
    holds the gated values on all six partitions). Derive them from which partitions answer."""
    return {0x30: 3 in fbpas, 0x31: 4 in fbpas and 5 in fbpas}


def program_bank(s, v, idx, bank, fbpas, patches=()):
    """nouveau gp100_ram_init, for one mode: select the write bank, run that mode's script."""
    from vbinit import Init, mode_scripts
    save = s.rd(FB + 0x65c)
    s.mask(FB + 0x65c, 0xf0, bank << 4, "write bank %d" % bank)
    it = Init(v, strap=idx, card=s.c, dry=not s.live, cond_override=cond_overrides(fbpas))
    it.run(mode_scripts(v)[bank])
    for w in it.writes:
        print("    script %d: %s" % (bank, ("wr %06x <- %08x" % w) if isinstance(w[0], int) else "%s %s" % w))
    apply_patches(s, patches)
    s.mask(FB + 0x65c, 0xf0, save & 0xf0, "restore write bank")


def train_tail(s, fbpas, rc, rm, soft=False):
    fb = lambda r: FB + r
    print("  [train]")
    data = (0x88020000 | (0x10000000 if rc["r07_04"] else 0) | (0 if rm["m08_10"] else 0x00080000)) \
        if not rc["r07_08"] else 0xa40e0000
    train(s, fbpas, 0xbc0f0000, data, "train", soft=soft)
    s.mask(fb(0x830), 0x01000000, 0x01000000)
    s.mask(fb(0x830), 0x01000000, 0)
    if rc["r07_02"]:
        train(s, fbpas, 0x80020000, 0x01000000, "train 07_02", soft=soft)
    s.mask(fb(0x200), 0x00000800, 0x800 if rm["m08_01"] else 0)


def retrain(c, v, idx, mhz, live, patches=(), soft=False):
    """Re-program the active band's bank and re-run the training tail, at the current clock."""
    band, rm, rc, tm = target_config(v, idx, mhz)
    fbpas = [i for i in range(12) if c.rd(0x90020c + i * 0x4000) >> 16 != 0xbadf]
    s = Seq(c, live)
    bank = rm["m11_0c"]
    if (c.rd(FB + 0x65c) & 0xf) != bank:
        raise SystemExit("active bank is %d, band %d wants %d: use set" % (c.rd(FB + 0x65c) & 0xf, band, bank))
    print("retrain band %d bank %d, FBPAs %s; %s" % (band, bank, fbpas, "LIVE" if live else "DRY RUN"))
    if bank:
        program_bank(s, v, idx, bank, fbpas, patches)
    else:
        apply_patches(s, patches)
    train_tail(s, fbpas, rc, rm, soft=soft)


def reclock(c, v, idx, mhz, live, patches=(), soft=False):
    regs = snapshot(c)
    cur = mclk_khz(regs)
    if cur["src"] != 1 or cur["mpll"][3]:
        raise SystemExit("not on the refpll path (0x1373f4 src %d, MPLL on %s): refusing" % (cur["src"], cur["mpll"][3]))
    if mhz > 1620:
        raise SystemExit("%d MHz needs the MPLL (mode 2): not implemented" % mhz)
    band, rm, rc, tm = target_config(v, idx, mhz)
    coef, fn = refpll_coef(mhz, cur["refpll"])
    fbpas = [i for i in range(12) if c.rd(0x90020c + i * 0x4000) >> 16 != 0xbadf]
    bank = rm["m11_0c"]
    print("target %d MHz: rammap band %d (%d..%d), timing %d, bank %d, refpll coef %08x, FBPAs %s; %s"
          % (mhz, band, rm["min"], rm["max"], rc["timing"], bank, coef, fbpas, "LIVE" if live else "DRY RUN"))
    print("  ramcfg %s" % {k: x for k, x in rc.items()})
    print("  timing WL %d CL %d WR %d, words %s" % (tm["WL"], tm["CL"], tm["WR"], " ".join("%08x" % w for w in tm["w"])))
    s = Seq(c, live)
    fb = lambda r: FB + r

    # 0. this band's FBPA register bank (devinit only programs bank 0)
    if bank:
        print("  [bank] program bank %d from the VBIOS mode script" % bank)
        program_bank(s, v, idx, bank, fbpas, patches)
    else:
        apply_patches(s, patches)

    # 1. into self-refresh (gk104_ram_calc_gddr5, from mode 1)
    print("  [enter] stop refresh, precharge, self-refresh")
    s.mask(fb(0x808), 0x40000000, 0x40000000)
    s.mask(fb(0x200), 0x00000800, 0)
    train(s, fbpas, 0x01020000, 0x000c0000, "train pre")
    s.mask(fb(0x210), 0x80000000, 0, "REFRESH_AUTO = 0")
    s.wr(fb(0x310), 1, "REFRESH")
    s.mask(fb(0x200), 0x80000000, 0x80000000)
    s.wr(fb(0x314), 1, "PRECHARGE")
    s.mask(fb(0x200), 0x80000000, 0)
    s.wr(fb(0x090), 0x00000061)
    s.wr(fb(0x090), 0xc000007f, "self-refresh enter")
    s.wr(fb(0x698), 0)
    s.wr(fb(0x69c), 0)
    s.mask(0x132040, 0x00010000, 0)

    # 2. the clock (r1373f4_init / _fini, from = 1, mode = 1)
    print("  [clock] refpll %08x -> %08x" % (regs[0x132024], coef))
    s.mask(0x1373f4, 0, 0x00010010)
    s.mask(0x1373f4, 0x00000003, 0, "mclk source off")
    s.mask(0x1373f4, 0x00000010, 0)
    if s.rd(0x132024) != coef or (s.rd(0x132034) & 0xffff) != fn:
        s.mask(0x132000, 1, 0)
        s.mask(0x132020, 1, 0, "refpll off")
        s.wr(0x137320, 0)
        s.mask(0x132030, 0xffff0000, fn << 16)
        s.mask(0x132034, 0x0000ffff, fn)
        s.wr(0x132024, coef, "refpll coefficients")
        s.mask(0x132028, 0x00080000, 0x00080000)
        s.mask(0x132020, 1, 1, "refpll on")
        s.wait(0x137390, 0x00020000, 0x00020000, why="refpll lock")
        s.mask(0x132028, 0x00080000, 0)
    s.mask(0x1373f4, 0, 0x00010100)
    s.mask(0x1373f4, 0, 0x00000010)
    s.mask(0x1373ec, 0x00030000, rc["r03_30"] << 16)
    s.mask(0x1373f0, 0x00000002, 0)
    s.mask(0x1373f4, 0x00000003, 0x00000001, "mclk source = refpll")
    s.mask(0x1373f4, 0x00010000, 0)
    s.mask(fb(0x800), 0x00000030, (rc["r03_c0"] ^ rc["r03_30"]) << 4)

    # 3. configuration for the band (the subset whose formula reproduces the boot state)
    print("  [config] band %d settings and timings" % band)
    if rc["r07_40"]:
        s.mask(fb(0x670), 0x80000000, 0x80000000)
    s.wr(fb(0x65c), 0x11 * rm["m11_0c"], "active bank")
    s.wr(fb(0x6b8), 0x01010101 * rc["r09"])
    s.wr(fb(0x6bc), 0x01010101 * rc["r09"])
    if not rc["r07_08"] and not rc["r07_04"]:
        s.wr(fb(0x698), 0x01010101 * rc["r04"]); s.wr(fb(0x69c), 0x01010101 * rc["r04"])
    elif not rc["r07_08"]:
        s.wr(fb(0x698), 0); s.wr(fb(0x69c), 0)
    s.mask(fb(0x694), 0xff00ff00, 0x01000100 * rc["r04"])
    s.mask(fb(0x60c), 0x00000080, 0)
    s.mask(fb(0x200), 0x00001000, 0 if rc["r01_08"] else 0x1000)
    if s.rd(fb(0x670)) & 0x80000000:
        time.sleep(10e-6)
        s.mask(fb(0x670), 0x80000000, 0)
    s.mask(fb(0x82c), 0x00100000, 0x00100000 if rc["r08_01"] else 0)
    s.mask(fb(0x830), 0x00007000, (rc["r08_08"] << 13) | (rc["r08_04"] << 12) | (rc["r08_02"] << 14))
    w = tm["w"]
    for reg, word in ((0x248, 10), (0x290, 0), (0x294, 1), (0x298, 2), (0x29c, 3), (0x2a0, 4), (0x2a4, 5),
                      (0x2a8, 6), (0x2ac, 7), (0x2cc, 8), (0x2e8, 9)):
        s.mask(fb(reg), 0xffffffff, w[word], "timing[%d]" % word)
    mask, data = 0x33f00000, 0
    if not rc["r01_04"]:
        data |= 0x20200000
    if not rc["r07_80"]:
        data |= 0x12800000
    if rc["r03_f0"]:
        if rm["m08_0c"]:
            if not rc["r07_80"]:
                mask |= 0x20
            else:
                data |= 0x20
            mask |= 0x4
    else:
        mask |= 0x40000020; data |= 0x4
    s.mask(fb(0x808), mask, data)
    s.wr(fb(0x870), 0x11111111 * rc["r03_0f"]) if s.rd(fb(0x870)) != 0x11111111 * rc["r03_0f"] else None
    s.mask(0x100778, 0x00000700, tm["t30_07"] << 8)
    s.mask(fb(0x250), 0x000003f0, tm["t2c_003f"] << 4)
    s.mask(fb(0x24c), 0x7f000000, max((w[10] & 0x7f000000) >> 24, tm["t2c_1fc0"]) << 24)
    s.mask(fb(0x224), 0x001f0000, tm["t30_f8"] << 16)

    # 4. out of self-refresh
    print("  [exit] self-refresh exit, refresh on")
    s.wr(fb(0x090), 0x4000007e, "self-refresh exit")
    time.sleep(2e-6)
    s.wr(fb(0x314), 1, "PRECHARGE")
    s.wr(fb(0x310), 1, "REFRESH")
    s.mask(fb(0x210), 0x80000000, 0x80000000, "REFRESH_AUTO = 1")

    # 5. mode registers (nvkm_gddr5_calc: MR0 / MR3 / MR8 reproduce the boot values)
    # MR3 bit 5: nouveau clears it from 1 GHz up ("XXX" in nvkm_gddr5_calc - a guess). On this
    # card's refpll path it stays as booted (1): that is what passed the VRAM tests at 1080 and
    # 1134 MHz (the 1296 attempt with it cleared did not train). --rq=N overrides.
    rq = 1
    for a in sys.argv:
        if a.startswith("--rq="):
            rq = int(a[5:])
    CL, WR, WL = tm["CL"] - 5, tm["WR"] - 4, tm["WL"]
    mr3 = s.rd(fb(0x338)); s.mask(fb(0x338), 0x020, rq << 5, "MR3") if (mr3 >> 5 & 1) != rq else None
    s.mask(fb(0x300), 0xf7f, ((WR & 0xf) << 8) | ((CL & 0xf) << 3) | (WL & 7), "MR0 WL %d CL %d WR %d" % (WL, CL + 5, WR + 4))
    xd = 0 if rc["DLLoff"] else 1   # MR1 = termination / drive; formula reproduces the boot value 0x030
    s.mask(fb(0x330), 0x0bf, (xd << 7) | (tm["t2e_c0"] << 4) | (tm["t2e_03"] << 2) | tm["t2f_03"], "MR1")
    s.mask(fb(0x354), 0x003, ((WR & 0x10) >> 3) | ((CL & 0x10) >> 4), "MR8")
    time.sleep(1e-6)
    s.mask(fb(0x200), 0x80000000, 0x80000000)
    s.wr(fb(0x318), 1, "NOP")
    s.mask(fb(0x200), 0x80000000, 0)
    s.mask(fb(0x830), 0, 0x00000001)

    # 6. retrain at the new clock
    train_tail(s, fbpas, rc, rm, soft=soft)
    print("  %d writes" % s.n)
    if live:
        after = snapshot(c)
        print("after: %s" % mclk_khz(after))
        dump_regs(after, regs)


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else "status"
    v = VBIOS(open(ROM, "rb").read())
    if cmd == "vbios":
        show_vbios(v)
        return 0
    from card import Card
    c = Card()
    strap_reg = c.rd(0x101000)
    idx, strap, xlat = ramcfg_index(v, strap_reg)
    print("0x101000 = %#010x: strap %d -> ramcfg index %d (BIT M xlat @%#x)" % (strap_reg, strap, idx, xlat))
    if cmd == "get":
        print(live_mclk_mhz(c))
        return 0
    if cmd == "status":
        show_vbios(v, idx)
        regs = snapshot(c)
        print("live clock: %s" % mclk_khz(regs))
        dump_regs(regs)
        return 0
    if cmd == "retrain":
        retrain(c, v, idx, int(sys.argv[2]), live="--live" in sys.argv,
                patches=cli_patches(sys.argv), soft="--soft" in sys.argv)
        return 0
    if cmd == "set":
        mhz = int(sys.argv[2])
        live = "--live" in sys.argv
        # Big jumps fail training (1620 -> 1296 did; 1242 -> 1296 did not): ramp in <= 54 MHz
        # steps of 27 MHz multiples, retraining at each, unless --jump.
        cur = live_mclk_mhz(c)
        steps = [mhz]
        if live and "--jump" not in sys.argv and cur:
            steps = []
            if cur <= 540 < mhz:                 # out of band 0: the validated first step
                steps.append(810); cur = 810
            sgn = 1 if mhz > cur else -1
            steps += [f for f in range(cur + sgn * 54, mhz, sgn * 54)] + ([mhz] if mhz != cur or not steps else [])
            print("ramping %d -> %d MHz via %s" % (live_mclk_mhz(c), mhz, steps))
        for f in steps:
            reclock(c, v, idx, f, live=live, patches=cli_patches(sys.argv), soft="--soft" in sys.argv)
        return 0
    print("unknown command %r" % cmd)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
