"""Port of nouveau gm200_devinit_post(): run the VBIOS PMU DEVINIT app (POSTs the card, trains GDDR5X)."""
import sys, time; sys.path.insert(0, __file__.rsplit("/", 1)[0])
from card import Card
from vbios import VBIOS

def falcon_reset(c, base):
    c.mask(base + 0x048, 3, 0); c.wr(base + 0x014, 0xffffffff)          # gm200_flcn_disable
    c.mask(base + 0x3c0, 1, 1); time.sleep(0.001); c.mask(base + 0x3c0, 1, 0)   # gp102_flcn_reset_eng
    ok = c.wait(lambda: not (c.rd(base + 0x10c) & 6), 0.5)              # mem scrubbing
    c.wr(base + 0x084, c.rd(0))
    return ok

def pmu_code(c, v, pmu, img, ln, sec):
    c.wr(0x10a180, 0x01000000 | (0x10000000 if sec else 0) | pmu)
    i = 0
    while i < ln:
        if (i & 0xff) == 0: c.wr(0x10a188, (pmu + i) >> 8)
        c.wr(0x10a184, v.rd32(img + i)); i += 4
    while i & 0xff: c.wr(0x10a184, 0); i += 4

def pmu_data(c, v, pmu, img, ln):
    c.wr(0x10a1c0, 0x01000000 | pmu)
    for i in range(0, ln, 4): c.wr(0x10a1c4, v.rd32(img + i))

def pmu_args(c, argp, argi):
    c.wr(0x10a1c0, argp); c.wr(0x10a1c0, c.rd(0x10a1c4) + argi); return c.rd(0x10a1c4)

def pmu_exec(c, addr):
    c.wr(0x10a104, addr); c.wr(0x10a10c, 0); c.wr(0x10a100, 2)

def pmu_load(c, v, typ):
    r = v.pmuR(typ)
    assert falcon_reset(c, 0x10a000), "PMU scrub timeout"
    pmu_code(c, v, r["boot_addr_pmu"], r["boot_addr"], r["boot_size"], False)
    pmu_code(c, v, r["code_addr_pmu"], r["code_addr"], r["code_size"], True)
    pmu_data(c, v, r["data_addr_pmu"], r["data_addr"], r["data_size"])
    return r

def snap(c, tag):
    print("[%s] PMU cpuctl=%08x sctl=%08x mbox0=%08x pc(trace)=%08x | FB 0x100ce0=%08x PFB_CFG0=%08x PRAMIN=%08x 0x12004c=%08x"
          % (tag, c.rd(0x10a100), c.rd(0x10a240), c.rd(0x10a040), c.rd(0x10a14c), c.rd(0x100ce0), c.rd(0x100200), c.rd(0x700000), c.rd(0x12004c)))

def post(c, v):
    I = v.bit('I'); assert I["version"] == 1 and I["length"] >= 0x1c
    r = pmu_load(c, v, 0x04)
    exec_, args = r["init_addr_pmu"], r["args_addr_pmu"]
    p = pmu_args(c, args + 0x08, 0x08); pmu_data(c, v, p, v.rd16(I["offset"] + 0x14), v.rd16(I["offset"] + 0x16))
    p = pmu_args(c, args + 0x08, 0x10); pmu_data(c, v, p, v.rd16(I["offset"] + 0x18), v.rd16(I["offset"] + 0x1a))
    c.wr(0x10a040, 0x00005000)
    snap(c, "loaded")
    t0 = time.time(); pmu_exec(c, exec_)
    ok = c.wait(lambda: c.rd(0x10a040) & 0x2000, 2.0)
    print("DEVINIT %s in %.0f ms" % ("COMPLETED" if ok else "TIMED OUT", (time.time() - t0) * 1000))
    snap(c, "after devinit")
    if ok:   # gm200_devinit_preos: PRE_OS app (fan control etc.)
        pmu_load(c, v, 0x01); pmu_exec(c, v.pmuR(0x01)["init_addr_pmu"]); time.sleep(0.2); snap(c, "preos")
    return ok

if __name__ == "__main__":
    c = Card(); v = VBIOS(open(sys.argv[1], "rb").read())
    snap(c, "before")
    post(c, v)
