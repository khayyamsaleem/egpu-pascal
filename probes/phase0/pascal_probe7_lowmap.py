"""Phase 0.5 probe: full falcon map of the low apertures.

probe6 found the PMU (0x10a000) and FECS (0x409000) falcons but nothing in the
0x800000-0x900000 SEC2 aperture, where nouveau's *GP102 ACR load* runs. One
hobbyist GV100 report puts SEC2 at a "corrected" base of 0x87000 in the low
aperture, which probe6 never scanned (it only looked at 0x0-0x2000).

So: enable the engine bits, then scan every 4 KB page of 0x0-0x100000 and the
0x400000/0x800000 apertures, with a distinctness test to reject pages that just
mirror one value (probe6's 0x61a000/0x61b000 hits were false positives of that
kind). PMC_ENABLE is restored at the end.
"""
import os

os.environ.setdefault("APL_REMOTE_SOCK", "/tmp/tinygpu.sock")

from tinygrad.runtime.support.system import APLRemotePCIDevice

dev = APLRemotePCIDevice("nv", "usb4")
bar0 = dev.map_bar(0, fmt="I")
def rd(a): return bar0[a // 4]
def wr(a, v): bar0[a // 4] = v
def poison(v): return (v >> 16) in (0xBADF, 0xBAD0)
def live(v): return not poison(v) and v not in (0, 0xFFFFFFFF)

PMC_ENABLE = 0x000200
BITS = (5, 8, 12, 13, 14, 15, 16, 18, 20, 21, 28, 29, 30)
orig = rd(PMC_ENABLE)
wr(PMC_ENABLE, orig | sum(1 << b for b in BITS))
print("PMC_ENABLE 0x%08x -> wrote 0x%08x, readback 0x%08x"
      % (orig, orig | sum(1 << b for b in BITS), rd(PMC_ENABLE)))

def falcon_at(base):
    vals = [rd(base + o) for o in (0x048, 0x100, 0x104, 0x108, 0x10c)]
    if sum(live(v) for v in vals) < 3: return None
    if len(set(vals)) == 1: return None          # mirrored page = false positive
    return vals

print("\n=== scan ===")
hits = []
for start, end in ((0x000000, 0x100000), (0x400000, 0x440000), (0x800000, 0x900000),
                   (0x100000, 0x102000), (0x10a000, 0x10b000)):
    found = []
    for base in range(start, end, 0x1000):
        vals = falcon_at(base)
        if vals: found.append((base, vals))
    hits += found
    print("  0x%06x-0x%06x : %s" % (start, end, ", ".join("0x%06x" % b for b, _ in found) or "none"))

print("\n=== details (IDLESTATE / CPUCTL / BOOTVEC / HWCFG / DMACTL) ===")
for base, vals in hits:
    idle, ctl, vec, hw, dma = vals
    print("  @0x%06x idle=0x%08x ctl=0x%08x vec=0x%08x HWCFG=0x%08x dma=0x%08x" % (base, idle, ctl, vec, hw, dma))

print("\n=== SEC2 hunt in the low aperture specifically ===")
for base in (0x84000, 0x87000, 0x88000, 0x840000, 0x8c0000):
    vals = falcon_at(base)
    print("  @0x%06x -> %s" % (base, "FALCON-ish %s" % [hex(v) for v in vals] if vals
                               else "poison (HWCFG=0x%08x)" % rd(base + 0x108)))

print("\n=== restore ===")
wr(PMC_ENABLE, orig)
print("  PMC_ENABLE = 0x%08x (orig 0x%08x) %s" % (rd(PMC_ENABLE), orig, "restored" if rd(PMC_ENABLE) == orig else "MISMATCH"))
