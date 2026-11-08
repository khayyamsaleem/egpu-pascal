"""Phase 0.4 probe: enable the engine bits and map every Falcon on the card.

Probe 5 established that "PRI poison" at an engine's register window means the
engine is switched off in PMC_ENABLE — enabling PGRAPH made FECS @0x409000
decode (HWCFG=0x20202060, IDLESTATE=0x4, the same idle signature as the PMU).

This widens the hunt: enable every documented GF100+/GM107+ engine bit, scan the
whole register space in 4 KB pages for Falcon blocks, then restore PMC_ENABLE.
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
ENGINE_BITS = {5: "PRING", 8: "PFIFO", 12: "PGRAPH", 13: "PDAEMON", 14: "PSEC",
               15: "PVDEC", 16: "PTIMER", 18: "PVENC", 20: "PBFB", 21: "PCOPY2",
               28: "PCOUNTER", 29: "PFFB", 30: "PDISPLAY"}

orig = rd(PMC_ENABLE)
en = orig
for b in ENGINE_BITS:
    en |= 1 << b
wr(PMC_ENABLE, en)
print("PMC_ENABLE 0x%08x -> 0x%08x (readback 0x%08x)" % (orig, en, rd(PMC_ENABLE)))

def falcon_at(base):
    hw, idle, ctl, vec, dmactl = (rd(base + o) for o in (0x108, 0x048, 0x100, 0x104, 0x10c))
    return sum(live(v) for v in (hw, idle, ctl, vec, dmactl)) >= 3, hw, idle, vec, ctl

print("\n=== falcon scan (4 KB pages, 5-register signature) ===")
ranges = [(0x000000, 0x002000), (0x100000, 0x102000), (0x10a000, 0x10b000),
          (0x400000, 0x440000), (0x600000, 0x620000), (0x800000, 0x900000)]
found = []
for start, end in ranges:
    hits = []
    for base in range(start, end, 0x1000):
        ok, hw, idle, vec, ctl = falcon_at(base)
        if ok:
            hits.append((base, hw, idle, vec, ctl))
    found += hits
    print("  0x%06x-0x%06x : %s" % (start, end, ", ".join("0x%06x" % h[0] for h in hits) or "none"))

print("\n=== falcon details ===")
for base, hw, idle, vec, ctl in found:
    print("  @0x%06x HWCFG=0x%08x IDLESTATE=0x%08x BOOTVEC=0x%08x CPUCTL=0x%08x" % (base, hw, idle, vec, ctl))

print("\n=== known-name mapping ===")
names = {"PMU": 0x10a000, "SEC2": 0x840000, "FECS": 0x409000, "GPCCS": 0x41a000}
for nm, base in names.items():
    ok, hw, idle, vec, ctl = falcon_at(base)
    print("  %-6s @0x%06x -> %s (HWCFG=0x%08x%s)" % (nm, base, "DECODED" if ok else "poison/blank",
          hw, "" if not poison(hw) else " = %s" % hex(hw)))

print("\n=== restore ===")
wr(PMC_ENABLE, orig)
print("  PMC_ENABLE = 0x%08x (orig 0x%08x) %s" % (rd(PMC_ENABLE), orig, "restored" if rd(PMC_ENABLE) == orig else "MISMATCH"))
