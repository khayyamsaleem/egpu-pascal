"""Phase 0.3 probe: is "poison" just "engine disabled"?

envytools' PMC doc says of PMC_ENABLE: "Most engines disappear from MMIO space
and reset to default state when disabled." Our earlier reads found PMU
(0x10a000) decoding but SEC2 (0x840000) and FECS/GPCCS (0x409000/0x41a000)
returning PRI poison — and PMC_ENABLE came up 0x40002020 (only PRING, PDAEMON
and PDISPLAY set). So the poison may simply mean those blocks are switched off.

This test writes a single bit at a time, reads back, and restores the original
value at the end. Every step is printed so the card's state is accounted for.

GF100+ PMC_ENABLE bit map (envytools pmc.rst):
  5 PRING  8 PFIFO  12 PGRAPH  13 PDAEMON  14 PSEC [GM107+]  15 PVDEC [GM107+]
  16 PTIMER  18 PVENC  20 PBFB  21 PCOPY[2]  28 PCOUNTER  29 PFFB  30 PDISPLAY
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
BITS = {5: "PRING", 8: "PFIFO", 12: "PGRAPH", 13: "PDAEMON", 14: "PSEC",
        15: "PVDEC", 16: "PTIMER", 18: "PVENC", 20: "PBFB", 21: "PCOPY2",
        28: "PCOUNTER", 29: "PFFB", 30: "PDISPLAY"}

orig = rd(PMC_ENABLE)
print("PMC_ENABLE = 0x%08x" % orig)
print("  set:   " + ", ".join("%s(%d)" % (n, b) for b, n in sorted(BITS.items())
                               if orig >> b & 1))
print("  clear: " + ", ".join("%s(%d)" % (n, b) for b, n in sorted(BITS.items())
                               if not orig >> b & 1))

print("\n=== write-path capability check (rewrite the same value) ===")
try:
    wr(PMC_ENABLE, orig)
    back = rd(PMC_ENABLE)
    print("  wrote 0x%08x, read back 0x%08x -> %s" % (orig, back, "WRITES WORK" if back == orig else "MISMATCH"))
except Exception as e:
    print("  write failed: %s: %s" % (type(e).__name__, e))
    raise SystemExit(1)

def probe_block(label, base):
    vals = {off: rd(base + off) for off in (0x048, 0x100, 0x104, 0x108, 0x10c)}
    n = sum(live(v) for v in vals.values())
    state = "DECODED (%d/5 live)" % n if n >= 2 else ("poison/blank (%d/5)" % n)
    print("  %-22s @0x%06x  %s  HWCFG=0x%08x IDLE=0x%08x" % (label, base, state, vals[0x108], vals[0x048]))
    return n >= 2

print("\n=== before: SEC2 / FECS / GPCCS ===")
probe_block("SEC2", 0x840000)
probe_block("FECS", 0x409000)
probe_block("GPCCS", 0x41a000)

for bit, name in ((14, "PSEC"), (12, "PGRAPH")):
    print("\n=== enabling bit %d (%s) ===" % (bit, name))
    wr(PMC_ENABLE, rd(PMC_ENABLE) | (1 << bit))
    print("  PMC_ENABLE now 0x%08x" % rd(PMC_ENABLE))
    probe_block("SEC2", 0x840000)
    probe_block("FECS", 0x409000)
    probe_block("GPCCS", 0x41a000)

print("\n=== falcon hunt across the SEC2 / GR apertures while enabled ===")
for start, end in ((0x800000, 0x8c0000), (0x400000, 0x420000), (0x100000, 0x102000)):
    found = []
    for base in range(start, end, 0x1000):
        hw, idle, ctl = rd(base + 0x108), rd(base + 0x048), rd(base + 0x100)
        if sum(live(v) for v in (hw, idle, ctl)) >= 2:
            found.append((base, hw, idle, ctl))
    print("  0x%06x-0x%06x: %s" % (start, end, ", ".join(
        "0x%06x(HWCFG=0x%08x)" % (b, hw) for b, hw, _, _ in found) or "none"))

print("\n=== restoring original PMC_ENABLE ===")
wr(PMC_ENABLE, orig)
print("  PMC_ENABLE = 0x%08x (original 0x%08x) -> %s"
      % (rd(PMC_ENABLE), orig, "restored" if rd(PMC_ENABLE) == orig else "MISMATCH"))
