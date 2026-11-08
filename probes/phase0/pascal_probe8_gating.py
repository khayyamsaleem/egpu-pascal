"""Phase 0.6 probe: which PMC_ENABLE bit gates which falcon?

probe7 found falcon-shaped windows in the LOW aperture at 0x84000 and 0x87000 —
not at 0x840000 where the community docs put SEC2 (a plausible hex-place typo,
0x84000 * 16 = 0x840000). This walks a few enable combinations and reports
which falcons appear for each, so the gate for each engine is known.

PMC_ENABLE is restored to its power-on value at the end.
"""
import os

os.environ.setdefault("APL_REMOTE_SOCK", "/tmp/tinygpu.sock")

from tinygrad.runtime.support.system import APLRemotePCIDevice

dev = APLRemotePCIDevice("nv", "usb4")
bar0 = dev.map_bar(0, fmt="I")
def rd(a): return bar0[a // 4]
def wr(a, v): bar0[a // 4] = v
def poison(v): return (v >> 16) in (0xBADF, 0xBAD0)

PMC_ENABLE = 0x000200
orig = rd(PMC_ENABLE)
WATCH = {"PMU?": 0x10a000, "FECS": 0x409000, "A(0x84000)": 0x84000,
         "B(0x87000)": 0x87000, "C(0x8b000)": 0x8b000, "GR-old(0x41a000)": 0x41a000}

def show(tag):
    enable = rd(PMC_ENABLE)
    row = []
    for nm, base in WATCH.items():
        hw, idle, ctl = rd(base + 0x108), rd(base + 0x048), rd(base + 0x100)
        alive = (not poison(hw)) and (not poison(idle)) and idle not in (0, 0xFFFFFFFF)
        row.append("%s=%s" % (nm, "LIVE(hw=%08x,idle=%08x)" % (hw, idle) if alive else "off"))
    print("  [%-18s] PMC_ENABLE=0x%08x  %s" % (tag, enable, "  ".join(row)))

print("watched windows:", ", ".join("0x%06x" % b for b in WATCH.values()))
print("original PMC_ENABLE = 0x%08x" % orig)
show("as found")

for tag, bits in (("PSEC(14)", (14,)), ("PGRAPH(12)", (12,)), ("PDAEMON(13)", (13,)),
                  ("PFIFO(8)", (8,)), ("PSEC+PGRAPH", (14, 12)),
                  ("all engines", (5, 8, 12, 13, 14, 15, 16, 18, 20, 21, 28, 29, 30))):
    wr(PMC_ENABLE, orig | sum(1 << b for b in bits))
    show(tag)

print("\n=== restoring ===")
wr(PMC_ENABLE, orig)
print("  PMC_ENABLE = 0x%08x (orig 0x%08x) %s" % (rd(PMC_ENABLE), orig, "restored" if rd(PMC_ENABLE) == orig else "MISMATCH"))
