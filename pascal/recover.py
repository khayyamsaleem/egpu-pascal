"""Recover from a downed PRI ring + garbage PCCSR bindings without a power cycle.

Written after the session-4 incident (a bad TPC count drove GR init loops across
all of MMIO).  Steps, each printed with the state it acts on:
  1. PCCSR: disable + unbind every bound channel except `keep` (host-side FIFO).
  2. PBUS/FIFO: clear the MMIO fault record and pending status.
  3. PRI ring: nvgpu gk20a_priv_ring_enable -- PMC_ENABLE bit 5 (PRIV_RING) off/on,
     0x12004c = 0x4 (enumerate and start), 0x122204 = 0x2 (sys decode config);
     then nouveau gk104_privring_init's timeout settings.
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from card import Card


def pccsr_clean(c, keep=()):
    w = c.bar0[0x800000 // 4:0x808000 // 4]
    n = 0
    for i in range(4096):
        if i in keep or not (w[2 * i] or w[2 * i + 1]):
            continue
        c.mask(0x800004 + i * 8, 0x00000800, 0x00000800)     # ENABLE_CLR
        c.wr(0x800000 + i * 8, 0)
        n += 1
    w = c.bar0[0x800000 // 4:0x808000 // 4]
    left = [i for i in range(4096) if (w[2 * i] or w[2 * i + 1]) and i not in keep]
    return n, left


def ring_reset(c):
    c.mask(0x200, 0x20, 0); c.rd(0x200); time.sleep(0.01)
    c.mask(0x200, 0x20, 0x20); c.rd(0x200); time.sleep(0.01)
    c.wr(0x12004c, 0x4)
    c.wr(0x122204, 0x2); c.rd(0x122204)
    ok = c.wait(lambda: (c.rd(0x12004c) & 0x3f) == 0, 1.0)
    for a, m, v in ((0x122318, 0x3ffff, 0x1000), (0x12231c, 0x3ffff, 0x200), (0x122310, 0x3ffff, 0x800),
                    (0x122348, 0x3ffff, 0x100), (0x1223b0, 0x3ffff, 0xfff), (0x122348, 0x3ffff, 0x200),
                    (0x122358, 0x3ffff, 0x2880)):
        c.mask(a, m, v)
    return ok


def probe(c, tag):
    regs = (("ring cmd 0x12004c", 0x12004c), ("ring intr 0x120058", 0x120058), ("hub/rop/gpc nr", 0x120070),
            ("0x120074", 0x120074), ("0x120078", 0x120078), ("FB 0x100ce0", 0x100ce0), ("LTC 0x12006c", 0x12006c),
            ("PMU cpuctl", 0x10a100), ("SEC2 cpuctl", 0x87100), ("PGRAPH 0x400700", 0x400700),
            ("claim 0x10e5f8", 0x10e5f8), ("temp 0x20460", 0x20460), ("top 0x22700", 0x22700))
    print("[%s] " % tag + "  ".join("%s=%08x" % (n, c.rd(a)) for n, a in regs))


if __name__ == "__main__":
    c = Card()
    keep = tuple(int(x) for x in sys.argv[1:]) or (1,)
    print("before: PMC_ENABLE=%08x PMC_INTR0=%08x PBUS_INTR=%08x FIFO_INTR=%08x 0x252c=%08x"
          % (c.rd(0x200), c.rd(0x100), c.rd(0x1100), c.rd(0x2100), c.rd(0x252c)))
    probe(c, "before")
    n, left = pccsr_clean(c, keep)
    print("PCCSR: unbound %d stray channels; still bound (excluding %s): %s" % (n, keep, left))
    c.wr(0x9084, 0); c.wr(0x1100, 0xe); c.wr(0x2100, c.rd(0x2100))
    print("cleared: PBUS_INTR=%08x FIFO_INTR=%08x" % (c.rd(0x1100), c.rd(0x2100)))
    ok = ring_reset(c)
    print("ring reset: command idle=%s  PMC_ENABLE=%08x PMC_INTR0=%08x PBUS_INTR=%08x" % (ok, c.rd(0x200), c.rd(0x100), c.rd(0x1100)))
    probe(c, "after")
