"""Which PMC_ENABLE bit makes the poisoned register blocks live?

The ACR app reads 0x10e5f8 (PM block, 0x10e000) and expects bits[23:20] to be
zero before it programs them; on this card the read returns PRI poison
(0xbadf51xx = "no target for that address").  Same for 0x100e10 (FB/MMU fault
block), 0x140100 (LTC0) and 0x17e0/0x17e8 (LTC CBC).

The rule this project already proved: poison at a window means the block is
switched off, and its PMC_ENABLE bit turns it back on.  So: for every PMC_ENABLE
bit that is currently clear, set it alone, re-read the four windows, and report
which bit makes each live.  PMC_ENABLE is restored afterwards.

Read-only apart from PMC_ENABLE (saved and restored).
"""

import sys

sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
from card import Card

WATCH = [("PM   0x10e5e4", 0x10E5E4), ("PM   0x10e5f8", 0x10E5F8),
         ("FB   0x100e10", 0x100E10), ("LTC0 0x140100", 0x140100),
         ("LTC  0x17e0", 0x17E0), ("PRI  0x100a2c", 0x100A2C),
         ("LTC  0x17e278", 0x17E278)]
POISON = 0xBAD00000, 0xBAE00000


def poison(v):
    return any((v & 0xFFFF0000) == p for p in POISON)


def main():
    c = Card()
    save = c.rd(0x200)
    print("PMC_ENABLE = %08x" % save)
    base = [(n, a, c.rd(a)) for n, a in WATCH]
    for n, a, v in base:
        print("  %s = %08x %s" % (n, v, "POISON" if poison(v) else ""))
    print()

    for bit in range(32):
        if save & (1 << bit):
            continue
        c.wr(0x200, save | (1 << bit))
        changed = []
        for (n, a, was) in base:
            v = c.rd(a)
            if v != was:
                changed.append("%s %08x->%08x%s" % (n, was, v, " *live*" if not poison(v) else ""))
        print("bit %2d set -> %s" % (bit, ", ".join(changed) if changed else "(no change)"))
        c.wr(0x200, save)
    print("\nPMC_ENABLE restored: %08x" % c.rd(0x200))
    for bit, name in ((25, "LTC"), (24, "PMU"), (27, "FB/hub")):
        c.wr(0x200, save | (1 << bit))
        print("with bit %d (%s): %s" % (bit, name, "  ".join(
            "%s=%08x" % (n, c.rd(a)) for n, a, _ in base)))
        c.wr(0x200, save)
    print("PMC_ENABLE restored: %08x" % c.rd(0x200))


if __name__ == "__main__":
    main()
