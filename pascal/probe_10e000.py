"""Is 0x10e000 a decoded block at all, and how does it compare with PDAEMON?

The ACR app reads 0x10e5f8 through SEC2's port and requires bits[23:20] == 0.
It is 4 on this card, neither the host nor the falcon can write it, and nouveau
never touches 0x10e000.  Two readings are possible and this probe separates them:

  * the block is decoded, and 4 is real state that some earlier init should have
    left at 0 (documented reset value);
  * the block is *not* decoded, and 0x10e5f8 returns a fixed sentinel -- in which
    case the ACR app is talking to hardware that has not been brought up at all,
    which is what the missing PMU firmware would explain (nouveau's own
    gp100_ltc_init is an empty function with "XXX: PMU LS call to setup tagram
    address", and 0x17e278 on this card reads 0 for the same reason).

A fixed value across many offsets, none of them writable, means sentinel.
0x10a5f8 is read as a control: envytools documents PDAEMON UNK5F8 as unbounded
but with reset value 0, so a real (decoded) register should not read like poison.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from card import Card                                          # noqa: E402
import acr                                                     # noqa: E402
import falcon_peek as fp                                       # noqa: E402

SWEEP_10E000 = [0x10E000, 0x10E040, 0x10E100, 0x10E200, 0x10E400,
                0x10E5F0, 0x10E5F4, 0x10E5F8, 0x10E5FC, 0x10E600, 0x10E700, 0x10EF00]
SWEEP_10A000 = [0x10A000, 0x10A040, 0x10A100, 0x10A240, 0x10A5F0, 0x10A5F4,
                0x10A5F8, 0x10A5FC, 0x10A600, 0x10A700, 0x10AF00]


def main():
    c = Card()
    f = acr.Falcon(c, acr.SEC2, pmc_bit=14)
    f.reset()
    f.mask(0x624, 0x80, 0x80)
    f.w(0x10c, 0)
    for name, base, addrs in (("0x10e000 (the app's block)", 0x10E000, SWEEP_10E000),
                              ("0x10a000 (PDAEMON/PMU)", 0x10A000, SWEEP_10A000)):
        vals, host = [], []
        for a in addrs:
            v, _cr, _st, _ok = fp.peek(c, f, a, settle=0.01, verbose=False)
            vals.append(v)
            host.append(c.rd(a))
        print("\n-- %s" % name)
        for a, v, h in zip(addrs, vals, host):
            print("   %#010x falcon=%08x host=%08x %s"
                  % (a, v, h, "<- all-same" if len(set(vals)) == 1 else ""))
        print("   distinct falcon values across the sweep: %d" % len(set(vals)))
    f.reset()
    print("\nSEC2 reset.")


if __name__ == "__main__":
    main()
