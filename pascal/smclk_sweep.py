#!/usr/bin/env python3
"""SM-clock register sweep: one variable at a time, measured, always restored.

Why a sweep and not a decode: the GP102 GPC clock is FLL/ADC and the VBIOS's PMU app sets it at boot
(pascal/devinit.py loads that app; there is no host init script that programs it), so -- exactly like
the memory clock before pascal/mclk.py existed -- the only way in is a measured register experiment.
The memory precedent is the argument for trying: the memory clock is PMU-set at boot too, and host
writes moved it 405 -> 1080 MHz.

Each candidate is a list of (address, value) writes.  For each one this script:

  1. probes the registers with a SAME-VALUE write and reads back -- if a register does not read back
     what was written, the candidate is skipped (no point risking a value write on a locked register);
  2. reads all four clock counters (gpc2clk, sys2clk, xbar2clk, dramdiv4) with mclk_measure's method;
  3. applies the candidate;
  4. reads the counters again and decides:
       moved      -> report which counter moved and by how much, KEEP it, and stop: a hit deserves a
                     careful look (memtest, then the bench) before sweeping further;
       unchanged  -> restore the previous values and carry on;
       out of range (below 0.5x or above 2x the previous reading) -> restore at once, report unsafe.

A register read-back only ever proved that the write landed -- RESULTS.md section 15.  The counters are
the judge.  Requires the card idle (`bin/local-gpu stop --all`); the snapshot in --snapshot is written
before the first write and every candidate restores from the live pre-values, not from the file.

    ~/.venvs/tinygpu/bin/python pascal/smclk_sweep.py --snapshot /tmp/sweep-before.json
    ~/.venvs/tinygpu/bin/python pascal/smclk_sweep.py --only 'pair N+1'      # one candidate
    ~/.venvs/tinygpu/bin/python pascal/smclk_sweep.py --list
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from card import Card                                                          # noqa: E402
import mclk_measure                                                            # noqa: E402
import smclk                                                                   # noqa: E402

COUNTERS = ["gpc2clk", "sys2clk", "xbar2clk", "dramdiv4_clk1"]

# The blocks found by smclk.py --scan: a PLL-shaped pair at 0x137020/24 and a second at 0x137040/44
# (each a control word + a coefficient that decodes on mclk.py's layout as M=1, N=0x42, P=1), and the
# divider-shaped words in the 0x137300 block.  dot1 was already tried alone and moved nothing.
A = 0x137000
CANDIDATES = [
    # both copies of the coefficient together (the second may be a shadow the hardware actually reads)
    ("pair N+1",         [(A + 0x24, 0x10814301), (A + 0x44, 0x10814301)]),
    # a strobe/update bit in the control word (bit 0 is set in the memory refpll's control, not here)
    ("ctrl bit0",        [(A + 0x20, 0x00030011)]),
    # re-latch: write the coefficient, then the control word again (some PLLs load on the control write)
    ("relatch N+1",      [(A + 0x24, 0x10814301), (A + 0x20, 0x00030010)]),
    # the second pair's control word, same idea
    ("ctrl2 bit0",       [(A + 0x40, 0x00010013)]),
    ("relatch2 N+1",     [(A + 0x44, 0x10814301), (A + 0x40, 0x00010012)]),
    # field-by-field on the coefficient: N is the multiplier on mclk.py's layout, so walk it up and down
    ("coef N-1",         [(A + 0x24, 0x10814101)]),
    ("coef P+1",         [(A + 0x24, 0x10824201)]),
    # the 0x137300 block: divider-shaped words, one field at a time
    ("139310 low+1",     [(0x137310, 0x87101617)]),
    ("139310 b1+1",      [(0x137310, 0x87101716)]),
    ("139330 low+1",     [(0x137330, 0x87100635)]),
    ("139330 b1+1",      [(0x137330, 0x87100734)]),
    # the block's first register (0x20000103) with its low bits stepped
    ("137300 low+1",     [(0x137300, 0x20000104)]),
]


def read_counters(c, samples=3):
    """Median of `samples` reads per counter: the counters jitter by a count or two run to run
    (0.01-0.08%), and a single-count difference is not a clock change."""
    got = {k: [] for k in COUNTERS}
    for _ in range(samples):
        for name in COUNTERS:
            n, ok = mclk_measure.measure(c, name, n=1000, delay_ms=2.0)
            if not ok:
                raise RuntimeError("%s counter cfg did not restore" % name)
            got[name].append(n)
    return {k: sorted(v)[len(v) // 2] for k, v in got.items()}


MOVED_PCT = 0.5          # below this the counters' own jitter explains it; way below any real change


def probe(c, addr, log):
    """Same-value write: does the host own this register at all?"""
    v = c.rd(addr)
    c.wr(addr, v)
    back = c.rd(addr)
    if back != v:
        log("    probe %#08x: %#010x -> %#010x  NOT writable, skipping" % (addr, v, back))
        return False
    log("    probe %#08x: %#010x  writable" % (addr, v))
    return True


def run(c, log, only=None):
    log("SM clock sweep: %d candidates, counters %s" % (len(CANDIDATES), " ".join(COUNTERS)))
    hits, unsafe = [], []
    for name, writes in CANDIDATES:
        if only and only not in name:
            continue
        log("\n== %s: %s" % (name, ", ".join("%#08x=%#010x" % w for w in writes)))
        if not all(probe(c, a, log) for a, _ in writes):
            continue
        before = read_counters(c)
        prev = {a: c.rd(a) for a, _ in writes}
        for a, v in writes:
            c.wr(a, v)
        time.sleep(0.02)
        after = read_counters(c)
        moved = {k: (before[k], after[k]) for k in COUNTERS
                 if abs(after[k] - before[k]) > before[k] * MOVED_PCT / 100.0}
        bad = {k: (before[k], after[k]) for k in COUNTERS
               if after[k] < before[k] * 0.5 or after[k] > before[k] * 2.0}
        if bad:
            log("    UNSAFE: %s -> restoring" % ", ".join("%s %d->%d" % (k, *v) for k, v in bad.items()))
            for a, v in prev.items():
                c.wr(a, v)
            unsafe.append((name, bad))
            break
        if moved:
            log("    MOVED: %s" % ", ".join("%s %d -> %d (%.2f%%)"
                                           % (k, v[0], v[1], 100.0 * (v[1] - v[0]) / v[0])
                                           for k, v in moved.items()))
            hits.append((name, writes, moved))
            log("    keeping it -- memtest and the bench next, then decide")
            break
        log("    no counter moved (gpc %d, dram %d)" % (after["gpc2clk"], after["dramdiv4_clk1"]))
        for a, v in prev.items():
            c.wr(a, v)
        time.sleep(0.01)
    log("\n---- summary ----")
    log("hits: %s" % (hits or "none"))
    log("unsafe: %s" % (unsafe or "none"))
    if not hits:
        log("no candidate moved a clock: the GPC source is still not located by this block")
    return hits, unsafe


def main(argv=None):
    ap = argparse.ArgumentParser(description="measured SM-clock register sweep")
    ap.add_argument("--snapshot", metavar="FILE", help="write the 413-register snapshot before any write")
    ap.add_argument("--only", metavar="SUBSTR", help="run only candidates whose name contains this")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--log", metavar="FILE", default=None)
    a = ap.parse_args(argv)

    if a.list:
        for name, writes in CANDIDATES:
            print("%-14s %s" % (name, ", ".join("%#08x=%#010x" % w for w in writes)))
        return 0

    lines = []

    def log(msg):
        print(msg)
        lines.append(msg)

    c = Card()
    if a.snapshot:
        regs, ck = smclk.snapshot(c)
        n = sum(len(v) for v in regs.values())
        with open(a.snapshot, "w") as f:
            json.dump({"card": "GTX 1080 Ti (GP102)", "when": time.ctime(), "checksum": ck,
                       "regs": regs}, f, indent=1)
        log("snapshot: %d registers -> %s (checksum %s)" % (n, a.snapshot, ck))
    log("GPC before: %.1f MHz" % smclk.verify(c)[0])
    hits, unsafe = run(c, log, a.only)
    log("GPC after : %.1f MHz" % smclk.verify(c)[0])
    if a.log:
        os.makedirs(os.path.dirname(a.log), exist_ok=True)
        with open(a.log, "a") as f:
            f.write("\n".join(lines) + "\n")
        print("logged to %s" % a.log)
    return 0 if not unsafe else 2


if __name__ == "__main__":
    sys.exit(main())
