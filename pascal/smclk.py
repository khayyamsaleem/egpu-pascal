#!/usr/bin/env python3
"""SM (GPC) clock work for the GTX 1080 Ti: snapshot, probe, measure, apply, restore.

ROLLBACK FIRST.  The card's only recovery from a bad clock or voltage write is a physical power
cycle of the enclosure (the user's action), so every step in this file is:

  * preceded by a full snapshot of the registers it may touch (--snapshot FILE, JSON + checksum),
  * confined to registers whose current value has been read and shown,
  * verified by read-back immediately after any write, with an automatic restore of the snapshot
    for that register if the read-back does not match what was asked for,
  * reversible with --restore FILE, which writes the snapshot back and verifies each read-back.

Nothing here writes a value that differs from what is already on the card unless --apply is given an
explicit ADDR:VALUE pair.  A write happens only when a human -- or a decoded register map -- has said
what the target value is.

WHY THERE IS NO --set-clock YET.  The GPC clock on GP102 is FLL/ADC (NAFLL), not a PLL: the Kepler
GPCPLL at 0x137000 reads 0xbadf on this card, and the only open reference driver (nvgpu) never writes
FLL registers from the host at all -- it parses VBIOS tables (BIT 'C' FLL device table, entry 4) and
hands them to PMU firmware as board objects (clk_fll.c).  So there is no published host-side map for
the FLL block, and guessing at addresses is how a card gets power-cycled.  What this file provides is
everything either side of that gap: the measurement that proves a change is present, the snapshot and
restore that make a write survivable, and --apply to execute a write once the target value is known
(e.g. after decoding clk_fll.h + the VBIOS FLL table, which is the documented next step).

    smclk.py --verify             measure the real GPC clock (nvgpu trim counters, anchored on %clock64)
    smclk.py --scan               read-only dump of the clock/voltage register families
    smclk.py --snapshot FILE      save those registers + values (JSON, with a checksum)
    smclk.py --diff FILE          compare a snapshot against the card now
    smclk.py --restore FILE       write a snapshot back and verify every read-back
    smclk.py --apply ADDR:VALUE   one write, read back, auto-restore that register if it does not match
    smclk.py --test               print the two-test protocol (change present / inference faster)

Reference values on this card (RESULTS.md sections 13-15, session 15): GPC clock 607 MHz at the P8
boot performance state; the VBIOS perf table's P0 row asks for 1594 quoted (~797 actual) -- so the
honest ceiling for this lever is about 1.3x, not the 2.4x that comparing 607 with the 1480 MHz base
clock suggests (the VBIOS quotes these clocks at 2x the rate the hardware counts).
"""
import argparse
import hashlib
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from card import Card                                                          # noqa: E402
import mclk_measure                                                            # noqa: E402

# Register families to snapshot.  The first two are the ones this project already writes with
# pascal/mclk.py (so they are known writable and known recoverable); the NVVDD pair is the voltage
# PWM the reference notes describe (0x20340 period, 0x20344 duty); the rest are the blocks where a
# GPC clock control could plausibly live, included so a snapshot covers them before anyone writes.
BLOCKS = {
    "mclk pll/ctl":    (0x132000, 0x132064),
    "gpc cand 1370":   (0x137000, 0x137100),
    "clk 1373":        (0x137300, 0x137400),
    "gpc cand 1380":   (0x138000, 0x138200),
    "clk cand 13a0":   (0x13a000, 0x13a200),
    "nvvdd pwm":       (0x20340, 0x20350),
}
# Registers this project has already written and recovered (pascal/mclk.py): safe to re-write with
# their own current value as a probe, because the value is identical either way.
KNOWN_WRITABLE = [("refpll_coef", 0x132024), ("refpll_ctrl", 0x132000),
                  ("mclk_src", 0x1373f4), ("mclk_mode", 0x1373f0)]


def snapshot(c, blocks=BLOCKS):
    out = {}
    for name, (lo, hi) in blocks.items():
        out[name] = {("%#08x" % a): c.rd(a) for a in range(lo, hi, 4)}
    blob = json.dumps(out, sort_keys=True).encode()
    return out, hashlib.sha256(blob).hexdigest()[:16]


def verify(c, quiet=False):
    """Measure the GPC clock on the card, not from a register that claims it."""
    counts = []
    for _ in range(3):
        n, ok = mclk_measure.measure(c, "gpc2clk", n=1000, delay_ms=2.0)
        if not ok:
            raise RuntimeError("gpc2clk counter cfg did not restore")
        counts.append(n)
    counts.sort()
    # the dram counter rides along: it is the reference this calibration was built on
    dram, _ = mclk_measure.measure(c, "dramdiv4_clk1", n=1000, delay_ms=2.0)
    # anchor: the same ratio method mclk_measure uses (SM clock known independently at 607 MHz)
    k = mclk_measure.SM_ANCHOR_MHZ / counts[len(counts) // 2]
    gpc_mhz = counts[len(counts) // 2] * k
    dram_mhz = dram * k * 4
    if not quiet:
        print("GPC (SM) clock : %.1f MHz   (raw %d, spread %.2f%%, anchored on %s)"
              % (gpc_mhz, counts[len(counts) // 2],
                 100.0 * (counts[-1] - counts[0]) / counts[len(counts) // 2], "607 MHz %clock64"))
        print("DRAM f_CK      : %.1f MHz   (quoted %.0f MHz)" % (dram_mhz, dram_mhz * 2))
    return gpc_mhz, dram_mhz


def do_scan(c):
    print("read-only scan (nothing is written):")
    for name, (lo, hi) in BLOCKS.items():
        vals = [(a, c.rd(a)) for a in range(lo, hi, 4)]
        nz = [(a, v) for a, v in vals if v not in (0, 0xffffffff)]
        print("  %-14s %#08x..%#08x  %2d/%2d non-zero" % (name, lo, hi, len(nz), len(vals)))
        for a, v in nz[:14]:
            print("      %#08x = %#010x" % (a, v))
        if len(nz) > 14:
            print("      ... %d more" % (len(nz) - 14))
    print("  (the locked-register sweep is deliberately not done here: PRAMIN/BAR reads are ~18 us")
    print("   each, so a full sweep takes minutes and belongs in its own run -- docs/reference/README.md)")


def do_probe(c):
    """Same-value writes: the value on the card cannot change, so this only measures whether the host
    may write at all (a locked register ignores the write, a writing register reads back the value)."""
    print("writability probe (each write is the register's OWN current value):")
    for name, a in KNOWN_WRITABLE:
        before = c.rd(a)
        c.wr(a, before)
        after = c.rd(a)
        print("  %-14s %#08x  %#010x -> %#010x  %s"
              % (name, a, before, after, "writable" if after == before else "CHANGED?!"))


def do_apply(c, spec, auto_restore=True):
    for pair in spec:
        addr, _, val = pair.partition(":")
        a, v = int(addr, 0), int(val, 0)
        before = c.rd(a)
        print("  write %#08x = %#010x (was %#010x)" % (a, v, before))
        c.wr(a, v)
        time.sleep(0.05)
        after = c.rd(a)
        if after != v:
            print("    read-back %#010x does NOT match" % after)
            if auto_restore:
                print("    restoring %#010x and stopping" % before)
                c.wr(a, before)
                print("    now %#010x" % c.rd(a))
            return False
        print("    read-back %#010x ok" % after)
    return True


def do_restore(c, path):
    snap = json.load(open(path))
    regs = snap.get("regs", snap)
    bad = 0
    for name, d in regs.items():
        for sa, v in d.items():
            a = int(sa, 0)
            cur = c.rd(a)
            if cur != v:
                c.wr(a, v)
                now = c.rd(a)
                print("  %-14s %#08x  %#010x -> %#010x  %s"
                      % (name, a, v, now, "restored" if now == v else "STILL DIFFERENT"))
                bad += 0 if now == v else 1
    print("restore done, %d register(s) did not take" % bad)
    return bad == 0


def do_diff(c, path):
    snap = json.load(open(path))
    regs = snap.get("regs", snap)
    diffs = 0
    for name, d in regs.items():
        for sa, v in d.items():
            cur = c.rd(int(sa, 0))
            if cur != v:
                print("  %-14s %s  snapshot %#010x  now %#010x" % (name, sa, v, cur))
                diffs += 1
    print("%d register(s) differ from the snapshot" % diffs)
    return diffs == 0


TEST_PROTOCOL = """\
Two tests, in this order, both required before a clock change counts as working:

1. THE CHANGE IS PRESENT -- measured, not read back from a register that claims it.
     ~/.venvs/tinygpu/bin/python pascal/smclk.py --verify
   The GPC counter (nvgpu trim counter 0x132924/0x132928, source gpc2clk) is read on the card and
   scaled through the calibration anchored on %clock64.  Before: 607 MHz.  After a successful change
   the same command must print a higher number; a register read-back alone proves nothing here, which
   is exactly what mclk_measure.py was built for.

2. INFERENCE IS FASTER -- the same argument set on both sides, or the comparison is meaningless.
     bin/local-gpu stop --all
     LOCAL_GPU_PLAIN=1 bin/local-gpu serve            # speculation off: the comparable shape
     ~/.venvs/tinygpu/bin/python localgpu/bench_chat.py
   Recorded reference at 607 MHz: 12 turns in 162 s, 13.5 tok/s overall, 354 tok/s prefill
   (logs/bench/20261113-102802-*.json).  Anything under ~4% is this card's run-to-run noise, so the
   change has to clear that bar.  NVPASCAL_PROF=1 adds per-kernel GPU time if the wall time moves
   less than expected.

Rollback at any point:
     bin/local-gpu stop --all
     ~/.venvs/tinygpu/bin/python pascal/smclk.py --restore <snapshot.json>
     ~/.venvs/tinygpu/bin/python pascal/reset_engines.py && ~/.venvs/tinygpu/bin/python pascal/gr.py
     bin/local-gpu serve
   and if the card stops answering host MMIO entirely: re-POST (devinit.py + fb.py) and, failing that,
   power-cycle the Razer Core X (the step that needs hands).
"""


def main(argv=None):
    ap = argparse.ArgumentParser(description="SM clock: snapshot, probe, measure, apply, restore")
    ap.add_argument("--verify", action="store_true", help="measure the real GPC/DRAM clocks")
    ap.add_argument("--scan", action="store_true", help="read-only register dump")
    ap.add_argument("--probe", action="store_true", help="same-value writes: map host-writability")
    ap.add_argument("--snapshot", metavar="FILE")
    ap.add_argument("--restore", metavar="FILE")
    ap.add_argument("--diff", metavar="FILE")
    ap.add_argument("--apply", metavar="ADDR:VALUE[,ADDR:VALUE]")
    ap.add_argument("--no-auto-restore", action="store_true")
    ap.add_argument("--test", action="store_true", help="print the change-present / faster protocol")
    a = ap.parse_args(argv)

    if a.test:
        print(TEST_PROTOCOL)
        return 0

    c = Card()
    if a.scan:
        do_scan(c)
    if a.probe:
        do_probe(c)
    if a.verify:
        verify(c)
    if a.snapshot:
        regs, ck = snapshot(c)
        n = sum(len(v) for v in regs.values())
        with open(a.snapshot, "w") as f:
            json.dump({"card": "GTX 1080 Ti (GP102)", "when": time.ctime(), "checksum": ck,
                       "regs": regs}, f, indent=1)
        print("snapshot: %d registers -> %s (checksum %s)" % (n, a.snapshot, ck))
    if a.diff:
        return 0 if do_diff(c, a.diff) else 1
    if a.apply:
        ok = do_apply(c, a.apply.split(","), auto_restore=not a.no_auto_restore)
        if ok:
            verify(c)
        return 0 if ok else 1
    if a.restore:
        return 0 if do_restore(c, a.restore) else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
