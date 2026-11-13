#!/usr/bin/env python3
"""Measure the ACTUAL clock in each GPU clock domain from the host, without running GPU work.

nvgpu keeps four "trim" counters the host can read directly (gp106/clk_gp106.c +
hw_trim_gp106.h).  Each counter counts cycles of a selected clock during a fixed
crystal-referenced window, so the answer does not depend on host timing -- which is why this is the
right instrument here: the PLL read-back is not a measurement, and the project's open question is
literally "what clock does the DRAM actually get?".

The one that matters is `trim_fbpa_bcast_clk_cntr_ncltcclk` (0x132128), whose source is
`dramdiv4_rec_clk1` -- the DRAM clock divided by 4, so the reading is scaled by 4 (nvgpu's
CTRL_CLK_DOMAIN_MCLK).  The other three are gpc2clk (SM, cross-checkable against %clock64),
sys2clk and xbar2clk.

    ~/.venvs/tinygpu/bin/python pascal/mclk_measure.py              # all four domains
    ~/.venvs/tinygpu/bin/python pascal/mclk_measure.py --repeat 5   # tighter numbers
    ~/.venvs/tinygpu/bin/python pascal/mclk_measure.py --dram 8     # DRAM only, 8 samples

Register layout (identical in all four; from hw_trim_gp106.h):
    bits[15:0] noofipclks (the window)  bit16 write_en  bit20 enable  bit24 reset  bits[31:28] source

The two readings are printed because the field name ("noofipclks") and the value the driver writes
into it (a crystal count) leave the direction of the ratio ambiguous on paper.  The known SM clock
(607 MHz, measured independently with %clock64 in bench.py) is the referee: whichever column
matches it is the correct interpretation, and then the DRAM column is trustworthy too.
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from card import Card                                                        # noqa: E402

XTAL4X_KHZ = 108000.0          # gp106_crystal_clk_hz: 27 MHz crystal x4

# name -> (cfg addr, cnt addr, source value, scale)
COUNTERS = {
    "gpc2clk":       (0x00132924, 0x00132928, 0x70000000, 1),
    "sys2clk":       (0x001373b0, 0x001373b4, 0x00000000, 1),
    "xbar2clk":      (0x001373c0, 0x001373c4, 0x20000000, 1),
    "dramdiv4_clk1": (0x00132128, 0x0013212c, 0x30000000, 4),
}

NOOFIPCLKS_M = 0xffff
WRITE_EN = 1 << 16
ENABLE = 1 << 20
RESET = 1 << 24


def measure(c, name, n=1000, delay_ms=2.0, settle_us=10, retries=50, verbose=False):
    """nvgpu's gp106_get_rate_cntr, by the book.  Returns (count, cfg_restored)."""
    cfg, cnt_addr, source, _ = COUNTERS[name]
    save = c.rd(cfg)

    # 1. reset and hold the counter, then wait for it to actually read 0
    c.wr(cfg, RESET)                                   # reset asserted, enable deasserted
    c.rd(cfg)                                          # write barrier
    got = None
    for _ in range(retries):
        got = c.rd(cnt_addr)
        if got == 0:
            break
        time.sleep(settle_us / 1e6)
    else:
        raise RuntimeError("%s: counter did not settle to 0 (read %#x)" % (name, got))

    # 2. program: enable, latch the source, count over a fixed crystal window
    c.wr(cfg, ENABLE | WRITE_EN | (n & NOOFIPCLKS_M) | source)
    c.rd(cfg)
    time.sleep(delay_ms / 1000.0)
    count = c.rd(cnt_addr)

    # 3. always put the register back exactly as we found it
    c.wr(cfg, RESET)
    c.rd(cfg)
    c.wr(cfg, save)
    c.rd(cfg)
    restored = c.rd(cfg) == save
    if verbose:
        print("  %-14s raw %8d  cfg %#010x -> %#010x (%s)"
              % (name, count, save, c.rd(cfg), "restored" if restored else "CHANGED!"))
    return count, restored


def derive(count, n, scale):
    """The two possible readings of (count, window).  Both are exact ratios; only one has the
    right absolute scale, and the SM clock's known value decides which."""
    # A: the window is n crystal-x4 cycles, so count = f_src * n / f_xtal
    a = count * (XTAL4X_KHZ / n) * scale / 1000.0
    # B: the window is n source cycles, so count = n * f_xtal / f_src
    b = (n * XTAL4X_KHZ * scale / count / 1000.0) if count else 0.0
    return a, b


SM_ANCHOR_MHZ = 607.0     # %clock64 in bench.py: the SM clock, measured independently of all this


def calibrate(out, anchor=SM_ANCHOR_MHZ):
    """The counters agree with each other to 0.01% but neither paper interpretation puts gpc2clk at
    its known 607 MHz, so the absolute scale is taken from that anchor and everything else is a
    ratio against it.  Returns MHz-per-count, or None if gpc2clk was not measured."""
    g = out.get("gpc2clk")
    if not g or not g["median"]:
        return None
    k = anchor / g["median"]
    for name, r in out.items():
        r["mhz"] = r["median"] * k * r["scale"]
    return k


def main(argv=None):
    ap = argparse.ArgumentParser(description="host-side GPU clock measurement (nvgpu trim counters)")
    ap.add_argument("--repeat", type=int, default=3, help="samples per counter (default 3)")
    ap.add_argument("--n", type=int, default=1000, help="counter window (default 1000)")
    ap.add_argument("--delay", type=float, default=2.0, help="ms to wait before reading (default 2)")
    ap.add_argument("--dram", type=int, default=0, help="DRAM counter only, this many samples")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)

    c = Card()
    names = ["dramdiv4_clk1"] if a.dram else list(COUNTERS)
    reps = a.dram or a.repeat

    out, bad = {}, []
    for name in names:
        counts = []
        for _ in range(reps):
            cnt, ok = measure(c, name, n=a.n, delay_ms=a.delay)
            if not ok:
                bad.append(name)
            counts.append(cnt)
        counts.sort()
        med = counts[len(counts) // 2]
        a_mhz, b_mhz = derive(med, a.n, COUNTERS[name][3])
        spread = (counts[-1] - counts[0]) / med * 100.0 if med else 0.0
        out[name] = dict(raw=counts, median=med, A_mhz=a_mhz, B_mhz=b_mhz, spread_pct=spread,
                         scale=COUNTERS[name][3])

    if a.json:
        import json
        print(json.dumps(out, indent=2))
        return 0

    print("nvgpu trim counters: window %d, %d samples, %s"
          % (a.n, reps, "median of each" if reps > 1 else "single sample"))
    k = calibrate(out)
    if k:
        print("scale: %.5f MHz/count, anchored on gpc2clk = %.1f MHz (%%clock64, bench.py)"
              % (k, SM_ANCHOR_MHZ))
    print("%-14s %10s %8s %10s %10s %10s" % ("counter", "raw", "spread", "MHz", "A (MHz)", "B (MHz)"))
    for name, r in out.items():
        print("%-14s %10d %7.2f%% %10.2f %10.2f %10.2f"
              % (name, r["median"], r["spread_pct"], r.get("mhz", 0.0), r["A_mhz"], r["B_mhz"]))
    if bad:
        print("WARNING: cfg register not restored for: %s" % ", ".join(sorted(set(bad))))
    if k and "dramdiv4_clk1" in out:
        fck = out["dramdiv4_clk1"]["mhz"]
        print()
        print("DRAM: f_CK = %.1f MHz  ->  quoted memory clock = %.0f MHz (GDDR5X convention, x2)"
              % (fck, fck * 2))
        print("      ceiling at that clock = %.1f GB/s (f_CK x 4 transfers x 44 B), "
              "vs %.1f GB/s measured at 1080" % (fck * 4 * 44 / 1000.0, 87.3))
        print("      it rounds the target to the nearest achievable refpll point, so compare "
              "against what you asked for")
    return 0


if __name__ == "__main__":
    sys.exit(main())
