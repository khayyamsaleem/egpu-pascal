"""Session-13 PHY trim sweep: band 2 (>=1300 MHz) never completes DRAM training on the refpll path.

What differs between band 1 (works to 1134 MHz) and band 2 is the VBIOS mode script's PHY trims
(per-lane delay seeds at FBPA 0x?6a0/0x?6a4 and 0x9846a8/ac, Vref at 0x9a070c, masks at 0x98467c /
0x984708) plus the timing entry (band 2 = CL 20).  The VBIOS only ships NVIDIA's trims for its own
reference board, so try others and watch the FBPA training-done bits (0x900974 & 0xf == 0), then
judge with pascal/memtest.py -- the status bits alone are not evidence.

Driver for `pascal/mclk.py retrain 1350 --live --soft --patch=...`; run it with the card at band 2
(`mclk.py set 1350 --live --soft`, which leaves the clock programmed though training is incomplete).

Usage: ~/.venvs/tinygpu/bin/python logs/mclk/s13-trim-sweep.py
"""
import os
import re
import subprocess
import sys

ROOT = os.path.expanduser("~/egpu-pascal")
PY = os.path.expanduser("~/.venvs/tinygpu/bin/python")
LOGS = os.path.join(ROOT, "logs", "mclk")
DELAY = ["9006a0", "9006a4", "9046a0", "9046a4", "9086a0", "9086a4",
         "90c6a0", "90c6a4", "9106a0", "9106a4", "9146a0", "9146a4"]     # per FBPA 0..5 (2 regs each)


def patches(delays=None, masks=None, vref=None, pairs=None):
    """delays=(lane, lane2): 0x?6a0/a4 repeated, 0x9846a8/ac repeated.  vref: 0x9a070c."""
    p = list(pairs or ())
    if delays is not None:
        d1, d2 = delays
        p += ["%s:%02x%02x%02x%02x" % ((a, d1, d1, d1, d1)) for a in DELAY]
        p += ["9846a8:%02x%02x%02x%02x" % ((d2,) * 4), "9846ac:%02x%02x%02x%02x" % ((d2,) * 4)]
    if masks is not None:
        p += ["98467c:%08x" % masks[0], "984708:%08x" % masks[1]]
    if vref is not None:
        p += ["9a070c:%04x" % vref]
    return ",".join(p)


COMBOS = [
    # name, kwargs for patches()
    ("band2-default",   dict()),                                            # baseline: what the script writes
    ("band1-trims",     dict(delays=(0x12, 0x0f), masks=(0xfff10000, 0x00030002), vref=0x4d4d)),
    ("band1-delays",    dict(delays=(0x12, 0x0f), vref=0x5353)),
    ("dly08",           dict(delays=(0x08, 0x08), vref=0x5353)),
    ("dly0a",           dict(delays=(0x0a, 0x0a), vref=0x5353)),
    ("dly10",           dict(delays=(0x10, 0x10), vref=0x5353)),
    ("dly18",           dict(delays=(0x18, 0x18), vref=0x5353)),
    ("dly00",           dict(delays=(0x00, 0x00), vref=0x5353)),
    ("vref-only-4d4d",  dict(vref=0x4d4d)),
    ("vref-only-5a5a",  dict(vref=0x5a5a)),
]


def run(args, timeout=300):
    return subprocess.run(args, cwd=ROOT, capture_output=True, text=True, timeout=timeout)


def main():
    mhz = sys.argv[1] if len(sys.argv) > 1 else "1350"
    results = []
    for name, kw in COMBOS:
        patch = patches(**kw)
        args = [PY, "pascal/mclk.py", "retrain", mhz, "--live", "--soft"]
        if patch:
            args.append("--patch=" + patch)
        r = run(args)
        out = (r.stdout or "") + (r.stderr or "")
        open(os.path.join(LOGS, "s13-sweep-%s.txt" % name), "w").write(out)
        ok = "INCOMPLETE" not in out and "Traceback" not in out and "training done" in out
        print("%-16s training %s%s" % (name, "COMPLETE" if ok else "incomplete",
                                       "" if ok else "  (%s)" % (out.strip().splitlines() or ["-"])[-1][:90]),
              flush=True)
        results.append((name, ok))
    print("\ncombos that completed training: %s" % [n for n, o in results if o])


if __name__ == "__main__":
    main()
