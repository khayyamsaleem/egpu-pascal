"""Read the GP102 die temperature from the card's own thermal sensor (read-only).

Register per nouveau's gm107/gp100 thermal code (`nvkm/subdev/therm/gp100.c`,
vendored at tools/nvkm/subdev/therm/gp100.c): BAR0 0x020460, valid when bit 29 is
set, temperature = (reg & 0x0001fff8) >> 8 degrees C. Bit 30 means the reading
came from the SHADOWed sensor lane.

Nothing here writes to the card: it maps BAR0, reads one word, and exits. The
point is to answer "is the card actually doing the work?" with the card's own
numbers rather than with fan noise.

Usage:
    python3 pascal/therm.py                 # one reading
    python3 pascal/therm.py --watch 30      # sample every second for 30 s
    python3 pascal/therm.py --watch 30 --tag loaded
"""
import argparse
import time

from card import Card

TEMP_REG = 0x020460
VALID = 0x20000000          # device-valid bit
SHADOW = 0x40000000         # reading came from the shadowed sensor


def read_temp(c):
    raw = c.rd(TEMP_REG)
    if not raw & VALID:
        return None, raw
    return (raw & 0x0001FFF8) >> 8, raw


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--watch", type=float, default=0, help="seconds to sample (0 = one reading)")
    ap.add_argument("--interval", type=float, default=1.0)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()

    c = Card()
    label = ("[%s] " % a.tag) if a.tag else ""

    if not a.watch:
        t, raw = read_temp(c)
        if t is None:
            print("%stemperature: sensor not valid (0x020460 = %#010x, valid bit clear)" % (label, raw))
            print("RESULT: NO READING")
            return 1
        print("%stemperature: %d C  (0x020460 = %#010x%s)" % (label, t, raw, ", shadowed" if raw & SHADOW else ""))
        print("RESULT: PASS")
        return 0

    t0 = time.time()
    temps = []
    while time.time() - t0 < a.watch:
        t, raw = read_temp(c)
        if t is None:
            print("%st=%.0fs  sensor not valid (0x020460 = %#010x)" % (label, time.time() - t0, raw), flush=True)
        else:
            temps.append(t)
            print("%st=%.0fs  %d C%s" % (label, time.time() - t0, t, "  (shadowed)" if raw & SHADOW else ""), flush=True)
        time.sleep(a.interval)

    if not temps:
        print("RESULT: NO READING")
        return 1
    print("%stemperature over %.0fs: min %d C, max %d C, first %d C, last %d C"
          % (label, a.watch, min(temps), max(temps), temps[0], temps[-1]))
    print("RESULT: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
