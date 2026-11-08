"""Start the ACR-bootstrapped FECS/GPCCS and talk to FECS (gf100_gr_init_ctxctl_ext tail).

Precondition: pascal/sec2_rtos.py has just run (HS ACR with the SEC2 RTOS in the
WPR, then ACR_BOOTSTRAP_FALCON for FECS and GPCCS), so both falcons sit halted
in LS mode (sctl 0x7021) with their bootloaders loaded.  `--full` runs that
chain first.

Round trip: FECS answers host "methods" through 0x409500 (data) / 0x409504
(method) with the result in 0x409800 (gf100_gr_fecs_*).  The host zeroes
0x409800 before each one, so a nonzero answer can only come from FECS code
running; three different methods should give three different sizes.
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from card import Card
import acr

FECS_MTHD_DATA, FECS_MTHD, FECS_MBOX = 0x409500, 0x409504, 0x409800


def fecs_mthd(c, mthd, data=0, wait=True, timeout=2.0):
    c.wr(FECS_MBOX, 0)
    c.wr(FECS_MTHD_DATA, data)
    c.wr(FECS_MTHD, mthd)
    if not wait:
        return None
    t0 = time.time()
    while time.time() - t0 < timeout:
        v = c.rd(FECS_MBOX)
        if v: return v
        time.sleep(0.0005)
    return None


def ctxctl_start(c, verbose=True):
    fecs, gpccs = acr.Falcon(c, acr.FECS), acr.Falcon(c, acr.GPCCS)
    if verbose:
        print("FECS  before: cpuctl=%08x sctl=%08x bootvec=%08x mbox0=%08x" % (fecs.r(0x100), fecs.r(0x240), fecs.r(0x104), fecs.r(0x040)))
        print("GPCCS before: cpuctl=%08x sctl=%08x bootvec=%08x mbox0=%08x" % (gpccs.r(0x100), gpccs.r(0x240), gpccs.r(0x104), gpccs.r(0x040)))
    c.wr(0x409800, 0); c.wr(0x41a10c, 0); c.wr(0x40910c, 0)
    gpccs.start(); fecs.start()
    t0 = time.time()
    up = c.wait(lambda: c.rd(0x409800) & 1, 2.0)
    dt = (time.time() - t0) * 1000
    if verbose:
        print("FECS ready bit (0x409800 & 1): %s after %.0f ms; 0x409800=%08x" % (up, dt, c.rd(0x409800)))
        print("FECS  after: cpuctl=%08x sctl=%08x pc=%08x mbox0=%08x mbox1=%08x" % (fecs.r(0x100), fecs.r(0x240), fecs.trace_pc(), fecs.r(0x040), fecs.r(0x044)))
        print("GPCCS after: cpuctl=%08x sctl=%08x pc=%08x mbox0=%08x mbox1=%08x" % (gpccs.r(0x100), gpccs.r(0x240), gpccs.trace_pc(), gpccs.r(0x040), gpccs.r(0x044)))
    if not up:
        return None
    fecs_mthd(c, 0x21, 0x7fffffff, wait=False)       # gf100_gr_fecs_set_watchdog_timeout
    sizes = {}
    for name, m in (("ctx image", 0x10), ("zcull image", 0x16), ("pm image", 0x25)):
        v = fecs_mthd(c, m)
        sizes[name] = v
        if verbose:
            print("  FECS method %#04x (%s size): %s" % (m, name, "TIMEOUT" if v is None else "%#x (%d bytes)" % (v, v)))
    return sizes


if __name__ == "__main__":
    c = Card()
    if "--full" in sys.argv:
        import subprocess
        here = os.path.dirname(os.path.abspath(__file__))
        r = subprocess.run([sys.executable, os.path.join(here, "sec2_rtos.py")], capture_output=True, text=True)
        print(r.stdout.strip().splitlines()[-1])
    s = ctxctl_start(c)
    ok = s is not None and all(s.values()) and len(set(s.values())) == len(s)
    print("GR CTXCTL RESULT:", "PASS (FECS running and answering methods)" if ok else "FAIL")
