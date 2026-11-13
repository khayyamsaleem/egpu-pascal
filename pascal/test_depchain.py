#!/usr/bin/env python3
"""Does a QMD dependency actually serialize grids on GP102?  (task B, the untried half)

The record says dropping the per-launch WAIT_FOR_IDLE fails: with a pushbuffer semaphore
4536/16384 pages came out wrong, and with the QMD's own release semaphore the completion counter
"went backwards" (done 16 after launch 18).  The conclusion was that grids from one channel are not
serialized on GP102 without the WFI, so real stream order needs dependent-QMD chaining -- which was
never tried.  This tries it.

Modes (one per process; a bad mode can wedge the FIFO, so recover between runs):

    off    the shipping path: WAIT_FOR_IDLE per launch.  Must be ordered.
    none   no WFI and no ordering field at all -- the NEGATIVE CONTROL.  Must show disorder,
           or the test cannot detect disorder and nothing else it says counts.
    dep    no WFI + DEPENDENT_QMD_SCHEDULE_ENABLE + DEPENDENT_QMD_TYPE=GRID (v02_01 has no
           DEPENDENT_QMD_POINTER, only these, so "the previous grid in this channel" is the
           only dependency it can express)
    group  no WFI + a distinct QMD_GROUP_ID per launch
    both   dep + group together

The test is a dependency chain: fill(buf i, pattern i) then copy(buf i -> out i).  out i must hold
pattern i; if the pair's two kernels overlap, out i holds a later pattern.  Sampled 4-byte reads at
three offsets per output keep the verification cheap enough to sweep.

    ~/.venvs/tinygpu/bin/python pascal/test_depchain.py off|none|dep|group|both
"""
import os
import struct
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import nvpascal

N = 48                      # (fill, copy) pairs in the chain
SZ = 0x40000                # 256 KiB: the helper kernels' grid is capped at 224 blocks of 256
                            # threads x 16 bytes = 896 KiB, so a bigger buffer is not covered
ROUNDS = 8                  # fills/copies per pair: a deeper queue is what makes disorder show
MODES = ("off", "none", "dep", "group", "both")


def run(dev, bufs, outs, mode):
    """Queue the chain in `mode`; return a dict of what happened."""
    dev.chain = mode
    last = dev.done_seq()
    backwards = 0
    t0 = time.perf_counter()
    for i in range(N):
        for r in range(ROUNDS):
            dev.fill_async(bufs[i].va, 0x1000 + i * ROUNDS + r, SZ)      # n is a BYTE count
            dev.copy_async(outs[i].va, bufs[i].va, SZ)                   # reads what the fill wrote
        d = dev.done_seq()
        if ((d - last) & 0xffffffff) >= 0x80000000:     # it moved backwards
            backwards += 1
        last = d
    t_queue = time.perf_counter() - t0
    dev.drain()
    wall = time.perf_counter() - t0
    wrong = []
    for i in range(N):
        want = struct.pack("<I", 0x1000 + i * ROUNDS + ROUNDS - 1)       # the last fill's pattern
        for off in (0, SZ // 2, SZ - 4):
            if outs[i].read(4, off=off) != want:
                wrong.append(i)
                break
    return dict(wrong=len(wrong), backwards=backwards, wall=wall,
                rate=N * ROUNDS * 2 / t_queue if t_queue else 0.0)


def main(argv):
    mode = (argv or ["off"])[0]
    if mode not in MODES:
        print("mode must be one of %s" % " ".join(MODES))
        return 2
    dev = nvpascal.Device()
    bufs = [dev.alloc(SZ) for _ in range(N)]
    outs = [dev.alloc(SZ, zero=True) for _ in range(N)]
    print("chain: %d x (fill %d KiB -> copy), mode %s" % (N, SZ // 1024, mode))
    try:
        r = run(dev, bufs, outs, mode)
    except Exception as e:
        print("  EXCEPTION: %s" % str(e).splitlines()[0][:110])
        print("MODE %s: ERROR" % mode)
        return 1
    ordered = r["wrong"] == 0 and r["backwards"] == 0
    print("  %d/%d outputs wrong, %d backwards counter reads, %.0f launch/s, wall %.2f s"
          % (r["wrong"], N, r["backwards"], r["rate"], r["wall"]))
    if mode == "off":
        print("MODE %s: %s (the shipping path must be ordered)" % (mode, "ORDERED" if ordered else "DISORDERED?"))
        return 0 if ordered else 1
    if mode == "none":
        print("MODE %s: %s (the control must be disordered, else the test is blind)"
              % (mode, "ORDERED?! blind test" if ordered else "DISORDERED -- detection works"))
        return 0 if not ordered else 1
    print("MODE %s: %s" % (mode, "ORDERED without WAIT_FOR_IDLE" if ordered else "still disordered"))
    return 0 if ordered else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
