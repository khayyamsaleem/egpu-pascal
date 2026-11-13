"""Full VRAM pattern test + SM-side bandwidth, for memory reclocking (kernels/memtest.cu).

Allocates --gib GiB of VRAM in 1 GiB buffers, and for each of several seeds fills every
word with an address-derived pattern and re-reads all of it on the GPU, counting
mismatches. Bandwidth comes from the same launches (fill = writes, check = reads), so a
reclock is judged on correctness and speed by one run.

Usage: ~/.venvs/tinygpu/bin/python pascal/memtest.py [--gib 8] [--seeds 3]
"""
import argparse
import os
import struct
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import nvpascal

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
GIB = 1 << 30
SEEDS = [0x00000000, 0xffffffff, 0x5a5aa5a5, 0x13579bdf, 0xdeadbeef]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gib", type=float, default=8)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--grid", type=int, default=28 * 16)
    ap.add_argument("--loops", type=int, default=1, help="repeat the whole pass (soak)")
    ap.add_argument("--hold", type=float, default=0,
                    help="seconds to wait between fill and check (retention: tests the refresh setting)")
    ap.add_argument("--inject", action="store_true",
                    help="negative control: flip one bit in buffer 0 after the fill; expect exactly 1 bad word")
    a = ap.parse_args()

    dev = nvpascal.Device()
    ks = dev.load(os.path.join(ROOT, "kernels", "memtest.sm_61.cubin"))
    chunk = GIB if a.gib >= 1 else int(a.gib * GIB)
    nbuf = max(1, int(a.gib * GIB) // chunk)
    bufs = [dev.alloc(chunk) for _ in range(nbuf)]
    out = dev.alloc(0x1000)
    print("testing %d x %d MiB = %.2f GiB of VRAM (PA %#x .. %#x)"
          % (nbuf, chunk >> 20, nbuf * chunk / GIB, bufs[0].pa, bufs[-1].pa + chunk))
    total_bad, fill_bw, check_bw = 0, [], []
    t_all = time.time()
    for seed in [x for _ in range(a.loops) for x in SEEDS[:a.seeds]]:
        for b in bufs:
            n4 = chunk // 16
            ns = dev.launch(ks["mt_fill"], (a.grid, 1, 1), (256, 1, 1), struct.pack("<QQI", b.va, n4, seed))
            fill_bw.append(chunk / ns)
        if a.inject:
            off = 0x123450
            w = dev.c.vram_r32(bufs[0].pa + off)
            dev.c.vram_w32(bufs[0].pa + off, w ^ 0x00010000)
            print("  injected: flipped bit 16 of the word at buffer 0 byte %#x (%08x -> %08x)" % (off, w, w ^ 0x10000))
        if a.hold:
            time.sleep(a.hold)
        bad = 0
        for i, b in enumerate(bufs):
            out.write(struct.pack("<4Q", 0, (1 << 64) - 1, 0, 0))
            n4 = chunk // 16
            ns = dev.launch(ks["mt_check"], (a.grid, 1, 1), (256, 1, 1), struct.pack("<QQIxxxxQ", b.va, n4, seed, out.va))
            check_bw.append(chunk / ns)
            nb, first, flips, val = struct.unpack("<4Q", out.read(32))
            if nb:
                print("  seed %08x buffer %d (PA %#x): %d bad words, first at byte %#x (read %08x), flipped bits %08x"
                      % (seed, i, b.pa, nb, first * 4, val, flips))
            bad += nb
        total_bad += bad
        print("seed %08x: %d bad words over %.2f GiB" % (seed, bad, nbuf * chunk / GIB), flush=True)
    print("write bandwidth %.1f GB/s (min %.1f), read bandwidth %.1f GB/s (min %.1f); %.0f s"
          % (sum(fill_bw) / len(fill_bw), min(fill_bw), sum(check_bw) / len(check_bw), min(check_bw),
             time.time() - t_all))
    if a.inject:
        ok = total_bad == a.seeds * a.loops
        print("RESULT: %s" % ("INJECTION DETECTED (negative control ok)" if ok else
                               "NEGATIVE CONTROL FAILED: expected %d bad words, got %d" % (a.seeds, total_bad)))
        return 0 if ok else 1
    print("RESULT: %s" % ("PASS" if total_bad == 0 else "FAIL (%d bad words)" % total_bad))
    return 0 if total_bad == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
