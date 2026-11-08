"""SM clock and SM-side DRAM bandwidth on the GP102 (kernels/bench.sm_61.cubin).

clk:     one thread; %clock64 / %globaltimer around a fixed spin -> SM clock in MHz.
bw_copy: grid-stride 16-byte loads/stores, VRAM -> VRAM; GPU time from report-semaphore
         timestamps around the launch; traffic = 2 * bytes / t.  Spot-checked byte-exact.
Also vadd from kernels/vadd.sm_61.cubin through the runtime, as a regression check.
"""
import os, struct, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import nvpascal

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")


def main():
    dev = nvpascal.Device()
    ks = dev.load(os.path.join(ROOT, "kernels", "bench.sm_61.cubin"))
    ks.update(dev.load(os.path.join(ROOT, "kernels", "vadd.sm_61.cubin")))
    print("loaded: " + ", ".join("%s@%#x (%d regs)" % (n, k["prog_off"], k["regs"]) for n, k in ks.items()))

    # ---- regression: vadd through the runtime, c in host memory
    N = 1 << 16
    a, b = dev.alloc(N * 4), dev.alloc(N * 4)
    out = dev.alloc_host(N * 4)
    av = [(i * 2654435761) & 0xffffffff for i in range(N)]
    bv = [(i * 97 + 5) & 0xffffffff for i in range(N)]
    a.write(struct.pack("<%dI" % N, *av)); b.write(struct.pack("<%dI" % N, *bv))
    out.write(b"\xef\xbe\xad\xde" * N)
    ns = dev.launch(ks["vadd"], (N // 256, 1, 1), (256, 1, 1), struct.pack("<QQQI", a.va, b.va, out.va, N))
    got = struct.unpack("<%dI" % N, out.read())
    bad = sum(got[i] != (av[i] + bv[i]) & 0xffffffff for i in range(N))
    print("vadd %d elements -> host memory: %d wrong, %.1f us" % (N, bad, ns / 1e3))

    # ---- SM clock
    clk_out = dev.alloc_host(0x1000)
    for spins in (10_000_000, 200_000_000):
        clk_out.write(bytes(32))
        ns = dev.launch(ks["clk"], (1, 1, 1), (1, 1, 1), struct.pack("<QQ", clk_out.va, spins))
        t0, t1, c0, c1 = struct.unpack("<4Q", clk_out.read(32))
        print("clk %d cycles: %d cycles in %d ns (globaltimer) -> SM clock %.0f MHz; launch %.2f ms"
              % (spins, c1 - c0, t1 - t0, (c1 - c0) / max(t1 - t0, 1) * 1e3, ns / 1e6))

    # ---- SM-side DRAM bandwidth
    for size in (64 << 20, 256 << 20):
        src, dst = dev.alloc(size), dev.alloc(size)
        probes = [0, 0x10000, size // 2 + 0x40, size - 16]
        for i, o in enumerate(probes):
            src.write(struct.pack("<4I", 0xa11ce000 | i, o, size, 0x600d600d), o)
            dst.write(b"\x00" * 16, o)
        n16 = size // 16
        for grid in (28 * 8, 28 * 32):
            ns = dev.launch(ks["bw_copy"], (grid, 1, 1), (256, 1, 1), struct.pack("<QQQ", src.va, dst.va, n16))
            ok = all(dst.read(16, o) == src.read(16, o) for o in probes)
            print("bw_copy %4d MiB grid %4d x 256: %.3f ms  traffic %.1f GB/s  spot-check %s"
                  % (size >> 20, grid, ns / 1e6, 2 * size / ns, "ok" if ok else "MISMATCH"))


if __name__ == "__main__":
    main()
