"""Large-allocation MMU check: map one buffer the size of MAX's pool (default 8.78 GiB), have the GPU
write a unique value into every 1 MiB chunk through its *virtual* address (fill32), then read each
chunk back by *physical* address over PRAMIN.  Catches aliasing / wrong PDE/PT setup for big ranges.
Prerequisite: reset_engines.py + gr.py, and no shimd running.  Prints RESULT: PASS/FAIL."""
import os, struct, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import nvpascal

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
GIB = float(sys.argv[1]) if len(sys.argv) > 1 else 8.78
dev = nvpascal.Device()
fill = dev.load(os.path.join(ROOT, "kernels", "bench.sm_61.cubin"))["fill32"]
size = int(GIB * 2**30) & ~0xfffff
t0 = time.time(); buf = dev.alloc(size); t1 = time.time()
CH = 1 << 20; n = size // CH
print("buffer %.2f GiB: va %#x .. %#x, pa %#x .. %#x (mapped in %.1f s)"
      % (size / 2**30, buf.va, buf.va + size, buf.pa, buf.pa + size, t1 - t0), flush=True)
for i in range(n):                                     # GPU writes 0xC0DE0000 | chunk into the chunk's first 4 KiB
    dev.launch(fill, (4, 1, 1), (256, 1, 1), struct.pack("<QI4xQ", buf.va + i * CH, 0xC0DE0000 | i, 1024))
print("GPU filled %d chunks in %.1f s" % (n, time.time() - t1), flush=True)
bad = []
for i in range(n):
    got = struct.unpack("<I", dev.c.vram_read(buf.pa + i * CH + 4092, 4))[0]
    if got != (0xC0DE0000 | i):
        bad.append((i, got))
print("chunks whose physical memory holds the wrong value: %d of %d" % (len(bad), n))
for i, got in bad[:10]:
    print("  chunk %5d (va %#x, pa %#x): read %#010x" % (i, buf.va + i * CH, buf.pa + i * CH, got))
print("RESULT:", "PASS" if not bad else "FAIL")
