"""Run the PTX that Mojo/MAX generated (captured by the shim) on the real GP102."""
import os, struct, sys
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
sys.path.insert(0, os.path.join(ROOT, "pascal"))
import nvpascal

N = 4096
dev = nvpascal.Device()
ks = dev.load(os.path.join(ROOT, "kernels", "captured", "mojo_vadd.sm_61.cubin"))
(name, k), = ks.items()
print("kernel %s: %d regs, params %s" % (name[:40] + "...", k["regs"], [(a, v.hex()) for a, v in k["info"] if a == "KPARAM_INFO"]))
a, b = dev.alloc(N * 4), dev.alloc(N * 4)
c = dev.alloc_host(N * 4)
a.write(struct.pack("<%df" % N, *[float(i) for i in range(N)]))
b.write(struct.pack("<%df" % N, *[float(2 * i) for i in range(N)]))
c.write(struct.pack("<f", -1.0) * N)
ns = dev.launch(k, (N // 128, 1, 1), (128, 1, 1), struct.pack("<QQQi", a.va, b.va, c.va, N))
out = struct.unpack("<%df" % N, c.read())
bad = sum(out[i] != 3.0 * i for i in range(N))
print("Mojo-generated vadd on the GTX 1080 Ti: c[0..3] = %s, c[N-1] = %s, %d/%d correct, %.1f us"
      % (out[:4], out[N - 1], N - bad, N, ns / 1e3))
print("RESULT:", "PASS" if bad == 0 else "FAIL")
