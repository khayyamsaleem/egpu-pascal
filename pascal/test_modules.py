"""CUDA-module ABI gate for nvpascal (what MAX's kernels need beyond rung 4):
local memory + stack pointer (c[0][0x20]), generic local/shared windows (c[0][0x0/0x4/0x100/0x104]),
module data (.nv.global.init) reached through a relocated constant bank 4, and the shared-memory
window.  Prerequisite: reset_engines.py + gr.py.   Prints RESULT: PASS/FAIL.
"""
import os, struct, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import nvpascal

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
dev = nvpascal.Device()
k = dev.load(os.path.join(ROOT, "kernels", "abi_probe.sm_61.cubin"))["abi_probe"]
print("abi_probe: frame %d, lmem %d bytes/thread, cbanks %s" % (
    k["frame"], k["lmem"], {b: (hex(v), n) for b, (v, n) in k["cbanks"].items()}))
ok = True
for idx in (0, 2, 3):
    out = dev.alloc(0x1000, zero=True)
    ns = dev.launch(k, (1, 1, 1), (1, 1, 1), struct.pack("<QI", out.va, idx))
    lptr, sptr, rt = struct.unpack_from("<QQQ", out.read(24))
    (g,) = struct.unpack_from("<I", out.read(4, off=24))
    exp_l = nvpascal.LOCAL_WINDOW + k["lmem"] - k["frame"]
    good = (lptr == exp_l and sptr == nvpascal.SHARED_WINDOW and rt == 0x1234 + idx
            and g == (11, 22, 33, 44)[idx])
    ok &= good
    print("  idx %d: &lbuf %#x (expect %#x)  &sbuf %#x  lbuf[idx] %#x  gtab[idx] %d  %.1f us  %s"
          % (idx, lptr, exp_l, sptr, rt, g, ns / 1e3, "ok" if good else "WRONG"))
print("RESULT:", "PASS" if ok else "FAIL")

# IEEE rcp/div: literals in the per-kernel constant bank 2.
import numpy as np
fd = dev.load(os.path.join(ROOT, "kernels", "abi_probe.sm_61.cubin"))["fdiv_probe"]
n = 256
a = np.random.default_rng(1).standard_normal(n).astype(np.float32) * 10
b = np.random.default_rng(2).standard_normal(n).astype(np.float32) * 10
a[:4] = [1.0, 3.0, 1e-40, 2.0 ** 126]                      # exact, inexact, denormal, huge
ba, bb, bo = dev.alloc(n * 4), dev.alloc(n * 4), dev.alloc(2 * n * 4)
ba.write(a.tobytes()); bb.write(b.tobytes())
print("fdiv_probe: cbanks %s" % {k: (hex(v), sz) for k, (v, sz) in fd["cbanks"].items()})
dev.launch(fd, (1, 1, 1), (n, 1, 1), struct.pack("<QQQI", bo.va, ba.va, bb.va, n))
got = np.frombuffer(bo.read(2 * n * 4), dtype=np.float32)
with np.errstate(all="ignore"):
    want = np.concatenate([np.float32(1) / a, a / b]).astype(np.float32)
div_ok = bool((got.view(np.uint32) == want.view(np.uint32)).all())
print("  rcp.rn/div.rn: %d of %d bit-exact vs numpy  %s" % ((got.view(np.uint32) == want.view(np.uint32)).sum(), 2 * n, "ok" if div_ok else "WRONG"))
ok &= div_ok

# Pushbuffer wrap: >4096 synchronous launches cycle the 1 MiB pushbuffer (and the 512-entry
# GPFIFO) more than once; an MMU fault at pb_va + pb_size used to end every long MAX run.
import time
fill = dev.load(os.path.join(ROOT, "kernels", "bench.sm_61.cubin"))["fill32"]
buf = dev.alloc(0x10000)
n, t0 = 5000, time.time()
for i in range(n):
    dev.launch(fill, (1, 1, 1), (256, 1, 1), struct.pack("<QI4xQ", buf.va, i, 0x10000 // 4))
last = struct.unpack("<I", buf.read(4))[0]
wrap_ok = last == n - 1
print("pushbuffer wrap: %d launches in %.1f s (%.2f ms each), last value %d  %s"
      % (n, time.time() - t0, (time.time() - t0) / n * 1e3, last, "ok" if wrap_ok else "WRONG"))
print("RESULT:", "PASS" if ok and wrap_ok else "FAIL")
