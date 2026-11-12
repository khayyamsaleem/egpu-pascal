"""Driver-level copy semantics through the shim (container, SHIM_MODE=remote): HtoD, DtoH, DtoD and
cuMemcpyBatchAsync with known patterns at the sizes/alignments MAX uses (3072, 43008, 8, odd)."""
import ctypes, os
import numpy as np

cu = ctypes.CDLL(os.environ.get("SHIM_LIB", "/repo/shim/build/libcuda.so.1"))
u64, sz = ctypes.c_uint64, ctypes.c_size_t


def ck(r, what):
    assert r == 0, "%s -> CUresult %d" % (what, r)


ck(cu.cuInit(0), "cuInit")
dev = ctypes.c_int(); ck(cu.cuDeviceGet(ctypes.byref(dev), 0), "cuDeviceGet")
ctx = ctypes.c_void_p(); ck(cu.cuDevicePrimaryCtxRetain(ctypes.byref(ctx), dev), "ctx")
pool = u64(); N = 64 << 20
ck(cu.cuMemAlloc_v2(ctypes.byref(pool), sz(N)), "cuMemAlloc")
base = pool.value
rng = np.random.default_rng(1)
ok = True


def htod(off, a): ck(cu.cuMemcpyHtoD_v2(u64(base + off), a.ctypes.data_as(ctypes.c_void_p), sz(a.nbytes)), "HtoD")
def dtoh(off, n):
    b = np.zeros(n, np.uint8); ck(cu.cuMemcpyDtoH_v2(b.ctypes.data_as(ctypes.c_void_p), u64(base + off), sz(n)), "DtoH"); return b
def dtod(dst, src, n): ck(cu.cuMemcpyDtoD_v2(u64(base + dst), u64(base + src), sz(n)), "DtoD")


def report(name, good):
    global ok
    ok &= good
    print("%-44s %s" % (name, "ok" if good else "WRONG"), flush=True)


for n, so, do in ((3072, 0x0f00, 0x2b00), (43008, 0x9900, 0x34100), (8, 0x8f00, 0x9a00), (4096 + 6, 0x40003, 0x80005), (1 << 20, 0x100000, 0x300000)):
    src = rng.integers(0, 256, n, dtype=np.uint8)
    canary = rng.integers(0, 256, n + 64, dtype=np.uint8)
    htod(so, src)
    htod(do - 32, canary)                                  # destination pre-filled, with margins
    dtod(do, so, n)
    got = dtoh(do - 32, n + 64)
    report("DtoD %7d B  src+%#x -> dst+%#x" % (n, so, do),
           np.array_equal(got[32:32 + n], src) and np.array_equal(got[:32], canary[:32])
           and np.array_equal(got[32 + n:], canary[32 + n:]) and np.array_equal(dtoh(so, n), src))

# cuMemcpyBatchAsync: H2D, D2H and D2D in one batch (CUDA 12.8 ABI)
a = rng.integers(0, 256, 3072, dtype=np.uint8); hb = np.zeros(3072, np.uint8)
htod(0x200000, rng.integers(0, 256, 3072, dtype=np.uint8))
before = dtoh(0x200000, 3072)
dsts = (u64 * 3)(base + 0x210000, hb.ctypes.data, base + 0x220000)
srcs = (u64 * 3)(a.ctypes.data, base + 0x200000, base + 0x200000)
sizes = (sz * 3)(3072, 3072, 3072); fail = sz(0)
ck(cu.cuMemcpyBatchAsync(dsts, srcs, sizes, sz(3), None, None, sz(0), ctypes.byref(fail), None), "batch")
report("cuMemcpyBatchAsync H2D", np.array_equal(dtoh(0x210000, 3072), a))
report("cuMemcpyBatchAsync D2H", np.array_equal(hb, before))
report("cuMemcpyBatchAsync D2D", np.array_equal(dtoh(0x220000, 3072), before))
print("RESULT:", "PASS" if ok else "FAIL")
