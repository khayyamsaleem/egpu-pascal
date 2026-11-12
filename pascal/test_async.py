"""Async engine checks (nvpascal host ring): stream-ordered copies/fills at any alignment,
ordering of queued work, launch throughput, and host<->VRAM bandwidth through the staging ring.
Prerequisite: reset_engines.py + gr.py.  Usage: ~/.venvs/tinygpu/bin/python pascal/test_async.py"""
import os, random, struct, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import nvpascal

rng = random.Random(1234)
dev = nvpascal.Device()
ok = True
base = dev.alloc(64 << 20)
mirror = bytearray(64 << 20)            # host model of the buffer
dev.htod(base.va, bytes(mirror))

# 1-3: random htod / dtod / fill at random offsets and lengths, then compare the whole buffer
t0 = time.time()
for i in range(400):
    op = rng.choice(("htod", "htod", "dtod", "fill1", "fill2", "fill4"))
    n = rng.choice((1, 2, 3, 5, 15, 16, 17, 255, 4096, 65537, rng.randrange(1, 1 << 20)))
    off = rng.randrange(0, (64 << 20) - n)
    if op == "htod":
        d = os.urandom(n)
        dev.htod(base.va + off, d); mirror[off:off + n] = d
    elif op == "dtod":
        src = rng.randrange(0, (64 << 20) - n)
        if abs(src - off) < n:
            continue                       # cuMemcpy forbids overlap
        dev.copy_async(base.va + off, base.va + src, n); mirror[off:off + n] = mirror[src:src + n]
    else:
        w = int(op[-1]); off -= off % w; n -= n % w
        if not n:
            continue
        v = rng.getrandbits(8 * w)
        pat = {1: v * 0x01010101, 2: v * 0x00010001, 4: v}[w]
        dev.fill_async(base.va + off, pat, n)
        mirror[off:off + n] = (struct.pack("<I", pat) * (n // 4 + 1))[:n]
    if i % 97 == 0:                         # spot-check mid-stream (dtoh orders after the queue)
        a = rng.randrange(0, (64 << 20) - 4096)
        if dev.dtoh(base.va + a, 4096) != bytes(mirror[a:a + 4096]):
            print("  mid-stream mismatch after op %d (%s)" % (i, op)); ok = False
got = dev.dtoh(base.va, 64 << 20)
bad = sum(1 for k in range(0, 64 << 20, 4096) if got[k:k + 4096] != mirror[k:k + 4096])
print("400 random htod/dtod/fill ops + full readback: %d bad pages of %d  (%.1f s)  %s"
      % (bad, (64 << 20) // 4096, time.time() - t0, "ok" if not bad else "WRONG"))
ok &= not bad

# 4: ordering -- a dependent chain with no host waits in between
x = dev.alloc(0x1000)
dev.fill_async(x.va, 0, 0x1000)
for i in range(200):                        # x[0:4] = i, copied forward each step
    dev.fill_async(x.va, i, 4)
    dev.copy_async(x.va + 4 + (i % 8) * 4, x.va, 4)
vals = struct.unpack("<9I", dev.dtoh(x.va, 36))
want = (199,) + tuple(max(i for i in range(200) if i % 8 == j) for j in range(8))
print("dependent chain of 400 queued launches: %s  %s" % (vals == want, "ok" if vals == want else "WRONG %r" % (vals,)))
ok &= vals == want

# 5: launch throughput, async vs synchronous
n = 3000
t0 = time.perf_counter()
for i in range(n):
    dev.fill_async(x.va, i, 4)
t_issue = time.perf_counter() - t0
dev.drain()
t_async = time.perf_counter() - t0
t0 = time.perf_counter()
for i in range(500):
    dev.launch(dev.k_xfill, (1, 1, 1), (256, 1, 1), struct.pack("<QI4xQ", x.va, i, 4))
t_sync = (time.perf_counter() - t0) / 500
last = struct.unpack("<I", dev.dtoh(x.va, 4))[0]
print("launches: async %.1f us each (issue %.1f us), sync %.1f us each; last value %d  %s"
      % (t_async / n * 1e6, t_issue / n * 1e6, t_sync * 1e6, last, "ok" if last == 499 else "WRONG"))
ok &= last == 499

# 6: bandwidth through the staging ring
d = os.urandom(64 << 20)
t0 = time.perf_counter(); dev.htod(base.va, d); dev.drain(); th = time.perf_counter() - t0
t0 = time.perf_counter(); r = dev.dtoh(base.va, 64 << 20); td = time.perf_counter() - t0
print("64 MiB: htod %.0f MB/s, dtoh %.0f MB/s, roundtrip %s" % (len(d) / th / 1e6, len(d) / td / 1e6, "ok" if r == d else "WRONG"))
ok &= r == d
print("RESULT:", "PASS" if ok else "FAIL")
