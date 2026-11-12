"""RT-4 gate: launch an sm_61 kernel on the GP102 through PASCAL_COMPUTE_B + a QMD.

Pieces (all on top of grchan.GRChannel, the GR channel that passes RT-4 step 1):
  * QMD v02_01, fields parsed from NVIDIA's tools/nvclass/clc1c0qmd.h; values as
    Mesa gp100_compute_setup_launch_desc (tools/mesa/nve4_compute.c);
  * compute state as Mesa nve4_screen_compute_setup: local memory (TLS) base +
    per-SM sizes, local/shared windows, program region, SPA version;
  * launch: SEND_PCAS_A = qmd >> 8, SEND_SIGNALING_PCAS_B = 3 (invalidate +
    schedule), WAIT_FOR_IDLE, then a PGRAPH report semaphore as completion;
  * constant bank 0 laid out the way ptxas' sm_6x code reads it (from the SASS of
    kernels/vadd.sm_61.cubin): c[0][0x8..0x10] = blockDim, c[0][0x14..0x1c] =
    gridDim, c[0][0x20] = initial stack pointer, params at c[0][0x140].

Kernel: kernels/vadd.sm_61.cubin (ptxas 12.4, -arch=sm_61): c[i] = a[i] + b[i].
a, b in VRAM; **c in host memory** (IOMMU, sys_coh) -- the rung-4 gate.
Negative controls: c prefilled with a pattern; n = N - 37 so the tail must keep it.

Usage: ~/.venvs/tinygpu/bin/python pascal/compute.py   (after reset_engines.py + gr.py)
"""
import os, re, struct, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from card import Card
import mmu as nvmmu
import fifo as fifomod
import gr as grmod
import grctx as grctxmod
import grchan
import cubin as cubinmod
import faults, fb

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
S = grchan.SUBC_CLASS

# --------------------------------------------------------------------------- QMD
class Qmd:
    """256-byte QMD; fields from clc1c0qmd.h `#define NVC1C0_QMDV02_01_<F>[(i)] MW(hi:lo)`."""
    FIELDS = None

    @classmethod
    def _load(cls, ver="V02_01"):
        if cls.FIELDS is None:
            cls.FIELDS = {}
            pat = re.compile(r"#define NVC1C0_QMD%s_(\w+?)(\(i\))?\s+MW\((.+)\)\s*$" % ver)
            for line in open(os.path.join(ROOT, "tools", "nvclass", "clc1c0qmd.h")):
                m = pat.match(line.strip())
                if m:
                    hi, lo = m.group(3).rsplit(":", 1) if m.group(3).count(":") == 1 else _split_mw(m.group(3))
                    cls.FIELDS[m.group(1)] = (hi, lo, bool(m.group(2)))
        return cls.FIELDS

    def __init__(self):
        self.f = self._load()
        self.bits = 0

    POS = {}                                   # (name, i) -> (lo, width): eval once, not per launch

    def set(self, name, value, i=None):
        pos = self.POS.get((name, i))
        if pos is None:
            hi, lo, idx = self.f[name]
            assert idx == (i is not None), name
            env = {"i": i or 0}
            h, l = eval(hi, {}, env), eval(lo, {}, env)
            pos = self.POS[(name, i)] = (l, h - l + 1)
        l, width = pos
        assert 0 <= value < (1 << width), "%s=%#x does not fit %d bits" % (name, value, width)
        self.bits &= ~(((1 << width) - 1) << l)
        self.bits |= value << l
        return self

    def bytes(self):
        return self.bits.to_bytes(256, "little")


def _split_mw(s):
    # "(640+(i)*1):(640+(i)*1)" -> split on the ':' at paren depth 0
    d = 0
    for k, ch in enumerate(s):
        d += ch == "("; d -= ch == ")"
        if ch == ":" and d == 0:
            return s[:k], s[k + 1:]
    raise ValueError(s)


# ------------------------------------------------------------------ the launch
def m(addr, *data):
    return fifomod.mthd(S, addr, *data)


def compute_state(tls_va, per_sm, code_va):
    """Mesa nve4_screen_compute_setup for class < GV100 (TIC/TSC/MS-offsets omitted)."""
    pb = []
    pb += m(0x0000, grchan.PASCAL_COMPUTE_B)                     # SET_OBJECT
    pb += m(0x0790, tls_va >> 32, tls_va & 0xffffffff)            # SET_SHADER_LOCAL_MEMORY_A/B
    pb += m(0x02e4, per_sm >> 32, per_sm & 0xffff8000, 0xff)      # ..._NON_THROTTLED_A/B/C
    pb += m(0x02f0, per_sm >> 32, per_sm & 0xffff8000, 0xff)      # ..._THROTTLED_A/B/C
    pb += m(0x077c, 0xff << 24)                                   # SET_SHADER_LOCAL_MEMORY_WINDOW
    pb += m(0x0214, 0xfe << 24)                                   # SET_SHADER_SHARED_MEMORY_WINDOW
    pb += m(0x1608, code_va >> 32, code_va & 0xffffffff)          # SET_PROGRAM_REGION_A/B
    pb += m(0x0310, 0x400)                                        # SET_SPA_VERSION (Mesa, >= NVF0)
    return pb


def build_qmd(k, grid, block, cb0_va, cb0_size, prog_off=0, crs=0x800, dyn_shared=0, lmem=0, cbanks=None):
    """lmem = per-thread local memory bytes (stack/spills); cbanks = {bank: (va, size)} beyond bank 0."""
    q = Qmd()
    q.set("SM_GLOBAL_CACHING_ENABLE", 1)
    q.set("RELEASE_MEMBAR_TYPE", 1)            # FE_SYSMEMBAR
    q.set("CWD_MEMBAR_TYPE", 1)                # L1_SYSMEMBAR
    q.set("API_VISIBLE_CALL_LIMIT", 1)         # NO_CHECK
    for f in ("INVALIDATE_TEXTURE_HEADER_CACHE", "INVALIDATE_TEXTURE_SAMPLER_CACHE",
              "INVALIDATE_TEXTURE_DATA_CACHE", "INVALIDATE_SHADER_DATA_CACHE",
              "INVALIDATE_SHADER_CONSTANT_CACHE"):
        q.set(f, 1)
    q.set("PROGRAM_OFFSET", prog_off)
    q.set("CTA_RASTER_WIDTH", grid[0]); q.set("CTA_RASTER_HEIGHT", grid[1]); q.set("CTA_RASTER_DEPTH", grid[2])
    q.set("CTA_THREAD_DIMENSION0", block[0]); q.set("CTA_THREAD_DIMENSION1", block[1])
    q.set("CTA_THREAD_DIMENSION2", block[2])
    q.set("SHARED_MEMORY_SIZE", nvmmu.align(k["shared"] + dyn_shared, 0x100))
    q.set("SHADER_LOCAL_MEMORY_LOW_SIZE", lmem)
    q.set("SHADER_LOCAL_MEMORY_HIGH_SIZE", 0)
    q.set("SHADER_LOCAL_MEMORY_CRS_SIZE", crs)
    q.set("REGISTER_COUNT", k["regs"])
    q.set("BARRIER_COUNT", k.get("barriers", 0))  # 0 with bar.sync in the code -> ESR 0xb ILLEGAL_INSTR_PARAM
    q.set("CONSTANT_BUFFER_ADDR_LOWER", cb0_va & 0xffffffff, 0)
    q.set("CONSTANT_BUFFER_ADDR_UPPER", cb0_va >> 32, 0)
    q.set("CONSTANT_BUFFER_SIZE_SHIFTED4", cb0_size >> 4, 0)
    q.set("CONSTANT_BUFFER_VALID", 1, 0)
    for i, (va, size) in sorted((cbanks or {}).items()):
        assert 0 < i < 8 and va % 0x100 == 0
        q.set("CONSTANT_BUFFER_ADDR_LOWER", va & 0xffffffff, i)
        q.set("CONSTANT_BUFFER_ADDR_UPPER", va >> 32, i)
        q.set("CONSTANT_BUFFER_SIZE_SHIFTED4", nvmmu.align(max(size, 0x10), 0x10) >> 4, i)
        q.set("CONSTANT_BUFFER_VALID", 1, i)
    if "--qmd-version" in sys.argv:
        q.set("QMD_VERSION", 1); q.set("QMD_MAJOR_VERSION", 2)
    return q.bytes()


_QMD_CACHE = {}


def qmd_fast(k, grid, block, cb0_va, cb0_size, dyn_shared=0):
    """build_qmd for the per-launch path: everything but constant bank 0's address is cached per
    (kernel, grid, block, cb0 size, dynamic shared); the address is OR'd into the template."""
    key = (id(k), grid, block, cb0_size, dyn_shared)
    t = _QMD_CACHE.get(key)
    if t is None:
        if len(_QMD_CACHE) > 4096:
            _QMD_CACHE.clear()
        q = Qmd()
        build = build_qmd(k, grid, block, 0, cb0_size, prog_off=k["prog_off"], dyn_shared=dyn_shared,
                          lmem=k.get("lmem", 0), cbanks=k.get("cbanks", {}))
        q.set("CONSTANT_BUFFER_ADDR_LOWER", 0, 0); q.set("CONSTANT_BUFFER_ADDR_UPPER", 0, 0)
        lo, hi = Qmd.POS[("CONSTANT_BUFFER_ADDR_LOWER", 0)][0], Qmd.POS[("CONSTANT_BUFFER_ADDR_UPPER", 0)][0]
        t = _QMD_CACHE[key] = (int.from_bytes(build, "little"), lo, hi, k)   # keep k alive (id key)
    bits, lo, hi, _ = t
    return (bits | ((cb0_va & 0xffffffff) << lo) | ((cb0_va >> 32) << hi)).to_bytes(256, "little")


def cbuf0(k, grid, block, params: bytes, stack=0, windows=(0xfe000000, 0xff000000)):
    """Driver area of constant bank 0 (sm_6x ABI, checked against ptxas output):
    c[0][0x0]/[0x100] shared window lo/hi, [0x4]/[0x104] local window lo/hi (generic pointers),
    [0x8] blockDim, [0x14] gridDim, [0x20] initial stack pointer (= local bytes per thread)."""
    img = bytearray(max(k["cbuf0_size"], k["param_off"] + len(params), 0x108))
    struct.pack_into("<II", img, 0x0, windows[0], windows[1])
    struct.pack_into("<II", img, 0x100, 0, 0)
    struct.pack_into("<3I", img, 0x8, *block)
    struct.pack_into("<3I", img, 0x14, *grid)
    struct.pack_into("<I", img, 0x20, stack)
    img[k["param_off"]:k["param_off"] + len(params)] = params
    return bytes(img)


def gr_error_dump(c):
    r = c.rd
    print("  PGRAPH intr 400100=%08x trap 400108=%08x 400118=%08x code 400110=%08x 400704=%08x 400708=%08x status 400700=%08x"
          % (r(0x400100), r(0x400108), r(0x400118), r(0x400110), r(0x400704), r(0x400708), r(0x400700)))
    print("  GPC0: exc 502c90=%08x  TPC0 exc 504508=%08x  SM esr 504648=%08x 50464c=%08x 504650=%08x 504654=%08x 504730=%08x"
          % (r(0x502c90), r(0x504508), r(0x504648), r(0x50464c), r(0x504650), r(0x504654), r(0x504730)))


def main():
    c = Card()
    cb = cubinmod.Cubin(open(os.path.join(ROOT, "kernels", "vadd.sm_61.cubin"), "rb").read())
    k = cb.kernel("vadd")
    print("kernel %s: sm_%d, %d bytes SASS, %d regs, params c[0][%#x]+%#x"
          % (k["name"], cb.sm, len(k["code"]), k["regs"], k["param_off"], k["param_size"]))

    if not grchan.gr_ready(c):
        print("GR not up -> run pascal/reset_engines.py + pascal/gr.py first"); return 1
    g = grmod.GR(c).oneinit(verbose=False)
    sm_nr = sum(g.tpc_nr) if hasattr(g, "tpc_nr") else 28

    heap = nvmmu.VramHeap(base=0x11000000)
    f = fifomod.Fifo(c); f.init()
    devs = {d["name"]: d for d in fb.top_parse(c)}
    for d in devs.values():                     # every engine on GR's runlist out of reset
        if d["runlist"] == devs["GR"]["runlist"] and d.get("reset") is not None:
            c.mask(0x200, 1 << d["reset"], 1 << d["reset"])
    if c.rd(0x262c) & 1:
        c.wr(0x262c, 1)
    faults.dump(c, clear=True)
    gc = grctxmod.GrCtx(c, g, heap)
    ch = grchan.GRChannel(f, chid=1, gc=gc)

    # ---- buffers
    N, BLOCK = 4096, 128
    n = N - 37
    CODE_VA, TLS_VA, QMD_VA, CB0_VA = 0x04000000, 0x05000000, 0x06000000, 0x06010000
    A_VA, B_VA, C_VA, SEM_VA = 0x07000000, 0x07100000, 0x07200000, 0x00600000
    code_pa = ch.alloc_map(CODE_VA, 0x10000)
    per_sm = 0x20000                                        # Mesa: (cstack 0x800) * 64 warps
    tls_pa = ch.alloc_map(TLS_VA, per_sm * 32)              # 32 >= SM count incl. disabled
    qmd_pa = ch.alloc_map(QMD_VA, 0x1000)
    cb0_pa = ch.alloc_map(CB0_VA, 0x1000)
    a_pa = ch.alloc_map(A_VA, N * 4)
    b_pa = ch.alloc_map(B_VA, N * 4)
    sem_pa = ch.alloc_map(SEM_VA, 0x1000)
    smem, iovas = c.dev.alloc_sysmem(N * 4)
    for i, iova in enumerate(iovas):
        ch.vmm.map(C_VA + i * 0x1000, iova, 0x1000, aper="sys_coh", flush=False)
    ch.vmm.flush()

    a = [(i * 2654435761) & 0x7fffffff for i in range(N)]
    b = [(i * 40503 + 12345) & 0x7fffffff for i in range(N)]
    c.vram_write(a_pa, struct.pack("<%dI" % N, *a))
    c.vram_write(b_pa, struct.pack("<%dI" % N, *b))
    POISON = 0xdeadbeef
    smem[0:N * 4] = struct.pack("<I", POISON) * N
    c.vram_write(code_pa, k["code"])
    grid, block = ((n + BLOCK - 1) // BLOCK, 1, 1), (BLOCK, 1, 1)
    params = struct.pack("<QQQI", A_VA, B_VA, C_VA, n)
    c.vram_write(cb0_pa, cbuf0(k, grid, block, params))
    qmd = build_qmd(k, grid, block, CB0_VA, 0x1000)
    c.vram_write(qmd_pa, qmd)
    print("grid %s block %s n=%d  code@%#x tls@%#x (%#x/SM) qmd@%#x cb0@%#x  a@%#x b@%#x c(sysmem)@%#x"
          % (grid, block, n, CODE_VA, TLS_VA, per_sm, QMD_VA, CB0_VA, A_VA, B_VA, C_VA))
    print("QMD: " + " ".join("%08x" % w for w in struct.unpack("<64I", qmd)))
    print("negative control before launch: c[0]=%08x c[n-1]=%08x c[N-1]=%08x"
          % tuple(struct.unpack("<I", bytes(smem[o:o + 4]))[0] for o in (0, (n - 1) * 4, (N - 1) * 4)))

    if not ch.setup():
        faults.dump(c); return 1

    pb = []
    pb += fifomod.mthd(grchan.SUBC_HOST, 0x0010, SEM_VA >> 32, SEM_VA & 0xffffffff, 0x51e4a001, 0x2 | (1 << 24))
    pb += compute_state(TLS_VA, per_sm, CODE_VA)
    pb += m(0x02b4, QMD_VA >> 8)                                  # SEND_PCAS_A
    pb += m(0x02bc, 0x3)                                          # SEND_SIGNALING_PCAS_B: invalidate | schedule
    pb += m(0x0110, 0)                                            # WAIT_FOR_IDLE
    pb += grchan.sem_release(SEM_VA + 0x10, 0xc0de0008)            # completion
    ch.push(pb)
    t0 = time.time(); ch.kick()
    ok_host = c.wait(lambda: c.vram_r32(sem_pa) == 0x51e4a001, 2.0)
    ok_done = c.wait(lambda: c.vram_r32(sem_pa + 0x10) == 0xc0de0008, 5.0)
    dt = (time.time() - t0) * 1e3
    out = struct.unpack("<%dI" % N, bytes(smem[0:N * 4]))
    exp = [(a[i] + b[i]) & 0xffffffff for i in range(N)]
    good = sum(out[i] == exp[i] for i in range(n))
    tail_ok = all(out[i] == POISON for i in range(n, N))
    print("after kick (%.0f ms): %s  host sem %s  completion sem %08x %s"
          % (dt, ch.status(), "OK" if ok_host else "no", c.vram_r32(sem_pa + 0x10), "OK" if ok_done else "not released"))
    print("  c[0..3] = %s   expected %s" % (" ".join("%08x" % x for x in out[:4]), " ".join("%08x" % x for x in exp[:4])))
    print("  correct: %d / %d   tail [n, N) untouched (poison): %s   c[n]=%08x" % (good, n, tail_ok, out[n]))
    gr_error_dump(c)
    faults.dump(c)
    rt4 = ok_done and good == n and tail_ok
    print("RT-4 (COMPUTE DISPATCH, sm_61 vadd -> sysmem) RESULT:", "PASS" if rt4 else "FAIL")
    return 0 if rt4 else 1


if __name__ == "__main__":
    sys.exit(main())
