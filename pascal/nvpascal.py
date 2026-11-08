"""nvpascal: a minimal compute runtime for the GP102 over TinyGPU (rung 4 -> 5).

One GR channel (grchan.GRChannel) with persistent PASCAL_COMPUTE_B state, and:
  Device.alloc(size)        VRAM buffer, mapped into the channel VMM  -> Buffer(va, pa, size)
  Device.alloc_host(size)   host memory through the IOMMU (sys_coh)   -> HostBuffer(va, mem, size)
  Device.load(cubin)        every kernel of a ptxas sm_61 cubin into the code region
  Device.launch(k, grid, block, params)   QMD v02_01 + cbuf0, PCAS, WFI; returns GPU ns
                            (from four-word report-semaphore timestamps around the launch)

This is the API a libcuda shim needs underneath (cuMemAlloc / cuMemcpy / cuModuleLoadData /
cuLaunchKernel / cuStreamSynchronize).  Launches are synchronous for now.

Prerequisite on a warm card: pascal/reset_engines.py + pascal/gr.py.
"""
import os, struct, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from card import Card
import mmu as nvmmu
import fifo as fifomod
import gr as grmod
import grctx as grctxmod
import grchan, cubin as cubinmod, compute, faults, fb

CODE_VA, CODE_SIZE = 0x04000000, 0x100000
TLS_VA, TLS_PER_SM, TLS_SMS = 0x05000000, 0x20000, 32
QMD_VA, CB_VA, RING = 0x06000000, 0x06100000, 64
SEM_VA = 0x00600000
DATA_VA0 = 0x100000000                     # buffers live above 4 GiB of VA
DATA_PA0 = 0x40000000                      # and above 1 GiB of VRAM (clear of WPR2 @ 0x18040000)


class Buffer:
    def __init__(self, dev, va, pa, size):
        self.dev, self.va, self.pa, self.size = dev, va, pa, size

    def write(self, data, off=0):
        self.dev.c.vram_write(self.pa + off, data)

    def read(self, n=None, off=0):
        return self.dev.c.vram_read(self.pa + off, self.size - off if n is None else n)


class HostBuffer:
    def __init__(self, dev, va, mem, size):
        self.dev, self.va, self.mem, self.size = dev, va, mem, size

    def write(self, data, off=0):
        self.mem[off:off + len(data)] = data

    def read(self, n=None, off=0):
        return bytes(self.mem[off:off + (self.size - off if n is None else n)])


class Device:
    def __init__(self, verbose=False):
        c = self.c = Card()
        if not grchan.gr_ready(c):
            raise RuntimeError("GR not up: run pascal/reset_engines.py + pascal/gr.py")
        self.g = grmod.GR(c).oneinit(verbose=verbose)
        heap = nvmmu.VramHeap(base=0x11000000, limit=0x18000000)      # channel + ctx, below WPR2
        self.data_heap = nvmmu.VramHeap(base=DATA_PA0)
        self.f = fifomod.Fifo(c); self.f.init(verbose=verbose)
        devs = {d["name"]: d for d in fb.top_parse(c)}
        for d in devs.values():                 # all engines on GR's runlist out of reset
            if d["runlist"] == devs["GR"]["runlist"] and d.get("reset") is not None:
                c.mask(0x200, 1 << d["reset"], 1 << d["reset"])
        if c.rd(0x262c) & 1:
            c.wr(0x262c, 1)
        faults.dump(c, clear=True) if verbose else self._clear_faults()
        out = sys.stdout
        if not verbose:
            sys.stdout = open(os.devnull, "w")
        try:
            gc = grctxmod.GrCtx(c, self.g, heap)
            self.ch = ch = grchan.GRChannel(self.f, chid=1, gc=gc)
            self.code_pa = ch.alloc_map(CODE_VA, CODE_SIZE)
            ch.alloc_map(TLS_VA, TLS_PER_SM * TLS_SMS)
            self.qmd_pa = ch.alloc_map(QMD_VA, RING * 0x100)
            self.cb_pa = ch.alloc_map(CB_VA, RING * 0x1000)
            self.sem_pa = ch.alloc_map(SEM_VA, 0x1000)
            ch.vmm.flush()
            ok = ch.setup()
        finally:
            if not verbose:
                sys.stdout.close(); sys.stdout = out
        if not ok:
            raise RuntimeError("GR channel setup failed")
        self.code_off = 0
        self.slot = 0
        self.seq = 0x100
        self.va_next = DATA_VA0
        self._submit(compute.compute_state(TLS_VA, TLS_PER_SM, CODE_VA), "compute state")

    def _clear_faults(self):
        import io, contextlib
        with contextlib.redirect_stdout(io.StringIO()):
            faults.dump(self.c, clear=True)

    # ------------------------------------------------------------ memory
    def _va(self, size, align=0x200000):
        va = nvmmu.align(self.va_next, align)
        self.va_next = va + nvmmu.align(size, 0x1000)
        return va

    def alloc(self, size, zero=False):
        size = nvmmu.align(size, 0x1000)
        pa = self.data_heap.alloc(size, 0x200000 if size >= 0x200000 else 0x1000)
        va = self._va(size)
        if zero:
            self.c.vram_write(pa, bytes(size))
        self.ch.vmm.map(va, pa, size)
        return Buffer(self, va, pa, size)

    def alloc_host(self, size):
        size = nvmmu.align(size, 0x1000)
        mem, iovas = self.c.dev.alloc_sysmem(size)
        va = self._va(size)
        for i, iova in enumerate(iovas):
            self.ch.vmm.map(va + i * 0x1000, iova, 0x1000, aper="sys_coh", flush=False)
        self.ch.vmm.flush()
        return HostBuffer(self, va, mem, size)

    # ------------------------------------------------------------ code
    def load(self, path):
        cb = cubinmod.Cubin(open(path, "rb").read())
        assert cb.sm == 61, "cubin is sm_%d, need sm_61" % cb.sm
        ks = {}
        for name in cb.kernels():
            k = cb.kernel(name)
            off = nvmmu.align(self.code_off, 0x100)
            assert off + len(k["code"]) + 0x100 <= CODE_SIZE, "code region full"
            self.c.vram_write(self.code_pa + off, k["code"])
            k["prog_off"] = off
            self.code_off = off + len(k["code"]) + 0x100      # gap: instruction prefetch
            ks[name] = k
        return ks

    # ------------------------------------------------------------ submission
    def _sem(self, payload, four_word=False):
        self.seq += 1
        off = (self.seq % 128) * 0x20
        return off, grchan.fifomod.mthd(grchan.SUBC_CLASS, 0x1b00, (SEM_VA + off) >> 32,
                                        (SEM_VA + off) & 0xffffffff, payload,
                                        0 if four_word else (1 << 28))

    def _submit(self, pb, what="", timeout=10.0, timed_sems=None):
        off, done = self._sem(0x5e000000 | (self.seq & 0xffffff))
        payload = struct.unpack_from("<I", struct.pack("<I", done[3]))[0]
        self.c.vram_w32(self.sem_pa + off, 0)
        self.ch.push(pb + grchan.fifomod.mthd(grchan.SUBC_CLASS, 0x0110, 0) + done)
        self.ch.kick()
        if not self.c.wait(lambda: self.c.vram_r32(self.sem_pa + off) == payload, timeout):
            compute.gr_error_dump(self.c)
            faults.dump(self.c)
            raise RuntimeError("submission %r did not complete: %s" % (what, self.ch.status()))

    def launch(self, k, grid, block, params: bytes):
        """Synchronous launch; returns GPU-measured kernel time in ns."""
        s = self.slot; self.slot = (self.slot + 1) % RING
        qmd_va, cb_va = QMD_VA + s * 0x100, CB_VA + s * 0x1000
        self.c.vram_write(self.cb_pa + s * 0x1000, compute.cbuf0(k, grid, block, params))
        self.c.vram_write(self.qmd_pa + s * 0x100,
                          compute.build_qmd(k, grid, block, cb_va, 0x1000, prog_off=k["prog_off"]))
        o0, s0 = self._sem(0x7a000000 | s, four_word=True)
        o1, s1 = self._sem(0x7b000000 | s, four_word=True)
        for o in (o0, o1):
            self.c.vram_write(self.sem_pa + o, bytes(16))
        m = compute.m
        pb = m(0x0110, 0) + s0 + m(0x02b4, qmd_va >> 8) + m(0x02bc, 0x3) + m(0x0110, 0) + s1
        self._submit(pb, k["name"])
        w0 = struct.unpack("<IIQ", self.c.vram_read(self.sem_pa + o0, 16))
        w1 = struct.unpack("<IIQ", self.c.vram_read(self.sem_pa + o1, 16))
        return w1[2] - w0[2]
