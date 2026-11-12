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
from sleepguard import SystemSlept   # re-exported for shimd
import mmu as nvmmu
import fifo as fifomod
import gr as grmod
import grctx as grctxmod
import grchan, cubin as cubinmod, compute, faults, fb

CODE_VA, CODE_SIZE = 0x04000000, 0x2000000  # 32 MiB: MAX loads one module per kernel
STUB_OFF, CODE_OFF0 = 0x0, 0x100             # code offset 0: shared stubs (vprintf -> ret)
STUB_SYMBOLS = ("vprintf",)                  # externals resolved to the ret stub
# Thread-local memory (stack, spills), sized like Mesa nvc0_screen_resize_tls_area(128*16, 0, cstack):
# per SM = (LMEM_MAX * 32 lanes + CRS per warp) * 64 warps.  Own VRAM range, clear of every heap.
LMEM_MAX, CRS_PER_WARP, WARPS_PER_SM, TLS_SMS = 0x800, 0x800, 64, 32
TLS_PER_SM = (LMEM_MAX * 32 + CRS_PER_WARP) * WARPS_PER_SM      # 0x420000 (multiple of 0x8000)
TLS_VA, TLS_PA = 0x10000000, 0x20000000                         # 132 MiB -> PA 0x28400000
# Module data: .nv.global / .nv.global.init / .nv.constantN (N > 0) of every loaded cubin.
MODDATA_VA, MODDATA_PA, MODDATA_SIZE = 0x30000000, 0x30000000, 0x1000000
# Generic-address windows (compute_state); the cbuf0 driver area tells kernels where they are.
SHARED_WINDOW, LOCAL_WINDOW = 0xfe000000, 0xff000000
QMD_VA, CB_VA, RING = 0x06000000, 0x06100000, 64
# Constant bank 0 per launch: driver area (0x140) + kernel params. sm_6x allows up to 0x1100 bytes of
# params, so a 4 KiB slot was too small: MAX's multi-output fused kernels (one output pointer per layer)
# reach ~4 KiB, and the params past 0x1000 spilled into the next slot / read as 0 in the kernel.
CB_SLOT = 0x2000
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
            ch.vmm.map(TLS_VA, TLS_PA, TLS_PER_SM * TLS_SMS, flush=False)
            ch.vmm.map(MODDATA_VA, MODDATA_PA, MODDATA_SIZE, flush=False)
            self.qmd_pa = ch.alloc_map(QMD_VA, RING * 0x100)
            self.cb_pa = ch.alloc_map(CB_VA, RING * CB_SLOT)
            self.sem_pa = ch.alloc_map(SEM_VA, 0x1000)
            ch.vmm.flush()
            ok = ch.setup()
        finally:
            if not verbose:
                sys.stdout.close(); sys.stdout = out
        if not ok:
            raise RuntimeError("GR channel setup failed")
        self.c.vram_write(self.code_pa + STUB_OFF, struct.pack(
            "<4Q", 0x7ef | 0x7ef << 21 | 0x7ef << 42,             # sched: stall 15, no barriers
            0xe32000000007000f, 0x50b0000000070f00, 0x50b0000000070f00))   # RET; NOP; NOP
        self.code_off = CODE_OFF0
        self.moddata_off = 0
        self.icache_dirty = True
        self.slot = 0
        self.seq = 0x100
        self.va_next = DATA_VA0
        self.allocs = []                        # (va, size, Buffer), sorted by va
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
        buf = Buffer(self, va, pa, size)
        self.allocs.append((va, size, buf))
        self.allocs.sort(key=lambda t: t[0])
        return buf

    def find(self, va):
        """(Buffer, offset) for a device address inside any allocation (MAX sub-allocates a pool)."""
        import bisect
        i = bisect.bisect_right([a[0] for a in self.allocs], va) - 1
        if i >= 0:
            base, size, buf = self.allocs[i]
            if va < base + size:
                return buf, va - base
        raise KeyError("device address %#x is not inside any allocation" % va)

    def free(self, va):
        """Unmap; VRAM is not recycled yet (bump heap) -- MAX allocates one pool."""
        for i, (base, size, buf) in enumerate(self.allocs):
            if base == va:
                self.ch.vmm.unmap(va, size)
                del self.allocs[i]
                return
        raise KeyError("free of unknown address %#x" % va)

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
        """Load every kernel of an sm_61 cubin (path or bytes) the way the CUDA driver does:
        code into the program region, module data sections (.nv.global, .nv.global.init,
        .nv.constantN) into module memory, R_CUDA_32/64 relocations applied, and per-kernel
        local-memory size and constant banks recorded for the launch."""
        data = open(path, "rb").read() if isinstance(path, str) else path
        cb = cubinmod.Cubin(data)
        assert cb.sm == 61, "cubin is sm_%d, need sm_61" % cb.sm
        images, sec_va, kbank = {}, {}, {}           # kbank: section -> (bank, owning kernel or None)
        kernels = set(cb.kernels())
        for sec in cb.section_list:
            n = sec["name"]
            if n.startswith(".nv.constant"):          # .nv.constantN (module) / .nv.constantN.<kernel>
                num, _, owner = n[len(".nv.constant"):].partition(".")
                if not num.isdigit() or num == "0" or (owner and owner not in kernels):
                    continue                          # bank 0 is built per launch (cbuf0)
                kbank[n] = (int(num), owner or None)
            if n in (".nv.global", ".nv.global.init") or n in kbank:
                off = nvmmu.align(self.moddata_off, max(sec["align"], 0x100))
                assert off + sec["size"] <= MODDATA_SIZE, "module data region full"
                self.moddata_off = off + sec["size"]
                sec_va[n] = MODDATA_VA + off
                images[n] = bytearray(cb.sec(n))
        ks, text_off = {}, {}
        for name in cb.kernels():
            k = cb.kernel(name)
            off = nvmmu.align(self.code_off, 0x100)
            assert off + len(k["code"]) + 0x100 <= CODE_SIZE, "code region full"
            self.code_off = off + len(k["code"]) + 0x100      # gap: instruction prefetch
            k["prog_off"] = text_off[".text." + name] = off
            images[".text." + name] = bytearray(k["code"])
            ks[name] = k
        for target, roff, si, typ, addend in cb.relocations():
            sname, _, shndx, sval, _ = cb.symbols[si]
            if shndx == 0:
                if sname not in STUB_SYMBOLS:
                    raise RuntimeError("cubin needs external %r (no driver implementation)" % sname)
                S = STUB_OFF
            else:
                ssec = cb.section_list[shndx]["name"]
                if ssec in sec_va:
                    S = sec_va[ssec] + sval
                elif ssec in text_off:
                    S = text_off[ssec] + sval
                else:
                    raise RuntimeError("relocation against %r in unsupported section %s" % (sname, ssec))
            img = images.get(target)
            if img is None:
                raise RuntimeError("relocation in unsupported section %s (%r)" % (target, sname))
            fmt = {1: "<I", 2: "<Q"}.get(typ)                  # R_CUDA_32, R_CUDA_64
            if fmt is None:
                raise NotImplementedError("relocation type %d (%r in %s)" % (typ, sname, target))
            A = struct.unpack_from(fmt, img, roff)[0] if addend is None else addend
            struct.pack_into(fmt, img, roff, (S + A) & (0xffffffff if typ == 1 else (1 << 64) - 1))
        for n, va in sec_va.items():
            self.c.vram_write(MODDATA_PA + (va - MODDATA_VA), bytes(images[n]))
        def banks_for(kernel):                         # module-level banks + the kernel's own
            b = {}
            for n, (num, owner) in kbank.items():
                if owner in (None, kernel):
                    assert num not in b or owner == kernel, "bank %d defined twice for %s" % (num, kernel)
                    b[num] = (sec_va[n], len(images[n]))
            return b
        for name, k in ks.items():
            k["code"] = bytes(images[".text." + name])
            self.c.vram_write(self.code_pa + k["prog_off"], k["code"])
            need = max(k["frame"], k["min_stack"], k["max_stack"])
            assert need <= LMEM_MAX - 0x100, "%s needs %d bytes of local memory per thread" % (name, need)
            k["lmem"] = 0 if not need else nvmmu.align(need, 0x10) + 0x100   # + slack for callees
            k["cbanks"] = banks_for(name)
        # The SM instruction cache prefetches past the end of the last kernel it ran, so bytes we
        # just wrote may be cached stale (zeros) -> warp ESR 0x9 ILLEGAL_INSTR_ENCODING on the next
        # module's first launch.  Invalidate it before the next launch.
        self.icache_dirty = True
        return ks

    def module_mark(self):
        return self.code_off, self.moddata_off

    def module_rewind(self, mark):
        """Forget every module loaded after `mark` (their code/data space is reused)."""
        self.code_off, self.moddata_off = mark
        self.icache_dirty = True

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
        done = lambda: self.c.vram_r32(self.sem_pa + off) == payload
        t0 = time.perf_counter()                 # spin first: time.sleep(0.001) costs ~1.5 ms on
        while not done() and time.perf_counter() - t0 < 0.02:   # macOS, kernels take 10-200 us
            pass
        if not done() and not self.c.wait(done, timeout):
            compute.gr_error_dump(self.c)
            faults.dump(self.c)
            raise RuntimeError("submission %r did not complete: %s" % (what, self.ch.status()))

    def launch(self, k, grid, block, params: bytes, smem=0):
        """Synchronous launch; returns GPU-measured kernel time in ns. smem = dynamic shared bytes."""
        s = self.slot; self.slot = (self.slot + 1) % RING
        qmd_va, cb_va = QMD_VA + s * 0x100, CB_VA + s * CB_SLOT
        lmem = k.get("lmem", 0)
        cb = compute.cbuf0(k, grid, block, params, stack=lmem, windows=(SHARED_WINDOW, LOCAL_WINDOW))
        if len(cb) > CB_SLOT:
            raise ValueError("%s: constant bank 0 needs %#x bytes (params %#x), slot is %#x"
                             % (k["name"], len(cb), len(params), CB_SLOT))
        self.c.vram_write(self.cb_pa + s * CB_SLOT, cb)
        self.c.vram_write(self.qmd_pa + s * 0x100, compute.build_qmd(
            k, grid, block, cb_va, nvmmu.align(len(cb), 0x100), prog_off=k["prog_off"], dyn_shared=smem,
            lmem=lmem, cbanks=k.get("cbanks", {})))
        o0, s0 = self._sem(0x7a000000 | s, four_word=True)
        o1, s1 = self._sem(0x7b000000 | s, four_word=True)
        for o in (o0, o1):
            self.c.vram_write(self.sem_pa + o, bytes(16))
        m = compute.m
        pb = []
        if self.icache_dirty:                   # INVALIDATE_SHADER_CACHES: INSTRUCTION | DATA | CONSTANT
            pb += m(0x021c, (1 << 0) | (1 << 4) | (1 << 12))
            self.icache_dirty = False
        pb += m(0x0110, 0) + s0 + m(0x02b4, qmd_va >> 8) + m(0x02bc, 0x3) + m(0x0110, 0) + s1
        self._submit(pb, k["name"])
        w0 = struct.unpack("<IIQ", self.c.vram_read(self.sem_pa + o0, 16))
        w1 = struct.unpack("<IIQ", self.c.vram_read(self.sem_pa + o1, 16))
        return w1[2] - w0[2]
