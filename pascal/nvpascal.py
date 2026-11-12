"""nvpascal: a minimal compute runtime for the GP102 over TinyGPU (rung 4 -> 5).

One GR channel (grchan.GRChannel) with persistent PASCAL_COMPUTE_B state, and:
  Device.alloc(size)        VRAM buffer, mapped into the channel VMM  -> Buffer(va, pa, size)
  Device.alloc_host(size)   host memory through the IOMMU (sys_coh)   -> HostBuffer(va, mem, size)
  Device.load(cubin)        every kernel of a ptxas sm_61 cubin into the code region
  Device.launch_async(k, grid, block, params)   queue a launch; returns its sequence number
  Device.htod / dtoh / copy_async / fill_async  stream-ordered copies (GPU kernels, host staging)
  Device.drain()            wait for everything queued;  Device.launch(...) = timed + synchronous

Launches are asynchronous and in order (one channel, WAIT_FOR_IDLE after each kernel). Everything
a launch needs lives in one GPU-mapped host buffer (the "host ring"): QMD, constant bank 0, the
pushbuffer segment, the GPFIFO and a completion counter the GPU releases after every launch. So a
launch costs one doorbell write over the transport, and waiting is reading host memory -- not
PRAMIN (70 MB/s write, 1 MB/s read) and 18 us BAR read round trips per launch.

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

# Program region: 240 MiB at a fixed VRAM range (MAX loads one module per kernel; llama.cpp's
# translation units are MBs each). Not zero-filled; code is uploaded through the staging ring.
CODE_VA, CODE_PA, CODE_SIZE = 0x40000000, 0x31000000, 0x0f000000
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
# Constant bank 0 per launch: driver area (0x140) + kernel params. sm_6x allows up to 0x1100 bytes of
# params, so a 4 KiB slot was too small: MAX's multi-output fused kernels (one output pointer per layer)
# reach ~4 KiB, and the params past 0x1000 spilled into the next slot / read as 0 in the kernel.
CB_SLOT = 0x2000
SEM_VA = 0x00600000
# Host ring (GPU-mapped host memory, sys_coh): see the module docstring.
HOST_VA, HOST_SIZE, NSLOT = 0x08000000, 0x3000000, 256
H_SEM, H_GPFIFO, H_PROF, H_FLUSH = 0x0, 0x1000, 0x2000, 0x4000
H_QMD, H_PB, H_CB, H_STAGE = 0x10000, 0x100000, 0x400000, 0x800000
PB_SLOT, STAGE_CHUNK = 0x2400, 0x400000                    # pushbuffer per slot: inline cbuf0 + QMD; 10 x 4 MiB staging
# Where a launch's QMD and constant bank 0 live (NVPASCAL_UPLOAD):
#   "host"   GPU-mapped host memory, written by the CPU (fetched over Thunderbolt by SKED / the SMs)
#   "cb"     cbuf0 uploaded to VRAM by the GPU from the pushbuffer (LOAD_INLINE_DATA), QMD in host
#   "all"    both uploaded inline to VRAM (default)
UPLOAD = os.environ.get("NVPASCAL_UPLOAD", "all")
VQMD_VA, VCB_VA = 0x06000000, 0x06100000                   # VRAM rings for the inline uploads
NSTAGE = (HOST_SIZE - H_STAGE) // STAGE_CHUNK
assert H_PB + NSLOT * PB_SLOT <= H_CB and H_CB + NSLOT * CB_SLOT <= H_STAGE
PROF = bool(os.environ.get("NVPASCAL_PROF"))                 # per-kernel GPU time (two timestamps/launch)
DATA_VA0 = 0x100000000                     # buffers live above 4 GiB of VA
DATA_PA0 = 0x40000000                      # and above 1 GiB of VRAM (clear of WPR2 @ 0x18040000)
HOSTMAP_VA0 = 0x800000000                  # alloc_host mappings


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
        self.data_heap = nvmmu.FreeListHeap(base=DATA_PA0)
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
            self.code_pa = CODE_PA
            ch.vmm.map(CODE_VA, CODE_PA, CODE_SIZE, flush=False)
            ch.vmm.map(TLS_VA, TLS_PA, TLS_PER_SM * TLS_SMS, flush=False)
            ch.vmm.map(MODDATA_VA, MODDATA_PA, MODDATA_SIZE, flush=False)
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
        self.seq = 0x100
        self.va_next = HOSTMAP_VA0                # host buffers (alloc_host): bump VA above the data range
        self.allocs = []                        # (va, size, Buffer), sorted by va
        self._submit(compute.compute_state(TLS_VA, TLS_PER_SM, CODE_VA), "compute state")
        self._host_ring_init()
        if UPLOAD != "host":
            self.vcb_pa = ch.alloc_map(VCB_VA, NSLOT * CB_SLOT)
            self.vqmd_pa = ch.alloc_map(VQMD_VA, NSLOT * 0x100)
            ch.vmm.flush()
        ks = self.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "kernels", "xfer.sm_61.cubin"))
        self.k_xcopy, self.k_xfill = ks["xcopy"], ks["xfill"]

    # ------------------------------------------------------------ host ring
    def _host_ring_init(self):
        c, ch = self.c, self.ch
        self.hmem, iovas = c.dev.alloc_sysmem(HOST_SIZE)
        self.hv = self.hmem.mv
        self.hv[0:H_STAGE] = bytes(H_STAGE)
        self.iovas = iovas
        # The FB's sysmem flush page (0x100c10, written on every system-memory barrier) must be a
        # page *this* process owns. gr.py points it at its own page, which TinyGPU frees when gr.py
        # exits; once the GPU touches host memory, barriers then write a dead or reused IOVA
        # (seen: a corrupted host-ring slot, then a cold card).
        c.wr(0x100c10, iovas[H_FLUSH // 0x1000] >> 8)
        i = 0
        while i < len(iovas):                     # map physically contiguous runs
            j = i + 1
            while j < len(iovas) and iovas[j] == iovas[j - 1] + 0x1000:
                j += 1
            ch.vmm.map(HOST_VA + i * 0x1000, iovas[i], (j - i) * 0x1000, aper="sys_coh", flush=False)
            i = j
        ch.vmm.flush()
        # Move the GPFIFO into host memory (idle here: the compute-state submit completed).
        n = ch.gp_entries * 8
        assert n <= H_PROF - H_GPFIFO
        self.hv[H_GPFIFO:H_GPFIFO + n] = c.vram_read(ch.gp_pa, n)
        ch.vmm.map(ch.gp_va, iovas[H_GPFIFO // 0x1000], nvmmu.align(n, 0x1000), aper="sys_coh")
        ch.gp_host = self.hv[H_GPFIFO:H_GPFIFO + n]
        self.hseq = 0                              # last sequence number queued
        self.hslot = 0
        self.slot_seq = [0] * NSLOT                # seq of the launch that last used each slot
        self.slot_prof = [None] * NSLOT            # kernel name, if that launch was timed
        self.stage_seq = [0] * NSTAGE
        self.stage_i = 0
        self.prof = {}                             # kernel name -> [launches, GPU ns]

    def done_seq(self):
        return struct.unpack_from("<I", self.hv, H_SEM)[0]

    def wait_seq(self, seq, timeout=20.0):
        """Wait until the launch numbered seq (and so everything before it) has completed."""
        if ((self.done_seq() - seq) & 0xffffffff) < 0x80000000:
            return
        t0 = time.perf_counter()
        while ((self.done_seq() - seq) & 0xffffffff) >= 0x80000000:
            dt = time.perf_counter() - t0
            if dt > 0.02:                          # spin first (kernels take 10 us - 10 ms)
                self.c.guard.check()
                time.sleep(0.0002)
            if dt > timeout:
                compute.gr_error_dump(self.c)
                faults.dump(self.c)
                raise RuntimeError("launch %d did not complete (done %d): %s"
                                   % (seq, self.done_seq(), self.ch.status()))

    def drain(self):
        self.wait_seq(self.hseq)
        if PROF:
            for s in range(NSLOT):
                self._prof_take(s)

    def _prof_take(self, s):
        name = self.slot_prof[s]
        if name is not None:
            o = H_PROF + s * 32
            t0, t1 = struct.unpack_from("<8xQ8xQ", self.hv, o)
            e = self.prof.setdefault(name, [0, 0])
            e[0] += 1; e[1] += t1 - t0
            self.slot_prof[s] = None

    @staticmethod
    def _inline(va, data):
        """Mesa nve4_compute_upload_input: the compute engine writes `data` to va (pitch, no
        sysmembar) from the pushbuffer -- OFFSET_OUT, LINE_LENGTH_IN/COUNT, then one ONE_INC
        header: LAUNCH_DMA, LOAD_INLINE_DATA x n."""
        assert len(data) % 4 == 0
        n = len(data) // 4
        hdr = 0xa0000000 | ((n + 1) << 16) | (grchan.SUBC_CLASS << 13) | (0x01b0 >> 2)
        return (compute.m(0x0188, va >> 32, va & 0xffffffff) + compute.m(0x0180, len(data), 1)
                + [hdr, 0x41] + list(struct.unpack("<%dI" % n, data)))

    def _hsem(self, va, payload, four_word):
        return grchan.fifomod.mthd(grchan.SUBC_CLASS, 0x1b00, va >> 32, va & 0xffffffff, payload,
                                   0 if four_word else (1 << 28))

    def launch_async(self, k, grid, block, params: bytes, smem=0, timed=False):
        """Queue a launch; returns its sequence number (see wait_seq)."""
        s = self.hslot; self.hslot = (s + 1) % NSLOT
        self.wait_seq(self.slot_seq[s])            # the GPU is done with this slot's last use
        self._prof_take(s)
        hv = self.hv
        lmem = k.get("lmem", 0)
        cb = compute.cbuf0(k, grid, block, params, stack=lmem, windows=(SHARED_WINDOW, LOCAL_WINDOW))
        if len(cb) % 16:                           # params need not end on a word (ggml: 117 bytes)
            cb += bytes(16 - len(cb) % 16)
        if len(cb) > CB_SLOT:
            raise ValueError("%s: constant bank 0 needs %#x bytes (params %#x), slot is %#x"
                             % (k["name"], len(cb), len(params), CB_SLOT))
        po = H_PB + s * PB_SLOT
        m = compute.m
        pb = []
        if UPLOAD == "host":
            cb_va, qmd_va = HOST_VA + H_CB + s * CB_SLOT, HOST_VA + H_QMD + s * 0x100
            hv[cb_va - HOST_VA:cb_va - HOST_VA + len(cb)] = cb
        else:
            cb_va = VCB_VA + s * CB_SLOT
            qmd_va = VQMD_VA + s * 0x100 if UPLOAD == "all" else HOST_VA + H_QMD + s * 0x100
            pb += self._inline(cb_va, cb)
        qmd = compute.qmd_fast(k, tuple(grid), tuple(block), cb_va, nvmmu.align(len(cb), 0x100), dyn_shared=smem)
        if UPLOAD == "all":
            pb += self._inline(qmd_va, qmd)
        else:
            hv[qmd_va - HOST_VA:qmd_va - HOST_VA + 0x100] = qmd
        self.hseq = seq = (self.hseq + 1) & 0xffffffff
        if self.icache_dirty:                   # INVALIDATE_SHADER_CACHES: INSTRUCTION | DATA | CONSTANT
            pb += m(0x021c, (1 << 0) | (1 << 4) | (1 << 12))
            self.icache_dirty = False
        timed = timed or PROF
        if timed:
            pb += self._hsem(HOST_VA + H_PROF + s * 32, 0, True)
        pb += m(0x02b4, qmd_va >> 8) + m(0x02bc, 0x3) + m(0x0110, 0)
        if timed:
            pb += self._hsem(HOST_VA + H_PROF + s * 32 + 16, 0, True)
        pb += self._hsem(HOST_VA + H_SEM, seq, False)
        data = struct.pack("<%dI" % len(pb), *pb)
        assert len(data) <= PB_SLOT
        hv[po:po + len(data)] = data
        self.slot_seq[s] = seq
        self.slot_prof[s] = k["name"] if timed else None
        self.ch.gp_entry(HOST_VA + po, len(data))
        self.ch.kick()
        return seq

    # ------------------------------------------------------------ stream-ordered copies
    @staticmethod
    def _grid(n):
        return (max(1, min(28 * 8, (n // 16 + 255) // 256)), 1, 1)

    def copy_async(self, dst, src, n):
        if n:
            return self.launch_async(self.k_xcopy, self._grid(n), (256, 1, 1), struct.pack("<QQQ", src, dst, n))

    def fill_async(self, dst, pat32, n):
        if n:
            return self.launch_async(self.k_xfill, self._grid(n), (256, 1, 1), struct.pack("<QI4xQ", dst, pat32, n))

    def _stage(self):
        i = self.stage_i; self.stage_i = (i + 1) % NSTAGE
        self.wait_seq(self.stage_seq[i])
        return i, H_STAGE + i * STAGE_CHUNK

    def htod(self, va, data):
        """Host -> device, in stream order (data is captured before this returns)."""
        mv = memoryview(data)
        for off in range(0, len(mv), STAGE_CHUNK):
            part = mv[off:off + STAGE_CHUNK]
            i, ho = self._stage()
            self.hv[ho:ho + len(part)] = part
            self.stage_seq[i] = self.copy_async(va + off, HOST_VA + ho, len(part))

    def dtoh(self, va, n):
        """Device -> host after everything queued before it."""
        out = bytearray(n)
        for off in range(0, n, STAGE_CHUNK):
            k = min(STAGE_CHUNK, n - off)
            i, ho = self._stage()
            self.stage_seq[i] = seq = self.copy_async(HOST_VA + ho, va + off, k)
            self.wait_seq(seq)
            out[off:off + k] = self.hv[ho:ho + k]
        return bytes(out)

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
        """VRAM buffer. Its VA is its PA plus a constant, so reused VRAM reuses its page tables
        (a fresh VA per allocation exhausted the page-table heap after ~15k llama.cpp allocations)."""
        size = nvmmu.align(size, 0x1000)
        pa = self.data_heap.alloc(size, 0x200000 if size >= 0x200000 else 0x1000)
        va = DATA_VA0 + (pa - DATA_PA0)
        if zero:
            self.c.vram_write(pa, bytes(size))
        try:
            self.ch.vmm.map(va, pa, size)
        except BaseException:
            self.data_heap.free(pa)
            raise
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
        """Unmap and return the VRAM to the heap (the VA is not reused)."""
        self.drain()                            # queued kernels may still use it
        for i, (base, size, buf) in enumerate(self.allocs):
            if base == va:
                self.ch.vmm.unmap(va, size)
                self.data_heap.free(buf.pa)
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
        put = self.htod if hasattr(self, "k_xcopy") else (lambda va, d: self.c.vram_write(va - CODE_VA + CODE_PA if va >= CODE_VA else MODDATA_PA + (va - MODDATA_VA), d))
        for n, va in sec_va.items():
            put(va, bytes(images[n]))
        def banks_for(kernel):                         # module-level banks + the kernel's own
            b = {}
            for n, (num, owner) in kbank.items():
                if owner in (None, kernel):
                    assert num not in b or owner == kernel, "bank %d defined twice for %s" % (num, kernel)
                    b[num] = (sec_va[n], len(images[n]))
            return b
        for name, k in ks.items():
            k["code"] = bytes(images[".text." + name])
            put(CODE_VA + k["prog_off"], k["code"])
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
        self.drain()
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
        s = self.hslot
        self.wait_seq(self.launch_async(k, grid, block, params, smem=smem, timed=True))
        t0, t1 = struct.unpack_from("<8xQ8xQ", self.hv, H_PROF + s * 32)
        if not PROF:
            self.slot_prof[s] = None
        return t1 - t0
