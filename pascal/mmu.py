"""Pascal MMU v2 (GP100-class) page tables, instance block, and falcon binding.

A faithful port of nouveau's nvkm MMU v2 for the GTX 1080 Ti (GP102), for the
one job the ACR needs first: give a falcon a *virtual* address space, so its DMA
stops being a physical access.  The FB hub refuses physical DMA from a falcon
that is not yet in secure mode (faults.py decodes it as SEC / PHYSICAL /
PRIV_VIOLATION), which is why the ACR HS bootloader hangs on its first transfer.

Ported from:
  nvkm/subdev/mmu/vmmgp100.c   PTE / PDE / PD0 formats, join, flush, 4 KiB desc
  nvkm/subdev/mmu/vmmgf100.c   gf100_vmm_join_, gf100_vmm_invalidate, aper
  nvkm/subdev/mmu/vmm.c        nvkm_vmm_ctor / nvkm_vmm_iter (the level walk)
  nvkm/subdev/acr/base.c       acr->inst = 4 KiB VRAM, acr->vmm = 4 KiB VMM, join
  nvkm/falcon/gm200.c          gm200_flcn_bind_inst + the "fw->inst" load path

Address space
=============
49-bit VA, 4 KiB pages.  nouveau derives that number from the level descriptors
of gp100_vmm_desc_12 (9+8+9+9+2 = 37 bits of index) plus the page shift (12).
Indices are taken from `va >> 12`, bottom-up, exactly as nvkm_vmm_iter does it
(`it.pte[lvl]`, lvl 0 = the page table):

    bits [11:0]    page offset
    bits [20:12]   PT  index   9 bits, 512 entries,  8 B each, 2 MiB covered
    bits [28:21]   PD0 index   8 bits, 256 entries, 16 B each, 512 MiB covered
    bits [37:29]   PD1 index   9 bits, 512 entries,  8 B each, 256 GiB covered
    bits [46:38]   PD2 index   9 bits, 512 entries,  8 B each, 128 TiB covered
    bits [48:47]   PD3 index   2 bits,   4 entries,  8 B each, top-level table

Entry formats (little-endian; every address field is `addr >> 4`, so page tables
and pages must both be 4 KiB aligned)
=====================================
  PDE, 8 B, points at the next page table (gp100_vmm_pd1_pde -> gp100_vmm_pde)
      [0]    0
      [2:1]  aperture of the *table*: 1 = VRAM, 2 = sysmem coherent,
             3 = sysmem noncoherent (0 = "no table here")
      [3]    VOL -- system memory only
      [63:4] table address >> 4
  PD0 PDE, 16 B, *dual* entry (gp100_vmm_pd0_pde).  The low 8 bytes point at the
      big-page (64 KiB) table, the high 8 at the small-page (4 KiB) table:
      nvkm_vmm_ref_hwpt puts the PT in `pgt->pt[desc[it->lvl-1].type == SPT]`,
      i.e. index 1, and gp100_vmm_pd0_pde writes that as data[1] -- the high half.
  PTE, 8 B, 4 KiB leaf (gp100_vmm_pgt_pte -> gp100_vmm_valid's map->type)
      [0]    VALID
      [2:1]  aperture of the *page*: 0 = VRAM, 2 = sysmem coherent,
             3 = sysmem noncoherent   (unlike a PDE, VRAM is 0 here)
      [3]    VOL -- system memory only
      [5]    PRIV
      [6]    RO
      [7]    atomic disable
      [63:4] page address >> 4   (the kind field is [63:56] but stays 0:
                                  gm200_mmu_kind[0] = 0x00, uncompressed)

Instance block (VRAM, 4 KiB aligned; gf100_vmm_join_ / gp100_vmm_join)
====================================================================
  0x200  u64  PD base | BIT(10) VER2 | BIT(11) 64 KiB | aperture (0 = VRAM)
  0x208  u64  VA limit - 1  (49-bit space -> 0x1ffffffffffff)

Tables are allocated by nvkm_mmu_ptc_get(mmu, size, align, zero) with zero =
true at every level we use (pages are not sparse, and gp100_vmm_desc_spt has no
`invalid` function), so each fresh table starts zeroed.
"""

import os
import struct
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ---------------------------------------------------------------- constants
# gf100_vmm_aper(): the aperture number carried in a PTE.
PAGE_APER = {"vram": 0, "sys_coh": 2, "sys_ncoh": 3}
# gp100_vmm_pde(): the aperture number carried in a PDE -- 1, not 0, for VRAM.
PT_APER = {"vram": 1, "sys_coh": 2, "sys_ncoh": 3}

PTE_VALID = 1 << 0
PTE_VOL = 1 << 3          # system memory is volatile
PTE_PRIV = 1 << 5
PTE_RO = 1 << 6
PTE_ATOMIC_OFF = 1 << 7

INST_PDB = 0x200          # instance block: page-directory base
INST_LIMIT = 0x208
PDB_VER2 = 1 << 10        # gp100_vmm_join: BIT_ULL(10)
PDB_BIG_PAGE_64K = 1 << 11  # gp100_vmm_join: BIT_ULL(11)

# MMU v2 PRI/MMU registers (gf100_vmm_invalidate / gp100_vmm_invalidate_pdb)
R_INVALIDATE = 0x100c80   # free-slot counter, trigger and completion bits
R_INVALIDATE_PDB_LO = 0x100cb8
R_INVALIDATE_PDB_HI = 0x100cec
R_INVALIDATE_CMD = 0x100cbc

FLUSH_PAGE_ALL = 0x00000001
FLUSH_ALL_PDB = 0x00000002
FLUSH_HUB_ONLY = 0x00000004

# falcon registers used by the "fw->inst" path (gm200_flcn_bind_inst,
# gm200_flcn_bind_stat, and gm200_flcn_fw_load)
F_INTR_CLEAR = 0x004
F_INTR = 0x008
F_FIFO_ENABLE = 0x048
F_CHANNEL_NEXT = 0x054     # the instance block address goes here
F_CHANNEL_CMD = 0x058
F_UNK090 = 0x090
F_ENG_CONTROL = 0x0a4
F_BIND_STATUS = 0x0dc      # bits 12-14
F_TRANSCFG = 0x600         # + 4 * dma index; that port's memory type
DMAIDX_VIRT = 1            # -> 0x604, the port the ACR bootloader DMA uses

PDE_BYTES = 8
PD0_BYTES = 16
PTE_BYTES = 8
PT_ENTRIES = 512          # 4 KiB pages: one PT = 512 PTEs = 2 MiB

# (va >> shift) index widths, bottom-up (gp100_vmm_desc_12)
SPLIT = ((12, 9), (21, 8), (29, 9), (38, 9), (47, 2))
# entry size for the table indexed by level L, and the level of the top table
LEVEL_BYTES = {0: PTE_BYTES, 1: PD0_BYTES, 2: PDE_BYTES, 3: PDE_BYTES, 4: PDE_BYTES}
TOP_LEVEL = 4


def align(x, a):
    return (x + a - 1) & ~(a - 1)


class FreeListHeap:
    """First-fit allocator with coalescing over [base, limit): VRAM for data buffers, which
    llama.cpp frees and reallocates (compute buffers grow during graph reservation)."""

    def __init__(self, base, limit=0x2b0000000):
        self.base, self.limit = base, limit
        self.free_list = [(base, limit - base)]       # sorted, non-adjacent (addr, size)
        self.used = {}                                 # addr -> size

    def alloc(self, size, align_to=0x1000):
        for i, (a, n) in enumerate(self.free_list):
            s = align(a, align_to)
            if s + size <= a + n:
                rest = []
                if s > a:
                    rest.append((a, s - a))
                if s + size < a + n:
                    rest.append((s + size, a + n - s - size))
                self.free_list[i:i + 1] = rest
                self.used[s] = size
                return s
        raise AssertionError("VRAM heap exhausted: no free block of %#x bytes (largest %#x)"
                             % (size, max([n for _, n in self.free_list] or [0])))

    def free(self, addr):
        size = self.used.pop(addr)
        import bisect
        i = bisect.bisect_left(self.free_list, (addr, 0))
        self.free_list.insert(i, (addr, size))
        j = max(i - 1, 0)                              # coalesce with the neighbours
        while j + 1 < len(self.free_list):
            a, n = self.free_list[j]; b, m = self.free_list[j + 1]
            if a + n == b:
                self.free_list[j:j + 2] = [(a, n + m)]
            elif j >= i:
                break
            else:
                j += 1


class VramHeap:
    """Bump allocator over physical VRAM owned by the driver.

    Everything it hands out is inside the low, non-VPR range that fb.py has
    proven readable and writable.
    """

    def __init__(self, base=0x10000000, limit=0x2b0000000):
        self.cur, self.limit, self.blocks = base, limit, []

    def alloc(self, size, align_to=0x1000):
        self.cur = align(self.cur, align_to)
        a = self.cur
        self.cur += size
        assert self.cur <= self.limit, "VRAM heap exhausted (%#x > %#x)" % (self.cur, self.limit)
        self.blocks.append((a, size))
        return a

    def __repr__(self):
        return "VramHeap(%d blocks, next=%#x)" % (len(self.blocks), self.cur)


class VMM:
    """A GP100-class (MMU v2) virtual address space with 4 KiB pages in VRAM."""

    LIMIT = 1 << 49

    def __init__(self, card, heap=None, va_base=0):
        self.c = card
        self.heap = heap if heap is not None else VramHeap()
        self.va_base = va_base
        self.tables = {}                       # path tuple -> table phys addr
        self.pd3 = self._table(TOP_LEVEL)      # nvkm_vmm_ctor: desc->size << bits
        self.tables[(TOP_LEVEL,)] = self.pd3
        self.mapped = []                       # (va, pa, size, aper) bookkeeping

    # ------------------------------------------------------------ tables
    @staticmethod
    def table_size(level):
        return LEVEL_BYTES[level] * (1 << SPLIT[level][1])

    def _table(self, level):
        """nvkm_mmu_ptc_get(..., zero=true): fresh GPU storage, zeroed."""
        a = self.heap.alloc(self.table_size(level), 0x1000)
        self.c.vram_write(a, bytes(self.table_size(level)))
        return a

    def indices(self, va):
        """nvkm_vmm_iter's it.pte[]: bottom-up (PT, PD0, PD1, PD2, PD3)."""
        return [(va >> sh) & ((1 << bits) - 1) for sh, bits in SPLIT]

    def _ensure(self, va):
        """Descend from the top, creating tables and writing their PDEs.

        Returns the page table (level 0) covering `va`.  This is
        nvkm_vmm_iter's walk-down step plus nvkm_vmm_ref_hwpt's PDE write.
        """
        idx = self.indices(va)                 # bottom-up
        path = (TOP_LEVEL,)
        parent = self.pd3
        for level, i in zip((TOP_LEVEL, 3, 2, 1), reversed(idx[1:])):
            key = path + (i,)
            child = self.tables.get(key)
            if child is None:
                child = self._table(level - 1)
                self.tables[key] = child
                self._write_pde(parent, i, child, level)
            parent, path = child, key
        return parent

    def _write_pde(self, parent, i, child, level):
        """gp100_vmm_pd0_pde at the dual level, gp100_vmm_pd1_pde above it."""
        v = (child >> 4) | (PT_APER["vram"] << 1)
        if level == 1:                         # PD0: 16-byte dual entry
            self.c.vram_write(parent + i * PD0_BYTES, struct.pack("<QQ", 0, v))
        else:                                  # PD1/PD2/PD3: 8-byte entry
            self.c.vram_write(parent + i * PDE_BYTES, struct.pack("<Q", v))

    # ------------------------------------------------------------ mapping
    def pte_value(self, pa, aper="vram", priv=False, ro=False):
        t = (PAGE_APER[aper] << 1) | PTE_VALID
        if aper != "vram":
            t |= PTE_VOL
        if priv:
            t |= PTE_PRIV
        if ro:
            t |= PTE_RO
        return (pa >> 4) | t

    def map(self, va, pa, size, aper="vram", priv=False, ro=False, flush=True):
        """nvkm_memory_map() for a contiguous range -- the only case we need."""
        assert va % 0x1000 == 0, "VA must be 4 KiB aligned"
        assert pa % 0x1000 == 0, "PA must be 4 KiB aligned"
        assert size % 0x1000 == 0, "size must be a multiple of 4 KiB"
        assert va + size <= self.LIMIT, "VA %#x outside the 49-bit space" % va
        # one PRAMIN write per page table (up to 512 PTEs) instead of one per page:
        # identical PTE values, ~500x fewer transport round trips for large maps
        off = 0
        while off < size:
            pt = self._ensure(va + off)
            i = self.indices(va + off)[0]
            n = min(PT_ENTRIES - i, (size - off) // 0x1000)
            ptes = struct.pack("<%dQ" % n, *(self.pte_value(pa + off + k * 0x1000, aper, priv, ro)
                                             for k in range(n)))
            self.c.vram_write(pt + i * PTE_BYTES, ptes)
            off += n * 0x1000
        self.mapped.append((va, pa, size, aper))
        if flush:
            self.flush()

    def unmap(self, va, size, flush=True):
        """gf100_vmm_pgt_unmap: zero the PTEs; the tables stay allocated."""
        for off in range(0, size, 0x1000):
            idx = self.indices(va + off)
            pt = self.tables.get((TOP_LEVEL, idx[4], idx[3], idx[2], idx[1]))
            if pt is not None:
                self.c.vram_write(pt + idx[0] * PTE_BYTES, bytes(PTE_BYTES))
        if flush:
            self.flush()

    # ------------------------------------------------------------ address space
    def join(self, inst):
        """gf100_vmm_join_ + gp100_vmm_join: publish the PD into an instance block."""
        base = self.pd3 | PDB_VER2 | PDB_BIG_PAGE_64K    # aperture 0 = VRAM
        self.c.vram_write(inst + INST_PDB, struct.pack("<QQ", base, self.LIMIT - 1))
        return base

    def part(self, inst):
        """gf100_vmm_part: clear the instance block's page-directory header."""
        self.c.vram_write(inst + INST_PDB, bytes(16))

    def flush(self, type=None):
        """gf100_vmm_invalidate + gp100_vmm_flush / gp100_vmm_invalidate_pdb."""
        c = self.c
        if type is None:
            type = FLUSH_PAGE_ALL
        # A "free flush slots" counter: wait for a slot before queueing another.
        c.wait(lambda: c.rd(R_INVALIDATE) & 0x00ff0000, 2.0)
        if not (type & FLUSH_ALL_PDB):
            pdb = (self.pd3 >> 12) << 4 | 0             # target 0 = VRAM
            c.wr(R_INVALIDATE_PDB_LO, pdb & 0xffffffff)
            c.wr(R_INVALIDATE_PDB_HI, pdb >> 32)
        c.wr(R_INVALIDATE_CMD, 0x80000000 | type)
        return c.wait(lambda: c.rd(R_INVALIDATE) & 0x00008000, 2.0)

    # ------------------------------------------------------------ verification
    def walk(self, va):
        """Software MMU: descend our own tables *out of VRAM* and translate `va`.

        Returns (aperture name, physical address), or None if any level is
        absent.  Because it re-reads what we actually wrote to VRAM, it checks
        the page tables rather than the intent behind them.
        """
        idx = self.indices(va)
        addr = self.pd3
        for level, i in zip((TOP_LEVEL, 3, 2, 1), reversed(idx[1:])):
            off = addr + i * (PD0_BYTES if level == 1 else PDE_BYTES)
            e = struct.unpack("<Q", self.c.vram_read(off + (8 if level == 1 else 0), 8))[0]
            aper = (e >> 1) & 3
            if aper == 0:
                return None
            # the stored address field is bits[63:4] = addr >> 4, so the flags
            # in bits[3:0] have to be masked off before shifting back up
            addr = (e & ~0xf) << 4
        pte = struct.unpack("<Q", self.c.vram_read(addr + idx[0] * PTE_BYTES, 8))[0]
        if not (pte & PTE_VALID):
            return None
        name = {v: k for k, v in PAGE_APER.items()}.get((pte >> 1) & 3, "aper%d" % ((pte >> 1) & 3))
        return name, (pte & ~0xf) << 4

    def table_bytes(self, addr, n):
        return self.c.vram_read(addr, n)

    def dump(self, va_list=(), name="vmm"):
        print("%s: pd3=%#010x  %d tables  %s" % (name, self.pd3, len(self.tables), self.heap))
        for key in sorted(self.tables):
            print("    %-22s @ %#010x" % (".".join("%x" % k for k in key), self.tables[key]))
        for va in va_list:
            got = self.walk(va)
            print("    va %#013x -> %s" % (va, "unmapped" if got is None else "%s pa=%#x" % got))


# ------------------------------------------------------------------ falcon side
def bind_inst(c, falcon, inst_addr, target=0):
    """gm200_flcn_bind_inst() plus the handshake in gm200_flcn_fw_load's inst path.

    When `fw->inst` is set -- and acr_oneinit always sets it, passing acr->inst
    into nvkm_falcon_fw_oneinit -- nouveau does *not* point the FBIF port at a
    physical aperture.  It binds the instance block and lets the VIRT port
    translate:

        mask(0x048, 1, 1)                     FIFO_ENABLE
        bind_inst(target, inst):
            mask(0x604, 7, 0)                 DMAIDX_VIRT memory type = virtual
            wr(0x054, (1<<30) | (target<<28) | (inst >> 12))
            mask(0x090, 1<<16, 1<<16)
            mask(0x0a4, 8, 8)
        wait until bind status == 5 with the bind interrupt pending
        mask(0x004, 8, 8); mask(0x058, 2, 2)
        wait until bind status == 0

    Returns (bound, status) -- `bound` is the first handshake.
    """
    falcon.mask(F_FIFO_ENABLE, 1, 1)
    falcon.mask(F_TRANSCFG + 4 * DMAIDX_VIRT, 0x7, 0)          # VIRT port -> virtual
    falcon.w(F_CHANNEL_NEXT, (1 << 30) | (target << 28) | (inst_addr >> 12))
    falcon.mask(F_UNK090, 0x00010000, 0x00010000)
    falcon.mask(F_ENG_CONTROL, 0x00000008, 0x00000008)

    def bind_stat(require_intr):
        if require_intr and not (falcon.r(F_INTR) & 0x00000008):
            return -1
        return (falcon.r(F_BIND_STATUS) & 0x00007000) >> 12

    bound = c.wait(lambda: bind_stat(True) == 5, 0.010)
    falcon.mask(F_INTR_CLEAR, 0x00000008, 0x00000008)
    falcon.mask(F_CHANNEL_CMD, 0x00000002, 0x00000002)
    c.wait(lambda: bind_stat(False) == 0, 0.010)
    return bound, (falcon.r(F_BIND_STATUS) & 0x00007000) >> 12


def inst_alloc(c, heap):
    """nvkm_memory_new(NVKM_MEM_TARGET_INST, 0x1000, 0, true): zeroed VRAM."""
    a = heap.alloc(0x1000, 0x1000)
    c.vram_write(a, bytes(0x1000))
    return a


# ------------------------------------------------------------------ self test
class _FakeCard:
    """4-byte-word addressable VRAM stand-in, so the encodings can be checked
    without touching the GPU."""

    def __init__(self):
        self.w = {}

    def vram_write(self, a, d):
        for i in range(0, len(d), 4):
            self.w[a + i] = bytes(d[i:i + 4]).ljust(4, b"\0")

    def vram_read(self, a, n):
        return b"".join(self.w.get(a + i, b"\0\0\0\0") for i in range(0, n, 4))[:n]


def selftest():
    ok = True

    def check(label, got, want):
        nonlocal ok
        if got != want:
            ok = False
        print("  %-44s %s" % (label, "ok" if got == want else
                              "FAIL got %s want %s" % (got, want)))

    check("VA space is 49 bits", sum(b for _, b in SPLIT) + SPLIT[0][0], 49)
    check("PT   table = 512 * 8 B", VMM.table_size(0), 0x1000)
    check("PD0  table = 256 * 16 B", VMM.table_size(1), 0x1000)
    check("PD1/PD2 table = 512 * 8 B", (VMM.table_size(2), VMM.table_size(3)), (0x1000, 0x1000))
    check("top  table = 4 * 8 B", VMM.table_size(4), 32)
    check("PT covers 2 MiB", VMM.table_size(0) // 8 * 0x1000, 2 << 20)
    check("PD0 covers 512 MiB", 256 * (2 << 20), 512 << 20)

    card = _FakeCard()
    v = VMM(card, VramHeap(base=0x10000000))
    heap_base = v.heap.cur - 32

    va = (0x3 << 47) | (0x1ab << 38) | (0x155 << 29) | (0x42 << 21) | (0x1ff << 12) | 0xfff
    idx = v.indices(va)
    check("va -> PT  index", idx[0], 0x1ff)
    check("va -> PD0 index", idx[1], 0x42)
    check("va -> PD1 index", idx[2], 0x155)
    check("va -> PD2 index", idx[3], 0x1ab)
    check("va -> PD3 index", idx[4], 0x3)

    # the PTE/PDE encodings nouveau writes for a plain VRAM mapping
    check("PTE vram pa=0x3000", v.pte_value(0x3000), 0x300 | 1)
    check("PTE syscoh pa=0x8000", v.pte_value(0x8000, "sys_coh"), 0x800 | (2 << 1) | 1 | PTE_VOL)
    check("PTE ro+priv", v.pte_value(0x1000, "vram", priv=True, ro=True),
          0x100 | 1 | PTE_PRIV | PTE_RO)

    # two mappings sharing PD3/PD2/PD1 but not PD0
    v.map(0x00400000, 0x05000000, 0x2000, flush=False)
    v.map(0x00400000 + (1 << 21), 0x06000000, 0x1000, flush=False)
    check("tables created", len(v.tables), 6)
    check("walk va 0x400000", v.walk(0x00400000), ("vram", 0x05000000))
    check("walk va 0x401000", v.walk(0x00401000), ("vram", 0x05001000))
    check("walk next PD0 slot", v.walk(0x00400000 + (1 << 21)), ("vram", 0x06000000))
    check("walk unmapped va", v.walk(0x00800000), None)

    # PDE values as stored: PD0 dual entry (low half empty, high half ours),
    # and an 8-byte PD1 entry
    pd0_tbl = v.tables[(4, 0, 0, 0)]
    lo, hi = struct.unpack("<QQ", v.table_bytes(pd0_tbl + 2 * PD0_BYTES, 16))
    check("PD0 low half (64 KiB table)", lo, 0)
    check("PD0 high half (4 KiB PT)", hi, (v.tables[(4, 0, 0, 0, 2)] >> 4) | (PT_APER["vram"] << 1))
    pd1_tbl = v.tables[(4, 0, 0)]
    check("PD1 entry -> PD0 table", struct.unpack("<Q", v.table_bytes(pd1_tbl, 8))[0],
          (pd0_tbl >> 4) | (PT_APER["vram"] << 1))
    check("top entry -> PD2 table", struct.unpack("<Q", v.table_bytes(v.pd3, 8))[0],
          (v.tables[(4, 0)] >> 4) | (PT_APER["vram"] << 1))

    # instance block
    base = v.join(0x40000)
    check("join: pd3 | VER2 | 64 KiB | vram", base, v.pd3 | PDB_VER2 | PDB_BIG_PAGE_64K)
    check("instance block 0x200", struct.unpack("<Q", v.table_bytes(0x40200, 8))[0], base)
    check("instance block 0x208", struct.unpack("<Q", v.table_bytes(0x40208, 8))[0], (1 << 49) - 1)
    v.part(0x40000)
    check("part clears the header", v.table_bytes(0x40200, 16), bytes(16))

    check("tables 4 KiB aligned", all(a % 0x1000 == 0 for a in v.tables.values()), True)
    check("pd3 came from the heap", v.pd3, heap_base)
    print("self test:", "PASS" if ok else "FAIL")
    return ok


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        sys.exit(0 if selftest() else 1)
    print(__doc__.strip().split("\n")[0])
    print("run `python3 mmu.py --selftest` (no GPU needed) to check the encodings")
