"""(RT-2) Prove the Pascal page tables in mmu.py actually translate, on the GPU.

Round-trip plan, rung RT-2: "a host buffer the GPU can read/write (MMU +
instance block)".  The falcon's own xfer engine (XFER_CTRL, envytools
falcon/xfer.rst) is the cheapest possible client of the MMU -- it needs no
firmware, no signatures and no FIFO channel, and it is the *same* engine the ACR
HS bootloader uses for its first DMA, which is exactly the transfer that is
currently hanging.

The tests, in order:
  1. pre-flight state (register dump; nothing is written)
  2. build the VMM in VRAM, map VRAM pages into it, verify every table by
     reading it back out of VRAM (mmu.VMM.walk is a software MMU)
  3. publish it with join() into a 4 KiB instance block and invalidate the TLB
  4. bind SEC2 to the instance block (gm200_flcn_bind_inst)
  5. **read round trip**: falcon xfer loads 256 B from a *virtual* address into
     SEC2's DMEM; compare against what we wrote to that VRAM page
  6. **write round trip**: falcon xfer stores 256 B from DMEM to a virtual
     address; read the VRAM page back and compare
  7. **negative control**: the same xfer from an unmapped VA must *not* deliver
     data and must raise an MMU fault -- otherwise "it worked" could just mean
     the address was ignored

Every step prints the raw register value it judged from, so "wrong value",
"engine disabled (poison)" and "MMU fault" are distinguishable at a glance.

Run:  APL_REMOTE_SOCK=/tmp/tinygpu.sock ~/.venvs/tinygpu/bin/python pascal/mmu_test.py
"""

import os
import struct
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from card import Card                      # noqa: E402
from acr import Falcon, SEC2               # noqa: E402

import fb                                  # noqa: E402
import mmu                                 # noqa: E402
import faults                              # noqa: E402

# VRAM scratch: inside the proven-good low range, below the 3.5 GB VPR and clear
# of the ACR's own areas (HS image 0x01000000, WPR 0x02000000).
VRAM_SCRATCH = 0x05000000
VRAM_SCRATCH2 = 0x05010000
# Virtual addresses.  VA_BASE and VA_BASE+2MiB deliberately land in different
# PD0 slots, so the walk has to create and use a second page table.
VA_BASE = 0x00400000
VA_SECOND = VA_BASE + (1 << 21)
VA_UNMAPPED = 0x00800000
PAGE = 0x1000

# falcon xfer registers (envytools falcon/xfer.rst)
F_XFER_EXT_BASE = 0x110
F_XFER_LOCAL = 0x114
F_XFER_CTRL = 0x118
F_XFER_EXT_OFF = 0x11c
F_XFER_STATUS = 0x120
XFER_DATA_LOAD, XFER_CODE_LOAD, XFER_DATA_STORE = 0, 1, 2
XFER_BUSY = 0x00000002
XFER_LOADS_PENDING = 0x7 << 24
XFER_STORES_PENDING = 0x7 << 16
XFER_CTRL_PENDING = 0x00000001

results = []


def check(label, ok, detail=""):
    results.append((label, ok))
    print("  [%s] %-46s %s" % ("PASS" if ok else "FAIL", label, detail))


def dump_state(c, tag):
    print("-- %s" % tag)
    for name, a in (("PMC_BOOT_0", 0x0), ("PMC_ENABLE", 0x200), ("PMC 0x260", 0x260),
                    ("MMU ctrl 0x100c80", 0x100c80), ("0x100cb8", 0x100cb8),
                    ("0x100cbc", 0x100cbc), ("0x100cec", 0x100cec),
                    ("FIFO_INTR 0x2100", 0x2100), ("MMU_FAULT_ID 0x259c", 0x259c),
                    ("SEC2 cpuctl", SEC2 + 0x100), ("SEC2 0x0dc", SEC2 + 0x0dc),
                    ("SEC2 0x048", SEC2 + 0x048), ("SEC2 0x054", SEC2 + 0x054),
                    ("SEC2 0x090", SEC2 + 0x090), ("SEC2 0x0a4", SEC2 + 0x0a4),
                    ("SEC2 xfer status", SEC2 + F_XFER_STATUS)):
        print("   %-20s %08x" % (name, c.rd(a)))
    print("   FBIF ports  %s" % " ".join("%08x" % c.rd(SEC2 + 0x600 + 4 * i) for i in range(8)))


def xfer(c, f, mode, ext_base, ext_offset, local, size_log=6, timeout=0.5):
    """One host-triggered falcon xfer.  Returns (completed, status_raw, ctrl_raw)."""
    # XFER_CTRL bit 0: the previous write is still waiting for a queue slot.
    c.wait(lambda: not (f.r(F_XFER_CTRL) & XFER_CTRL_PENDING), 0.5)
    f.w(F_XFER_EXT_BASE, ext_base)
    f.w(F_XFER_LOCAL, local)
    f.w(F_XFER_EXT_OFF, ext_offset)
    f.w(F_XFER_CTRL, (mode << 4) | (size_log << 8) | (mmu.DMAIDX_VIRT << 12))
    done = c.wait(lambda: not (f.r(F_XFER_STATUS) & (XFER_BUSY | XFER_LOADS_PENDING
                                                     | XFER_STORES_PENDING)), timeout)
    return done, f.r(F_XFER_STATUS), f.r(F_XFER_CTRL)


def main():
    print("=== RT-2: Pascal MMU v2 page tables, on the card ===")
    c = Card()
    dump_state(c, "pre-flight (no writes yet)")

    # ---- 0. maison: fb_init installs the flush page and the FB/MMU page config,
    #         nouveau does this before any MMU work (nvkm_fb_init -> acr->init).
    mem, iovas = c.dev.alloc_sysmem(0x1000, contiguous=True)
    fb.fb_init(c, iovas[0])
    print("-- fb_init done, flush page IOVA %#x" % iovas[0])

    # a stale fault would make every judgement below ambiguous
    faults.dump(c, clear=True)

    # ---- 1. page tables in VRAM
    print("-- building the address space")
    heap = mmu.VramHeap()
    v = mmu.VMM(c, heap)
    inst = mmu.inst_alloc(c, heap)
    print("   instance block @ %#x, tables from @ %#x" % (inst, v.pd3))

    # two pages of known content in VRAM to be mapped
    pat_a = bytes((i * 7 + 0x11) & 0xff for i in range(256))
    pat_b = bytes((i * 13 + 0x40) & 0xff for i in range(256))
    c.vram_write(VRAM_SCRATCH, pat_a)
    c.vram_write(VRAM_SCRATCH2, pat_b)
    check("VRAM scratch written and readable", c.vram_read(VRAM_SCRATCH, 256) == pat_a)

    v.map(VA_BASE, VRAM_SCRATCH, PAGE)
    v.map(VA_SECOND, VRAM_SCRATCH2, PAGE)
    # VA_BASE..VA_BASE+0x2000 spans two 4 KiB pages, and VA_SECOND is one PD0
    # slot further up -- so this exercises both the PT and the PD0 dual entry.
    v.map(VA_BASE + PAGE, VRAM_SCRATCH + PAGE, PAGE)
    v.dump([VA_BASE, VA_BASE + PAGE, VA_SECOND, VA_UNMAPPED], name="vmm")

    check("tables read back from VRAM: VA_BASE", v.walk(VA_BASE) == ("vram", VRAM_SCRATCH),
          "%s" % (v.walk(VA_BASE),))
    check("tables read back: VA_BASE + 4 KiB", v.walk(VA_BASE + PAGE) == ("vram", VRAM_SCRATCH + PAGE))
    check("tables read back: second PD0 slot", v.walk(VA_SECOND) == ("vram", VRAM_SCRATCH2))
    check("unmapped VA has no translation", v.walk(VA_UNMAPPED) is None)
    check("PTE really says VRAM (aper 0)", (struct.unpack(
        "<Q", v.table_bytes(v.tables[(4, v.indices(VA_BASE)[4], v.indices(VA_BASE)[3],
                                      v.indices(VA_BASE)[2], v.indices(VA_BASE)[1])]
                            + v.indices(VA_BASE)[0] * 8, 8))[0] & 0x7) == 1)

    # ---- 2. instance block + TLB invalidate
    base = v.join(inst)
    check("instance block 0x200 = pd3 | VER2 | 64 KiB",
          struct.unpack("<Q", v.table_bytes(inst + 0x200, 8))[0] == base, "%#x" % base)
    check("instance block 0x208 = VA limit - 1",
          struct.unpack("<Q", v.table_bytes(inst + 0x208, 8))[0] == (1 << 49) - 1)
    t0 = time.time()
    flushed = v.flush()
    check("MMU invalidate completed", flushed, "%.1f ms, 0x100c80=%08x"
          % ((time.time() - t0) * 1000, c.rd(mmu.R_INVALIDATE)))

    # ---- 3. bind SEC2 to the instance block
    f = Falcon(c, SEC2, pmc_bit=14)
    f.reset()
    bound, status = mmu.bind_inst(c, f, inst, target=0)
    check("SEC2 bound to the instance block", bound, "bind status %d, 0x054=%08x"
          % (status, f.r(0x054)))
    check("VIRT port is in virtual mode", (f.r(0x604) & 0x7) == 0,
          "0x604=%08x (memtype 0 = virtual)" % f.r(0x604))

    # ---- 4. read round trip: VA -> DMEM through the page tables
    print("-- read round trip: falcon xfer, VA %#x -> DMEM" % VA_BASE)
    f.dmem_pio(bytes(256) + b"\x00" * 256, 0)          # clear DMEM window
    done, st, ctrl = xfer(c, f, XFER_DATA_LOAD, VA_BASE >> 8, 0, 0)
    got = f.dmem_read(0, 256)
    check("xfer from mapped VA completed", done,
          "status=%08x ctrl=%08x pending=%d" % (st, ctrl, st >> 24 & 7))
    check("DMEM holds the mapped VRAM contents", got == pat_a,
          "first 8: %s vs %s" % (got[:8].hex(), pat_a[:8].hex()))

    # ---- 5. write round trip: DMEM -> VA through the page tables
    print("-- write round trip: falcon xfer, DMEM -> VA %#x" % VA_SECOND)
    f.dmem_pio(pat_b, 0)
    done, st, ctrl = xfer(c, f, XFER_DATA_STORE, VA_SECOND >> 8, 0, 0)
    back = c.vram_read(VRAM_SCRATCH2, 256)
    check("xfer to mapped VA completed", done, "status=%08x ctrl=%08x" % (st, ctrl))
    check("mapped VRAM page received the DMEM contents", back == pat_b,
          "first 8: %s vs %s" % (back[:8].hex(), pat_b[:8].hex()))

    # ---- 6. negative control
    print("-- negative control: the same xfer from an unmapped VA must fault")
    faults.dump(c, clear=True)
    f.dmem_pio(bytes(256), 0)
    done, st, ctrl = xfer(c, f, XFER_DATA_LOAD, VA_UNMAPPED >> 8, 0, 0, timeout=0.3)
    got = f.dmem_read(0, 256)
    check("no data delivered from an unmapped VA", got != pat_b and got == bytes(256),
          "first 8: %s" % got[:8].hex())
    print("   fault registers after the negative control:")
    faults.dump(c)
    check("an MMU fault was raised", bool(c.rd(0x259c) & 0x7fffffff) or (c.rd(0x2100) != 0),
          "0x259c=%08x 0x2100=%08x" % (c.rd(0x259c), c.rd(0x2100)))
    faults.dump(c, clear=True)

    # ---- 7. leave the falcon and the tables in a known state
    v.part(inst)
    check("instance block detached (0x200 zeroed)", v.table_bytes(inst + 0x200, 16) == bytes(16))
    c.vram_write(VRAM_SCRATCH, bytes(PAGE))
    c.vram_write(VRAM_SCRATCH2, bytes(PAGE))
    f.reset()
    dump_state(c, "post-test")

    failed = [l for l, ok in results if not ok]
    print("\n=== %d/%d checks passed ===" % (len(results) - len(failed), len(results)))
    if failed:
        print("failed: " + "; ".join(failed))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
