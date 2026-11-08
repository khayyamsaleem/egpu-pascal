"""VRAM bandwidth at boot clocks, measured with the copy engine (CE2, runlist 5).

Timing comes from the CE's own four-word semaphore timestamps (GPU ns clock), so
Thunderbolt round-trip latency does not enter the number.  Copies use physical
addressing (LAUNCH_DMA SRC/DST_TYPE = PHYSICAL, LOCAL_FB), so no page tables are
needed for the large buffers.

Each copy reads N and writes N bytes: reported "traffic" = 2N / dt.  A single CE
is a lower bound on what the SMs can pull from DRAM, not the ceiling.

Control: the 4 KiB warm-up copy is checked byte-exact; the big copy is spot
checked at a few offsets against a pattern written through PRAMIN beforehand.

Usage: ~/.venvs/tinygpu/bin/python pascal/ce_bw.py [--sysmem]
"""
import os, struct, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from card import Card
import fb, faults, fifo
import mmu as nvmmu
from fifo import mthd, PASCAL_DMA_COPY_B

SRC_PA, DST_PA = 0x40000000, 0xC0000000        # 1 GiB and 3 GiB into VRAM; clear of WPR2/heap
SIZES = (16 << 20, 64 << 20, 256 << 20, 1 << 30)
S = 4                                          # CE subchannel

# LAUNCH_DMA: NON_PIPELINED | FLUSH | FOUR_WORD_SEMAPHORE | SRC/DST PITCH | SRC/DST PHYSICAL
LAUNCH_PHYS = 0x2 | 0x4 | (2 << 3) | 0x80 | 0x100 | (1 << 12) | (1 << 13)
LAUNCH_VIRT = 0x2 | 0x4 | (2 << 3) | 0x80 | 0x100


def copy(src, dst, n, sem_va, payload, launch, src_tgt=0, dst_tgt=0):
    pb = []
    if launch & (1 << 12):
        pb += mthd(S, 0x0260, src_tgt)
    if launch & (1 << 13):
        pb += mthd(S, 0x0264, dst_tgt)
    pb += mthd(S, 0x0400, src >> 32, src & 0xffffffff, dst >> 32, dst & 0xffffffff, n, n, n, 1)
    pb += mthd(S, 0x0240, sem_va >> 32, sem_va & 0xffffffff, payload)
    pb += mthd(S, 0x0300, launch)
    return pb


def main():
    c = Card()
    devs = {d["name"]: d for d in fb.top_parse(c)}
    ce = devs["CE2"]
    runl = ce["runlist"]
    c.mask(0x200, 1 << ce["reset"], 1 << ce["reset"])
    faults.dump(c, clear=True)
    f = fifo.Fifo(c); f.init()
    heap = nvmmu.VramHeap(base=0x10200000)
    ch = fifo.Channel(f, chid=1, runl=runl, heap=heap)
    sem_va, warm_va = 0x00600000, 0x00400000
    sem_pa = ch.alloc_map(sem_va, 0x1000)
    warm_pa = ch.alloc_map(warm_va, 0x2000)
    ch.vmm.flush()
    if not ch.setup():
        faults.dump(c); sys.exit(1)

    pb = mthd(S, 0x0000, PASCAL_DMA_COPY_B)
    ch.push(pb); ch.kick()

    # spot-check pattern at a few offsets of the largest copy
    probes = [0, 0x1000, (1 << 30) - 0x1000, 0x12345000]
    for i, o in enumerate(probes):
        c.vram_write(SRC_PA + o, struct.pack("<4I", 0xb0b0_0000 | i, o, ~o & 0xffffffff, 0x5a5a5a5a))
        c.vram_write(DST_PA + o, b"\xee" * 16)

    results = []
    seq = 1
    for n in SIZES:
        # warm-up 4 KiB virtual copy -> T0, then the big physical copy -> T1, back to back
        c.vram_write(warm_pa, bytes((i * 7 + seq) & 0xff for i in range(0x1000)))
        s0, s1 = sem_va + 0x40 * seq, sem_va + 0x40 * seq + 0x20
        p0, p1 = 0x7000_0000 | seq * 2, 0x7000_0000 | seq * 2 + 1
        pb = copy(warm_va, warm_va + 0x1000, 0x1000, s0, p0, LAUNCH_VIRT)
        pb += copy(SRC_PA, DST_PA, n, s1, p1, LAUNCH_PHYS)
        ch.push(pb)
        t_host = time.time(); ch.kick()
        o0, o1 = sem_pa + 0x40 * seq, sem_pa + 0x40 * seq + 0x20
        ok = c.wait(lambda: c.vram_r32(o1) == p1, 20.0)
        wall = time.time() - t_host
        w0 = struct.unpack("<4I", c.vram_read(o0, 16)); w1 = struct.unpack("<4I", c.vram_read(o1, 16))
        t0 = w0[2] | (w0[3] << 32); t1 = w1[2] | (w1[3] << 32)
        dt = (t1 - t0) * 1e-9
        warm_ok = c.vram_read(warm_pa + 0x1000, 0x1000) == c.vram_read(warm_pa, 0x1000)
        print("copy %5d MiB: sem0 %08x sem1 %08x (%s)  gpu dt %.3f ms  host wall %.1f ms  "
              "traffic %.1f GB/s  warm-up byte-exact %s"
              % (n >> 20, w0[0], w1[0], "ok" if ok else "TIMEOUT", dt * 1e3, wall * 1e3,
                 (2 * n / dt / 1e9) if dt > 0 else float("nan"), warm_ok))
        results.append((n, dt, ok))
        seq += 1
        if not ok:
            print("  status:", ch.status(), "PBDMA", fifo.pbdma_dump(c, next(i for i in range(4) if c.rd(0x2390 + 4 * i) & (1 << runl))))
            faults.dump(c)
            break

    good = 0
    for i, o in enumerate(probes):
        if o + 16 > SIZES[-1]:
            continue
        exp = struct.pack("<4I", 0xb0b0_0000 | i, o, ~o & 0xffffffff, 0x5a5a5a5a)
        got = c.vram_read(DST_PA + o, 16)
        good += got == exp
        print("  probe dst+%#010x: %s %s" % (o, got.hex(), "==" if got == exp else "!= " + exp.hex()))
    print("spot check: %d/%d" % (good, len(probes)))

    if "--sysmem" in sys.argv:
        # host memory -> VRAM through the IOMMU (model-upload path), 4 MiB chunks
        n = 4 << 20
        smem, iovas = c.dev.alloc_sysmem(n)
        contiguous = all(iovas[i + 1] == iovas[i] + 0x1000 for i in range(len(iovas) - 1))
        print("sysmem %d MiB, %d pages, IOVA contiguous: %s" % (n >> 20, len(iovas), contiguous))
        va = 0x08000000
        for i, iova in enumerate(iovas):
            ch.vmm.map(va + i * 0x1000, iova, 0x1000, aper="sys_coh", flush=False)
        ch.vmm.flush()
        smem[0:16] = b"\x11" * 16
        seq += 1
        s0, s1 = sem_va + 0x40 * seq, sem_va + 0x40 * seq + 0x20
        p0, p1 = 0x7100_0000 | seq * 2, 0x7100_0000 | seq * 2 + 1
        pb = copy(warm_va, warm_va + 0x1000, 0x1000, s0, p0, LAUNCH_VIRT)
        pb += copy(va, DST_PA, n, s1, p1, LAUNCH_VIRT & ~0 | (1 << 13))   # src virtual (sysmem), dst physical
        ch.push(pb); ch.kick()
        o0, o1 = sem_pa + 0x40 * seq, sem_pa + 0x40 * seq + 0x20
        ok = c.wait(lambda: c.vram_r32(o1) == p1, 20.0)
        w0 = struct.unpack("<4I", c.vram_read(o0, 16)); w1 = struct.unpack("<4I", c.vram_read(o1, 16))
        dt = ((w1[2] | (w1[3] << 32)) - (w0[2] | (w0[3] << 32))) * 1e-9
        print("sysmem->VRAM %d MiB: %s  gpu dt %.3f ms  %.2f GB/s  dst[0:16]=%s"
              % (n >> 20, "ok" if ok else "TIMEOUT", dt * 1e3, n / dt / 1e9 if dt > 0 else float("nan"),
                 c.vram_read(DST_PA, 16).hex()))
    faults.dump(c)
    ch.teardown()


if __name__ == "__main__":
    main()
