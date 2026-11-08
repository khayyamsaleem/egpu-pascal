"""Two copy engines at once (CE2 runlist 5 / PBDMA2, CE3 runlist 6 / PBDMA3).
If aggregate traffic ~= one CE's, the limit is DRAM, not the copy engine."""
import os, struct, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from card import Card
import fb, faults, fifo
import mmu as nvmmu
from fifo import mthd, PASCAL_DMA_COPY_B
from ce_bw import copy, LAUNCH_PHYS, LAUNCH_VIRT, S

N = 1 << 30
c = Card()
devs = {d["name"]: d for d in fb.top_parse(c)}
for n in ("CE2", "CE3"):
    c.mask(0x200, 1 << devs[n]["reset"], 1 << devs[n]["reset"])
faults.dump(c, clear=True)
f = fifo.Fifo(c); f.init()
heap = nvmmu.VramHeap(base=0x10200000)
chans = []
for chid, name, src, dst in ((1, "CE2", 0x40000000, 0xC0000000), (2, "CE3", 0x80000000, 0x100000000)):
    ch = fifo.Channel(f, chid=chid, runl=devs[name]["runlist"], heap=heap)
    sem_pa = ch.alloc_map(0x600000, 0x1000); ch.vmm.flush()
    assert ch.setup(), name
    ch.push(mthd(S, 0, PASCAL_DMA_COPY_B)); ch.kick()
    chans.append((ch, name, src, dst, sem_pa))
time.sleep(0.05)

def run(which):
    for i in which:
        ch, name, src, dst, sem_pa = chans[i]
        c.vram_write(sem_pa, bytes(0x100))
        ch.push(copy(0x600000 + 0x80, 0x600000 + 0xc0, 0x40, 0x600000, 0xa0, LAUNCH_VIRT)
                + copy(src, dst, N, 0x600000 + 0x20, 0xa1, LAUNCH_PHYS))
    for i in which:
        chans[i][0].kick()
    out = []
    for i in which:
        ch, name, src, dst, sem_pa = chans[i]
        ok = c.wait(lambda: c.vram_r32(sem_pa + 0x20) == 0xa1, 20.0)
        w0 = struct.unpack("<4I", c.vram_read(sem_pa, 16)); w1 = struct.unpack("<4I", c.vram_read(sem_pa + 0x20, 16))
        out.append((name, ok, w0[2] | w0[3] << 32, w1[2] | w1[3] << 32))
    return out

for label, which in (("CE2 alone", [0]), ("CE3 alone", [1]), ("CE2+CE3 together", [0, 1])):
    res = run(which)
    t0 = min(r[2] for r in res); t1 = max(r[3] for r in res)
    per = ", ".join("%s %s %.1f ms" % (n, "ok" if ok else "TIMEOUT", (b - a) / 1e6) for n, ok, a, b in res)
    print("%-18s %s | span %.1f ms, aggregate traffic %.1f GB/s" % (label, per, (t1 - t0) / 1e6, 2 * N * len(res) / ((t1 - t0) * 1e-9) / 1e9))
r = c.rd
print("refpll 132020=%08x 132024=%08x 132030=%08x  div 137320=%08x 137330=%08x  1373f4=%08x  mpll 132000=%08x"
      % (r(0x132020), r(0x132024), r(0x132030), r(0x137320), r(0x137330), r(0x1373f4), r(0x132000)))
faults.dump(c)
for ch, *_ in chans: ch.teardown()
