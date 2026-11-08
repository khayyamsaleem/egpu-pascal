import sys, os, random; sys.path.insert(0, __file__.rsplit("/", 1)[0])
from card import Card
c = Card()
def pramin_set(addr): c.wr(0x1700, addr >> 16)    # 64 KB window units, VRAM target
ok = True
for mb in (1, 16, 512, 4096, 8192, 11000):
    a = mb << 20; pramin_set(a)
    pat = [random.getrandbits(32) for _ in range(64)]
    for i, p in enumerate(pat): c.wr(0x700000 + i * 4, p)
    got = [c.rd(0x700000 + i * 4) for i in range(64)]
    good = got == pat; ok &= good
    print("VRAM @%5d MB: %s  (first: wrote %08x read %08x)" % (mb, "OK" if good else "MISMATCH %d/64" % sum(a != b for a, b in zip(got, pat)), pat[0], got[0]))
# aliasing check: different windows must hold different data
pramin_set(1 << 20); a = c.rd(0x700000); pramin_set(4096 << 20); b = c.rd(0x700000)
print("no aliasing 1MB vs 4GB:", a != b)
print("PFB size reg 0x100ce0 = %08x -> %d MB" % (c.rd(0x100ce0), ((c.rd(0x100ce0) >> 4) & 0x3f) << (c.rd(0x100ce0) & 0xf)))
print("RESULT:", "VRAM WORKS" if ok else "VRAM BROKEN")
