import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from card import Card
from acr import Falcon, SEC2
import fb
c = Card()
mem, iovas = c.dev.alloc_sysmem(0x4000, contiguous=True)
fb.fb_init(c, iovas[0])
f = Falcon(c, SEC2, pmc_bit=14)
VR = 0x05000000
def dma_ok():
    f.reset(); f.mask(0x624, 0x80, 0x80); f.w(0x10c, 0); f.mask(0x600, 0x10007, 0x4)
    f.dmem_pio(bytes(256), 0); pat = os.urandom(256); c.vram_write(VR, pat)
    f.w(0x110, VR >> 8); f.w(0x128, 0); f.w(0x114, 0); f.w(0x11c, 0); f.w(0x118, 6 << 8)
    idle = c.wait(lambda: f.r(0x118) & 2, 0.3)
    return idle, f.dmem_read(0, 256) == pat
print("baseline PMC=%08x dma=%s" % (c.rd(0x200), dma_ok()))
for bit, name in ((27, "FB/hub"), (28, "BUS"), (9, "FAULT"), (25, "LTC"), (26, "DISP?"), (24, "PMU"), (20, "TIMER"), (18, "THERM")):
    before = c.rd(0x200); c.mask(0x200, 1 << bit, 1 << bit); after = c.rd(0x200)
    print("set bit %2d (%-6s): PMC %08x -> %08x  stuck=%s  0x260=%08x 0x17e8=%08x  dma=%s" % (
        bit, name, before, after, bool(after & (1 << bit)), c.rd(0x260), c.rd(0x17e8), dma_ok()))
