"""Host-triggered falcon DMA (DMATRF) VRAM/sysmem -> DMEM, to isolate the FBIF/hub path."""
import sys, os, time, struct
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from card import Card
from acr import Falcon, SEC2
import fb

c = Card()
mem, iovas = c.dev.alloc_sysmem(0x4000, contiguous=True)
fb.fb_init(c, iovas[0])
f = Falcon(c, SEC2, pmc_bit=14)

def trial(label, transcfg, ctx, src_addr, fill):
    f.reset(); f.mask(0x624, 0x80, 0x80); f.w(0x10c, 0)
    f.mask(0x600 + 4 * ctx, 0x10007, transcfg)
    f.dmem_pio(bytes(256), 0)
    fill(os.urandom(0)); pat = os.urandom(256); fill(pat)
    f.w(0x110, src_addr >> 8); f.w(0x128, 0)
    f.w(0x114, 0); f.w(0x11c, 0); f.w(0x118, (6 << 8) | (ctx << 12))
    done = c.wait(lambda: f.r(0x118) & 2, 0.5)
    got = f.dmem_read(0, 256)
    print("%-34s transcfg=%05x ctx=%d: idle=%s cmd=%08x  match=%s  first=%s" % (
        label, f.r(0x600 + 4 * ctx), ctx, done, f.r(0x118), got == pat, got[:8].hex()))

VR = 0x05000000
vram_fill = lambda b: c.vram_write(VR, b) if b else None
def sys_fill(b):
    if b: mem[0x1000:0x1000 + len(b)] = b
for ctx in (0, 2):
    trial("VRAM phys", 0x4, ctx, VR, vram_fill)
for ctx in (0, 3):
    trial("sysmem coherent phys", 0x5, ctx, iovas[1], sys_fill)
    trial("sysmem noncoherent phys", 0x6, ctx, iovas[1], sys_fill)
print("PRI/hub status: 0x100c80=%08x 0x100e10=%08x  SEC2 0x008(irq)=%08x 0x0dc=%08x" % (c.rd(0x100c80), c.rd(0x100e10), f.r(0x008), f.r(0x0dc)))
