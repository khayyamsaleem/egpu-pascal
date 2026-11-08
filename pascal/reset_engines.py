"""Full engine-level reset: PFIFO (bit 8) + GR (bit 12) via PMC_ENABLE."""
import sys, time
sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
from card import Card
c = Card()
print("before: PMC=%08x FECS 0x409800=%08x sctl=%08x PGRAPH 0x400500=%08x"
      % (c.rd(0x200), c.rd(0x409800), c.rd(0x409240), c.rd(0x400500)))
for bit in (8, 12):
    c.mask(0x200, 1 << bit, 0); time.sleep(0.05)
    c.mask(0x200, 1 << bit, 1 << bit); time.sleep(0.1)
    print("  reset bit %d -> PMC=%08x" % (bit, c.rd(0x200)))
time.sleep(0.2)
print("after:  PMC=%08x FECS sctl=%08x cpuctl=%08x 0x409800=%08x GPCCS sctl=%08x PGRAPH intr=%08x 0x400500=%08x"
      % (c.rd(0x200), c.rd(0x409240), c.rd(0x409100), c.rd(0x409800), c.rd(0x41a240), c.rd(0x400100), c.rd(0x400500)))
print("  FIFO_INTR=%08x 0x25a0=%08x PBDMA0 0x40108=%08x" % (c.rd(0x2100), c.rd(0x25a0), c.rd(0x40108)))
