"""Reset PFIFO (PMC_ENABLE bit 8) so the wedged scheduler/PBDMA comes back."""
import sys, time
sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
from card import Card
import fifo

c = Card()
print("before: PMC 0x200=%08x  0x2100=%08x 0x25a0=%08x 0x40108=%08x 0x40100=%08x 0x2200=%08x"
      % (c.rd(0x200), c.rd(0x2100), c.rd(0x25a0), c.rd(0x40108), c.rd(0x40100), c.rd(0x2200)))
c.mask(0x200, 1 << 8, 0); time.sleep(0.05)
print("  PMC with PFIFO in reset: %08x" % c.rd(0x200))
c.mask(0x200, 1 << 8, 1 << 8); time.sleep(0.05)
print("after reset:  PMC 0x200=%08x  0x2100=%08x 0x25a0=%08x 0x40108=%08x 0x40100=%08x 0x2200=%08x"
      % (c.rd(0x200), c.rd(0x2100), c.rd(0x25a0), c.rd(0x40108), c.rd(0x40100), c.rd(0x2200)))
print("  PBDMA runlist masks: " + " ".join("%08x" % c.rd(0x2390 + 4 * i) for i in range(4)))
print("  chan 0x800000/0x800004 = %08x/%08x" % (c.rd(0x800000), c.rd(0x800004)))
c.wr(0x2200, 1)                                  # gf100_fifo_init: PFIFO enable
f = fifo.Fifo(c); f.init()
print("after fifo init: 0x2100=%08x 0x25a0=%08x 0x40108=%08x 0x2254=%08x"
      % (c.rd(0x2100), c.rd(0x25a0), c.rd(0x40108), c.rd(0x2254)))
