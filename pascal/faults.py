import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from card import Card
REASON = "PDE PDE_SIZE PTE VA_LIMIT UNBOUND_INST PRIV_VIOL RO_VIOL WO_VIOL PITCH KIND? APERTURE COMPRESSION UNSUPP_KIND REGION_VIOL BOTH_PTES POISONED".split()
HUB = "VIP CE0 CE1 DNISO FE FECS HOST HOST_CPU HOST_CPU_NB ISO MMU MSPDEC MSPPP MSVLD NISO P2P PD PERF PMU RASTER2D SCC SCC_NB SEC SSYNC GR_CE CE2 XV MMU_NB MSENC DFALCON SKED AFALCON".split()
ENG = {1:"DISPLAY",3:"IFB",4:"BAR1",5:"BAR2",0x13:"PERF",0x17:"PMU",0x18:"PTP",0x1f:"PHYSICAL"}
def dump(c, clear=False):
    print("FIFO_INTR=%08x MMU_FAULT_ID(0x259c)=%08x" % (c.rd(0x2100), c.rd(0x259c)))
    m = c.rd(0x259c)
    for u in range(32):
        if m & (1 << u):
            inst, lo, hi, t = (c.rd(0x2800 + u * 0x10 + 4 * i) for i in range(4))
            print("  unit %2d (%s): inst=%#x addr=%#x hub=%d gpc=%d access=%d client=%s reason=%s raw=%08x" % (
                u, ENG.get(u, "eng%d" % u), inst << 12, (hi << 32) | lo, (t >> 20) & 1, (t >> 24) & 0x1f,
                (t >> 16) & 7, HUB[(t >> 8) & 0x1f] if (t >> 20) & 1 else (t >> 8) & 0x7f, REASON[t & 0x1f] if (t & 0x1f) < 16 else t & 0x1f, t))
            if clear: c.wr(0x259c, 1 << u)
    if clear: c.wr(0x2100, 0x18000000)
if __name__ == "__main__":
    dump(Card(), clear="--clear" in sys.argv)
