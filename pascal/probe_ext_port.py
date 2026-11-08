"""Read/write an MMIO register the way the ACR app does: through SEC2's external port.

The app's read32/write32 (imem 0x27fc / 0x283d) are:

    I[0x1c300] = 0x20000                      setup
    wait: poll I[0x1c000] bits[14:12] == 0    (extr $r10 $r10 0xc:0xe)
    I[0x1c100] = address
    I[0x1c000] = 0x800000f1                   read   (data comes back in I[0x1c200])
    I[0x1c000] = 0x800000f2                   write  (data first in I[0x1c200])
    wait again, then I[0x1c200] is the data

envytools' falcon/io.rst gives the I-space mapping as "0x040 / I[0x01000]" for
SCRATCH0, i.e. I = MMIO * 0x40, so for the SEC2 falcon at 0x87000 the port
registers are 0x87700 (command) / 0x87704 (address) / 0x87708 (data out) /
0x8770c (setup).  The app also touches I[0x18900] = 0x87624, which is a falcon
register nouveau itself pokes -- confirmation of the stride.

Controls: two registers whose host-side value is already known, so the port's
addressing can be trusted before the interesting reads.

Read-only apart from one attempted write to the register in question.
"""

import sys

sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
from card import Card
import acr
import fb
import mmu

CMD_READ = 0x800000F1
CMD_WRITE = 0x800000F2
P_CMD, P_ADDR, P_DATA, P_SETUP = 0x700, 0x704, 0x708, 0x70C


def main():
    c = Card()
    mem, iovas = c.dev.alloc_sysmem(0x1000, contiguous=True)
    fb.fb_init(c, iovas[0])

    # same falcon state the ACR app sees: reset, with an instance block bound
    f = acr.Falcon(c, acr.SEC2, pmc_bit=14)
    f.reset()
    heap = mmu.VramHeap()
    v = mmu.VMM(c, heap)
    inst = mmu.inst_alloc(c, heap)
    v.join(inst)
    v.flush()
    bound, status = mmu.bind_inst(c, f, inst)
    print("SEC2 bound=%s status=%d  port regs 0x700..0x70c = %s"
          % (bound, status, " ".join("%08x" % f.r(P_CMD + 4 * i) for i in range(4))))

    def wait_done():
        return c.wait(lambda: ((f.r(P_CMD) >> 12) & 7) == 0, 0.05)

    def ext_read(addr):
        f.w(P_SETUP, 0x20000)
        ok = wait_done()
        f.w(P_ADDR, addr)
        f.w(P_CMD, CMD_READ)
        ok2 = wait_done()
        return f.r(P_DATA), (ok, ok2, f.r(P_CMD), f.r(0x244))

    def ext_write(addr, val):
        f.w(P_SETUP, 0x20000)
        wait_done()
        f.w(P_ADDR, addr)
        f.w(P_DATA, val)
        f.w(P_CMD, CMD_WRITE)
        return wait_done(), f.r(P_CMD)

    print("\n-- controls (host-side value in brackets)")
    for name, addr, host in (("0x100ce0 fb size", 0x100CE0, c.rd(0x100CE0)),
                             ("0x12006c ltc count", 0x12006C, c.rd(0x12006C)),
                             ("0x100cd8 wpr2 lo", 0x100CD8, c.rd(0x100CD8))):
        val, diag = ext_read(addr)
        print("   %-20s port=%08x host=%08x %s  (waits=%s cmd=%08x 0x244=%08x)"
              % (name, val, host, "MATCH" if val == host else "DIFFER", diag[:2], diag[2], diag[3]))

    print("\n-- the registers the app checks")
    for name, addr in (("0x10e5f8 UNK5F8", 0x10E5F8), ("0x10e5e4", 0x10E5E4),
                       ("0x17e278", 0x17E278), ("0x100cd4 wpr idx", 0x100CD4)):
        val, diag = ext_read(addr)
        print("   %-20s port=%08x host=%08x  bits[23:20]=%d  (waits=%s cmd=%08x)"
              % (name, val, c.rd(addr), (val >> 20) & 0xF, diag[:2], diag[2]))

    before = c.rd(0x10E5F8)
    pv, _ = ext_read(0x10E5F8)
    ok, cmd = ext_write(0x10E5F8, pv & ~0x00F00000)
    after_port, _ = ext_read(0x10E5F8)
    print("\n-- write through the port: 0x10e5f8 %08x -> %08x (port read %08x, host now %08x, cmd=%08x)"
          % (pv, pv & ~0x00F00000, after_port, c.rd(0x10E5F8), cmd))
    print("   host write of the same value:", end=" ")
    c.wr(0x10E5F8, before & ~0x00F00000)
    print("%08x -> %08x" % (before & ~0x00F00000, c.rd(0x10E5F8)))

    f.reset()
    print("\nSEC2 reset.  (SEC2+0x244=%08x 0x700=%08x)" % (f.r(0x244), f.r(P_CMD)))


if __name__ == "__main__":
    main()
