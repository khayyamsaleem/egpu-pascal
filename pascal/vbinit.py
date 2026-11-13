"""A small VBIOS init-script interpreter (port of the subset of nouveau nvkm/subdev/bios/init.c
that the GP102 rammap mode scripts use).

The rammap table header (BIT 'P' + 4) points at one init script per memory mode
(nouveau gp100_ram_init: count at +0x14, u32 list at +0x10). Each programs the FBPA
register bank selected by 0x9a065c bits 7:4. devinit only programs the boot mode's bank,
so a reclock into another rammap band needs that band's script run first.

Unknown opcodes stop the interpreter (never guess a length). `dry=True` records writes
without touching the card; reads in a dry run come from `regs` (a snapshot) or 0.

Usage: ~/.venvs/tinygpu/bin/python pascal/vbinit.py [script-index ...]   (dry run, no card)
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from vbios import VBIOS

ROM = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "firmware", "gp102_vbios.rom")


def ramcfg_count(v):
    M = v.bit("M")
    if M["version"] == 1 and M["length"] >= 5:
        return v.rd8(M["offset"] + 2)
    if M["version"] == 2 and M["length"] >= 3:
        return v.rd8(M["offset"] + 0)
    return 0


def mode_scripts(v):
    P = v.bit("P")
    rm = v.rd32(P["offset"] + 4)
    cnt, lst = v.rd8(rm + 0x14), v.rd32(rm + 0x10)
    return [v.rd32(lst + 4 * i) for i in range(cnt)]


class Init:
    def __init__(self, v, strap, card=None, dry=True, regs=None, log=print, cond_override=None):
        self.v, self.strap, self.c, self.dry = v, strap, card, dry
        self.regs = dict(regs or {})
        self.count = ramcfg_count(v)
        self.writes = []
        self.log = log
        self.execute = True
        # GP10x VBIOSes have no legacy condition table (BIT 'I' + 4 is 0): the PMU DEVINIT app
        # evaluates conditions from its own data. nouveau then treats every condition as false,
        # which skips writes the card needs; callers pass what the hardware says instead.
        self.cond_override = dict(cond_override or {})

    # register access
    def rd(self, a):
        """Reads come from the card when one is given, even in a dry run (writes never do);
        a dry run sees its own earlier writes."""
        if a in self.regs and (self.dry or self.c is None):
            return self.regs[a]
        if self.c is None:
            return 0
        return self.c.rd(a)

    def cond_met(self, cond):
        if cond in self.cond_override:
            ok = bool(self.cond_override[cond])
            self.writes.append(("cond", "[%#04x] -> %s (override)" % (cond, ok)))
            return ok
        I = self.v.bit("I")
        table = self.v.rd16(I["offset"] + 0x04) if I and I["length"] >= 6 else 0
        if not table:
            self.writes.append(("cond", "[%#04x] -> False (no condition table)" % cond))
            return False
        reg, msk, val = (self.v.rd32(table + cond * 12 + k) for k in (0, 4, 8))
        got = self.rd(reg)
        ok = (got & msk) == val
        self.writes.append(("cond", "[%#04x] (R[%06x]=%08x & %08x) == %08x -> %s" % (cond, reg, got, msk, val, ok)))
        return ok

    def wr(self, a, d):
        if not self.execute:
            return
        self.writes.append((a, d))
        self.regs[a] = d
        if not self.dry:
            self.c.wr(a, d)

    def run(self, off, limit=4096):
        v = self.v
        start = off
        while off - start < limit:
            op = v.rd8(off)
            if op == 0x71:                                   # INIT_DONE
                return off + 1
            elif op == 0x8f:                                 # INIT_RAM_RESTRICT_ZM_REG_GROUP
                addr, incr, num = v.rd32(off + 1), v.rd8(off + 5), v.rd8(off + 6)
                p = off + 7
                for i in range(num):
                    for j in range(self.count):
                        if j == self.strap:
                            self.wr(addr, v.rd32(p))
                        p += 4
                    addr += incr
                off = p
            elif op == 0x7a:                                 # INIT_ZM_REG
                self.wr(v.rd32(off + 1), v.rd32(off + 5)); off += 9
            elif op == 0x6e:                                 # INIT_NV_REG: (rd & mask) | data
                a, m, d = v.rd32(off + 1), v.rd32(off + 5), v.rd32(off + 9)
                self.wr(a, (self.rd(a) & m) | d); off += 13
            elif op == 0x58:                                 # INIT_ZM_REG_SEQUENCE
                base, n = v.rd32(off + 1), v.rd8(off + 5)
                for i in range(n):
                    self.wr(base + 4 * i, v.rd32(off + 6 + 4 * i))
                off += 6 + 4 * n
            elif op == 0x91:                                 # INIT_ZM_REG_GROUP
                a, n = v.rd32(off + 1), v.rd8(off + 5)
                for i in range(n):
                    self.wr(a, v.rd32(off + 6 + 4 * i))
                off += 6 + 4 * n
            elif op == 0x97:                                 # INIT_ZM_MASK_ADD: (rd & mask) + add
                a, m, add = v.rd32(off + 1), v.rd32(off + 5), v.rd32(off + 9)
                o = self.rd(a)
                self.wr(a, ((o & m) | ((o & ~m) + add)) & 0xffffffff); off += 13
            elif op == 0x77:                                 # INIT_ZM_REG16
                self.wr(v.rd32(off + 1), v.rd16(off + 5)); off += 7
            elif op == 0x90:                                 # INIT_COPY_ZM_REG
                self.wr(v.rd32(off + 5), self.rd(v.rd32(off + 1))); off += 9
            elif op == 0x75:                                 # INIT_CONDITION
                if not self.cond_met(v.rd8(off + 1)):
                    self.execute = False
                off += 2
            elif op == 0x72:                                 # INIT_RESUME
                self.execute = True; off += 1
            elif op == 0x38:                                 # INIT_NOT
                self.execute = not self.execute; off += 1
            elif op == 0x74:                                 # INIT_TIME (usec)
                if self.execute:
                    self.writes.append(("time_us", v.rd16(off + 1)))
                    if not self.dry:
                        import time; time.sleep(v.rd16(off + 1) / 1e6)
                off += 3
            elif op == 0x57:                                 # INIT_LTIME (msec)
                if self.execute:
                    self.writes.append(("time_ms", v.rd16(off + 1)))
                    if not self.dry:
                        import time; time.sleep(v.rd16(off + 1) / 1e3)
                off += 3
            elif op in (0x8c, 0x8d, 0x8e, 0x92):             # reserved / GPIO (no-op here)
                off += 1
            elif op == 0xaa:
                off += 4
            else:
                raise NotImplementedError("init opcode %#04x at %#x (script @%#x)" % (op, off, start))
        raise RuntimeError("script @%#x ran past %d bytes" % (start, limit))


def main():
    v = VBIOS(open(ROM, "rb").read())
    scripts = mode_scripts(v)
    which = [int(x) for x in sys.argv[1:]] or range(len(scripts))
    card = None
    if os.environ.get("VBINIT_CARD_READS") == "1":
        from card import Card
        card = Card()
    over = {}
    for kv in os.environ.get("VBINIT_COND", "").split(","):
        if kv:
            k, x = kv.split("=")
            over[int(k, 0)] = x not in ("0", "false", "False")
    for i in which:
        it = Init(v, strap=0, card=card, dry=True, cond_override=over)
        end = it.run(scripts[i])
        print("script %d @%#x .. %#x: %d writes (strap 0, ramcfg count %d)" % (i, scripts[i], end, len(it.writes), it.count))
        for w in it.writes:
            print("    %s" % (("%06x <- %08x" % w) if isinstance(w[0], int) else "%s %s" % w))


if __name__ == "__main__":
    main()
