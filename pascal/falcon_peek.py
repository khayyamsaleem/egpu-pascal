"""Read a register from the falcon's side -- the same path the ACR app uses.

Why this exists
===============
The ACR app (gp102 acr/ucode_load.bin) reads MMIO registers through SEC2's
external port, I[0x1c000..0x1c300] (= host MMIO 0x700..0x70c).  It fails a step
at imem 0x1600 with error 0x29 unless bits[23:20] of 0x10e5f8 are zero, and the
host cannot settle what that register really holds: BAR0 reads there give
0x80400000, writes do not stick, and using the port itself is refused from the
host (pascal/probe_ext_port.py gets the 0xdead5ec1 access-denied sentinel for
*every* address, including ones whose host value is known).  Only falcon code can
see what the app sees, so that is what runs here.

Mechanism (no branches: envytools' envyas has no labels, so the port is driven
by two short programs booted one after the other, with the host pausing in
between -- the port registers live in the engine, not the falcon core):

    cmd:  I[0x1c300] = 0x20000        port setup
          I[0x1c100] = address
          I[0x1c000] = 0x800000f1     read command
    get:  I[0x1c200] -> D[0x1000]     the value
          I[0x1c000] -> D[0x1004]     the command register afterwards
          I[0x9100]  -> D[0x1008]     host 0x244: the status the app polls first

Controls 0x100ce0 (FB size, host 0x2c8) and 0x12006c (LTC count, host 0xb) are
read first, so the port's addressing can be trusted before the interesting read.
DMEM 0x1000-0x103f is saved and restored.
"""

import os
import re
import struct
import subprocess
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from card import Card          # noqa: E402
import acr                     # noqa: E402

ENVYAS = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                                       "tools", "envytools", "build", "envydis", "envyas"))
DMEM_BASE = 0x1000

# envytools envyas takes no comments and has no labels, and prints a C-style byte
# list rather than a binary.  Comments live here in python instead.
CMD_PROG = [
    "mov $r15 0x1000",              # DMEM result pointer
    "mov $r11 0x20000",             # port setup value
    "mov $r10 0x1c300",
    "iowrs I[$r10] $r11",
    "mov $r11 {ADDR}",              # register to read
    "mov $r10 0x1c100",
    "iowrs I[$r10] $r11",
    "mov $r11 0x800000f1",          # read command
    "mov $r10 0x1c000",
    "iowrs I[$r10] $r11",
    "exit",
]
GET_PROG = [
    "mov $r15 0x1000",
    "mov $r10 0x1c200",             # data out
    "iord $r0 I[$r10]",
    "st b32 D[$r15] $r0",
    "mov $r10 0x1c000",             # command register after the transfer
    "iord $r0 I[$r10]",
    "st b32 D[$r15+0x4] $r0",
    "mov $r10 0x9100",              # host 0x244 -- what the app polls first
    "iord $r0 I[$r10]",
    "st b32 D[$r15+0x8] $r0",
    "exit",
]
# The same port, write side: the app's write32 (imem 0x283d) is this plus data.
POKE_PROG = [
    "mov $r15 0x1000",
    "mov $r11 0x20000",
    "mov $r10 0x1c300",
    "iowrs I[$r10] $r11",
    "mov $r11 {ADDR}",
    "mov $r10 0x1c100",
    "iowrs I[$r10] $r11",
    "mov $r11 {VAL}",
    "mov $r10 0x1c200",
    "iowrs I[$r10] $r11",
    "mov $r11 0x800000f2",          # write command
    "mov $r10 0x1c000",
    "iowrs I[$r10] $r11",
    "exit",
]


def _envyas(lines, **kw):
    with tempfile.NamedTemporaryFile("w", suffix=".asm", delete=False) as fh:
        fh.write("\n".join(_sub(l, kw) for l in lines) + "\n")
        tmp = fh.name
    out = subprocess.run([ENVYAS, "-m", "fuc", "-V", "fuc6", tmp], capture_output=True, text=True)
    os.unlink(tmp)
    txt = out.stdout + out.stderr
    if out.returncode or "error" in txt or "No match" in txt:
        raise SystemExit("envyas failed: %s" % txt[:400])
    return bytes(int(x, 16) for x in re.findall(r"0x([0-9a-fA-F]{2})", out.stdout))


def _sub(line, kw):
    for k, v in kw.items():
        line = line.replace("{%s}" % k, v if isinstance(v, str) and v.startswith("0x") else "0x%x" % v)
    return line


def disassemble(code, base=0):
    envydis = os.path.join(os.path.dirname(ENVYAS), "envydis")
    with tempfile.NamedTemporaryFile("wb", suffix=".bin", delete=False) as fh:
        fh.write(code)
        tmp = fh.name
    out = subprocess.run([envydis, "-m", "fuc", "-V", "fuc6", "-i", "-b", "0x%x" % base, tmp],
                         capture_output=True, text=True).stdout
    os.unlink(tmp)
    return re.sub(r"\x1b\[[0-9;]*m", "", out)


def boot(c, f, prog):
    f.imem_pio(prog, 0, 0)
    f.w(0x040, 0xcafebeef)
    f.w(0x104, 0)
    f.w(0x100, 2)
    return c.wait(lambda: f.r(0x100) & 0x10, 1.0)


def _session(c, f, progs, settle):
    """Reset SEC2 once, then boot each program in turn (the port lives in the
    engine, so its state survives between boots)."""
    f.reset()
    f.mask(0x624, 0x80, 0x80)
    f.w(0x10c, 0)
    oks = []
    for p in progs:
        oks.append(boot(c, f, p))
        time.sleep(settle)
    val, cmdreg, stat = struct.unpack("<3I", f.dmem_read(DMEM_BASE, 12))
    return (val, cmdreg, stat), all(oks)


def peek(c, f, addr, settle=0.02, verbose=True):
    (val, cmdreg, stat), ok = _session(c, f, [_envyas(CMD_PROG, ADDR=addr), _envyas(GET_PROG)], settle)
    if verbose:
        print("  %#010x -> %08x   cmd=%08x  I[0x9100]=%08x  (boots ok=%s)" % (addr, val, cmdreg, stat, ok))
    return val, cmdreg, stat, ok


def poke(c, f, addr, val, settle=0.02, verbose=True):
    """Write through the port, then read the register back the same way (peek)."""
    (after, cmdreg, stat), ok = _session(
        c, f, [_envyas(POKE_PROG, ADDR=addr, VAL=val), _envyas(CMD_PROG, ADDR=addr), _envyas(GET_PROG)], settle)
    if verbose:
        print("  %#010x <- %08x   readback %08x   cmd=%08x  (boots ok=%s)" % (addr, val, after, cmdreg, ok))
    return after, ok


def main():
    print("round-tripping the two programs through envydis before touching the card:")
    for label, prog in (("cmd", CMD_PROG), ("get", GET_PROG), ("poke", POKE_PROG)):
        code = _envyas(prog, ADDR=0x10E5F8, VAL=0x80000000)
        print("  -- %s (%d bytes)" % (label, len(code)))
        for line in disassemble(code).splitlines():
            if line.strip():
                print("    ", line)

    c = Card()
    f = acr.Falcon(c, acr.SEC2, pmc_bit=14)
    before = f.dmem_read(DMEM_BASE, 16)
    try:
        print("\n-- controls (host value in brackets)")
        for addr in (0x100CE0, 0x12006C):
            host = c.rd(addr)
            val, _cr, _st, _ok = peek(c, f, addr)
            print("     host=%08x  %s" % (host, "MATCH" if val == host else "DIFFER"))

        print("\n-- registers the ACR app judges (host value shown after)")
        for addr in (0x10E5F8, 0x10E5E4, 0x17E278, 0x100CD4, 0x100CD8, 0x17E0, 0x140100, 0x100E10):
            val, _cr, _st, _ok = peek(c, f, addr, verbose=False)
            print("  %#010x falcon=%08x  host=%08x%s" % (
                addr, val, c.rd(addr),
                "   bits[23:20]=%d (app needs 0)" % ((val >> 20) & 0xF) if addr == 0x10E5F8 else ""))

        print("\n-- can the falcon write it? (host writes do not stick)")
        cur = c.rd(0x10E5F8)
        want = cur & ~0x00F00000
        after, ok = poke(c, f, 0x10E5F8, want)
        print("     wrote %08x (bits[23:20]=0), read back %08x -> %s; host now reads %08x"
              % (want, after, "WRITE STUCK" if after == want else "write ignored", c.rd(0x10E5F8)))
    finally:
        f.dmem_pio(struct.pack("<16I", *before), DMEM_BASE)   # restore the result block
        f.reset()
        print("\nDMEM 0x%x-0x%x restored, SEC2 reset." % (DMEM_BASE, DMEM_BASE + 0x3c))


if __name__ == "__main__":
    main()
