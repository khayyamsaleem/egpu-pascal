"""RT-0: data round-trip into a falcon's DMEM.

The smallest honest round trip on this hardware: host -> falcon data memory ->
host. It exercises exactly the PIO path every later stage depends on, and it is
the first test in the harness that *writes* to the GPU and verifies the effect.

Falcon v1 PIO (nouveau nvkm/falcon/v1.c):
    DMEM write: wr32(0x1c0 + port*8, start | BIT(24)); then wr32(0x1c4 + port*8, data)
    DMEM read : wr32(0x1c0 + port*8, start | BIT(25)); then rd32(0x1c4 + port*8)
Falcon info (nouveau nvkm/falcon/base.c):
    rd32(0x12c): version = b0-3, secret = b4-5, code.ports = b8-11, data.ports = b12-15
    rd32(0x108): code.limit (IMEM bytes) = (reg & 0x1ff) << 8
                 data.limit (DMEM bytes) = (reg & 0x3fe00) >> 1

Writes 16 bytes near the top of DMEM, reads them back, restores the original
contents, and verifies the restore. Nothing else is touched.
"""
import os
import sys

os.environ.setdefault("APL_REMOTE_SOCK", "/tmp/tinygpu.sock")

from tinygrad.runtime.support.system import APLRemotePCIDevice

FALCONS = {"SEC2": 0x87000, "PMU": 0x10a000, "FECS": 0x409000}
PORT = 0
PATTERN = [0xDEADBEEF, 0xCAFEBABE, 0x12345678, 0xA5A5A5A5]

dev = APLRemotePCIDevice("nv", "usb4")
bar0 = dev.map_bar(0, fmt="I")
def rd(a): return bar0[a // 4]
def wr(a, v): bar0[a // 4] = v
def poison(v): return (v >> 16) in (0xBADF, 0xBAD0)

class Falcon:
    def __init__(self, name, base):
        self.name, self.base = name, base
        info = rd(base + 0x12c)
        hwcfg = rd(base + 0x108)
        self.version = info & 0xF
        self.secret = (info >> 4) & 0x3
        self.code_ports = (info >> 8) & 0xF
        self.data_ports = (info >> 12) & 0xF
        self.imem_bytes = (hwcfg & 0x1FF) << 8
        self.dmem_bytes = (hwcfg & 0x3FE00) >> 1
        self.hwcfg, self.info = hwcfg, info
        self.alive = not (poison(info) or poison(hwcfg))
    def r(self, off): return rd(self.base + off)
    def w(self, off, v): wr(self.base + off, v)
    def dmem_write(self, start, words, port=PORT):
        self.w(0x1C0 + port * 8, (start & 0xFFFFFF) | (1 << 24))
        for v in words:
            self.w(0x1C4 + port * 8, v & 0xFFFFFFFF)
    def dmem_read(self, start, count, port=PORT):
        self.w(0x1C0 + port * 8, (start & 0xFFFFFF) | (1 << 25))
        return [self.r(0x1C4 + port * 8) for _ in range(count)]
    def dmem_peek(self, start, count, port=PORT):
        """Read without disturbing the control register twice."""
        self.w(0x1C0 + port * 8, (start & 0xFFFFFF) | (1 << 25))
        out = []
        for i in range(count):
            out.append(self.r(0x1C4 + port * 8))
        return out

print("=== falcon inventory ===")
falcons = {}
for name, base in FALCONS.items():
    f = Falcon(name, base)
    falcons[name] = f
    if f.alive:
        print("  %-5s @0x%06x v%d secret=%d ports(code=%d,data=%d) IMEM=%d B DMEM=%d B HWCFG=0x%08x"
              % (name, base, f.version, f.secret, f.code_ports, f.data_ports,
                 f.imem_bytes, f.dmem_bytes, f.hwcfg))
    else:
        print("  %-5s @0x%06x not responding (info=0x%08x hwcfg=0x%08x)" % (name, base, f.info, f.hwcfg))

target = falcons["SEC2"]
if not target.alive:
    print("\nSEC2 not responding — aborting")
    sys.exit(1)

# Stay clear of anything the falcon might be using: use a slot near the top of DMEM.
offset = (target.dmem_bytes - 0x40) & ~0xF
print("\n=== RT-0: DMEM round trip on SEC2 @0x%06x (offset 0x%x, port %d) ==="
      % (target.base, offset, PORT))

saved = target.dmem_peek(offset, len(PATTERN))
print("  saved  :", " ".join("0x%08x" % v for v in saved))

target.dmem_write(offset, PATTERN)
back = target.dmem_peek(offset, len(PATTERN))
print("  wrote  :", " ".join("0x%08x" % v for v in PATTERN))
print("  read   :", " ".join("0x%08x" % v for v in back))

ok = back == PATTERN
print("  match  :", "YES — round trip WORKS" if ok else "NO — data did not survive")

# A second, different pattern catches "always returns the same thing".
second = [0x00000000, 0xFFFFFFFF, 0x0F0F0F0F, 0xF0F0F0F0]
target.dmem_write(offset, second)
back2 = target.dmem_peek(offset, len(second))
print("  wrote2 :", " ".join("0x%08x" % v for v in second))
print("  read2  :", " ".join("0x%08x" % v for v in back2))
ok2 = back2 == second
print("  match2 :", "YES — memory is live, not a fixed response" if ok2 else "NO")

print("\n=== restore ===")
target.dmem_write(offset, saved)
restored = target.dmem_peek(offset, len(PATTERN))
print("  now    :", " ".join("0x%08x" % v for v in restored))
print("  restored:", "YES" if restored == saved else "NO — ORIGINAL DMEM NOT RECOVERED")

print("\nRESULT: RT-0 %s" % ("PASS" if (ok and ok2 and restored == saved) else "FAIL"))
