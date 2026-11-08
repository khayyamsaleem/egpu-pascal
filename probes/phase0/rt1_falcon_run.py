"""RT-1: load our own microcode into SEC2 and run it.

The program (assembled by envytools' envyas for `fuc6`, verified by round-tripping
through envydis):

    mov $r1 0x1000          ; base  -> 41 00 10
    mov $r0 0xdeadbeef      ; marker -> d0 ef be ad de
    st b32 D[$r1+0x10] $r0  ; store  -> b5 10 04
    exit                    ; halt   -> f8 02

So if the falcon executes it, DMEM byte 0x1010 becomes 0xDEADBEEF. IMEM cannot be
read back on this core (secure-read mask: zeros on SEC2, 0xdead5ec1 on the PMU),
so behaviour is the only verification available — which is exactly what makes this
a round trip rather than a write-and-hope.

Load path and start sequence are nouveau's (nvkm/falcon/v1.c):
    IMEMC = 0x180 + port*16: start | BIT(24) | (secure ? BIT(28) : 0)
    IMEMT = 0x188 + port*16: fresh tag every 256 B (0x40 words)
    IMEMD = 0x184 + port*16: instruction words; image padded to 256 B
    start: if CPUCTL(0x100) has BIT(6) then wr32(0x130, 2) else wr32(0x100, 2)

DMEM around the target is saved and restored; IMEM is not restorable (unreadable),
which is acceptable: it is blank (zeros) and overwriting it from offset 0 is the
normal, documented operation.
"""
import os
import struct
import sys
import time

os.environ.setdefault("APL_REMOTE_SOCK", "/tmp/tinygpu.sock")

from tinygrad.runtime.support.system import APLRemotePCIDevice

PROG = "/tmp/rt1.bin"
SEC2 = 0x87000
PORT = 0
SCAN_FROM, SCAN_TO = 0x1000, 0x1040          # watch window in DMEM
MARKER = 0xDEADBEEF
EXPECT_AT = 0x1010

dev = APLRemotePCIDevice("nv", "usb4")
bar0 = dev.map_bar(0, fmt="I")
def rd(a): return bar0[a // 4]
def wr(a, v): bar0[a // 4] = v
def poison(v): return (v >> 16) in (0xBADF, 0xBAD0)

class Falcon:
    def __init__(self, base):
        self.base = base
    def r(self, o): return rd(self.base + o)
    def w(self, o, v): wr(self.base + o, v)
    def dmem_write(self, start, words):
        self.w(0x1C0 + PORT * 8, (start & 0xFFFFFF) | (1 << 24))
        for v in words: self.w(0x1C4 + PORT * 8, v & 0xFFFFFFFF)
    def dmem_read(self, start, count):
        self.w(0x1C0 + PORT * 8, (start & 0xFFFFFF) | (1 << 25))
        return [self.r(0x1C4 + PORT * 8) for _ in range(count)]
    def load_imem(self, data, start=0, tag0=0):
        rem = len(data) % 4
        pad = data + b"\x00" * ((4 - rem) % 4)
        words = list(struct.unpack("<%dI" % (len(pad) // 4), pad))
        while len(words) % 0x40:                      # pad image to 256 B
            words.append(0)
        self.w(0x180 + PORT * 16, (start & 0xFFFFFF) | (1 << 24))
        tag = tag0
        for i, w in enumerate(words):
            if i % 0x40 == 0:
                self.w(0x188 + PORT * 16, tag)
                tag = (tag + 1) & 0xFFFF
            self.w(0x184 + PORT * 16, w)
        return len(words) * 4
    def start(self, bootvec):
        self.w(0x104, bootvec)
        ctl = self.r(0x100)
        if ctl & (1 << 6):
            self.w(0x130, 0x2)
            return "0x130"
        self.w(0x100, 0x2)
        return "0x100"
    def poll_halt(self, timeout=0.5):
        t0 = time.time()
        while time.time() - t0 < timeout:
            if self.r(0x100) & (1 << 4):
                return True, time.time() - t0
            time.sleep(0.005)
        return False, time.time() - t0
    def snap(self):
        return dict(CPUCTL=self.r(0x100), BOOTVEC=self.r(0x104), SCTL=self.r(0x240),
                    TIDX=self.r(0x148), IP=self.r(0x14C), MB0=self.r(0x040), MB1=self.r(0x044))

prog = open(PROG, "rb").read()
print("program: %d bytes  %s" % (len(prog), prog.hex(" ")))
f = Falcon(SEC2)
print("falcon info: HWCFG=0x%08x CPUCTL=0x%08x IP=0x%x" % (f.r(0x108), f.r(0x100), f.r(0x14C)))

n = (SCAN_TO - SCAN_FROM) // 4
before = f.dmem_read(SCAN_FROM, n)
print("DMEM before: %s" % " ".join("0x%08x" % v for v in before[:8]))

results = []
for label, bootvec, start_off in (("A: bootvec=0x0, load@0", 0x0, 0),
                                  ("B: bootvec=0x8, load@0", 0x8, 0),
                                  ("C: bootvec=0x0, load@8", 0x0, 0x8)):
    print("\n--- attempt %s ---" % label)
    img = (b"\x00" * start_off) + prog if start_off else prog
    loaded = f.load_imem(img, start=0, tag0=0)
    print("  IMEM: %d bytes loaded (tag every 256 B), BOOTVEC=0x%x" % (loaded, bootvec))
    pre = f.snap()
    print("  pre : CPUCTL=0x%08x IP=0x%x" % (pre["CPUCTL"], pre["IP"]))
    reg = f.start(bootvec)
    halted, dt = f.poll_halt()
    post = f.snap()
    after = f.dmem_read(SCAN_FROM, n)
    hit = [(SCAN_FROM + 4 * i, v) for i, v in enumerate(after) if v == MARKER]
    print("  start via %s; halted=%s after %.1f ms" % (reg, halted, dt * 1000))
    print("  post: CPUCTL=0x%08x IP=0x%x TIDX=0x%08x SCTL=0x%08x MB0=0x%08x MB1=0x%08x"
          % (post["CPUCTL"], post["IP"], post["TIDX"], post["SCTL"], post["MB0"], post["MB1"]))
    print("  DMEM after: %s" % " ".join("0x%08x" % v for v in after[:8]))
    print("  marker 0xdeadbeef found at: %s" % (hit if hit else "NOWHERE"))
    results.append((label, bool(hit), halted))
    if hit:
        break

print("\n=== restore DMEM ===")
f.dmem_write(SCAN_FROM, before)
now = f.dmem_read(SCAN_FROM, n)
print("  restored:", "YES" if now == before else "NO")

win = [r for r in results if r[1]]
print("\nRT-1 RESULT: %s" % ("PASS — our code ran on the GPU's own microcontroller"
                             if win else "FAIL — no marker appeared in DMEM"))
