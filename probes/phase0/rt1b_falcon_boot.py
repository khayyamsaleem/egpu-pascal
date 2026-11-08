"""RT-1 (attempt 2): follow nouveau's falcon enable -> load -> boot order exactly.

Attempt 1 failed because it only wrote IMEM and poked START_TRIGGER. UC_CTRL read
back 0x12 (STOPPED still set, START_TRIGGER still pending) — the core was never
enabled, so the trigger had nothing to start.

The real sequence, from nouveau:

  gm200_flcn_enable:   reset_eng -> select -> (reset_pmc: nvkm_mc_enable) ->
                       reset_wait_mem_scrubbing -> wr32(0x084, PMC_BOOT_0)
  gp102_flcn_reset_eng: mask(0x3c0, 1, 1); udelay(10); mask(0x3c0, 1, 0);
                        then wait for mem scrubbing
  gm200_flcn_reset_wait_mem_scrubbing:
                        poll until (rd32(0x10c) & 0x6) == 0
  gm200_flcn_fw_load (PIO path):
                        mask(0x624, 0x80, 0x80); wr32(0x10c, 0)
                        then PIO the code into IMEM
  gm200_flcn_fw_boot:  wr32(0x040, 0xcafebeef)  (mailbox sentinel)
                       wr32(0x104, entry); wr32(0x100, 0x2)
                       poll up to 2000 ms for (rd32(0x100) & 0x10)  (STOPPED)

Our program still writes 0xDEADBEEF to DMEM 0x1010, and that remains the
verification — IMEM reads are masked on this core.
"""
import os
import struct
import time

os.environ.setdefault("APL_REMOTE_SOCK", "/tmp/tinygpu.sock")

from tinygrad.runtime.support.system import APLRemotePCIDevice

SEC2_BASE = 0x87000
PORT = 0
SCAN_FROM, SCAN_TO = 0x1000, 0x1040
MARKER = 0xDEADBEEF
PMC_ENABLE, PSEC = 0x000200, 14

dev = APLRemotePCIDevice("nv", "usb4")
bar0 = dev.map_bar(0, fmt="I")
def rd(a): return bar0[a // 4]
def wr(a, v): bar0[a // 4] = v

class Falcon:
    def __init__(self, base):
        self.base = base
    def r(self, o): return rd(self.base + o)
    def w(self, o, v): wr(self.base + o, v)
    def mask(self, o, m, v): self.w(o, (self.r(o) & ~m) | (v & m))
    def dmem_write(self, start, words):
        self.w(0x1C0 + PORT * 8, (start & 0xFFFFFF) | (1 << 24))
        for v in words: self.w(0x1C4 + PORT * 8, v & 0xFFFFFFFF)
    def dmem_read(self, start, count):
        self.w(0x1C0 + PORT * 8, (start & 0xFFFFFF) | (1 << 25))
        return [self.r(0x1C4 + PORT * 8) for _ in range(count)]
    def load_imem(self, data, start=0, tag0=0):
        pad = data + b"\x00" * ((4 - len(data) % 4) % 4)
        words = list(struct.unpack("<%dI" % (len(pad) // 4), pad))
        while len(words) % 0x40:
            words.append(0)
        self.w(0x180 + PORT * 16, (start & 0xFFFFFF) | (1 << 24))
        tag = tag0
        for i, w in enumerate(words):
            if i % 0x40 == 0:
                self.w(0x188 + PORT * 16, tag)
                tag = (tag + 1) & 0xFFFF
            self.w(0x184 + PORT * 16, w)

def wait(fn, timeout, step=0.002):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if fn(): return True, time.time() - t0
        time.sleep(step)
    return False, time.time() - t0

prog = open("/tmp/rt1.bin", "rb").read()
f = Falcon(SEC2_BASE)
print("program (%d bytes): %s" % (len(prog), prog.hex(" ")))
print("start state: UC_CTRL=0x%08x UC_STATUS=0x%08x 0x10c=0x%08x 0x3c0=0x%08x 0x624=0x%08x PMC_ENABLE=0x%08x"
      % (f.r(0x100), f.r(0x128), f.r(0x10C), f.r(0x3C0), f.r(0x624), rd(PMC_ENABLE)))

def run(label, set_psec):
    print("\n=== %s ===" % label)
    saved = f.dmem_read(SCAN_FROM, (SCAN_TO - SCAN_FROM) // 4)

    # --- disable (gm200_flcn_disable) ---
    f.mask(0x048, 0x3, 0x0)
    f.w(0x014, 0xFFFFFFFF)
    print("  disable: 0x048/0x014 written")

    # --- reset_eng (gp102_flcn_reset_eng) ---
    f.mask(0x3C0, 0x1, 0x1)
    time.sleep(0.001)
    f.mask(0x3C0, 0x1, 0x0)
    ok, dt = wait(lambda: (f.r(0x10C) & 0x6) == 0, 0.05)
    print("  reset_eng: 0x3c0 toggled, mem-scrub clear=%s after %.1f ms (0x10c=0x%08x)"
          % (ok, dt * 1000, f.r(0x10C)))

    # --- enable: reset_pmc + scratch ---
    if set_psec:
        wr(PMC_ENABLE, rd(PMC_ENABLE) | (1 << PSEC))
        print("  reset_pmc: PMC_ENABLE=0x%08x (PSEC bit %d set)" % (rd(PMC_ENABLE), PSEC))
    f.w(0x084, rd(0x000000))
    print("  scratch 0x084 <- PMC_BOOT_0 = 0x%08x" % f.r(0x084))

    # --- load: unlock + PIO ---
    f.mask(0x624, 0x80, 0x80)
    f.w(0x10C, 0x00000000)
    f.load_imem(prog, start=0, tag0=0)
    print("  load: 0x624 bit7 set, 0x10c cleared, IMEM PIO'd (%d bytes padded to 256)"
          % max(256, ((len(prog) + 3) // 4 + 0x3F) // 0x40 * 0x40 * 4))

    # --- boot ---
    f.w(0x040, 0xCAFEBEEF)
    f.w(0x104, 0x0)
    pre = f.r(0x100)
    f.w(0x100, 0x00000002)
    stopped, dt = wait(lambda: f.r(0x100) & 0x10, 2.0)
    print("  boot: UC_CTRL 0x%08x -> 0x%08x ; STOPPED within 2000ms = %s (%.1f ms)"
          % (pre, f.r(0x100), stopped, dt * 1000))
    print("  mailbox0=0x%08x mailbox1=0x%08x (sentinel was 0xcafebeef)"
          % (f.r(0x040), f.r(0x044)))
    print("  UC_STATUS=0x%08x 0x10c=0x%08x" % (f.r(0x128), f.r(0x10C)))

    after = f.dmem_read(SCAN_FROM, (SCAN_TO - SCAN_FROM) // 4)
    hit = [(SCAN_FROM + 4 * i, v) for i, v in enumerate(after) if v == MARKER]
    print("  DMEM 0x1000..0x1040: %s" % " ".join("0x%08x" % v for v in after[:8]))
    print("  marker: %s" % (hit if hit else "not found"))

    f.dmem_write(SCAN_FROM, saved)
    return bool(hit)

win = run("A: full sequence + PSEC set", True)
if not win:
    win = run("B: full sequence, PMC untouched", False)

print("\nRT-1 RESULT: %s" % ("PASS — our code executed on SEC2" if win
                             else "FAIL — core still not running our code"))
