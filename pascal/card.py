"""Shared transport for the GP102 over TinyGPU."""
import os, time, struct
os.environ.setdefault("APL_REMOTE_SOCK", "/tmp/tinygpu.sock")
from tinygrad.runtime.support.system import APLRemotePCIDevice
from sleepguard import SleepGuard, SystemSlept, hold_awake   # sleep -> TB link down -> MMIO panic

class Card:
    def __init__(self):
        hold_awake()                      # no idle/system sleep while this process holds the card
        self.guard = SleepGuard()         # ...and no MMIO at all once the Mac has slept
        self.dev = APLRemotePCIDevice("nv", "usb4")
        self.bar0 = self.dev.map_bar(0, fmt="I")
    BAR0_SIZE = 0x1000000
    def rd(self, a):
        assert 0 <= a < self.BAR0_SIZE and not a & 3, "BAR0 read out of range: %#x" % a
        self.guard.check()
        return self.bar0[a // 4]
    def wr(self, a, v):
        # out-of-range writes used to go out silently (session 4 PRI-ring incident)
        assert 0 <= a < self.BAR0_SIZE and not a & 3, "BAR0 write out of range: %#x" % a
        self.guard.check()
        self.bar0[a // 4] = v & 0xffffffff
    def mask(self, a, m, v):
        o = self.rd(a); self.wr(a, (o & ~m) | (v & m)); return o
    def wait(self, fn, timeout=2.0, step=0.001):
        t0 = time.time()
        while time.time() - t0 < timeout:
            if fn(): return True
            time.sleep(step)
        return False

    # --- physical VRAM through the 64 KB PRAMIN window (BAR0 + 0x700000) ---
    def _win(self, addr):
        self.wr(0x1700, addr >> 16); return 0x700000 + (addr & 0xffff)
    # Any byte address and length.  The window moves 32-bit words, so a copy that doesn't start
    # or end on a word boundary read-modify-writes its first/last word.  (It used to write at the
    # address rounded down and zero-pad the tail, i.e. shift 2-byte-aligned bf16 copies and clobber
    # up to 3 bytes past the end -- MAX's bf16 slices hit that, fp32 workloads never did.)
    def _write_words(self, addr, data):            # addr % 4 == 0, len(data) % 4 == 0
        off = 0
        while off < len(data):
            a = addr + off; n = min(len(data) - off, 0x10000 - (a & 0xffff))
            w = self._win(a)
            self.bar0[w // 4:(w + n) // 4] = list(struct.unpack("<%dI" % (n // 4), data[off:off + n]))
            off += n
    def _read_words(self, addr, n4):               # addr % 4 == 0, n4 % 4 == 0
        out = b""
        while len(out) < n4:
            a = addr + len(out); k = min(n4 - len(out), 0x10000 - (a & 0xffff))
            w = self._win(a)
            out += struct.pack("<%dI" % (k // 4), *self.bar0[w // 4:(w + k) // 4])
        return out
    def vram_write(self, addr, data: bytes):
        self.guard.check()
        data = bytes(data)
        if not data:
            return
        head, end = addr & 3, addr + len(data)
        if not head and not end & 3:
            return self._write_words(addr, data)
        a0, a1 = addr - head, (end + 3) & ~3
        buf = bytearray(a1 - a0)
        if head:
            buf[0:4] = self._read_words(a0, 4)
        if end & 3:
            buf[-4:] = self._read_words(a1 - 4, 4)
        buf[head:head + len(data)] = data
        self._write_words(a0, bytes(buf))
    def vram_read(self, addr, n) -> bytes:
        self.guard.check()
        if n <= 0:
            return b""
        head = addr & 3
        return self._read_words(addr - head, (head + n + 3) & ~3)[head:head + n]
    def vram_w32(self, addr, v): self.vram_write(addr, struct.pack("<I", v))
    def vram_r32(self, addr): return struct.unpack("<I", self.vram_read(addr, 4))[0]
