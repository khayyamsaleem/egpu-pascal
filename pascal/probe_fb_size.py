"""Probe the registers the ACR app's VRAM-size helper (0x1393) reads."""
import sys
sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
from card import Card

c = Card()
names = [("0x17e278 LTC size raw", 0x17e278), ("0x12006c LTC count", 0x12006c),
         ("0x100ce0 fb size?", 0x100ce0), ("0x100c1c", 0x100c1c), ("0x100c20", 0x100c20),
         ("0x10e5e4", 0x10e5e4), ("0x17e8", 0x17e8), ("0x17e0", 0x17e0),
         ("0x120070", 0x120070), ("0x100cd8", 0x100cd8), ("0x100cdc", 0x100cdc),
         ("0x100cd4", 0x100cd4), ("0x100cd0", 0x100cd0), ("0x100c14", 0x100c14),
         ("0x100c10", 0x100c10), ("0x100c80", 0x100c80)]
for n, a in names:
    print("%-22s %08x" % (n, c.rd(a)))

r0 = c.rd(0x17e278) & 0x3FFFFFF
cnt = c.rd(0x12006c) & 0x1F
v = (r0 * cnt) << 11
print("app 0x1393 formula: (0x17e278 & 0x3ffffff)=%#x X count=%d << 11 = %#x (%d MiB)"
      % (r0, cnt, v, v >> 20))
print("our WPR start/end: %#x / %#x   (threshold in the 0x25 check = %#x)"
      % (0x2040000, 0x2080000, 0x17FFFFFF))
