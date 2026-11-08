"""Stage-4 probe (Phase 0 of the escalation plan): read-only reconnaissance of
what a Pascal bring-up would need to talk to, from the macOS/TinyGPU side.

1. Are the Falcon microcontroller blocks (PMU/SEC2/FECS/GPCCS) addressable and
   decoded, or poison? The whole ACR secure-boot chain runs on these.
2. Is the card's VBIOS visible (needed for straps/devinit)?
No writes, no resets.
"""
import os

os.environ.setdefault("APL_REMOTE_SOCK", "/tmp/tinygpu.sock")

from tinygrad.runtime.support.system import APLRemotePCIDevice

dev = APLRemotePCIDevice("nv", "usb4")
bar0 = dev.map_bar(0, fmt="I")

def rd(addr): return bar0[addr // 4]
def poison(v): return (v & 0xFFFF0000) == 0xBADF0000

print("=== global state ===")
for nm, a in (("PMC_BOOT_0", 0x000000), ("PMC_ENABLE", 0x000200), ("PBUS_DEVID", 0x001000)):
    v = rd(a)
    print("  %-12s @0x%06x = 0x%08x%s" % (nm, a, v, "  POISON" if poison(v) else ""))

print("\n=== Falcon blocks (the ACR / secure-boot chain lives here) ===")
falcons = {"PMU": 0x10a000, "SEC2": 0x840000, "FECS": 0x409000, "GPCCS": 0x41a000}
fregs = {"IRQSTAT": 0x008, "MAILBOX0": 0x040, "MAILBOX1": 0x044,
         "IDLESTATE": 0x048, "OS": 0x050, "CPUCTL": 0x100,
         "BOOTVEC": 0x104, "HWCFG": 0x108, "DMACTL": 0x10c}
for name, base in falcons.items():
    dec = alive = 0
    cells = []
    for rn, off in fregs.items():
        v = rd(base + off)
        if poison(v):
            cells.append("%s=POISON" % rn)
        else:
            dec += 1
            alive += v != 0 and v != 0xFFFFFFFF
            cells.append("%s=0x%08x" % (rn, v))
    print("  %-6s @0x%06x  decoded=%d/%d  nonzero=%d  | %s"
          % (name, base, dec, len(fregs), alive, " ".join(cells[:4])))

print("\n=== VBIOS visibility (straps / devinit source) ===")
print("  PCI ROM BAR (config 0x30) = 0x%08x" % dev.read_config(0x30, 4))
for cand in (0x088000, 0x300000, 0x000000):
    words = [rd(cand + 4 * i) for i in range(4)]
    sig = words[0] & 0xFFFF
    print("  BAR0[0x%06x..] = %s   rom-sig=%s" % (cand, " ".join("0x%08x" % w for w in words),
          "0xAA55 (VBIOS!)" if sig == 0xAA55 else "0x%04x" % sig))
