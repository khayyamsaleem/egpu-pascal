"""RT-1a: read-only reconnaissance of SEC2/PMU before loading any microcode.

Answers three questions that decide how RT-1 is written:
  1. what is already in IMEM at reset (a resident boot ROM? zeros?)
  2. where is the falcon's program counter (TRACEPC / IDX / IP) — i.e. where did
     it stop, and does that look like a halted boot ROM or a blank core?
  3. does the IMEM window support PIO reads, so we can diff IMEM before/after a
     load later (the write path is confirmed; the read path for IMEM is not).

Falcon v1 offsets (nouveau):
  0x100 CPUCTL   0x104 BOOTVEC   0x108 HWCFG   0x10c DMACTL
  0x12c version/ports        0x148 TIDX   0x14c IP   0x240 SCTL
  0x180 IMEMC    0x184 IMEMD   0x188 IMEMT
  0x1c0 DMEMC    0x1c4 DMEMD
IMEMC/DMEMC mode bits: BIT(24) = write, BIT(25) = read (DMEM confirmed in RT-0;
IMEM assumed by symmetry — that assumption is what this probe tests).
"""
import os

os.environ.setdefault("APL_REMOTE_SOCK", "/tmp/tinygpu.sock")

from tinygrad.runtime.support.system import APLRemotePCIDevice

dev = APLRemotePCIDevice("nv", "usb4")
bar0 = dev.map_bar(0, fmt="I")
def rd(a): return bar0[a // 4]

PORT = 0
TARGETS = {"SEC2": 0x87000, "PMU": 0x10a000}

for name, base in TARGETS.items():
    print("=== %s @0x%06x ===" % (name, base))
    hw, ctl, vec, dma, sctl, tidx, ip = (rd(base + o) for o in (0x108, 0x100, 0x104, 0x10c, 0x240, 0x148, 0x14c))
    print("  HWCFG=0x%08x CPUCTL=0x%08x BOOTVEC=0x%08x DMACTL=0x%08x" % (hw, ctl, vec, dma))
    print("  SCTL=0x%08x TIDX=0x%08x IP=0x%08x" % (sctl, tidx, ip))
    imem = (hw & 0x1FF) << 8
    # decode a few CPUCTL bits the way nouveau/falcon docs describe them
    print("  CPUCTL: halted=%d started=%d alias=%d" % ((ctl >> 4) & 1, (ctl >> 1) & 1, (ctl >> 0) & 1))

    print("  --- IMEM contents via PIO read (mode BIT(25)) ---")
    for off, label in ((0x000, "offset 0"), (imem - 0x40, "top-0x40")):
        try:
            rd(base + 0x180 + PORT * 16)          # reset the window
            bar0[(base + 0x180 + PORT * 16) // 4] = (off & 0xFFFFFF) | (1 << 25)
            words = [rd(base + 0x184 + PORT * 16) for _ in range(8)]
            print("    %-10s: %s" % (label, " ".join("0x%08x" % w for w in words)))
        except Exception as e:
            print("    %-10s: read failed %s: %s" % (label, type(e).__name__, e))

    print("  --- IMEM contents via PIO read (mode 0, plain window) ---")
    try:
        bar0[(base + 0x180 + PORT * 16) // 4] = 0x00000000 | (0 << 24)
        words = [rd(base + 0x184 + PORT * 16) for _ in range(8)]
        print("    offset 0  : %s" % " ".join("0x%08x" % w for w in words))
    except Exception as e:
        print("    read failed %s: %s" % (type(e).__name__, e))
    print()
