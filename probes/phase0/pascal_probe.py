"""Stage-1 probe: read-only register access to a Pascal GPU over TinyGPU.

Uses tinygrad's own register definitions (autogen/nv_regs) for offsets, so the
reads are against the same maps the framework itself uses. No writes, no reset,
no GSP boot.
"""
import os, time

os.environ.setdefault("APL_REMOTE_SOCK", "/tmp/tinygpu.sock")

from tinygrad.helpers import getbits
from tinygrad.runtime.support.system import APLRemotePCIDevice
import tinygrad.runtime.autogen.nv_regs as nv_regs

print("=== connect ===")
dev = APLRemotePCIDevice("nv", "usb4")
print("connected: pcibus=%s dev_id=%d" % (dev.pcibus, dev.dev_id))

cfg = dev.read_config(0, 4)
print("PCI config[0:4] = 0x%08x  vendor=0x%04x device=0x%04x  (%s)"
      % (cfg, cfg & 0xFFFF, (cfg >> 16) & 0xFFFF,
         "GTX 1080 Ti" if (cfg & 0xFFFF) == 0x10DE and ((cfg >> 16) & 0xFFFF) == 0x1B06 else "?"))

print("\n=== BARs ===")
bars = {}
for b in range(6):
    try:
        addr, size = dev.bar_info(b)
        bars[b] = (addr, size)
        print("  BAR%d: addr=0x%x size=0x%x (%.1f MB)" % (b, addr, size, size / 1e6))
    except Exception as e:
        print("  BAR%d: n/a (%s)" % (b, type(e).__name__))

bar0 = dev.map_bar(0, fmt="I")
print("\n=== BAR0 MMIO (mapped %d bytes) ===" % bar0.nbytes)

def reg_offset(name):
    """(base, off, fields) out of nv_ref's register map."""
    return nv_regs.nv_ref.regs[name]

# --- PMC_BOOT_0 : chip id
b, off, _ = reg_offset("NV_PMC_BOOT_0")
boot0 = bar0[(b + off) // 4]
print("NV_PMC_BOOT_0  (0x%06x) = 0x%08x" % (b + off, boot0))

# --- PMC_BOOT_42 : architecture/implementation, decoded with tinygrad's field map
b, off, fields = reg_offset("NV_PMC_BOOT_42")
boot42 = bar0[(b + off) // 4]
print("NV_PMC_BOOT_42 (0x%06x) = 0x%08x" % (b + off, boot42))
dec = {name: getbits(boot42, s, e) for name, (s, e) in fields.items()}
print("  decoded fields:", {k: hex(v) for k, v in dec.items()})

# --- PTIMER: is the GPU's clock alive?
ptimer = {k: v for k, v in nv_regs.nv_ref.regs.items() if "PTIMER" in k and "TIME" in k}
print("\n=== PTIMER registers known to this map: %s ===" % sorted(ptimer))
if ptimer:
    rb, roff, _ = sorted(ptimer.items())[0][1]
    r0 = bar0[(rb + roff) // 4]
    time.sleep(0.05)
    r1 = bar0[(rb + roff) // 4]
    print("  %s (0x%06x): 0x%08x -> 0x%08x  delta=%d ns (%s)"
          % (sorted(ptimer)[0], rb + roff, r0, r1, (r1 - r0) & 0xFFFFFFFF, "TICKING" if r1 != r0 else "STUCK"))

# --- the gate, demonstrated from the device's own registers
print("\n=== tinygrad's architecture gate ===")
supported = {0x17: "GA1", 0x19: "AD1", 0x1b: "GB2"}
arch = None
for key in ("architecture", "arch"):
    if key in dec:
        arch = dec[key]
        break
print("  nvdev.py: chip_name = {0x17:'GA1', 0x19:'AD1', 0x1b:'GB2'}[chip_details['architecture']]")
print("  device reports architecture = 0x%x" % arch if arch is not None else "  (architecture field not present)")
if arch is not None and arch not in supported:
    print("  => KeyError: 0x%x is not in the map. tinygrad stops here." % arch)
elif arch in supported:
    print("  => in map (would proceed to GSP boot)")

print("\n=== VRAM size register used by tinygrad ===")
try:
    b, off, _ = reg_offset("NV_PGC6_AON_SECURE_SCRATCH_GROUP_42")
    raw = bar0[(b + off) // 4]
    print("  raw=0x%08x -> %d MB" % (raw, raw))
except Exception as e:
    print("  n/a:", type(e).__name__, e)
