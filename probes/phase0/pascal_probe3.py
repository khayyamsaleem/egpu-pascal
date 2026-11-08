"""Stage-3 probe: read the registers tinygrad evaluates *before* its failing
architecture lookup, using tinygrad's own include maps. Determines whether
attempting to open the device through NVDev would trigger a PCIe function-level
reset (which, on this Intel-bridge enclosure, can drop the card off the bus)."""
import os, inspect

os.environ.setdefault("APL_REMOTE_SOCK", "/tmp/tinygpu.sock")

from tinygrad.runtime.support.system import APLRemotePCIDevice
import tinygrad.runtime.autogen.nv_regs as nv_regs

dev = APLRemotePCIDevice("nv", "usb4")
bar0 = dev.map_bar(0, fmt="I")

def find_reg(name):
    """Every (module, variant) whose register map defines `name`."""
    hits = []
    for mod_name in dir(nv_regs):
        mod = getattr(nv_regs, mod_name)
        if not inspect.ismodule(mod):
            continue
        for variant, table in vars(mod).items():
            if isinstance(table, dict) and name in table and isinstance(table[name], tuple):
                base, off, fields = table[name]
                hits.append((mod_name, variant, base, off, fields))
    return hits

for name in ("NV_PFB_PRI_MMU_WPR2_ADDR_HI", "NV_PGC6_AON_SECURE_SCRATCH_GROUP_42", "NV_PMC_BOOT_1"):
    hits = find_reg(name)
    print("=== %s ===" % name)
    if not hits:
        print("  not defined in any include map")
        continue
    for mod_name, variant, base, off, fields in hits[:4]:
        val = bar0[(base + off) // 4]
        print("  %s/%s @0x%06x = 0x%08x" % (mod_name, variant, base + off, val))

print("\n=== what tinygrad's dev_fb/tu102 map points at on a Pascal part ===")
print("  (a nonzero WPR2 read makes NVDev issue a PCIe reset before it ever")
print("   reaches the architecture lookup; on this Intel-bridge enclosure a FLR")
print("   can drop the card off the bus until the enclosure is power-cycled)")
