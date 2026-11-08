"""Stage-2 probe: identity + liveness + a read-only look at the registers
tinygrad touches before its architecture gate. All reads; nothing written."""
import os, time, inspect

os.environ.setdefault("APL_REMOTE_SOCK", "/tmp/tinygpu.sock")

from tinygrad.runtime.support.system import APLRemotePCIDevice
import tinygrad.runtime.autogen.nv_regs as nv_regs

dev = APLRemotePCIDevice("nv", "usb4")
bar0 = dev.map_bar(0, fmt="I")

def find_reg(name):
    """Search every include map for a register name; return list of (module, variant, base, off, fields)."""
    hits = []
    for mod_name in dir(nv_regs):
        mod = getattr(nv_regs, mod_name)
        if not inspect.ismodule(mod):
            continue
        for variant, table in vars(mod).items():
            if not isinstance(table, dict) or "regs" not in table:
                continue
            regs = table["regs"]
            if name in regs:
                v = regs[name]
                if isinstance(v, tuple) and len(v) == 3:
                    hits.append((mod_name, variant, v[0], v[1], v[2]))
    return hits

print("=== PRISE identity ===")
b, off, fields = nv_regs.nv_ref.regs["NV_PMC_BOOT_0"]
print("PMC_BOOT_0 = 0x%08x  (chip id 0x%x -> %s)"
      % (bar0[0], (bar0[0] >> 20) & 0xfff,
         {0x130: "GP100", 0x132: "GP102 (1080 Ti)", 0x134: "GP104", 0x136: "GP106", 0x137: "GP107", 0x138: "GP108"}.get((bar0[0] >> 20) & 0xfff, "?")))

print("\n=== PTIMER: is the GPU clock alive? ===")
hits = find_reg("NV_PTIMER_TIME_0")
print("register map hits for NV_PTIMER_TIME_0:", [(m, v, hex(bb + oo)) for m, v, bb, oo, _ in hits])
# classic Fermi+ PTIMER base; formula: PTIMER_TIME_0 at 0x009400
for label, addr in (("0x009400 (probed)", 0x9400), ("0x009410 (probed)", 0x9410)):
    v = bar0[addr // 4]
    print("  BAR0[%s] = 0x%08x" % (label, v))
t0 = bar0[0x9400 // 4] | (bar0[0x9410 // 4] << 32)
time.sleep(0.05)
t1 = bar0[0x9400 // 4] | (bar0[0x9410 // 4] << 32)
dt = (t1 - t0) & 0xFFFFFFFFFFFFFFFF
print("  PTIMER 64-bit: 0x%x -> 0x%x  delta=%d ns  => %s"
      % (t0, t1, dt if dt < (1 << 40) else -1, "TICKING" if t1 != t0 else "STUCK"))

print("\n=== registers tinygrad reads before its gate ===")
for name in ("NV_PFB_PRI_MMU_WPR2_ADDR_HI", "NV_PGC6_AON_SECURE_SCRATCH_GROUP_42"):
    hits = find_reg(name)
    if not hits:
        print("  %s: not present in any include map" % name)
        continue
    mod, variant, base, off, _ = hits[0]
    val = bar0[(base + off) // 4]
    print("  %s (%s/%s @0x%06x) = 0x%08x" % (name, mod, variant, base + off, val))

print("\n=== BAR1 (VRAM aperture) read test ===")
bar1 = dev.map_bar(1, fmt="B")
head = bytes(bar1[i] & 0xFF for i in range(32))
print("  BAR1 size = %d MB" % (bar1.nbytes // (1024 * 1024)))
print("  first 32 bytes:", head.hex(" "))
print("  all-ones (dead aperture):", all(c == 0xFF for c in head))
print("  all-zero:", all(c == 0x00 for c in head))
