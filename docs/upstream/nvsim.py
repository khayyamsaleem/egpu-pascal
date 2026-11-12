"""Simulate NVDev._early_ip_init on an unsupported chip, patched vs unpatched.

Uses tinygrad's own autogen register maps; no hardware. The MMIO stub returns what a
GP102 (arch 0x13) actually returns for the Turing-only WPR2 offset: PRI poison.
"""
import importlib.util, sys, types

import tinygrad.runtime.support.memory as _mem
import tinygrad.runtime.support.system as _sys
import tinygrad.runtime.autogen.nv_regs as nv_regs

# The installed tinygrad may predate some names master's nvdev.py imports; stub what's missing.
for _m, _names in ((_mem, ["TLSFAllocator", "MemoryManager", "AddrSpace", "MMIOInterface"]),
                   (_sys, ["PCIDevice"])):
    for _n in _names:
        if not hasattr(_m, _n):
            setattr(_m, _n, type(_n, (), {}))

print("autogen sections: dev_fb.tu102=%s dev_gc6_island.ga102=%s" % (
    hasattr(nv_regs.dev_fb, "tu102"), hasattr(nv_regs.dev_gc6_island, "ga102")))

WPR2_ADDR, BOOT0_ADDR, BOOT42_ADDR = 0x1FA828, 0x000000, 0x000A00


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


class StubPCI:
    def __init__(self): self.resets, self.cfg = 0, {}

    def write_config_flush(self, off, val, size): self.cfg[off] = val
    def read_config(self, off, size): return self.cfg.get(off, 0)
    def reset(self): self.resets += 1


def run(mod, arch, wpr2):
    dev = mod.NVDev.__new__(mod.NVDev)
    dev.pci_dev, dev.devfmt = StubPCI(), "sim"
    dev.mmio = type("M", (), {"__getitem__": lambda s, i: 0, "__setitem__": lambda s, i, v: None})()

    def rreg(addr):
        if addr == WPR2_ADDR: return wpr2                      # the Turing offset, undecoded on this chip
        if addr == BOOT0_ADDR: return 0x13200000                # chip id
        if addr == BOOT42_ADDR: return (arch << 24)             # architecture field is bits 24..29
        return 0
    dev.rreg, dev.wreg = rreg, lambda a, v: None

    try:
        mod.NVDev._early_ip_init(dev)
        err = None
    except Exception as e:
        err = "%s: %s" % (type(e).__name__, str(e)[:90])
    return dev.pci_dev.resets, err


POISON = 0xBADF1100
rows = []
for label, path in (("unpatched (master)", "/tmp/nvdev.py"), ("patched", "/tmp/tg/tinygrad/runtime/support/nv/nvdev.py")):
    mod = load(path, "nvsim_" + str(len(rows)) + str(id(path)))
    for arch, wpr2, note in ((0x13, POISON, "GP102 (Pascal), WPR2 read = PRI poison"),
                             (0x16, POISON, "TU100 (Turing)"),
                             (0x19, POISON, "AD100 (Ada), WPR2 read = poison"),
                             (0x17, 0, "GA100 (Ampere), a normal WPR2 read")):
        resets, err = run(mod, arch, wpr2)
        rows.append((label, "0x%02x" % arch, "reset=YES" if resets else "reset=no ", err, note))

print("\n%-20s %-6s %-11s %-46s %s" % ("variant", "arch", "PCIe FLR", "error", "case"))
for r in rows:
    print("%-20s %-6s %-11s %-46s %s" % r)
