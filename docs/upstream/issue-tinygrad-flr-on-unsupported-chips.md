### Summary

`NVDev._early_ip_init` probes `NV_PFB_PRI_MMU_WPR2_ADDR_HI` through the Turing-era `dev_fb/tu102` map *before* it knows which chip it is talking to. On a pre-Ampere part that offset is undecoded and reads back PRI poison (`0xbadf....`), which is non-zero, so the driver concludes "WPR2 is up" and issues a **PCIe function-level reset** - before the `chip_name` lookup below raises `KeyError: 0x13`.

### Environment

- GPU: NVIDIA GTX 1080 Ti (GP102, `architecture = 0x13`), Razer Core X V2 Thunderbolt enclosure, Intel-bridge, on a MacBook Air M1
- Host: macOS 26.6.2, TinyGPU dext `org.tinygrad.tinygpu.driver2` 1.0.0/3
- tinygrad: master, driven through `APLRemotePCIDevice` (`nvdev.py` as of `7bb6492`)

### What happens

1. `self.include("dev_fb", "tu102")`, then `self.reg("NV_PFB_PRI_MMU_WPR2_ADDR_HI").read()` -> the offset is `0x1FA828`, which on Pascal is undecoded. It reads back `0xBADF1100` (NVIDIA's PRI poison, "no target for that address") instead of a real address.
2. `!= 0` is true, so bus mastering is disabled and `self.pci_dev.reset()` runs - a PCIe FLR on a chip the driver cannot drive.
3. Behind an Intel-bridge Thunderbolt enclosure that FLR does not return: the card is gone until the enclosure is power-cycled. That is the only recovery we found.
4. If it did return, line 111 would raise `KeyError: 0x13` anyway, since `chip_name`/`fw_name`/GSP are Ampere-and-newer only.

The `dev_fb/tu102` map is used unconditionally, so this is not specific to Pascal: a Turing part (0x16) hits the same path and the same `KeyError`.

### Minimal reproduction, no GPU needed

tinygrad's own autogen maps plus an MMIO stub that returns what the chip actually returns for the WPR2 offset (`0xBADF1100`), calling `NVDev._early_ip_init` on a stub `NVDev`:

```
variant              arch   PCIe FLR    error
unpatched (master)   0x13   reset=YES   KeyError: 19     <- GP102: FLR, then the unsupported-chip error
unpatched (master)   0x16   reset=YES   KeyError: 22     <- TU100: same
unpatched (master)   0x19   reset=YES   TimeoutError     <- Ada with a poison read
unpatched (master)   0x17   reset=no    TimeoutError     <- Ampere, normal WPR2 read (0)
```

(The `TimeoutError` rows are the simulation running on past the guard into the falcon/GSP setup with no hardware behind it; the FLR column is the part that matters.)

### Expected

A chip this driver can't drive should say so, not reset hardware. Since `docs/tinygpu.md` requires Ampere+ for NVIDIA and the code below this point is already Ampere/Ada/Blackwell-only, the architecture check can happen before any arch-specific register access.

### Notes

- I have a PR for this: read `NV_PMC_BOOT_0`/`NV_PMC_BOOT_42` (both arch-independent, from `nv_ref`) first, then raise a clear `RuntimeError` unless the architecture is `0x17`/`0x19`/`0x1b`. For supported chips the flow is unchanged.
- If you'd rather not reorder, skipping the reset when the read is poison would also avoid the FLR, but the misleading `KeyError` would stay.
- Context: this surfaced while bringing a GP102 up from user space over TinyGPU (https://github.com/khayyamsaleem/egpu-pascal). Written with AI assistance from our session notes; the hardware observations are ours.
