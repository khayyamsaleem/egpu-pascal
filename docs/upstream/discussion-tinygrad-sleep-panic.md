### Summary

If the Mac sleeps while a tinygrad client has the GPU open over TinyGPU, the wake path can kernel-panic the machine and reboot it. Reproduced twice, two days apart, same signature. The panicked task is `TinyGPU`.

(Posting in Q&A because `tinygrad/tinygrad` has issues disabled - happy to move this to an issue in `tinygrad/tinygpu_releases`, which is where the app/dext artifacts live, if that's the preferred tracker.)

### Environment

- MacBook Air M1 (MacBookAir10,1), macOS 26.6.2
- TinyGPU dext `org.tinygrad.tinygpu.driver2` 1.0.0/3; app from `tinygrad/tinygpu_releases` at the pin in `APLRemotePCIDevice.ensure_app` (`c0d024f9`)
- Client: `APLRemotePCIDevice` over `/tmp/tinygpu.sock`, long GPU runs (a driver holding BARs, thousands of MMIO reads/writes per second)
- Card: GTX 1080 Ti (GP102) in a Thunderbolt enclosure, Intel bridge

### What happens

Panic reason strings, both runs:

```
LLC Bus error ... addr=0x480400100 rd_ld
LLC Bus error ... addr=0x480703f6c ncwr
panicked task TinyGPU
```

Sequence: a GPU run in progress -> the Mac sleeps (twice: `Clamshell Sleep` from closing the lid, then `Idle Sleep` with `pmset sleep 1`) -> on wake the Thunderbolt link is down and the next MMIO faults.

The MMIO is issued by the TinyGPU server process, not by the client: client BAR access is an RPC (`RemoteMMIOInterface.__getitem__` -> `dev._bulk_read`, `runtime/support/system.py:312`), so the task touching the device is the server spawned by `APLRemotePCIDevice.__init__`. `TinyGPUDriverUserClient` does guard for a missing device (`if (!ivars->pci) return kIOReturnNotReady;` at `TinyGPUDriver.cpp:161/181/190`), but an already-established mapping and the MMIO after the wake don't go through that check.

### Impact

A full reboot and lost work, with no userspace defence. We now hold `caffeinate -i -s` in every process that has the card open, and refuse all MMIO after a sleep/wake boundary - that mitigates it, but userspace cannot stop a panic caused by a stale mapping, which is why this is filed rather than worked around again.

### Asking for

Any one of:

- a power assertion (or equivalent) while a client has the device open, so the system cannot sleep mid-run; or
- tearing down / invalidating mappings on sleep and re-establishing them on wake; or
- making MMIO through a stale mapping fail with `kIOReturnNotReady` instead of reaching the device.

### Why there is no patch from me

I can't build or sign the dext here, and the reproduction requires inducing a panic that reboots the machine, so I have no way to test a DriverKit change before asking for review. I do have the hardware, the reproduction, and a scripted way to make the Mac sleep mid-run, so I'm happy to test a proposed change - or to prototype the userspace power-assertion option if that's the direction you'd prefer.

### Related

- #15606 (DART IOMMU panic from BAR1 writes on eGPU) and #15605 look like the same family of "access after the Thunderbolt link state changed" faults, different trigger.
- Also filed against the NV driver path: a PCIe FLR gets issued to unsupported (pre-Ampere) chips by the WPR2 probe in `nvdev.py` - PR up shortly.
