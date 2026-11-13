# Reference material for the GP102 / GDDR5X work

Local copies live in `docs/reference/files/` (git-ignored: vendor licensing). Re-fetch with the
URLs below. Collected 2026-11-12 (session 12).

## Card

| | |
|---|---|
| GPU | GP102 (16 nm), 3584 CUDA cores, 28 SMs, 6 GPCs, 88 ROPs, CC 6.1 |
| Reference clocks | 1480 MHz base / 1582 MHz boost (we run 607 MHz: the VBIOS boot clock) |
| Memory | 11 GB GDDR5X, 352-bit (11 x 32-bit channels on 6 FBPAs; FBPA0 half-populated), 11 Gb/s, 484 GB/s |
| Board | EVGA, subsystem 3842:6696; 250 W board power |

Sources: [AnandTech review](https://www.anandtech.com/show/11180),
[GamersNexus specs](https://gamersnexus.net/news-pc/2820-official-nvidia-gtx-1080-ti-specs-announced),
[TechPowerUp-style spec listing](https://www.gpu-monkey.com/en/gpu-asus_turbo_geforce_gtx_1080_ti).

## GDDR5X DRAM (Micron MT58K256M32JA, 8 Gb x32)

`files/micron-gddr5x-8gb-datasheet.pdf` (product brief, Rev. C 2/17,
[mirror](https://www.micro-semiconductor.ru/datasheet/be-MT58K256M321JA-100-A.pdf); an
[arrow.com copy](http://static6.arrow.com/aropdfconversion/9105c2cb27fa7e77211e197f32e2236811d5b9f8/6974060544539718gb_gddr5x_sgram_brief.pdf)
timed out). Facts that matter here:

* Grades -100/-110/-120 = 10/11/12 Gb/s. VDD = VDDQ = 1.35 V, VPP = 1.8 V.
* WCK runs at 2x CK. **QDR mode** (16n prefetch, BL16): the DRAM's PLL is mandatory;
  f_CK 687..1375 MHz on -110. **DDR mode** (8n prefetch, BL8): the PLL is **not supported**;
  f_CK 50..1375 MHz on -110 (so up to 5.5 Gb/s per pin). RDQS mode 50..500 MHz.
* 16 banks in 4 bank groups (tCCDL = 3 or 4 tCK); READ latency 9..20, WRITE latency 5..7
  (matches the VBIOS timing CL 9 / 20 and WL 5 we program).
* Trainings: address, WCK2CK (phase via EDC; WCK2CK alignment is "DDR mode only" in the
  clocking figure), read/write via the read FIFO.

What this says about our reclock: at "1080 MHz" we measure ~2.0 Gb/s per pin, i.e. DDR mode
with f_CK = 540 MHz and WCK = 1080 MHz = our refpll output. The DRAM's own DDR-mode ceiling is
f_CK 1375 MHz (WCK 2750), so the failures from 1188 MHz up are on the **controller side**
(bank-1 PHY settings / training), not a DRAM limit. The mode-register definitions (needed to
configure band 2 properly) are in JEDEC JESD232, not in this brief.

## How the memory clock is selected on this chip (from nouveau's source)

Three separate things, easy to conflate (all in `drivers/gpu/drm/nouveau/nvkm/`):

| choice | decided by | where |
|---|---|---|
| **band** (rammap entry: `min..max`, timings index, masks) | the target frequency | `nvbios_rammapEe`, `gk104_ram_calc_data` |
| **FBPA register bank** (which set of PHY trims the hardware reads) | the band's `rammap_11_11_0c` field → write bank `0x10f65c[7:4]`, active bank `[3:0]` | `gk104_ram_prog_0`/`gk104_ram_init` (the comment there: "each script is for a separate mode (likely selected by 0x10f65c's lower bits?)").  Pascal moved the block: 0x10fXXX → 0x9a0XXX |
| **clock source / mode** (refpll vs mempll) | **frequency only**: `ram->mode = (next->freq > fuc->refpll.vco1.max_freq) ? 2 : 1` (`ramgk104.c`) — 1620 MHz here, the refpll VCO max | mode 1 = refpll `0x132024` (src bit `0x1373f4[1:0]` = 1); mode 2 = mempll `0x132004` (src = 2) |

Mode-2 writes mode 1 has no counterpart for (`r1373f4_init`/`_fini`, `gk104_ram_calc_gddr5`):
`0x1373f4 |= 0x1100` then `|= 0x10`; mempll: `0x132000 |= 0x80000000`, `0x132004 = (x &
~0x103fffff) | mcoef` with `mcoef = ((P2-1) << 28) | (N2 << 8) | M2`, wait `0x137390` bit 1; source
`0x1373f4[1:0] = 2` and `0x1373f4 &= ~0x1100`; plus `0x9a0800 &= ~4`, `0x1373f0 |= 2`, `0x9a0830 =
(x & ~0x00800001) | 0x00408010`, `0x9a0808 |= 0x00080000`, `0x9a0200 = (x & ~0x00808000) |
0x00800000`, and *no* write to `0x9a0694`. `gk104_pll_calc_hiclk` picks the refpll reference
frequency (~180 MHz, P1 5..7, N1 0x25..0x2b) and N2 so that the mempll output is the target.

Reading a clock back (`nvkm/subdev/clk/gk104.c`, `read_mem`/`read_pll`):
`src 1 → read_pll(0x132020)` (coef at +4, `0x132030` for fN), `src 2 → read_pll(0x132000)` with
`P = (coef & 0x10000000) ? 2 : 1`, `N = (coef >> 8) & 0xff`, `M = coef & 0xff`, `sclk` = the
refpll's output, `fN = 0xf000`; `out = (sclk * N + (((u16)(fN + 4096) * sclk) >> 13)) / (M * P)`.

Band-dependent registers `gk104_ram_prog_0` writes after a band change, with the rammap fields
(`nvbios_rammapEp` v0x11 reads them from one `rd32(entry + 0x09)`; the field names give byte+mask):
0x9a0468 ← `09_01ff`, `0a_03fe << 12`; 0x9a0420 ← `0a_0400`; 0x9a0430 ← `0a_0800`; 0x9a0400 ←
`0b_01f0`; 0x9a0410 ← `0b_0200 << 9`; 0x9a0440 ← `0d << 16`, `0f << 8`; 0x9a0444 ← `0e << 8`,
`0b_0800 << 7`, `0b_0400 << 5`. On this card DEVINIT/the PMU already leave **band 2's** values
there at boot, so they are not what a band-2 reclock is missing (RESULTS.md section 12).

nouveau sources used above (local copies: `tools/nvkm/subdev/fb/ramgk104.c`, and the other two
fetched on demand): `drivers/gpu/drm/nouveau/nvkm/subdev/fb/ramgk104.c`,
`nvkm/subdev/bios/rammap.c`, `nvkm/subdev/clk/gk104.c` (torvalds/linux, `master`).

## NVIDIA's own documentation (open-gpu-doc, MIT-licensed)

* [MemoryClockTable](https://github.com/NVIDIA/open-gpu-doc/tree/master/MemoryClockTable)
  (`files/open-gpu-doc-MemoryClockTable.html`): the table we call "rammap" (BIT 'P' + 4),
  version 0x11. Base entry: min/max MHz, read/write config words. Strap entry byte 0 = MemTweak
  (timing) index; Flags0 bit 7 = alignment mode (0 phase detector, 1 pin); Flags4 bit 7 = MRS7
  (GDDR5); **Flags5 bit 6 = GDDR5X internal VrefC (0: 70%, 1: 50%)**. This card's strap 0 has
  both bits clear in every band.
* [MemoryTweakTable](https://github.com/NVIDIA/open-gpu-doc/tree/master/MemoryTweakTable)
  (`files/open-gpu-doc-MemoryTweakTable.html`): the timing table (BIT 'P' + 8), version 0x20:
  CONFIG0..5 (RC/RFC/RAS/RP, CL/WL/RD_RCD/WR_RCD, ...), drive strength, voltages, R2P, RDCRC.
* [BIOS-Information-Table](https://github.com/NVIDIA/open-gpu-doc/tree/master/BIOS-Information-Table):
  BIT layout and token IDs ('C' clock pointers v2, 'P' perf pointers v2, 'I', 'M', 'p', ...).
* [pascal/gp100-fbpa.txt](https://github.com/NVIDIA/open-gpu-doc/blob/master/pascal/gp100-fbpa.txt):
  confirms NV_PFB_FBPA 0x10F000 moved to 0x9A0000 on Pascal (what pascal/mclk.py found).

## nvgpu (NVIDIA's open driver, the GP106 discrete-Pascal code)

The current [OE4T/linux-nvgpu](https://github.com/OE4T/linux-nvgpu) branches dropped dGPU; the
`l4t/l4t-r32.7.6` branch still has `drivers/gpu/nvgpu/gp106/`. Local copies in `files/nvgpu-*`.

* `bios.h`: clock-pointer (BIT 'C') indices `CLOCKS_TABLE = 2, CLOCK_PROGRAMMING_TABLE = 3,
  FLL_TABLE = 4, VIN_TABLE = 5, FREQUENCY_CONTROLLER_TABLE = 6`; perf-pointer (BIT 'P') indices
  `VOLTAGE_RAIL_TABLE = 26, VOLTAGE_DEVICE_TABLE = 27, VOLTAGE_POLICY_TABLE = 28` (4-byte
  pointers). This VBIOS's BIT 'C' has entries 0 (PLL), 2, 3, 4 (FLL) and 5 (VIN) populated:
  **the GP102 GPC clock is FLL/ADC-based (NAFLL), not a PLL** (the Kepler GPCPLL at 0x137000
  reads badf on this card).
* `clk_fll.c`, `clk_vin.c`, `volt_dev.c`: the host only parses those VBIOS tables and sends them
  to **PMU firmware**, which programs the FLL lookup table and the voltage PWM
  (`NV_PMU_PMGR_PWM_SOURCE_THERM_VID_PWM_0` for logic, `_1` for SRAM). The host never writes
  the FLL/LUT or PWM registers.
* `hw_trim_gp106.h` / `clk_gp106.c`: host-side clock access is limited to counters, e.g.
  `trim_gpc_bcast_clk_cntr_ncgpcclk_cfg 0x132924` / `_cnt 0x132928` (source gpc2clk =
  0x70000000), sys 0x1373b0/0x1373b4, ltc 0x1373c0/0x1373c4; crystal x4 = 108 MHz.

Consequence: an SM (GPC) clock change on GP102 means doing what NVIDIA's closed PMU firmware
does (FLL LUT + voltage), against undocumented and possibly privilege-locked registers.
nouveau never did it for desktop Pascal ([Phoronix](https://www.phoronix.com/news/NVIDIA-Signed-Firmware-Pascal);
its 2025 [gp10b reclocking patch](https://www.mail-archive.com/nouveau@lists.freedesktop.org/msg47901.html)
is Tegra, where the BPMP owns the clock).

## The "gpu unlock" repos

* [Zagitta/gpu-unlock](https://github.com/Zagitta/gpu-unlock) (GA10x stable; TU10x/AD10x
  experimental) and its Turing port [nlzy/turing-unlock](https://github.com/nlzy/turing-unlock):
  set the FEAT "speed selector" registers (GA10x `0x82381c`/`0x823820`, TU102 `0x409664`/`0x40966c`,
  value `0x88888888`/`0x8`) to run **tensor-core FP16/BF16 at full rate**. The selectors sit
  behind a priv-level mask that the host cannot open; the module runs NVIDIA's signed SEC2
  Booter with a crafted write chain (cmpunlocker's technique) to open it, before the NVIDIA
  driver loads. Press: [letsdatascience](https://letsdatascience.com/news/gpu-unlock-mod-lifts-rtx-30-tensor-core-throughput-up-to-80-fbc34ff8).
* **Not applicable to GP102**: no tensor cores, its 1/64-rate FP16 is the silicon (few FP16x2
  units per SM), not a selector, and our decode is dp4a + bandwidth bound anyway. What does
  transfer is the *technique* (borrowing a signed falcon's privilege to make writes the host
  may not), relevant only if the GPC clock / voltage registers turn out to be PLM-locked.
* Related: [cmpunlocker](https://github.com/Rabbit714205/cmpunlocker) (CMP 170HX),
  [vGPU-Unlock-patcher](https://github.com/61s/vGPU-Unlock-patcher).

## llama.cpp on pre-Ampere

CUDA graphs were introduced with "Disable CUDA graphs for old GPU arch"
([PR #6766](https://app.semanticdiff.com/gh/ggml-org/llama.cpp/pull/6766/overview); NVIDIA's
[blog](https://developer.nvidia.com/blog/optimizing-llama-cpp-ai-inference-with-cuda-graphs)
notes graph launch stays O(n) before Ampere), so graphs are not a lever on this card.

## Pascal architecture

`files/nvidia-gtx1080-pascal-whitepaper.pdf`
([NVIDIA](https://international.download.nvidia.com/geforce-com/international/pdfs/GeForce_GTX_1080_Whitepaper_FINAL.pdf)):
GP104 SM layout, GDDR5X memory subsystem and delta color compression, GPU Boost 3.0 (the V/F
curve the PMU firmware drives).
