# egpu-pascal

A from-scratch, user-space bring-up of an **NVIDIA GTX 1080 Ti (GP102, Pascal)** in a
Thunderbolt enclosure on an **Apple Silicon MacBook Air**. It needs no NVIDIA driver and
runs over tinygrad's TinyGPU DriverKit transport.

**Status (2026-11-09):** the card POSTs from macOS, secure-boots its graphics firmware,
and runs **ptxas-compiled sm_61 CUDA kernels** dispatched through our own channel/QMD code.
**Modular MAX runs LLMs on it** through a libcuda/NVML shim and local MAX patches
([`maxpatch/`](maxpatch/)): SmolLM2-135M (fp32, text identical to MAX-on-CPU, 6.6 tok/s)
and **Gemma 4 E2B** (bf16 on the MAX nightly, "The capital of France is **Paris**." as
transformers gives, 1.9 tok/s).
Full results: **[RESULTS.md](RESULTS.md)**.

```
rung 0  transport + VBIOS POST from macOS + 11 GB VRAM   DONE
rung 1  our microcode on a falcon                         DONE
rung 2  ACR secure boot, FECS/GPCCS running               DONE
rung 3  MMU v2 + GPFIFO channel + copy engine             DONE
rung 4  compute dispatch: sm_61 vadd -> host memory        DONE
rung 5  model inference (MAX): SmolLM2-135M, Gemma 4 E2B  DONE (speed work next)
```

Boot clocks only for now: 607 MHz SM, 405 MHz memory (~30 GB/s measured).

## Requirements

* Apple Silicon Mac, a TB3/USB4 eGPU enclosure, a GP102 card (validated: EVGA GTX 1080 Ti)
* tinygrad's **TinyGPU** DriverKit extension installed and enabled; Python venv with
  `tinygrad==0.14.0` (scripts assume `~/.venvs/tinygpu/bin/python`; the transport
  socket is `/tmp/tinygpu.sock`)
* firmware: `firmware/fetch.sh` (NVIDIA signed blobs from linux-firmware) and
  `pascal/dump_vbios.py` (this card's VBIOS); see [firmware/README.md](firmware/README.md)
* kernels: Docker (OrbStack works) for CUDA 12.4 `ptxas` via `kernels/build.sh`
* optional: [envytools](https://github.com/envytools/envytools) cloned and built in
  `tools/envytools` (falcon/SASS disassembly, rnndb)

## Run

```sh
PY=~/.venvs/tinygpu/bin/python
# cold card (after an enclosure power cycle)
$PY pascal/devinit.py firmware/gp102_vbios.rom
$PY pascal/fb.py
# every session
$PY pascal/reset_engines.py      # PFIFO + GR engine reset
$PY pascal/gr.py                 # ACR -> SEC2 RTOS -> FECS/GPCCS -> GR init   (GR BRING-UP RESULT: PASS)
$PY pascal/compute.py            # RT-4: sm_61 vadd -> host memory             (RESULT: PASS)
$PY pascal/bench.py              # SM clock + DRAM bandwidth via the runtime
```

Other gates: `pascal/mmu.py --selftest` (offline), `pascal/mmu_test.py` (RT-2, 18/18),
`pascal/fifo.py` (RT-3 copy engine), `pascal/grchan.py` (GR channel), `pascal/ce_bw.py`,
`pascal/test_modules.py` (CUDA-module ABI: local memory, globals, constant banks, 5000-launch wrap).

## Run an LLM with MAX

```sh
# Mac: card up (above), then the driver server
$PY pascal/shimd.py --vram-limit-gib 4
# Linux container with modular==26.6.0 and this repo at /repo (see shim/README.md)
git clone --depth 1 --branch max/v26.6.0 https://github.com/modular/modular.git upstream/modular
docker exec maxlab bash /repo/maxpatch/install.sh            # sm_61 target + SIMT matmul
docker exec maxlab bash /repo/shim/tests/generate.sh gpu "The capital of France is"
#   -> The capital of France is Paris.
# Gemma 4 E2B: MAX nightly container (see maxpatch/README.md), then
docker exec maxnightly bash -lc 'MODEL=google/gemma-4-E2B-it ENCODING=bfloat16 MAX_GEMMA4_ACTIVATION_MIB=1024 \
    bash /repo/shim/tests/generate.sh gpu "The capital of France is" --no-prefer-module-v3'
#   -> The capital of France is **Paris**.
```

## Using the runtime

```python
import nvpascal, struct
dev = nvpascal.Device()                         # GR channel + compute state
k = dev.load("kernels/vadd.sm_61.cubin")["vadd"]
a, b = dev.alloc(4096 * 4), dev.alloc(4096 * 4) # VRAM
c = dev.alloc_host(4096 * 4)                    # Mac memory through the IOMMU
ns = dev.launch(k, (32, 1, 1), (128, 1, 1), struct.pack("<QQQI", a.va, b.va, c.va, 4096))
```

## Layout

| Path | What |
|---|---|
| `pascal/` | the driver: `card` (BAR0/PRAMIN), `devinit`/`vbios`, `fb`, `mmu`, `acr`/`acr_unload`, `sec2_rtos`, `gr`/`gr_ctxctl`/`grctx`, `fifo`, `grchan`, `compute`, **`nvpascal`** (runtime), `cubin`, `bench`, `fecs_snap`, probes, recovery |
| `kernels/` | PTX sources, built sm_61 cubins, `Dockerfile` + `build.sh` |
| `shim/` | `libcuda.so.1` + `libnvidia-ml.so.1` replacements for MAX (Linux container) and their tests |
| `maxpatch/` | the local MAX 26.6.0 patches for sm_61 and their installer |
| `tools/nvclass`, `tools/nvkm`, `tools/mesa` | vendored references (MIT): NVIDIA class/QMD headers, nouveau, Mesa |
| `docs/` | the chronological evidence trail and per-session handoffs |
| `logs/` | hardware output cited by RESULTS.md and the docs |
| `probes/phase0/` | the first read-only probes and RT-0/RT-1 microcode tests |
| `PLAN.md` | the plan of record (phases, routes, odds) |

## Safety notes

* An out-of-range or wrong BAR0 write can take the PRI ring down; recovery is an
  enclosure power cycle. `card.py` asserts the BAR0 range.
* Never write falcon `0x000/0x004/0x010/0x014` (IRQ set/clear registers) from a probe.
* **Don't let the Mac sleep while anything holds the card.** Sleep takes the Thunderbolt link
  down, and the next MMIO on wake is a fatal `LLC Bus error` kernel panic (it happened twice:
  lid close, then a 1-minute idle sleep). `pascal/card.py` now holds a `caffeinate` assertion for
  as long as a process has the card open, and refuses all MMIO once the Mac has slept
  (`pascal/sleepguard.py`). Closing the lid can still sleep a MacBook without an external
  display, so keep it open during runs. After any sleep the card is cold: devinit + fb + gr.
* tinygrad's own NV runtime will issue a PCIe FLR on Pascal (it misreads a Turing WPR2
  register); don't point `NV=1` tinygrad at this card.

## Prior art and credits

* [TheTom/pascal-egpu](https://github.com/TheTom/pascal-egpu): the earlier Pascal-over-TinyGPU
  attempt (register access on GP106). This repo is an independent implementation, not a fork.
  Its "POST is impossible" conclusion didn't hold on GP102.
* [Davinchy/macuda](https://github.com/Davinchy/macuda): the CUDA-shim design for eGPUs on macOS (GSP cards).
* nouveau, Mesa, envytools, NVIDIA open-gpu-doc, tinygrad: see [NOTICE.md](NOTICE.md).
