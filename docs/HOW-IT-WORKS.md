# How it works

A short tour of the stack: what runs where, what happens when a token is produced, and how to
tell that the card did the work. Depth lives elsewhere — [RESULTS.md](../RESULTS.md) for measured
numbers, [handoffs/HANDOFF-TO-OPUS.md](handoffs/HANDOFF-TO-OPUS.md) for what is in flight, and
the [evidence trail](README.md) for the route that got here. This page is the map.

## The shape of it

Four layers, and only the bottom one is close to the hardware. Nothing from NVIDIA is installed
on the Mac; the card is driven by our own user-space code over tinygrad's TinyGPU DriverKit dext.

```
  ┌─ Mac (macOS, Apple Silicon) ───────────────────────────────────────────────────────────────┐
  │                                                                                            │
  │   clients      Hermes agent (bin/hermes-local) · chat harness (local-gpu chat)             │
  │                Open WebUI / Zed / curl / any OpenAI or Ollama client                       │
  │          │   http://127.0.0.1:11434                                                        │
  │          ▼                                                                                 │
  │   proxy        localgpu/ollama_proxy.py — one port for Ollama and OpenAI                   │
  │          │   http://127.0.0.1:8080                                                         │
  │          ▼                                                                                 │
  │                                                                                            │
  │   ┌─ Docker (OrbStack) ───────────────────────────────────────────────────────────────┐    │
  │   │                                                                                   │    │
  │   │   llamacpp   llama-server built for sm_61, all layers on the GPU                  │    │
  │   │      │                                                                            │    │
  │   │      │   LD_LIBRARY_PATH = shim/build-cuda12                                      │    │
  │   │   shim       libcudart / libcublas -> libcuda.so.1 + libnvidia-ml.so.1            │    │
  │   │      │                                                                            │    │
  │   │      │   SHIM_MODE=remote                                                         │    │
  │   │                                                                                   │    │
  │   └──────┼────────────────────────────────────────────────────────────────────────────┘    │
  │          │   TCP  host.docker.internal:7461                                                │
  │          ▼                                                                                 │
  │                                                                                            │
  │   shimd      pascal/shimd.py — the CUDA driver ABI, implemented on the Mac                 │
  │          │                                                                                 │
  │   nvpascal   pascal/nvpascal.py — channel, VRAM, async launches of sm_61 cubins            │
  │          │                                                                                 │
  │   card       pascal/card.py — BAR0 / PRAMIN / BAR1 over the TinyGPU socket                 │
  │          │                                                                                 │
  │          │   /tmp/tinygpu.sock                                                             │
  │                                                                                            │
  └──────────┼─────────────────────────────────────────────────────────────────────────────────┘
             ▼  TinyGPU DriverKit dext  ->  Thunderbolt 3 (PCIe Gen3 x4)  ->  GTX 1080 Ti
```

The two hops worth knowing: the container never touches hardware (it speaks the CUDA driver ABI
to a TCP socket), and the Mac never loads a kernel extension of ours (TinyGPU owns the PCI device
and exposes BARs plus a lock over a unix socket).

## What a token costs, end to end

llama.cpp in the container calls `cuLaunchKernel` like it would on any NVIDIA box. The shim turns
that into `op 9 LAUNCH` on the shimd protocol; shimd's `nvpascal` queue is what actually owns the
GPU work. Per launch, in order:

1. `shimd` writes a QMD (the kernel's grid/block/param descriptor), constant bank 0 and a
   pushbuffer segment into one GPU-mapped **host ring** buffer, and rings a doorbell — one MMIO
   write over the transport, not a copy of the parameters through PRAMIN.
2. The PBDMA fetches the GPFIFO entry, the GR engine switches to the channel's context
   (PASCAL_COMPUTE_B, built from the golden context image `golden_ctx_gp102.bin`) and executes the
   pushbuffer. The kernel's code came from a ptxas `-arch=sm_61` cubin loaded into a 240 MiB
   program region.
3. Every launch ends with `WAIT_FOR_IDLE` and a pushbuffer semaphore the GPU releases into host
   memory — so "the kernel finished" is a host-memory read, and results are visible without
   polling PRAMIN (70 MB/s write, 1 MB/s read). Dropping that per-launch synchro was tried and
   **does not work** on this card (RESULTS.md section 14).

A decode step is ~900 of those launches for one token, which is why the measured 11.5-20 tok/s is
launch-bound and SM-clock-bound, not bandwidth-bound (section 13). Numbers, per model:
[RESULTS.md](../RESULTS.md).

## What `bin/local-gpu serve` does

One command, and it skips anything already up:

1. **the card** — `devinit` (runs the VBIOS PMU app, which POSTs the card and trains GDDR5X) and
   `fb` if it is cold; reclock memory 405 → 1080 MHz and prove it with a 9 GiB pattern test,
   stepping down to 810 then 405 on failure; `reset_engines` + `gr` (ACR secure boot of
   FECS/GPCCS, PGRAPH init, then the falcon context); then `shimd`, which holds the TinyGPU
   device lock exclusively and keeps the Mac awake.
2. **the container** — `llamacpp` (or `maxnightly` for `*-max` models), started with
   `-p 127.0.0.1:8080:8080`; Docker is started first if the daemon is down.
3. **the model** — the default is now Gemma 4 E2B (`local-gpu serve -m gemma4-e4b` serves the
   stronger E4B), downloaded once, resumably, into the container's `/models` volume.
4. **the server** — `llama-server` with all layers on the GPU (`-ngl 99 -fa on`, `--jinja`).
5. **the proxy** — `localgpu/ollama_proxy.py` on `127.0.0.1:11434`, so Ollama clients hit
   `/api/*` and OpenAI clients hit `/v1/*` against one address.

`local-gpu watch` runs alongside: the card has to be present in the IO registry, and if the
enclosure is unplugged or powered off the servers and the driver server are stopped before
anything can MMIO to a card that is gone. `local-gpu stop --all` hands the card back.

## The bring-up chain, and why each piece exists

Climbing it is the project; each rung is independently verifiable (PLAN.md section 0):

| Step | Module | What it buys |
|---|---|---|
| Transport + VBIOS POST | `card.py`, `vbios.py`, `devinit.py` | BAR0 access, the card POSTed, 11 GB VRAM usable |
| Falcon code | `falcon_peek.py`, `probes/` | our microcode runs on a falcon; mailbox echo |
| Secure boot | `acr.py` + `mmu.py` | WPR2, FECS/GPCCS booted from signed firmware |
| MMU v2 + FIFO | `mmu.py`, `fifo.py` | page tables, a GPFIFO/PBDMA channel, copy engine |
| Compute dispatch | `gr.py`, `grchan.py`, `grctx.py`, `compute.py` | a channel that runs sm_61 cubins |
| Inference | `nvpascal.py` → `shimd` → shim | llama.cpp and MAX generate on the card |

The ACR is where the interesting constraint lives: its bootloader DMA must go through page tables
(a falcon not yet in secure mode is refused *physical* DMA — `SEC / PHYSICAL / PRIV_VIOLATION`), so
MMU v2 came before secure boot, not after.

## How to tell the card really did the work

This project's rule: every claim is a round trip with a negative control, and the raw value behind
the judgement gets printed. "The server answered" and "the GPU computed" are different claims.

* `SHIM_QUIET=1` logs module loads and held stubs; `SHIMD_TRACE=1` logs copies and pointers;
  `NVPASCAL_PROF=1` (on the server) prints per-kernel GPU time when a client disconnects.
* `probes/freeze_card_test.py` freezes `shimd` mid-generation and shows the token stream stop —
  the strongest cheap proof that the tokens come from this card.
* A stub that is never hit is not a pass: the shim logs unsupported entry points, and
  `test-backend-ops` in the `llamacpp` container names any missing symbol.
* Register read-back proves a write landed, not that a clock changed: `mclk_measure.py` /
  `smclk.py --verify` measure the counters, and `memtest.py` is the only judge of a memory reclock
  (a PRAMIN spot check passes through L2 even with broken DRAM).

## Safety invariants

* **Don't sleep the Mac while anything holds the card.** Sleep takes the Thunderbolt link down and
  the next MMIO is a fatal `LLC Bus error` panic. `card.py` holds a `caffeinate` assertion and
  `sleepguard.py` refuses MMIO after a sleep; keep the lid open (a MacBook can still sleep with
  the lid closed when no external display is attached).
* **BAR0 writes are range-asserted** (`card.py`) — an out-of-range write can take the PRI ring down
  and only an enclosure power cycle recovers it.
* **Never point tinygrad's own NV runtime (`NV=1`) at this card**: it issues a PCIe FLR that Pascal
  misreads, and it never comes back behind this enclosure's Intel bridge.
* **A wedged engine usually does not need a power cycle** — engines reset individually through
  PMC_ENABLE, and the FIFO/GR init chains can be re-run (`reset_engines.py`, `recover_fifo.py`,
  `gr.py`). See the recovery ladder in the handoff before reaching for the plug.
* Voltage and SM-clock writes need the user's explicit OK: a bad frequency/voltage request is the
  one mistake whose only recovery is a physical power cycle.
