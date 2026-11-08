# Round-trip test plan: proving a tiny model runs on the 1080 Ti

## Status — ACR + FECS/GPCCS boot PASS (2026-11-08, session 4)

HS ACR load passes (WPR2 programmed, host refused inside it), the SEC2 RTOS boots
and bootstraps FECS + GPCCS over its command queue, and FECS answers host
methods. RT-3 (FIFO) is now unblocked on the firmware side. See `MMU-AND-ACR.md`
(session 4). Run order on a warm, POSTed card:
`pascal/sec2_rtos.py` then `pascal/gr_ctxctl.py` (or `gr_ctxctl.py --full`).

Every stage up to inference is otherwise unfalsifiable — you cannot tell "the
register write worked" from "the write went nowhere" by looking. A round trip
fixes that: the host writes a known input, the GPU transforms it, the host reads
it back and compares against a reference computed on the CPU. Each rung is a
smaller, cheaper version of the real thing, and each one is a runnable test.

## Status — RT-2 PASSES (2026-11-08, session 3)

`pascal/mmu_test.py`: the MMU v2 page tables from `pascal/mmu.py` are built in
VRAM, published through a 4 KiB instance block, and SEC2 is bound to them. A
falcon xfer then reads 256 B from a **virtual** address into DMEM and stores
DMEM back to a second virtual address — both byte-identical to the VRAM pages we
wrote. The negative control is what makes it a round trip rather than a lucky
read: the same xfer from an unmapped VA delivers nothing and raises an MMU fault
whose record names our instance block (`client=SEC reason=PDE addr=0x800000
inst=0x10001000`). 18/18 checks. See `MMU-AND-ACR.md`.

## Status — RT-0 PASSES (2026-11-08)

`rt0_dmem_roundtrip.py`: 16 bytes written into SEC2's DMEM and read back
identical, a second pattern proving the memory is live rather than a fixed
response, and the original contents restored and verified. The PIO data path
works end to end.

Falcon facts measured while doing it (`0x108` / `0x12c` decode):

| Falcon | Base | Version | Secret | Ports (code/data) | IMEM | DMEM |
|---|---|---|---|---|---|---|
| SEC2 | 0x87000 | **6** | 3 | 1 / 1 | 64 KB | 64 KB |
| PMU | 0x10a000 | **6** | 3 | 1 / 4 | 64 KB | 64 KB |

Version 6 is decisive: envytools' `envyas`/`envydis` ship a `fuc6` variant
("falcon version 6, GP102 and up"), so a real assembler and disassembler exist
for exactly this core. That is the RT-1 tooling.

Reset state, also measured: SEC2 is **halted at PC=0x8**, `BOOTVEC=0`,
`DMACTL=0x1`, `started=0` — a virgin, never-run core, which is what we want to
load into. The PMU has run (PC=0xa4d, `SCTL=0x3002`).

IMEM **reads** come back masked: zeros on SEC2, and a `0xdead5ec1` fill on the
PMU (a "dead" sentinel, not the `0xBADF` poison family) — most likely
secure-read protection on an HS-capable core (`secret=3`). Consequence: RT-1
cannot verify the load by reading the code back. Verification must be
behavioural — the program writes a known value into DMEM and we read that back
with the RT-0 path that already works.

Start sequence for this core (nouveau `nvkm_falcon_v1_start`): if `CPUCTL`
(0x100) has bit 6 set, start by writing `0x130 = 0x2`; otherwise write
`0x100 = 0x2`. Both our falcons have bit 6 clear, so they take the classic path.
IMEM load is `wr32(0x180 + port*16, start | BIT(24) | (secure ? BIT(28) : 0))`,
a fresh tag to `0x188 + port*16` every 256 bytes, data to `0x184 + port*16`, and
the image padded to 256 bytes.

## RT-1 PASSES — our microcode ran on SEC2 (2026-11-08)

Program (13 bytes, assembled with `envyas -m fuc -V fuc6`, and verified by
round-tripping through `envydis` first, because IMEM cannot be read back to check
it):

```
mov $r1 0x1000          ; 41 00 10
mov $r0 0xdeadbeef      ; d0 ef be ad de
st b32 D[$r1+0x10] $r0  ; b5 10 04
exit                    ; f8 02
```

Result: **`0xdeadbeef` appeared at DMEM 0x1010** — the address the program
computes for itself. Verification is behavioural by design; IMEM reads on this
core are masked.

The sequence that works — nouveau's enable → load → boot, with nothing skipped:

1. **disable**: `0x048 &= ~3`, `0x014 = 0xffffffff`
2. **reset_eng** (`gp102_flcn_reset_eng`): `mask(0x3c0, 1, 1)`; ~1 ms; `mask(0x3c0, 1, 0)`
3. **mem scrub**: poll `(0x10c & 6) == 0`
4. **reset_pmc**: set `PMC_ENABLE` bit 14 (PSEC); then `0x084 = PMC_BOOT_0`
5. **load**: `mask(0x624, 0x80, 0x80)`, `0x10c = 0`, PIO the image into IMEM
   (`0x180 = start | BIT(24) | (secure ? BIT(28) : 0)`, fresh tag to `0x188`
   every 256 B, instruction words to `0x184`, padded to 256 B)
6. **boot**: `0x040 = 0xcafebeef` (mailbox sentinel), `0x104 = entry`,
   `0x100 = 2`, poll `0x100 & 0x10` for STOPPED

Attempt 1 did only steps 5–6 and failed with `UC_CTRL` stuck at `0x12` (STOPPED
set, START_TRIGGER still pending): **the enable/reset half is not optional — a
falcon that has not been reset and enabled will not start.**

Side observations: mailbox0 kept its `0xcafebeef` sentinel (our program never
touches it, as expected), and STOPPED was already set on the first poll — a
four-instruction program runs and exits in well under 2 ms.

Note: the successful run left `PMC_ENABLE` bit 14 (PSEC) set, and the falcon's
IMEM now holds our 256-byte image rather than zeros. Both are deliberate and
easily reverted; IMEM cannot be restored to its original contents (unreadable),
but it was blank.

## The ladder

| Rung | Round trip | Proves | Cost |
|---|---|---|---|
| **RT-0** | 16 bytes into SEC2's DMEM and back | the PIO data path works at all | today |
| **RT-1** | our microcode into IMEM → runs → known value out | we can execute our own code on the GPU | days |
| **RT-2** | a host buffer the GPU can read/write (MMU + instance block) | shared memory between host and GPU | 1–2 weeks |
| **RT-3** | a pushbuffer the GPU fetches and retires (FIFO/QMD) | command submission | 2–4 weeks |
| **RT-4** | `C = ReLU(A@B + bias)` vs numpy | real compute on the SMs | after ACR |
| **RT-5** | a tiny model's logits vs a CPU reference | **the actual goal** | months |

RT-1 alone is a genuine "we wrote a program and the GPU ran it" claim, and it
exercises the *same* mechanics ACR needs (load microcode, set bootvec, run,
poll halt, read mailbox) — just without the signatures.

## RT-5 target, defined so "it works" is unambiguous

Escalating, each with a numpy reference implementation and tolerances:

- **Model v0 — one fused op.** `C[8,16] = ReLU(A[8,32] @ B[32,16] + bias)`,
  fp32, fixed seed. Pass = `allclose(C, ref, rtol=1e-4, atol=1e-5)`.
  Bit-exactness is *not* required at first; the f32 reduction order on SMs will
  differ from numpy's.
- **Model v1 — a small MLP.** 784→64→32→10, ReLU, ~52 k parameters, weights
  loaded from `.npz`. Pass = the 10 logits match the CPU reference.
- **Model v2 — a tiny transformer LM.** 2 layers, 4 heads, `d_model=64`,
  vocab 256, ≤1 M params, char-level. Greedy-decode 32 tokens on the GPU and
  compare the token string to a reference decode of the same weights.
  This is the first thing that genuinely deserves the word "inference": it
  exercises matmul, softmax, layernorm and attention — the same op set as a real
  LLM, just small enough to debug.
- **Round-trip properties tested at every level, not just numerics:**
  1. same inputs → same outputs across 10 consecutive runs;
  2. same outputs after closing and reopening the device (server restart,
     re-mapped BARs, fresh instance) — the property that separates a real
     driver from a lucky first run;
  3. restore-after-test: any state the test touched is put back and verified.

## Harness

`~/egpu-pascal/rt/` as a package, so this stops being a pile of probes:

```
rt/
  card.py      transport, BAR mapping, register read/write, poison decoding
  falcon.py    falcon info, reset, PIO IMEM/DMEM load+read, start/halt/mailbox
  acr.py       WPR build + ACR bootloader (RT-2/3)
  mmu.py       instance blocks, page tables, sysmem mapping (RT-2)
  fifo.py      GPFIFO channel, pushbuffer, QMD submission (RT-3)
  compute.py   kernel descriptors, dispatch, result readback (RT-4)
  ref.py       numpy reference implementations for each model
  tests/test_rt0.py … test_rt5.py
  run_all.sh
```

Rules for every test: save the register/state it will touch → run → restore →
verify the restore → assert. Failures must print the raw read so "engine
disabled (poison)", "wrong address (poison, different code)" and "real value
mismatch" are distinguishable at a glance.

## What gates what (honest ordering)

- **RT-0 → RT-1** need nothing we don't have: SEC2 is located (`0x87000`), live
  at power-on, and BAR0 writes are proven.
- **RT-2/RT-3** need the ACR (rung 2). Establishing the WPR is MMIO-driven and
  reachable now; bootstrapping FECS/GPCCS is *message-based* over SEC2's command
  queue, which needs the instance block and page tables — so MMU work moves
  before the end of the ACR phase, not after it.
- **RT-4/RT-5** additionally need SASS for `sm_61` (CUDA 12.x `ptxas`; CUDA 13
  dropped Pascal) and our own QMD v01_07 emitter.
- The tiny model runs through *our* runtime first. Bridging to llama.cpp still
  needs the cudart/cuBLAS shim layer — a separate project after RT-5, not part
  of it.

## Microcode for RT-1 — three options, in order of preference

1. **Hand-assemble a 3–5 instruction falcon program** that stores a constant to
   DMEM and exits. Needs the falcon ISA encoding (envytools' falcon docs).
   Cheapest, fully under our control, and the DMEM store is read back with the
   RT-0 path.
2. **Load the real `nvidia/gp102/acr/bl.bin`** — a ready-made falcon program
   designed to be loaded exactly this way. Its mailbox/error code is the
   round-trip signal (a `0xdeada5a5`-style sentinel untouched means it did not
   run; a different code means it did, and tells us what it wanted).
3. **Use nouveau's fuc assembler** if its ISA/assembler is available, to build a
   richer program. Most work, most capability later.

## Immediate next steps

1. **RT-0** — implement and run (in flight).
2. **Falcon facts** — read `0x12c` (version/secret/ports) and decode `0x108`
   (`code.limit`, `data.limit`) on SEC2/PMU/FECS, so we know each engine's
   addressable IMEM/DMEM before writing anything.
3. **RT-1 option 1** — get the falcon ISA and hand-assemble the smallest program
   that produces an observable effect.
4. Wire the `rt/` package and the pytest harness around what already exists.
