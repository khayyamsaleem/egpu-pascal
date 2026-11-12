# Handoff: session 8 (2026-11-12)

## Where we stopped

**Gemma 4 E2B runs correctly on the GTX 1080 Ti through MAX** (nightly
`26.8.0.dev2026110805`, bf16, PLE table in host memory): "The capital of France is **Paris**."
(identical to transformers), deterministic, 530 ms/token. SmolLM2-135M still matches MAX-CPU
on both 26.6 and the nightly. Details: RESULTS.md §9 "Session 8", blockers 21–23 in §4.

The two Mac reboots were **kernel panics on wake from sleep while the driver held the card**
(lid close; 1-minute idle sleep). `pascal/sleepguard.py` now holds `caffeinate` for every
card-holding process and refuses MMIO after a sleep. **Keep the lid open during runs.** After
any sleep or reboot the card is cold: devinit + fb + reset_engines + gr.

## State

* Containers: `maxlab` (modular 26.6.0, maxpatch 26.6 applied) and `maxnightly`
  (26.8.0.dev2026110805, maxpatch nightly set applied, CPU torch + pytest installed for the
  upstream Gemma tests in `/tmp/g4t`).
* Source checkouts (git-ignored): `upstream/modular` (tag max/v26.6.0) and
  `upstream/modular-main` (sparse, at 2841b31631e7, with the nightly patches applied).
* E2B weights are in the `maxnightly` HF cache (9.6 GB).

## Resume

```sh
PY=~/.venvs/tinygpu/bin/python
$PY pascal/devinit.py firmware/gp102_vbios.rom && $PY pascal/fb.py     # only if cold
$PY pascal/reset_engines.py && $PY pascal/gr.py && $PY pascal/test_modules.py
$PY pascal/shimd.py --vram-limit-gib 9.5 &                              # SHIMD_TRACE=1 logs copies + kernel pointers
docker exec maxnightly bash -lc 'MODEL=google/gemma-4-E2B-it ENCODING=bfloat16 MAX_GEMMA4_ACTIVATION_MIB=1024 \
    bash /repo/shim/tests/generate.sh gpu "The capital of France is" --no-prefer-module-v3'
```

## Debugging method that worked (reuse it)

1. A reference that runs the same model elsewhere: transformers on the Mac CPU (bf16), or numpy
   from the memory-mapped checkpoint for a sub-block (the per-layer inputs).
2. Env-gated `ops.print` of per-layer hidden states (last token, first 8 values + RMS) in MAX;
   compare layer by layer to find the first layer that diverges.
3. Run twice: identical-config runs that differ point at our runtime (races, uninitialized or
   misplaced memory), not at MAX's math.
4. `SHIMD_TRACE=1`: every HtoD/DtoD destination and every device pointer a kernel receives.
   Follow a buffer's address through the trace to see who writes it next.
5. Prints change MAX's fusion; if a bug disappears under prints, suspect memory layout/limits.

## Next

1. Launch overhead (decode ~700 launches/token for E2B): asynchronous launches, fire-and-forget
   launch RPCs.
2. FA2 gating for head depth 128 (Llama-3-8B, Qwen) — see maxpatch/README "Not patched yet".
3. Gemma 4 E4B needs quantized device weights (MAX gemma4 loads bf16/fp16/NVFP4 only).
4. Memory reclock (writes need consent).
