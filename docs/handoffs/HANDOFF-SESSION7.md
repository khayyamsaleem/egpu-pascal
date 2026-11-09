# Handoff: session 7 (2026-11-09)

## Where we stopped

**Rung 5 is reached: MAX runs an LLM on the GTX 1080 Ti.**
`max generate --model HuggingFaceTB/SmolLM2-135M-Instruct --quantization-encoding float32
--devices gpu --top-k 1` produces text identical to MAX-on-CPU over 64 greedy tokens,
at 6.6 tok/s decode with a 0.30 s time to first token. Details are in RESULTS.md §9
"Session 7" and blockers 12–20 in §4.

State at the end of the session:

* `maxlab` container (python:3.12, `modular==26.6.0`, `/repo` = this repo) has
  `maxpatch/install.sh` applied: std PTX 6.3 target + linalg pre-Volta matmul.
  `std.mojoc.orig` and `linalg.mojoc.orig` are kept next to the patched ones.
* `upstream/modular` is a clone at tag `max/v26.6.0` with patch 0001 applied
  (git-ignored; `install.sh` resets and re-applies it).
* The card is warm. Start the server with `reset_engines.py`, then `gr.py`, then
  `shimd.py --vram-limit-gib 4`.

## Resume

```sh
PY=~/.venvs/tinygpu/bin/python
$PY pascal/reset_engines.py && $PY pascal/gr.py && $PY pascal/test_modules.py   # RESULT: PASS
$PY pascal/shimd.py --vram-limit-gib 4 &
docker exec maxlab bash /repo/shim/tests/generate.sh gpu "The capital of France is"
```

After any SM trap (`SM esr` in the shimd log), GR is wedged. Stop shimd, then run
`reset_engines.py` + `gr.py` again (no power cycle needed).

## How the blockers were found (method worth reusing)

1. LLVM "Cannot select" means an arch-gated path; read the dispatch in
   `upstream/modular/max/kernels/src`.
2. A launch failure: read the shimd log's `SM esr` low byte (nouveau `gf100_mp_warp_error`:
   0x9 illegal encoding, 0xb illegal param, 0xe out-of-range address) and the PC
   (`0x504654`, program-region offset). Then disassemble the dumped cubin (`SHIMD_DUMP=dir`)
   with `envydis -m gm107`.
3. Wrong numbers: `shim/tests/ops_check.py` / `exp_check.py` bisect by op. Then capture
   the PTX with `shim/tools/ptxas-capture.sh` and inspect the cubin sections.

## Next (RESULTS.md §7)

1. Launch overhead: ~80% of decode time. Make launches asynchronous (sync only on
   copies/stream sync) and the LAUNCH RPC fire-and-forget.
2. Llama-3.2-1B fp32 (head depth 64, so naive MHA). For depth-128 models, gate FA2 in
   `nn/attention/gpu/mha.mojo` on `compute >= 8.0`.
3. `tanh` (`tanh.approx.f32` is sm_75+) for GELU-tanh models.
4. Tiled SIMT matmul for prefill; then the memory reclock (needs consent for writes).
