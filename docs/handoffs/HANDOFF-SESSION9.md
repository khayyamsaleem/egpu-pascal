# Handoff: session 9 (2026-11-12)

## Where we stopped

Session 9 turned rung 5 into **two working runtimes** and made both of them fast. It ended at
Claude's 5-hour usage limit, mid-task, with everything on disk committed in
`611fe85`, `9ebe6aa`, `56cdb71`, `c4f50f5`, `f45e3dc` — the last two were written in the final
minutes and are the loose ends described below.

**MAX (rung 5).** Launches are now asynchronous, through one GPU-mapped host ring
(`pascal/nvpascal.py`, commit `611fe85`): the QMD and constant bank 0 are uploaded inline from
the pushbuffer, so a launch costs one doorbell write (~12 µs end to end, was 120–170 µs).

* Gemma 4 E2B bf16: 529 → **200 ms/token**, same text, and the model loads in seconds instead
  of ~80 s.
* SmolLM2-135M fp32: 6.6 → **22.7 tok/s**, text identical to MAX-CPU.
* Gates that pass: `pascal/test_async.py`, `pascal/test_modules.py`, `shim/tests/copy_check.py`,
  `shim/tests/ops_check.py --dtype bfloat16` 12/12.

**llama.cpp (new).** The CUDA backend reaches the card through new `libcudart.so.12` and
`libcublas.so.12` shims over our `libcuda.so.1` (`shim/cudart`, `shim/cublas`, `shim/build-cuda12.sh`;
commit `9ebe6aa`).

* SmolLM2-135M Q4_0: **61 tok/s** generation, 447 tok/s prompt; greedy text matches CPU.
* `test-backend-ops`: all 2,682 MUL_MAT cases match the CPU backend, and 10,000+ cases of the
  full suite passed with 0 failures — then the suite **aborted (exit 134) at 13,380 cases**
  (see below).
* `local-gpu` (`bin/local-gpu`, `localgpu/`) brings the whole thing up in one command: card
  init if cold, `shimd`, the right container, a resumable model download, then either
  llama-server with all layers on the GPU or `max serve`, plus an Ollama-compatible endpoint on
  `127.0.0.1:11434` that also passes OpenAI `/v1` through, a terminal chat and a
  `localgpu/bench_chat.py` multi-turn benchmark. Default model: `gemma4-e4b`.

The session reached this state without rebooting the Mac; the sleep guard from session 8 is
still the thing keeping the panics away (keep the lid open during runs).

## The open task: MAX_GEMMA4_INT8 never finds its kernel

The last user ask was model performance, including quantized weights for Gemma 4 E4B. Patch
`0004` (commit `f45e3dc`) adds the int8 weight-only path: `_matmul_int8` on a non-Apple GPU
routes to the `w8_linear` custom op (`maxpatch/kernels/w8`, benchmarked standalone by
`maxpatch/kernels/w8_check.py`: 1.2–2.8× decode, 5.5–11× prefill over MAX's own bf16
`ops.matmul` at 2–3e-3 relative error), and `MAX_GEMMA4_INT8=1` gives the model an INT8_W8A8
`QuantConfig` plus RTN-quantized weights at load.

**A full model compile still fails:**

```
error: 'mo.custom' op MOToMAP: no kernel registered for 'w8_linear'
RuntimeError: Failed to compile the model
```

Two registration paths were tried and neither reached the pipeline's grouped compile:
`Gemma4ForConditionalGenerationConfig` appends the package to
`max.graph.graph._default_custom_extensions`, and `_w8_linear_gpu` calls
`Graph.current._import_kernels([pkg])` before the custom op. The trace was at
`pipelines/lib/graph_specs.py:130` — `GraphSpec` carries a
`custom_extensions: Sequence[Path] = ()` field — when the session ended, so the next thing to
read is how the `GraphSpec(...)` built in `architectures/gemma4/model.py` is compiled and
where that field is consumed.

## The exit-134 abort in the op suite

`test-backend-ops` passed 13,380 cases and then aborted. The shim's generated stubs log and
abort on an unimplemented entry point; the only cuBLAS entry point added between that abort and
the end of the session is `cublasStrsmBatched` (ggml's SOLVE_TRI, commit `c4f50f5`). That is a
plausible cause, not a confirmed one — the suite has not been re-run since, and the library
rebuild itself is the only thing verified (`shim/build-cuda12/libcublas.so.12` exports
`cublasStrsmBatched`). Re-run the op suite before trusting either.

## State

* Containers: `maxlab` (modular 26.6.0, patch set 26.6), `maxnightly` (26.8.0.dev2026110805,
  patch set 0001–0004 applied), and the llama.cpp container `llama/build.sh` +
  `shim/build-cuda12.sh` build in. The shim libraries differ per runtime: MAX is served with
  `shim/build` (see `localgpu/local_gpu.py`), llama.cpp with `shim/build-cuda12` — both are
  git-ignored build outputs.
* Source checkouts (git-ignored): `upstream/modular` (tag `max/v26.6.0`),
  `upstream/modular-main` (sparse, at `2841b31631e7`) and `upstream/llama.cpp`
  (`2a4daf0`, built for sm_61 into `/opt/llama-build` in the container). Patch 0004 also
  touches `max/python/max/nn/_graph_ops/quant.py`, which is **not** in the sparse-checkout set
  yet.
* Weights: Gemma 4 E2B bf16 in the `maxnightly` HF cache (9.6 GB), SmolLM2 Q4_0 GGUF.
  Gemma 4 E4B QAT Q4_0 (`local-gpu`'s default model) died twice on a flaky link during the
  session and finished afterwards at 20:22 UTC — `/models/gemma-4-E4B_q4_0-it.gguf` in the
  llama.cpp container is complete at 5,154,941,280 bytes, but nothing has served a token from
  it yet: the default path of `local-gpu` is still unexercised end to end.
* The card holds its state across processes but not across sleep/reboot: after either, it is
  cold (devinit + fb + reset_engines + gr). `local-gpu serve` does that check itself.

## Resume

```sh
bin/local-gpu status                    # card, containers, servers
bin/local-gpu chat                      # serve + terminal chat on the default model
bin/local-gpu -m gemma4-e2b-max serve   # MAX path (compiles kernels on first start)
docker exec maxnightly bash -lc 'MODEL=google/gemma-4-E2B-it ENCODING=bfloat16 \
    MAX_GEMMA4_ACTIVATION_MIB=1024 MAX_GEMMA4_INT8=1 \
    bash /repo/shim/tests/generate.sh gpu "The capital of France is" --no-prefer-module-v3'
```

The last command is the reproduction of the open blocker. Add `MAX_GEMMA4_INT8=1` to the `env`
of a `runtime="max"` entry in `localgpu/local_gpu.py`'s `MODELS` to drive it through the helper.

## What to reuse

* `NVPASCAL_PROF=1` prints GPU time per kernel when a client disconnects — the fastest way to
  attribute wall time between the driver and the model.
* `SHIMD_TRACE=1` logs every copy and every device pointer a kernel receives (session 8's
  technique; still the way to find who writes a buffer).
* `test-backend-ops` (`-o MUL_MAT` for a quick pass) is the conformance gate for the whole
  CUDA-runtime surface; a failure there names the entry point the shim is missing.
* `localgpu/bench_chat.py` measures TTFT, prefill/decode tok/s and prompt-cache reuse over a
  12-turn conversation, so performance claims come with a repeatable shape.

## Next

1. **Finish patch 0004** (above): get `w8_linear` into the pipeline's compile, then run Gemma 4
   E4B through MAX with int8 weights — that is what unblocks the larger models on this card.
2. **Memory clock.** Still boot clocks: 607 MHz SM, 405 MHz memory (~30 GB/s measured). The next
   step was raising the memory clock from the VBIOS BIT 'P' performance table; `pascal/vbios_perf.py`
   parses it (per-domain entries, memory = domain 2, RAMMAP bands). The open question when the
   session moved on was the pointer arithmetic for the per-strap subentries. **Writes to the
   card's clocks need the user's consent.**
3. **Re-run the llama.cpp op suite** after `c4f50f5` and read the abort if it recurs.
4. Reference numbers to beat: 200 ms/token for Gemma 4 E2B bf16 on MAX, 61 tok/s for
   SmolLM2-135M Q4_0 on llama.cpp.
