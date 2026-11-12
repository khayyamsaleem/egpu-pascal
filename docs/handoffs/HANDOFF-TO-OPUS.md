HANDOFF — GTX 1080 Ti (GP102) bring-up over TinyGPU on a MacBook Air M1
Rewritten 2026-11-12, end of session 10.  This file replaces the session-5
handoff entirely: the rung-4 FECS blocker it was built around has been closed
for five sessions, and rungs 1-5 have all landed since.  Nothing here is
stale-by-supersession — the session-5 file is kept only as history.

Read first, in this order (do not re-derive any of it):
  docs/handoffs/HANDOFF-SESSION9.md  previous session: the async launch ring,
                                     both runtimes working, the patch-0004 blocker
  maxpatch/README.md                 what the local MAX patch set is, per version
  RESULTS.md                         the measured evidence trail, rung by rung
  PLAN.md                            the 8-phase plan; Phase 6 targets MAX
Then the code, in this order: pascal/shimd.py + pascal/nvpascal.py (the GPU side
of the stack), shim/ (the libcuda/NVML replacement), localgpu/local_gpu.py (the
one-command bring-up + serve), localgpu/bench_chat.py (the multi-turn benchmark),
maxpatch/ (the MAX patches + installer).

GOAL (user's, standing): run LLM inference on this 1080 Ti over the Thunderbolt /
USB4 enclosure from this Mac.  Phase 6 was originally decided as "MAX instead of
llama.cpp"; in practice BOTH runtimes work and the user drives whichever fits.
The user wants persistence and "know exactly where we stopped" over optimism:
every claim in this file is measured on this card, not inferred.

=====================================================================
1. ENVIRONMENT AND HOW TO RUN ANYTHING
=====================================================================
Host: MacBook Air M1, macOS 26.6.2.  Enclosure: Razer Core X V2 on a TB3/USB4
port, PCIe Gen3 x4.  Card: GTX 1080 Ti, GP102, 10de:1b06, 28 SMs, CC 6.1.
Transport: tinygrad's TinyGPU DriverKit dext + its socket /tmp/tinygpu.sock.
python = ~/.venvs/tinygpu/bin/python (3.12), APL_REMOTE_SOCK defaulted in
pascal/card.py.

One command drives the whole stack (card, driver server, container, model,
proxy, chat):

  bin/local-gpu status                 card, containers, servers, endpoint
  bin/local-gpu serve -m <model>       bring it up (also the default command)
  bin/local-gpu chat                   serve + terminal chat
  bin/local-gpu pull <model> | models | bench <model> | logs [server|shimd|proxy]
  bin/local-gpu stop [--all]           stop servers; --all also releases the card

  NOTE: -m/--model goes AFTER the subcommand.  `bin/local-gpu -m X serve` does
  not parse (argparse puts --model on the subparser).  Handoff-session-9 had
  this wrong; it is fixed there as of session 10.

Card cold path (after any enclosure power cycle, sleep, or reboot), in order:

  ~/.venvs/tinygpu/bin/python pascal/devinit.py firmware/gp102_vbios.rom
  ~/.venvs/tinygpu/bin/python pascal/fb.py
  ~/.venvs/tinygpu/bin/python pascal/reset_engines.py
  ~/.venvs/tinygpu/bin/python pascal/gr.py          # expect: GR BRING-UP RESULT: PASS

`local-gpu serve` does that check itself; `local-gpu stop --all` is the way to
hand the card back (shimd holds the TinyGPU device lock exclusively — see 5).

Containers (all reachable with `docker exec`):
  llamacpp    llama.cpp built for sm_61 at /opt/llama-build + shim/build-cuda12
              (/models holds the GGUF files; port 8080 -> 127.0.0.1:8080)
  maxnightly  modular 26.8.0.dev2026110805, maxpatch set 0001-0004 installed,
              HF cache holds gemma-4-E2B-it bf16; `max serve` on container :8000
  maxlab      modular 26.6.0, patch set 26.6 (the older, still-working MAX path)

Ports on the Mac: 11434 = the Ollama+OpenAI proxy (what clients use), 7461 =
shimd.  llama-server (8080) and max serve (8000) live inside the containers.

Shim libraries differ per runtime and MUST NOT be mixed:
  shim/build         libcuda.so.1 + libnvidia-ml.so.1              <- MAX path
  shim/build-cuda12  those + libcudart/libcublas/libcublasLt .so.12 <- llama.cpp
MAX is always served with LD_LIBRARY_PATH=/repo/shim/build; llama.cpp with
shim/build-cuda12 (localgpu/local_gpu.py).  Both dirs are git-ignored build
outputs; rebuild with shim/build.sh and shim/build-cuda12.sh.

Server command lines actually used (from local_gpu.py, quoted because the flags
are load-bearing):
  llama-server -m /models/<file> --alias <name> -ngl 99 -fa on -c <ctx> -np 1
               -cram 2048 --jinja --host 0.0.0.0 --port 8080
  max serve --model <hf-id> --served-model-name <alias> --quantization-encoding
            <enc> --devices gpu --no-device-graph-capture --max-length <ctx>
            --max-batch-size 1 --host 0.0.0.0 --port 8000 [--no-prefer-module-v3]
  (MAX env for E-series: MAX_GEMMA4_ACTIVATION_MIB=1024, extra --no-prefer-module-v3)

=====================================================================
2. WHERE THE WORK STANDS
=====================================================================
Rung ladder (PLAN.md section 0): 0 transport+ID, 1 falcon runs our code,
2 ACR secure boot, 3 MMU + FIFO channel, 4 compute dispatch, 5 inference.

  Rung 0-4  DONE (transport, VBIOS POST, ACR secure boot, MMU v2, PFIFO/PBDMA
            channels, copy engine, GR compute dispatch of ptxas sm_61 cubins).
  Rung 5    DONE, TWO RUNTIMES.  Modular MAX (via a libcuda+NVML shim) and
            llama.cpp (via that shim plus cudart/cuBLAS shims) both generate
            correct tokens on the card over TinyGPU.

  Boot clocks only: 607 MHz SM, 405 MHz memory, ~30 GB/s measured.  That single
  fact caps every number below — no PMU reclocking has been attempted (writes to
  the card's clocks need the user's explicit consent; see 4.2).

What runs, and how fast (session-9 numbers, all greedy, batch 1):
  Gemma 4 E2B bf16 on MAX       ~200 ms/token (5 tok/s), 9.6 GB checkpoint
  SmolLM2-135M fp32 on MAX      22.7 tok/s, text identical to MAX-CPU
  SmolLM2-135M Q4_0 on llama.cpp 61 tok/s generation, 447 tok/s prompt
  Gemma 4 E4B QAT Q4_0 on llama.cpp  6.8-6.9 tok/s decode (session 10)

Models configured in localgpu/local_gpu.py MODELS: gemma4-e4b (DEFAULT, llama,
QAT Q4_0 5.15 GB, desc "fast, tools"), gemma4-12b (llama, 6.98 GB), llama3.1-8b,
qwen3-8b (marker thinking=True), smollm2 (llama, "tiny, for testing"),
gemma4-e2b-max (MAX, bf16 9.6 GB), smollm2-max (MAX, fp32).
Only e4b, smollm2 and e2b have ever been served; 12b / llama3.1-8b / qwen3-8b
have no weights on disk yet.

=====================================================================
3. WHAT SESSION 10 DID (2026-11-12, evening)
=====================================================================
The multi-turn benchmark existed since session 9 (commit 56cdb71) and had NEVER
been run — logs/bench/ did not exist.  Session 10 ran it on three models, which
is where the numbers in section 3.1 come from, and fixed two bugs found doing it.

3.1 The bench, three models, same 12-turn script (logs/bench/, do not re-run —
    each run costs 40 s / 5 min / 17 min of card time)

  model        runtime   wall    decode     prefill / cache      TTFT med/max  tool
  SmolLM2-135M llama.cpp 39.5 s  47.1 t/s   1623 t/s, 31k cached  0.05/0.9 s    NOT made
  E4B QAT Q4_0 llama.cpp 302 s   6.88 t/s   270 t/s, 21.5k cached  1.15/6.95 s   ok
  E2B bf16     MAX       1025 s  4.12 t/s   49 t/s, 0 cached(art)  11.0/390 s    ok
  (E4B was run twice, 301.7 s / 6.88 and 304.3 s / 6.83 t/s — repeatable.)

  * Tool calling works end to end on a model that can do it: E4B and E2B both
    emitted get_weather{"city":"Tokyo"} + {"city":"Paris"} and then used the
    tool results verbatim (14 C light rain / 9 C overcast; the harness's fake
    weather).  SmolLM2-135M cannot: it hallucinated an answer by recycling the
    RESULTS.md document it had been fed earlier in the conversation, and turn 7
    was skipped because no tool call was pending.  tool_call_ok is a property of
    the model, not of the stack.
  * MAX returns no llama.cpp `timings` block, so its `cached` is always 0 and its
    "prefill" is prompt/ttft — a derived figure, not a measurement.  Do not
    compare those two columns across runtimes.  Its two TTFT spikes (216 s on
    long-doc, 390 s on the first tool-call request) are graph compiles at new
    input shapes, not prompt processing: that turn generated 29 tokens after a
    390 s first token, where llama.cpp did the same request in 7 s.
  * E4B has a thinking channel and it eats the budget: with --max-tokens 200,
    6 of 12 turns produced NO visible answer — the model planned in the
    reasoning channel and was cut off before answering (finish=length).  Turn 1's
    reasoning opens "Here's a plan to answer the request: 1. Analyze the
    request... 2. Constraint Checklist: Three short paragraphs... Under 200
    words."  The 5 turns that did answer (code, tool result, translate, haiku,
    JSON) were competent.  Two knobs before any "E4B quality" claim: raise
    --max-tokens, or disable the thought channel with
    chat_template_kwargs {"enable_thinking": false} in the /v1 body.

3.2 Fixes made in session 10 (see the uncommitted list at the end)

  * localgpu/local_gpu.py — the MAX alias bug.  The proxy rewrites every
    request's "model" to the LOCAL alias, but `max serve` only accepts the HF
    repo id, so every MAX request through the proxy died with
    `400 Unknown model 'gemma4-e2b-max', currently serving ['google/gemma-4-E2B-it']`.
    Fixed by passing `--served-model-name <alias>`; verified HTTP 200 through
    127.0.0.1:11434 before re-running.  This is why the default MAX path had
    never been exercised end to end — its first real request failed.
  * localgpu/bench_chat.py — recorded only `content`, so a thinking-channel
    model looked like it produced nothing.  Now stores `reasoning` per turn,
    prints `[think: N chars, no content]` instead of a blank line, counts
    `thinking_turns` (tool-call turns excluded), and warns in the report when the
    backend supplied no timings block.
  * docs/handoffs/HANDOFF-SESSION9.md — the -m argument-order typo in its resume
    block.

3.3 Proof that the tokens are computed on the CARD (the user asked, twice)

  Four independent checks, all reproduced in session 10:
  1. The server's own flags: llama-server runs `-ngl 99` — all layers on the
     GPU device, which is our shim reporting the 1080 Ti (CC 6.1, 28 SMs, 10 GiB).
  2. What the shim pushed: the container's SHIM_LOG shows ggml's real Q4_0
     kernels arriving as `cuModuleLoadDataEx(cubin, N bytes) -> status 0` and
     buffers as `cuMemAlloc(n) -> 0x119...` (device VAs from our own MMU).
  3. A live libcudart/cuBLAS conformance test through the RUNNING shimd:
     shim/cudart/test_runtime.cu built with nvcc and run with
     LD_LIBRARY_PATH=/repo/shim/build-cuda12 SHIM_MODE=remote -> kernels with
     dynamic shared memory + __syncthreads, cudaMemcpy2D, memset, and four
     cuBLAS GEMMs against host references, all "ok", RESULT: PASS.
  4. The decisive one (probes/freeze_card_test.py): SIGSTOP the driver server
     (`~/.local/state/local-gpu/shimd.pid`) mid-generation.  The token stream
     stopped dead at that instant — 29 tokens before the freeze, ZERO for the
     next three minutes — and SIGCONT brought the server straight back (fresh
     request: 54 tok/s, healthy).  A CPU-side token would not have noticed.
     The shim has no socket timeouts (checked), so the freeze is safe; the
     in-flight client's stream does not resume, but the server is unharmed.

  Why the fan cannot be heard (do not chase this): the stack runs the card's own
  PMU — pascal/devinit.py ports nouveau's gm200_devinit_post — so the card's
  onboard thermal/fan policy is in charge, and we never program a fan curve.
  Nothing in this tree reads temperature or RPM (pascal/dma_probe2.py only
  touches the THERM interrupt bit), and at 607/405 MHz boot clocks with a small
  model in sub-second bursts the board is drawing a few tens of watts, below
  where any 1080 Ti ramps its fan.  Silence proves nothing either way.

=====================================================================
4. OPEN TASKS, RANKED
=====================================================================
1. **Finish MAX patch 0004 (int8 W8A16 weights).**  The `w8_linear` custom op
   (maxpatch/kernels/w8) is written and benchmarked standalone
   (kernels/w8_check.py: 1.2-2.8x decode, 5.5-11x prefill over MAX's bf16
   ops.matmul), and MAX_GEMMA4_INT8=1 makes the gemma4 config declare INT8_W8A8
   with load-time RTN quantization — but a full model compile still fails with
   `error: 'mo.custom' op MOToMAP: no kernel registered for 'w8_linear'`.  Two
   registration paths were tried; the trace was at pipelines/lib/graph_specs.py
   (GraphSpec.custom_extensions) when session 9 ended.  Read how the GraphSpec
   built in architectures/gemma4/model.py is compiled and where that field is
   consumed.  Session 10 saw MAX's shim log load `mogg_foreach_r1_w8_b256_*`
   modules during an E2B bring-up, so the package does reach some compile path —
   worth comparing against what the failing pipeline compile does.
   This is what would let Gemma 4 E4B (9.4 GB of device weights, over the card's
   ~8.8 GiB pool) run on MAX at all.
2. **Memory clock** — the single biggest performance lever.  Still boot clocks
   (405 MHz memory / ~30 GB/s); every decode number in this file is capped by it
   (E4B at 5.15 GB/token cannot exceed ~6 tok/s).  The VBIOS BIT 'P' performance
   table is parsed by pascal/vbios_perf.py (per-domain entries, memory = domain
   2, RAMMAP bands); the open question is the pointer arithmetic for the
   per-strap subentries.  **Register writes to the card's clocks require the
   user's explicit consent — ask first.**
3. **Re-run the llama.cpp op suite** after commit c4f50f5 (which added
   cublasStrsmBatched for ggml's SOLVE_TRI).  Session 9's run passed 13,380 cases
   then aborted with exit 134 (the shim's generated stubs log and abort on an
   unimplemented entry point); the rebuild is verified, the re-run is not.
   `test-backend-ops -o MUL_MAT` is the quick pass; a failure names the missing
   entry point.
4. **Bench knobs before judging model quality** (see 3.1): raise --max-tokens and
   / or disable the thought channel, so a thinking model's plan and its answer
   both fit.  Optionally add the same switch to bench_chat.py's body.
5. **Die temperature.**  pascal/therm.py reads BAR0 0x020460 per
   nouveau's gp100_temp_get (valid bit 0x20000000, (reg & 0x1fff8) >> 8).  It
   cannot run while shimd owns the card: the TinyGPU transport takes an exclusive
   lock (nv_usb4.lock) and a second opener dies with "Failed to acquire lock
   file".  Either `local-gpu stop --all` for the ~5 s the read takes, or add an
   op 11 TEMP to shimd's protocol (its docstring documents ops 1-10; the handler
   is three lines and it already owns a Card handle) so it is readable any time.
6. Reference numbers to beat: E4B 6.8-6.9 tok/s and SmolLM2 61 tok/s on
   llama.cpp; E2B 200 ms/token on MAX (both at boot clocks).

=====================================================================
5. STANDING CORRECTIONS AND GOTCHAS (do not re-learn these)
=====================================================================
* `bin/local-gpu -m X serve` does not parse — `bin/local-gpu serve -m X`.
* MAX needs `--served-model-name <local alias>` or every proxied request 400s
  (fixed in local_gpu.py in session 10; do not remove it).
* Never put shim/build-cuda12 on MAX's LD_LIBRARY_PATH: MAX ships its own cuBLAS
  bindings (_cublas.mojoc) and a vendor-BLAS fallback in linalg, which MAX's
  kernel-compile gate (maxpatch 0001) disables for compute < 7.0 anyway. MAX is
  driven only through libcuda + libnvidia-ml (it dlsyms ~118 cu* + 16 nvml*).
* MAX TTFT spikes are graph compiles at new input shapes, not prefill. MAX also
  reports no per-request timings, so cached/prefill columns are derived.
* E4B (QAT Q4_0) plans in a thinking channel before answering; a 200-token budget
  is often spent entirely on the plan. Its channel is `reasoning_content`; the
  GGUF carries the `<|channel>` markers.
* shimd holds the TinyGPU device exclusively: only one client-side process family
  at a time, and no second process may open the card while it runs.
* SHIM_QUIET=1 logs module loads, allocations, stubs and errors — NOT launches.
  Per-kernel GPU time is `NVPASCAL_PROF=1` on the *server* (printed on client
  disconnect) and `SHIMD_TRACE=1` logs copies and device pointers. Set those on
  shimd, which `local-gpu` starts for you.
* Recreating a container does not pick up edits to bind-mounted files; the repo
  is mounted at /repo in every container, so edits are live, but a *process*
  started before the edit keeps the old code.
* tinygrad's own NV runtime must never be pointed at this card (NV=1): it issues
  a PCIe FLR that Pascal misreads (tinygrad PR #19323 carries the write-up).
* Sleep takes the Thunderbolt link down and the next MMIO is a kernel panic
  (`LLC Bus error`). pascal/sleepguard.py holds a caffeinate assertion in every
  card-holding process and refuses MMIO after a sleep; keep the lid open.
* Ground rule from all sessions: every claim is a round trip with a negative
  control, and print the raw value the judgement came from.

=====================================================================
6. FILE MAP
=====================================================================
  pascal/shimd.py        the macOS-side driver server: the CUDA driver ABI over
                         nvpascal, TCP protocol ops 1-10 documented in its
                         docstring; PTX is JIT'd with ptxas -arch=sm_61
  pascal/nvpascal.py     the runtime: GR channel, QMD/constant-bank upload from
                         the pushbuffer (the async launch ring, commit 611fe85)
  pascal/card.py         BAR0/PRAMIN/BAR1 access: rd/wr/vram_read/vram_write
  pascal/therm.py        die temperature read (0x020460) — needs the card free
  pascal/devinit.py      VBIOS POST (PMU DEVINIT app), fb.py VRAM/VPR unlock
  pascal/gr.py, fifo.py, grchan.py, mmu.py, acr.py, grctx.py, faults.py,
  reset_engines.py, recover_fifo.py, bench.py, test_modules.py, test_async.py
  shim/                  libcuda.so.1 + libnvidia-ml.so.1 (build/) and the
                         cudart/cublas layers (build-cuda12/); README lists the
                         env knobs (SHIM_MODE=remote, SHIM_LOG, SHIM_CAPTURE_DIR)
  shim/tests/            vadd.mojo, shfl.mojo, graph_add.py, ops_check.py,
                         exp_check.py, generate.sh (the MAX/llama end-to-end)
  shim/cudart/test_runtime.cu  the libcudart+cuBLAS conformance test
  localgpu/local_gpu.py  card/stack bring-up, model table, serve/chat/status,
                         the Ollama+OpenAI proxy launcher (ollama_proxy.py)
  localgpu/bench_chat.py the 12-turn multi-turn benchmark (TTFT, prefill/decode,
                         prompt-cache reuse, tool call, thinking channel)
  maxpatch/              patches/<modular-version>/ + install.sh (+ --revert),
                         kernels/w8 (the int8 custom op), patch_target.py
  kernels/               PTX sources + sm_61 cubins + the ptxas Docker build
  probes/freeze_card_test.py   proves the token stream is gated by the driver
                         server (SIGSTOP the server, watch the stream stop)
  logs/bench/            the four session-10 benchmark reports (.json + .md)
  docs/                  evidence trail, per-session handoffs, docs/upstream/
                         (the three draft Modular PRs + what stays local)

Card state as handed over: warm and serving gemma4-e4b on llama.cpp (context
16384, all layers on the card, 6.8 GiB of 10.0 GiB free).  shimd is up
(--vram-limit-gib 10).  SM 607 MHz, memory 405 MHz.  `bin/local-gpu stop --all`
hands the card back and clears the lock that pascal/therm.py needs.

Git state as handed over: HEAD 1f94923 ("Modular patches are open as draft PRs"),
17 commits.  Uncommitted when this file was written: localgpu/local_gpu.py and
localgpu/bench_chat.py (the two session-10 fixes), docs/handoffs/
HANDOFF-SESSION9.md (the -m typo), new pascal/therm.py and
probes/freeze_card_test.py, and logs/bench/ (the reports, untracked).
