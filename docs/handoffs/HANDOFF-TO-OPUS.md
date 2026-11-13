HANDOFF — GTX 1080 Ti (GP102) bring-up over TinyGPU on a MacBook Air M1

Written at the end of session 15 (2026-11-13) for whoever picks this up next — human or model.  Every
claim here is measured on this card unless it says otherwise: the project's standing rule is that a
claim is a round trip with a negative control, and that the raw value the judgement came from gets
printed.  RESULTS.md is the full evidence trail, section by section (it now runs to section 15); the
per-session handoffs in docs/handoffs/ are history.  Sections 1, 5 and 6 (environment, gotchas, file
map) are the parts that stay true across sessions — read them before running anything.

=====================================================================
0. READ THIS FIRST
=====================================================================

THE STACK WORKS.  Rung 5 (LLM inference on the card) has been done for several sessions, in two
runtimes: llama.cpp (the one that serves) and Modular MAX (still wired up).  One command drives
everything, the card runs above its boot clocks, and the chat harness has working tool calling.
Today's numbers:

  Gemma 4 E4B QAT Q4_0 on llama.cpp, 12-turn bench, memory 1080 MHz (measured 537 f_CK), SM 607 MHz:
    162 s / 13.5 tok/s decode / 354 tok/s prefill    n-gram speculation OFF (the comparable shape)
     67 s / 42.8 tok/s overall                       speculation ON (what `serve` does by default)
  Memory read bandwidth 87.4 GB/s (was 31.1 at the 405 MHz boot clock).

WHAT SESSION 15 SETTLED — three levers, each closed by measurement so nobody re-opens them on
intuition:

  1. MEMORY CLOCK: now measurable, and no longer the constraint.  `pascal/mclk_measure.py` reads
     nvgpu's trim counters (the DRAM one is `CTRL_CLK_DOMAIN_MCLK`, source `dramdiv4_rec_clk1`,
     scale 4) and measures the real clock rather than trusting a PLL read-back: 1080 -> 537.1 f_CK,
     1134 -> 564.0, and mode 2 asked for 1350 reaches 593.3 — so the mempll at 0x132000 DOES steer
     the memory clock, and the band-2 wall is the timing/training side, not a missing PLL.  1134 buys
     87.3 -> 92.0 GB/s of bandwidth and ZERO end to end (162 s vs 159 s, same args), because decode
     already moves 69.5 GB/s of an achievable 92 = 75% of the ceiling: the card is launch- and
     SM-bound.  Band 2 (1350 -> 118 GB/s) would buy nothing until that changes.  Default stays
     1080; 1134 is opt-in (`--mclk 1134` / LOCAL_GPU_MCLK) because it also flips rare bits warm.
     RESULTS.md section 13.

  2. LAUNCH PATH (drop the per-launch WAIT_FOR_IDLE for ~5-7%): CLOSED — no.  A QMD RELEASE0
     semaphore aimed at host memory never signals (the model load hangs waiting for completions that
     never arrive); no-WFI with a pushbuffer semaphore corrupts results; `DEPENDENT_QMD_SCHEDULE_ENABLE`
     + `TYPE=GRID` deadlocks the channel at the second launch; `QMD_GROUP_ID` does nothing.  WAIT_FOR_IDLE
     is load-bearing.  RESULTS.md section 14; harness `pascal/test_depchain.py`.

  3. SM CLOCK (607 MHz; the biggest untouched lever at ~1.3x): NOT REACHABLE BY HOST REGISTERS.  The
     PLL-shaped block at 0x137000-0x137100 accepts every write and moves no clock (12-candidate
     sweep), and the boot clock comes from the VBIOS's PMU app — nvgpu, the only open reference,
     never writes FLL registers from the host either.  The SM clock is PMU-firmware territory.
     RESULTS.md section 15; tools `pascal/smclk.py` + `pascal/smclk_sweep.py`.

THE ONE REMAINING PERFORMANCE AVENUE is therefore the **PMU board-object path** (task 4A): parse the
VBIOS FLL/VIN/volt tables into PMU board objects, load them, then request a VFE (voltage/frequency)
point — what nvgpu does in clk_fll.c / clk_vin.c / volt_dev.c against ctrl/ctrlclk.h and
ctrl/ctrlvolt.h.  It is a project, not a session, and a bad frequency/voltage request can wedge the
card, whose only recovery is a physical power cycle of the enclosure (the user's hands).  Scope it,
write down what breaks if it is wrong, and get the user's explicit OK before writing anything.

WORKING RULES THAT HAVE PAID OFF (keep them):
  * Rollback first for anything touching clocks or voltage: snapshot the registers, probe writability
    with SAME-VALUE writes, verify by read-back, auto-restore on mismatch.  `pascal/smclk.py
    --snapshot/--diff/--restore` exists for this and has been exercised for real (restore, diff = 0,
    clock unchanged).  Escalation if a write wedges the card: stop -> restore -> reset_engines + gr
    -> serve -> re-POST (devinit + fb) -> power cycle.
  * Measure the clock, not the register.  A read-back only proves a write landed.  The trim counters
    are the judge (`smclk.py --verify`), and for memory it is `pascal/memtest.py`.
  * A one-count jitter is not a clock change: the counters move 0.01-0.08% run to run, so compare
    medians of ~3 reads against a threshold (0.5% was used).  A single-read equality test fires on
    noise and "keeps" a no-op write — that bug cost the first sweep run.
  * Performance numbers are only comparable within one argument set: `-bs --spec-type ngram-mod`
    moves the same bench from 162 s to 67 s.  Pin LOCAL_GPU_PLAIN=1 for any A/B.
  * Writes to voltage or the SM clock need the user's explicit OK.  They gave it once, for the
    register investigation above; that is not standing consent for the PMU work.
  * After a FAILED memory reclock, never run GPU work (corrupted page tables / GPFIFO made the card
    reset itself: PMC defaults, FB reads badf3000, no PCIe re-enumeration).  Recover with the cold
    path, then reclock.

=====================================================================
1. ENVIRONMENT, AND HOW TO RUN ANYTHING
=====================================================================
Host: MacBook Air M1, macOS 26.6.2.  Enclosure: Razer Core X V2 on a TB3/USB4 port, PCIe Gen3 x4.
Card: GTX 1080 Ti, GP102, 10de:1b06, 28 SMs, CC 6.1.  Transport: tinygrad's TinyGPU DriverKit dext
plus its socket /tmp/tinygpu.sock (APL_REMOTE_SOCK defaulted in pascal/card.py).
python = `~/.venvs/tinygpu/bin/python` (3.12).

  bin/local-gpu status                  card, containers, servers, endpoint
  bin/local-gpu serve [-m <model>]      bring it up (also the default subcommand)
                                             --mclk N   reclock to N (default 1080, env LOCAL_GPU_MCLK)
  bin/local-gpu chat [--tools]          serve + terminal chat; --tools turns on the tool harness
  bin/local-gpu sandbox [--status|--guest|--build|--seed P|--reset|--shell|--run CODE]
  bin/local-gpu pull <model> | models | bench <model> | logs [server|shimd|proxy]
  bin/local-gpu stop [--all]            stop servers; --all also releases the card

  NOTE: `-m/--model` goes AFTER the subcommand.  `bin/local-gpu -m X serve` does not parse.

Card cold path (after any enclosure power cycle, sleep or reboot), in order:
  ~/.venvs/tinygpu/bin/python pascal/devinit.py firmware/gp102_vbios.rom
  ~/.venvs/tinygpu/bin/python pascal/fb.py
  ~/.venvs/tinygpu/bin/python pascal/reset_engines.py
  ~/.venvs/tinygpu/bin/python pascal/gr.py          # expect: GR BRING-UP RESULT: PASS
`local-gpu serve` does that check itself.  `local-gpu stop --all` hands the card back — shimd holds
the TinyGPU device lock exclusively, and a second opener dies with "Failed to acquire lock file".

Containers (all reachable with `docker exec`):
  llamacpp    llama.cpp built for sm_61 at /opt/llama-build + shim/build-cuda12 (/models = GGUFs,
              port 8080)
  maxnightly  modular 26.8.0.dev2026110805, maxpatch set 0001-0004 installed, HF cache holds
              gemma-4-E2B-it bf16 (`max serve` on container :8000)
  maxlab      modular 26.6.0, patch set 26.6 (the older, still-working MAX path)

Ports on the Mac: 11434 = the Ollama+OpenAI proxy (what clients use), 7461 = shimd.  llama-server
(8080) and max serve (8000) live inside the containers.

Shim libraries differ per runtime and MUST NOT be mixed:
  shim/build         libcuda.so.1 + libnvidia-ml.so.1               <- MAX path
  shim/build-cuda12  those + libcudart/libcublas/libcublasLt .so.12 <- llama.cpp
MAX is always served with LD_LIBRARY_PATH=/repo/shim/build, llama.cpp with shim/build-cuda12 (in
localgpu/local_gpu.py).  Both are git-ignored build outputs; rebuild with shim/build.sh and
shim/build-cuda12.sh.

Server command lines actually used (flags are load-bearing):
  llama-server -m /models/<file> --alias <name> -ngl 99 -fa on -c <ctx> -np 1 -cram 2048 --jinja
               --host 0.0.0.0 --port 8080 -bs --spec-type ngram-mod
               -ot per_layer_token_embd.weight=CUDA0,token_embd.weight=CUDA0
  max serve --model <hf-id> --served-model-name <alias> --quantization-encoding <enc> --devices gpu
            --no-device-graph-capture --max-length <ctx> --max-batch-size 1 --host 0.0.0.0 --port 8000
            [--no-prefer-module-v3]        (E-series env: MAX_GEMMA4_ACTIVATION_MIB=1024)

=====================================================================
2. WHERE THE WORK STANDS
=====================================================================
Rung ladder (PLAN.md section 0): 0 transport+ID, 1 falcon runs our code, 2 ACR secure boot, 3 MMU +
FIFO channel, 4 compute dispatch, 5 inference.

  Rungs 0-4  DONE (transport, VBIOS POST, ACR secure boot, MMU v2, PFIFO/PBDMA channels, copy engine,
             GR compute dispatch of ptxas sm_61 cubins).
  Rung 5     DONE, TWO RUNTIMES (MAX through a libcuda+NVML shim; llama.cpp through that shim plus
             cudart/cuBLAS shims).  Both generate correct tokens on the card over TinyGPU.

Card state at handover: warm and serving gemma4-e4b on llama.cpp, context 16384, 1080 MHz memory
(measured 537 f_CK) / 607 MHz SM, 4.7 GiB of 10 GiB free, endpoint http://127.0.0.1:11434 up, shimd
running.  `bin/local-gpu status` is the authority.

Models in localgpu/local_gpu.py MODELS: gemma4-e4b (DEFAULT, llama, QAT Q4_0 5.15 GB, "fast, tools"),
gemma4-12b (6.98 GB), llama3.1-8b, qwen3-8b (thinking), smollm2 (tiny, for testing), gemma4-e2b-max
(MAX bf16 9.6 GB), smollm2-max (MAX fp32).  Only e4b and smollm2 have weights on disk; the others
need `local-gpu pull` (llama3.1-8b was 20% through a pull when a 900 s cap cut it — it resumes).

Performance history, for orientation (all greedy, batch 1):
  405 MHz memory (boot): E4B 304 s / 6.83 tok/s decode; SmolLM2-135M Q4_0 61 tok/s; MAX: SmolLM2-135M
    fp32 6.6 tok/s, Gemma 4 E2B bf16 1.9 tok/s.
  1080 MHz memory (since session 11): E4B 159 s / 13.4-13.5 tok/s, byte-identical output to the
    405 MHz run.
  Caveat worth reconciling: the bench's speculation-ON number is recorded twice with very different
    results — 153 s in the session-12 notes, 67 s measured in session 15 (11 turns, tool call not
    made).  Neither is explained; state which argument set a number came from before quoting it.

=====================================================================
3. INSTRUMENTS (sessions 13-15 — use these, do not rebuild them)
=====================================================================
  pascal/mclk.py            memory reclock: status | get | set <MHz> [--live] [--mode2]
                            [--patch=ADDR:VAL,...] | retrain ... --soft.  Ramps in <= 54 MHz steps.
  pascal/mclk_measure.py    measures the REAL clock per domain from nvgpu's trim counters (--verify
                            is what smclk.py calls; the counters and the calibration are in the doc
                            string).  Calibrated on the SM clock (607 MHz, %clock64) because the
                            paper reading of the counter's window field does not land on it.
  pascal/smclk.py           SM-clock harness: --verify, --scan, --probe, --snapshot/--diff/--restore
                            (413 registers, JSON + checksum), --apply ADDR:VALUE (read-back verified,
                            auto-restore on mismatch), --test (the two-test protocol).
  pascal/smclk_sweep.py     measured register sweep: same-value probe first, one variable at a time,
                            medians of 3 reads, 0.5% threshold, restore unless it moves a clock.
  pascal/memtest.py         full-VRAM GPU pattern test (+ --inject negative control, --loops, --hold).
                            THE judge of a memory reclock.
  pascal/test_depchain.py   launch-path dependency test (modes off/none/dep/group/both); needs
                            NVPASCAL_CHAIN_ALLOW_UNSAFE=1 to arm a non-default mode.
  localgpu/bench_chat.py    12-turn benchmark: TTFT, prefill/decode, cache reuse, tool call, thinking
                            channel.  Reports land in logs/bench/*.{json,md}.
  localgpu/sandbox_exec.py  the code sandbox (hvi microVM default, docker fallback): per-libc deps,
                            `--seed` profiles.  Docs in sandbox/README.md.
  sandbox/tools/candles.py  OHLCV CSV -> candlestick PNG, shipped in the repo so a model can run it
                            read-only at /repo.
  scripts/… nothing; `shim/build.sh`, `shim/build-cuda12.sh` rebuild the shims.

=====================================================================
4. OPEN TASKS, RANKED
=====================================================================
A. **PMU clock control** — the only remaining performance avenue (RESULTS.md section 15).
   What it is: on this card the GPC clock and its voltage are programmed by PMU firmware from VBIOS
   tables.  nvgpu's route: parse the BIT 'C' FLL/VIN and BIT 'P' perf tables into PMU *board
   objects*, load them, then request a voltage/frequency point (VFE).  Read first:
   docs/reference/files/nvgpu-r32.7.6-clk_fll.c, -clk_vin.c, -clk_gp106.c, plus ctrl/ctrlclk.h and
   ctrl/ctrlvolt.h upstream (the board-object wire format), and pascal/vbios.py / vbios_perf.py for
   the tables.
   The prize: SM 607 -> ~797 MHz (the perf table's P0 row) = ~1.3x, NOT the 2.4x that comparing 607
   with the 1480 MHz base clock suggests — the VBIOS quotes these clocks at 2x the hardware rate, the
   same convention the memory clock turned out to have.  If decode is launch-bound at ~14%, expect
   single digits to ~20% end to end; measure, do not assume.
   The risk: a bad frequency/voltage request can wedge the card and only a physical power cycle
   recovers it.  Scope it, write down what breaks if it is wrong, ask the user explicitly.
   How to know it worked: `pascal/smclk.py --verify` prints a higher GPC clock (a measurement, not a
   register), then bench_chat.py with LOCAL_GPU_PLAIN=1 on BOTH sides (4% is this card's noise floor).
   Anything else is not evidence.

B. **Finish MAX patch 0004 (int8 W8A16 weights).**  The `w8_linear` custom op (maxpatch/kernels/w8)
   is written and benchmarked standalone (1.2-2.8x decode, 5.5-11x prefill over MAX's bf16
   ops.matmul), and MAX_GEMMA4_INT8=1 makes the gemma4 config declare INT8_W8A8 with load-time RTN
   quantization — but a full model compile still fails with `error: 'mo.custom' op MOToMAP: no kernel
   registered for 'w8_linear'`.  Two registration paths were tried; the trace was at
   pipelines/lib/graph_specs.py (GraphSpec.custom_extensions).  Session 10 saw MAX's shim load
   `mogg_foreach_r1_w8_b256_*` modules during an E2B bring-up, so the package does reach some compile
   path — compare it with the failing pipeline compile.  This is what would let Gemma 4 E4B (9.4 GB of
   device weights, over the card's ~8.8 GiB pool) run on MAX at all.  Honest expectation: int8 weights
   read 2x the bytes of llama.cpp's Q4_0, so this is a coverage task, not a tok/s one.

C. **Re-run the llama.cpp op suite** after commit c4f50f5 (added cublasStrsmBatched for ggml's
   SOLVE_TRI).  Session 9's run passed 13,380 cases then aborted with exit 134 (the shim's generated
   stubs log and abort on an unimplemented entry point); the rebuild is verified, the re-run is not.
   `test-backend-ops -o MUL_MAT` is the quick pass; a failure names the missing entry point.

D. **Model-side knobs, no hardware.**  Requantising the Q6_K output tensor is untried (~170 MB/token
   less, ~+3%, inside the noise).  KV cache q8_0 was measured and REJECTED (168 s / 12.6 tok/s vs
   159 s / 13.4).  E4B plans in its thinking channel and can spend a 200-token budget on the plan:
   raise --max-tokens, or set chat_template_kwargs {"enable_thinking": false} in the /v1 body.

E. **Housekeeping.**  Die temperature is DONE: `local-gpu status` reports it through shimd's op 11
   TEMP (37 C whilst serving), so the old "can't read it while shimd holds the card" problem is
   solved — `pascal/therm.py` is only useful when the card is free.  The two MAX containers
   (maxnightly/maxlab) sit idle holding RAM; the user was offered a stop and has not answered.

F. **Left unclaimed on purpose:** the v01_07 QMD layout as a serialization encoding (v02_01 has no
   DEPENDENT_QMD_POINTER; v01_07 does, at bits 191:160).  Only worth it if someone wants the ~5-7%
   badly enough to hand the hardware a different QMD layout — and the synthetic chain test cannot
   certify it, only the real workload can.

=====================================================================
5. STANDING CORRECTIONS AND GOTCHAS (do not re-learn these)
=====================================================================
* `bin/local-gpu -m X serve` does not parse — `bin/local-gpu serve -m X`.
* MAX needs `--served-model-name <local alias>` or every proxied request 400s.  Do not remove it.
* Never put shim/build-cuda12 on MAX's LD_LIBRARY_PATH: MAX ships its own cuBLAS bindings
  (_cublas.mojoc) and a vendor-BLAS fallback that its compile gate disables for compute < 7.0
  anyway.  MAX is driven only through libcuda + libnvidia-ml (it dlsyms ~118 cu* + 16 nvml*).
* MAX TTFT spikes are graph compiles at new input shapes, not prefill, and MAX reports no per-request
  timings, so its cached/prefill columns are derived — never compare those across runtimes.
* E4B (QAT Q4_0) plans in a thinking channel (`reasoning_content`) before answering; the GGUF carries
  the `<|channel>` markers; a 200-token budget is often spent entirely on the plan.
* shimd holds the TinyGPU device exclusively: one client-side process family at a time, and no second
  process may open the card while it runs.
* SHIM_QUIET=1 logs module loads, allocations, stubs and errors — NOT launches.  Per-kernel GPU time
  is NVPASCAL_PROF=1 on the *server* (printed on client disconnect); SHIMD_TRACE=1 logs copies and
  device pointers; SHIMD_OPSTAT=1 gives per-op host timing.
* Recreating a container does not pick up edits to bind-mounted files: the repo is mounted at /repo in
  every container so edits are live, but a *process* started before the edit keeps the old code.
* tinygrad's own NV runtime must never be pointed at this card (NV=1): it issues a PCIe FLR that
  Pascal misreads (tinygrad PR #19323).
* Sleep takes the Thunderbolt link down and the next MMIO is a kernel panic (`LLC Bus error`).
  pascal/sleepguard.py holds a caffeinate assertion in every card-holding process and refuses MMIO
  after a sleep; keep the lid open.  A replug can bring the card back cold — TinyGPU reattaches by
  itself; then run the cold path.
* A PRAMIN write+read spot check (vram_test.py) passes even when DRAM is broken: PRAMIN goes through
  the 2.75 MB L2.  Only memtest.py counts.
* Reclock recovery after a bad excursion is a clock *cycle*, not a down-ramp: `mclk.py set 810 --live`
  then `set 1080 --live`.  `local-gpu serve` now cycles and retries the target before settling for
  the 810 fallback (it used to land there silently, costing ~30% of the bandwidth).
* devinit + fb do NOT rewrite the refpll coefficient or the DRAM band config, so `mclk.py get` can
  name a clock the DRAM is no longer configured for after a re-POST; memtest is the judge.
* Ground rule: every claim is a round trip with a negative control, and print the raw value the
  judgement came from.  "The model served tokens" and "the server is healthy" are different claims.

=====================================================================
6. FILE MAP
=====================================================================
  pascal/shimd.py        the macOS-side driver server: the CUDA driver ABI over nvpascal; its
                         docstring documents protocol ops 1-11 (11 = TEMP); PTX JIT'd with ptxas
                         -arch=sm_61
  pascal/nvpascal.py     the runtime: GR channel, the async launch ring, QMD/constant-bank upload from
                         the pushbuffer, stream-ordered copies, launch_async (WFI + pushbuffer semaphore)
  pascal/card.py         BAR0/PRAMIN/BAR1 access: rd/wr/vram_read/vram_write (the TinyGPU lock)
  pascal/mclk.py         memory reclock;  mclk_measure.py  the clock counters;  memtest.py  the judge
  pascal/smclk.py        SM-clock harness (snapshot/probe/apply/restore/verify);  smclk_sweep.py  the sweep
  pascal/vbios.py, vbios_perf.py, vbinit.py    VBIOS parsing: BIT tables, perf/rammap, init scripts
  pascal/devinit.py      VBIOS POST (the PMU DEVINIT app — this is what sets the boot clocks);  fb.py
  pascal/gr.py, fifo.py, grchan.py, mmu.py, acr.py, grctx.py, faults.py, reset_engines.py,
  recover_fifo.py, bench.py, test_modules.py, test_async.py, test_depchain.py, therm.py
  shim/                  libcuda.so.1 + libnvidia-ml.so.1 (build/) and the cudart/cublas layers
                         (build-cuda12/); README lists the env knobs (SHIM_MODE=remote, SHIM_LOG,
                         SHIM_CAPTURE_DIR)
  shim/tests/, shim/cudart/test_runtime.cu     the MAX/llama end-to-end and the cudart+cuBLAS gate
  localgpu/local_gpu.py  bring-up, model table, serve/chat/status, the Ollama+OpenAI proxy launcher
  localgpu/bench_chat.py the 12-turn benchmark;  localgpu/chat.py  the chat client + tool harness
  localgpu/sandbox_exec.py, pascal/hvi_guest.py  the sandbox (hvi guest + docker backend)
  sandbox/               README.md (backends, mounts, deps, seeding), tools/candles.py
  maxpatch/              patches/<modular-version>/ + install.sh (+ --revert), kernels/w8, patch_target.py
  kernels/               PTX sources + sm_61 cubins + the ptxas Docker build
  probes/freeze_card_test.py   proves the token stream is gated by the driver server
  logs/                  bench/ (benchmark reports), mclk/ (reclock evidence), smclk/ (the sweep log)
  docs/                  evidence-log.md, reference/ (+ files/ with the vendored nvgpu sources to
                         build on), handoffs/ (per-session + this file), upstream/ (the draft Modular PRs)

=====================================================================
7. HANDOVER STATE
=====================================================================
Card: warm, serving gemma4-e4b on llama.cpp (context 16384), 1080 MHz memory / 607 MHz SM, endpoint
127.0.0.1:11434 up, shimd running.  Nothing is left modified from any experiment: `pascal/smclk.py
--diff /tmp/sweep2-before.json` reported 0 registers differing from the pre-sweep snapshot.

Git: branch `main`, working tree clean, pushed.  Session 15 commits, newest last:
  4668d3a  Sandbox: one pre-seeded stack for both backends, per-libc deps, and a candlestick helper
  67b6427  chat: make the python tool's files= usable, and keep the candlestick nudge
  2b8c5be  Memory clock: measure it from the host, and act on what the measurement says
  e534b8f  Task B closed: the per-launch WAIT_FOR_IDLE cannot be dropped on GP102
  9405108  SM clock: the harness, the rollback, and what the card actually allows
  677f4bf  SM clock sweep: 12 candidate registers, all writable, none effective
  Earlier in session 15: 13fe319 (SearXNG + fetch_url), 90c1a14 (tool harness registry/bundles),
  f35806e (slash TAB completion), 34ac4e2, b0f5c66, 61c2289.
  Earlier sessions: b24b010 (the 405 -> 1080 reclock), 1f94923 (Modular draft PRs), 611fe85 (the
  async launch ring), 56cdb71 (bench_chat).

Also worth knowing, from earlier in session 15:
  * The chat harness (`bin/local-gpu chat --tools`) has a tool registry in bundles that load by
    keyword, by the model calling `use_tools`, or on demand — only the active tools go into a request.
    `edit` never auto-loads.  `--tools` pins nothing, `--tools=all` pins everything, `--no-auto`
    disables keyword loading.  Slash commands list on the bare word and complete with TAB.
  * The DOCN task — "read DOCN's open and close for the last 7 days, chart it with python, open the
    image" — runs end to end on E4B in three tool calls (stock_history -> python -> show), with the
    chart verified as real candlesticks and the Mac actually opening it.  It is the harness's
    reference task; both sandbox backends carry matplotlib for it.
  * The sandbox is an hvi microVM by default (one VM per call, 0.3-0.7 s, no network, Alpine guest)
    with Docker as the fallback; deps are per-libc (glibc for the container, musl for the guest) and
    `sandbox --seed data|plot|web|text|math` pre-loads named sets.  The curated package list lives in
    exactly one place (APK_SEED / PIP_SEED) so the two backends cannot drift again.
