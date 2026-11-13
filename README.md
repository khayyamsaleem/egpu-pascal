# egpu-pascal

A from-scratch, user-space bring-up of an **NVIDIA GTX 1080 Ti (GP102, Pascal)** in a
Thunderbolt enclosure on an **Apple Silicon MacBook Air**. It needs no NVIDIA driver and
runs over tinygrad's TinyGPU DriverKit transport.

**Status (2026-11-13):** the card POSTs from macOS, secure-boots its graphics firmware,
and runs **ptxas-compiled sm_61 CUDA kernels** dispatched through our own channel/QMD code.
Two runtimes reach it through a libcuda/NVML shim over TinyGPU:

* **Modular MAX** (via [`maxpatch/`](maxpatch/)): SmolLM2-135M fp32 (text identical to
  MAX-on-CPU) and **Gemma 4 E2B** bf16 — "The capital of France is **Paris**." as transformers
  gives it, and 200 ms/token since launches became asynchronous (was 529); int8 weights (patch
  0004) take that to 13.4 tok/s decode / 37.7 tok/s prefill with byte-identical text.
* **llama.cpp** (via `shim/cudart` + `shim/cublas`): SmolLM2-135M Q4_0 at 61 tok/s
  generation / 447 tok/s prompt, and `test-backend-ops` 20,290 cases with 0 failures.
  **Gemma 4 E4B (QAT Q4_0) at ~13 tok/s** with tool calling and its vision projector — the
  serving path, and the one the chat harness and Hermes ride on. Only ~25 of those tokens-per-turn
  are the answer; the rest is the model's thinking, which is what
  [Thinking is the cost](#thinking-is-the-cost-not-the-answer) is about.

**The memory is reclocked** from the 405 MHz boot clock to **1080 MHz** (31 -> 87 GB/s), the
first Pascal memory reclock by an open driver (`pascal/mclk.py`; RESULTS.md section 10). The
VBIOS bands above 1134 MHz are out of reach: band 2 never completes DRAM training and the first
GPU work after it hangs. The memory PLL (`0x132000`) was then ported and measured — it *does*
steer the memory clock (a 1350 target reaches 593 f_CK), so band 2's wall is the timing/training
side, not a missing PLL — but the memory is **no longer the constraint**: decode already moves 75%
of the achievable bandwidth, so 1134 MHz's extra 5% buys nothing end to end and the card is bound
by its **607 MHz SM clock and ~900 launches per token** (RESULTS.md sections 13 and 15; the SM
clock is not host-writable, so the remaining avenue is the PMU voltage/frequency path).

[`bin/local-gpu`](localgpu/) brings the whole stack up with one command: card init if it is
cold, the driver server, the right container, a resumable model download, an Ollama-compatible
endpoint on `127.0.0.1:11434`, and a terminal chat. How the pieces fit together, and how to tell
the card did the work: **[docs/HOW-IT-WORKS.md](docs/HOW-IT-WORKS.md)**. Full results:
**[RESULTS.md](RESULTS.md)**; what is in flight:
**[docs/handoffs/HANDOFF-TO-OPUS.md](docs/handoffs/HANDOFF-TO-OPUS.md)**.

```
rung 0  transport + VBIOS POST from macOS + 11 GB VRAM   DONE
rung 1  our microcode on a falcon                         DONE
rung 2  ACR secure boot, FECS/GPCCS running               DONE
rung 3  MMU v2 + GPFIFO channel + copy engine             DONE
rung 4  compute dispatch: sm_61 vadd -> host memory        DONE
rung 5  model inference: llama.cpp (QAT 4-bit, tool calling), MAX (SmolLM2, Gemma 4 E2B)   DONE
```

**llama.cpp is the serving path**, on quantized weights; MAX stays for what only it can do (bf16
checkpoints) and as the thing that exercised the libcuda shim hardest — it has no int4 path for
Pascal (GP102 has no tensor cores), so it pays roughly twice the weight bytes: Gemma 4 E2B is
9.8 tok/s in bf16 and 13.4 tok/s with int8 weights (patch 0004).

## Requirements

* Apple Silicon Mac, a TB3/USB4 eGPU enclosure, a GP102 card (validated: EVGA GTX 1080 Ti)
* tinygrad's **TinyGPU** DriverKit extension installed and enabled; Python venv with
  `tinygrad==0.14.0` (scripts assume `~/.venvs/tinygpu/bin/python`; the transport
  socket is `/tmp/tinygpu.sock`)
* firmware: `firmware/fetch.sh` (NVIDIA signed blobs from linux-firmware) and
  `pascal/dump_vbios.py` (this card's VBIOS); see [firmware/README.md](firmware/README.md)
* kernels: Docker (OrbStack works) for CUDA 12.4 `ptxas` via `kernels/build.sh`
* optional: [hvi](https://github.com/brig-sh/hvi-vmm) for the code sandbox (a signed release binary
  in `~/.local/bin/hvi` plus the guest image — `bin/local-gpu sandbox --status` says what is
  missing); without it the sandbox falls back to Docker
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

MAX is the alternative runtime, not the serving path (see the note above rung 5) — it is here
because it is what proved the libcuda shim against a real framework, and because it is the only way
to run the bf16 checkpoints llama.cpp's 4-bit paths do not cover.

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

## Chat and the tool harness

`local-gpu chat` (or `bin/gpu-chat` for a new window) talks to the local endpoint; adding
`--tools` gives it the tool harness. Tools are a registry grouped into bundles, and **only the
active ones go into a request** -- the default is one tool wide (`use_tools`) and grows on demand:

| bundle | tools | loads when |
|---|---|---|
| `web` | `web_search` (your own SearXNG first, then keyless fallbacks), `fetch_url` | search / latest / url / news |
| `repo` | `list_files`, `read_file`, `grep_repo`, `git_cmd` (read-only git) | grep / git / diff / file / repo |
| `market` | `stock_quote` (live, or a past date/time via `when=`; Nasdaq/Yahoo/CoinGecko), `fx_rate` (ECB) | stock / ticker / trading at / exchange rate |
| `compute` | `calc` (AST-checked arithmetic, no side effects) | how many / calculate / percent |
| `sandbox` | `python`, `shell` (run in an hvi microVM or a locked-down container) | python / script / pandas / plot / parse / unit test |
| `edit` | `write_file`, `run_command` (ON THE HOST) | **never automatically**, and each call asks first |
| `memory` | `remember`, `recall` (notes in `~/.local/state/local-gpu/chat-notes.md`) | remember / recall |
| `time` | `now` | what time / the date |
| `demo` | `get_weather` | (offline demo) |

Three ways a tool gets loaded: the model calls `use_tools`, the model calls a tool that isn't
loaded (the harness loads it and lets it retry), or you do it by hand -- `/tools` lists the registry
and what is pinned, `/load web`, `/unload repo`, `/load all`, `/auto` toggles the keyword
heuristic, `/yes` toggles confirmations.  `--tools=all` pins everything up front, `--tools=web,repo`
pins just those, `--no-auto` turns the keyword heuristic off.

Slash commands autocomplete: `/` then TAB lists every command, TAB completes command names and the
bundle/tool names after `/load`/`/unload` (an ambiguous TAB prints the candidates), `/help <fuzzy>`
ranks the commands by a query, and a typo gets a did-you-mean.  Up-arrow history comes with it.

Checked on the card (session 14): "how many seconds are in 3 days, with the calc tool" loaded
`compute` and called `calc` -> 259200; "what is the current time" loaded `time` and called `now`;
asking for Python loaded `edit` on demand, got the confirmation refusal, and re-ran the arithmetic
with `calc` instead.  A market question ran three tools across three bundles: "what is NVDA trading
at, in yen?" auto-loaded `market`, called `stock_quote` then `fx_rate`, and the model then summoned
`compute` on its own (`use_tools(["compute"])`) to do the multiplication.

Where the market data comes from, and why: **Stooq's keyless CSV is gone** (the old
`/q/l/?s=...&e=csv` URL 404s now, behind a JS proof-of-work check), so `stock_quote` picks a route
per symbol -- Nasdaq's `api.nasdaq.com` for plain US tickers (no rate limit, and it carries the
company name, market cap and 52-week range), Yahoo's keyless chart endpoint for foreign listings
and indices, CoinGecko for crypto pairs.  One measured gotcha worth keeping: Yahoo's edge returns
**429 to a full browser User-Agent but 200 to a plain `Mozilla/5.0`**, same request -- the finance
fetches therefore send the short one.

`stock_quote` also takes `when` for a **historical** price instead of the live one:

| `when` | what comes back |
|---|---|
| `"2026-11-10"` | that session's OHLC and close (interval `1d`) |
| `"2026-11-10 14:30"` | the price **at that minute** (1m bars), times read in the exchange's own zone |
| `"yesterday"`, `"today"` | shorthand for the above |

Crypto goes to CoinGecko for this too (its history and `market_chart/range` endpoints, both
keyless), and Nasdaq's `historical` endpoint is the second opinion for US tickers.  Two details
worth knowing: Yahoo answers **HTTP 422** if an intraday window is too wide or too old, so the
interval is chosen by the age of the target (1m within a week, 5m within two months, 1h within two
years, daily beyond) and the window shrinks with it; and when the requested minute doesn't exist
(the market was closed, or the date is a weekend) the nearest bar is returned **and labelled as
such** rather than passed off as exact.  Verified: AAPL `2026-11-10` returns 348.51 / O 351.35 /
H 353.87 / L 346.46 from Yahoo, matching Nasdaq's own history rows to the cent.

For a *series* rather than one price there is `stock_history(symbols, days=7)`: the daily
open/high/low/close/volume rows for the last N trading days, returned as CSV
(`date,open,high,low,close,volume`) in **one** request per symbol, which is what a chart or a trend
question wants -- seven `stock_quote` calls would be seven round trips for the same numbers.

## Images, Hermes, and unplugging

**Images.** `gemma4-e4b` is served with its vision projector, so the OpenAI endpoint takes
`image_url` content parts (132 prompt tokens per image). `LOCAL_GPU_NO_VISION=1` serves text-only
and frees ~1 GiB of VRAM. The model reads images; it does not produce them.

**Hermes Agent** (CLI, TUI and the desktop app share one config). Add to `~/.hermes/config.yaml`:

```yaml
providers:
  local-gpu:
    api: http://127.0.0.1:11434/v1
    api_key: local-gpu
    discover_models: false
    models:
      gemma4-e4b:
        context_length: 65536
model_aliases:
  local:
    model: gemma4-e4b
    provider: local-gpu
```

Then `local-gpu serve` and `hermes chat -m local`, `/model local` in a running session, or pick
**local-gpu / gemma4-e4b** in the model picker. Hermes needs a 64K context, which is why 65,536 is
the default for this model. Its default prompt is ~13.9K tokens (~10.4K of that JSON tool
schemas), which the card prefills at ~560 tok/s: ~25 s of prompt processing, ~35 s to the first
token on a cold server (RESULTS.md section 17); later turns reuse the cache.

Use **`bin/hermes-local`** rather than bare `hermes chat -m local`: it checks the endpoint and then
launches the same session with a four-tool surface (`file,terminal,code_execution,web`), which is
6.9K tokens of prompt instead of 13.8K — 10.6 s of prompt processing instead of 24.6 s, because ~10.4K
of that default prompt is JSON tool schemas. `HERMES_LOCAL_TOOLSETS=file,terminal` cuts it further.
Section 17 of [RESULTS.md](RESULTS.md) has the multi-turn and subagent measurements behind that
advice, plus the two levers (temperature, `max_tokens`) that were tried and measured as nothing.

**The card has to be there.** `local-gpu serve` refuses to start when no NVIDIA PCI device is in
the IO registry, and a watcher it leaves running stops the servers and the driver server if the
enclosure is unplugged or powered off (MMIO to a card that has gone is a kernel panic). After a
reconnect, `local-gpu serve` again: the card is cold and is re-initialised. Docker is started for
you if it is not running. From everything-down to a healthy endpoint is about 40 s.

## Running code: the sandbox

The model can write and run code without the question "may I run this?" having to be "only if I read
it first", because the code never runs on the Mac. Two backends, picked automatically:

* **hvi** (default) — a [microVMM](https://github.com/brig-sh/hvi-vmm) that talks to
  Hypervisor.framework directly, so every call boots its **own Linux guest with its own kernel**.
  Measured 0.3–0.7 s per call, boot included.
* **docker** — `egpu-sandbox:3.12`, no network, read-only rootfs, caps dropped, used when hvi is
  missing and always for `deps` installs (that step needs egress).

Both see the same `/work` (writable, results come back through it), `/deps` (on `PYTHONPATH`) and a
read-only `/repo`; neither sees the host home, the docker socket, or a network by default.  Host
folders are shared only when *you* declare them — `--mount ~/data:/data:ro`, `LOCAL_GPU_SANDBOX_MOUNTS`,
or `/mount` in a session — and appear at `/mnt/<tag>`; there is deliberately no tool argument for a
mount, so the model cannot ask for `~/.ssh`.

Dependencies: numpy pandas matplotlib scipy sympy tabulate requests bs4 lxml pyyaml dateutil are
preinstalled **in both backends, from one list** (`APK_SEED`/`PIP_SEED`) so they cannot drift, and
`deps=["pkg"]` pip-installs into a per-backend deps dir (glibc for the container, musl for the
guest -- a glibc wheel on the guest's `PYTHONPATH` breaks every import, which is exactly how that
was learned).  `sandbox --seed data|plot|web|text|math` pre-loads a named set.  Files a run leaves
in `/work` are listed back with their host paths, and the `show` tool opens one on the Mac (asks
first, and only inside the work dir).  `bin/local-gpu sandbox --status | --guest | --build | --seed
| --reset | --shell`; full notes in [sandbox/README.md](sandbox/README.md).

Worked example -- the task this harness was measured against ("review DOCN's open and close every day
for the last seven days, chart it with python, open the image") runs end-to-end on the 1080 Ti with
the E4B model in three tool calls and no hints from the human:

    -> stock_history({"days": 7, "symbols": "DOCN"})        one request, 7 daily OHLCV rows as CSV
    -> python({"code": "... candles.py --out /work/chart.png ..."})      0.6 s in the microVM
    -> show({"path": "/work/chart.png"})                    "opened chart.png on the Mac (26733 bytes)"

Getting there was almost entirely tool *interface* work, not model work -- small models drive a tool
correctly only when the obvious invocation is the right one: the helper takes its CSV by path, by
`--csv`, or on stdin (E4B tried all three), the guest has `python` as well as `python3`, the repo is
at `/repo` in both backends rather than `/mnt/repo` in one, and `stock_history` emits a real header
row so its output can be pasted straight into a file.  Each of those was a failed round trip first.

Verified on the card: asked to read a host file mounted at `/mnt/data`, the model wrote Python that
read it inside the guest (0.4 s) and printed its SHA-256 — which matched `shasum -a 256` on the Mac
exactly.  A read-only mount refuses writes with `Read-only file system`; a read-write mount's file
appears on the host afterwards.

## Thinking is the cost, not the answer

On this card decode runs at ~13 tok/s, so the price of a turn is the tokens the model *thinks*, and
the stock question "what is the stock price for docn?" made a good specimen: one tool call plus a
one-line answer cost **270 generated tokens and ~21 s** in the session, of which the answer itself
was ~25 tokens.

Where it went (each config replayed the same prompt through the harness loop -- request, run the
tool, request again -- measuring `reasoning_content` against `content`; temp 1.0 is the server
default the client used to inherit):

| prompt shape | temp | generated | of which thinking | answer | wall | round trips |
|---|---|---|---|---|---|---|
| original prompt | 1.0 | 236 | 169 | 29 | 18.8 s | 2 |
| original prompt | 0.0 | 139 | 87 | 16 | 10.3 s | 2 |
| terse prompt | 0.0 | 202 | 134 | 16 | 14.9 s | 2 |
| **terse + terse-answer few-shot** | **0.0** | **156** | 108 | 9 | **9.6 s** | 2 |

And on a question that needs no tool ("how many seconds are in 3 days?") the difference is not a
percentage but a failure: the original prompt **never answered** -- 643 tokens, 3 round trips, 50.7 s
-- while terse + few-shot answered in **10 tokens and 1 s** with no thinking channel at all.

What the three changes actually are:

1. **Send a temperature.** The client sent none, so llama-server's own default (1.0) applied; 0.0 is
   what a tool-calling agent wants and it halved the thinking.  `--temp` / `LOCAL_GPU_TEMP`, or
   `/temp` in the session.
2. **Stop writing a numbered rule list in the system prompt.** The old prompt enumerated its rules
   ("use a tool when...", "use the result verbatim", "state assumptions") and the model mirrored them
   back as a *constraint checklist with a confidence score* inside its thinking -- ~80 tokens every
   turn.  The prompt is now two sentences.
3. **Give a terse-answer few-shot** (`FORMAT_SHOT`) that demonstrates the answer format *without*
   demonstrating a tool call.  A few-shot that showed a tool call scored about the same on the tool
   question but taught the model to call a tool for arithmetic it could do itself -- and every extra
   round trip pays for another thinking block.

Ruled out, on evidence rather than taste:

- **`chat_template_kwargs: {"enable_thinking": false}`** -- the model's template opens the thinking
  channel when `enable_thinking or tools or the first message is a system turn`, i.e. offering any
  tool forces it on.  Passing false does suppress the channel and the model then returns an *empty*
  answer to a tool question and a wrong one ("86400" for three days).  Not a lever.
- **`--reasoning-effort`** -- this build accepts it, but the template never reads it, so it is a
  no-op for this model.  Verified by grepping the template from `/props`.
- **Sending no system prompt at all** -- the most verbose result of the lot (~380 generated tokens on
  the stock question).

## Layout

| Path | What |
|---|---|
| `pascal/` | the driver: `card` (BAR0/PRAMIN), `devinit`/`vbios`, `fb`, `mmu`, `acr`/`acr_unload`, `sec2_rtos`, `gr`/`gr_ctxctl`/`grctx`, `fifo`, `grchan`, `compute`, **`nvpascal`** (runtime), `cubin`, `bench`, `fecs_snap`, probes, recovery |
| `kernels/` | PTX sources, built sm_61 cubins, `Dockerfile` + `build.sh` |
| `shim/` | `libcuda.so.1` + `libnvidia-ml.so.1` replacements for MAX (Linux container), `cudart/` + `cublas/` for llama.cpp, and their tests |
| `localgpu/` | `local-gpu`: card/stack bring-up, model fetch, Ollama + OpenAI proxy, terminal chat (plain, or `--tools` for the tool harness: bundles loaded on demand), multi-turn bench |
| `bin/` | `local-gpu`, `gpu-chat`, `gpu-chat-tools`, `hermes-local` — the launchers, including the Ghostty window and the lean-toolset Hermes session ([HOW-IT-WORKS](docs/HOW-IT-WORKS.md)) |
| `sandbox/` | the code sandbox's image and notes ([README](sandbox/README.md)); the mechanics are `localgpu/hvi_guest.py` (guest image + virtio-fs run protocol) and `localgpu/sandbox_exec.py` (backends, mounts, deps) |
| `llama/` | llama.cpp build for sm_61 (`--no-compress`, no VMM, no CUDA graphs) and its container |
| `maxpatch/` | the local MAX patches for sm_61 (and the `w8_linear` int8 custom-op kernel) + their installer |
| `tools/nvclass`, `tools/nvkm`, `tools/mesa` | vendored references (MIT): NVIDIA class/QMD headers, nouveau, Mesa |
| `docs/` | the chronological evidence trail and per-session handoffs |
| `docs/upstream/`, `upstream/` | what was sent to Modular and tinygrad (issues, draft PRs, the exact diffs) and why the rest stays local; `upstream/` is the early MAX/GPU work |
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
