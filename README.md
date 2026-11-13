# egpu-pascal

A from-scratch, user-space bring-up of an **NVIDIA GTX 1080 Ti (GP102, Pascal)** in a
Thunderbolt enclosure on an **Apple Silicon MacBook Air**. It needs no NVIDIA driver and
runs over tinygrad's TinyGPU DriverKit transport.

**Status (2026-11-12):** the card POSTs from macOS, secure-boots its graphics firmware,
and runs **ptxas-compiled sm_61 CUDA kernels** dispatched through our own channel/QMD code.
Two runtimes reach it through a libcuda/NVML shim over TinyGPU:

* **Modular MAX** (via [`maxpatch/`](maxpatch/)): SmolLM2-135M fp32 (text identical to
  MAX-on-CPU) and **Gemma 4 E2B** bf16 — "The capital of France is **Paris**." as transformers
  gives it, and 200 ms/token since launches became asynchronous (was 529).
* **llama.cpp** (via `shim/cudart` + `shim/cublas`): SmolLM2-135M Q4_0 at 61 tok/s
  generation / 447 tok/s prompt, and `test-backend-ops` 10,000+ cases with 0 failures.
  **Gemma 4 E4B (QAT Q4_0) at 13.5 tok/s** with tool calling.

**The memory is reclocked** from the 405 MHz boot clock to **1080 MHz** (31 -> 87 GB/s), the
first Pascal memory reclock by an open driver (`pascal/mclk.py`; RESULTS.md section 10). The
VBIOS bands above 1134 MHz are out of reach on that (refpll) path — band 2 never completes DRAM
training and the first GPU work after it hangs — so the next lever is the memory PLL the high
bands need (RESULTS.md section 12).

[`bin/local-gpu`](localgpu/) brings the whole stack up with one command: card init if it is
cold, the driver server, the right container, a resumable model download, an Ollama-compatible
endpoint on `127.0.0.1:11434`, and a terminal chat. Full results: **[RESULTS.md](RESULTS.md)**;
what is in flight: **[docs/handoffs/HANDOFF-TO-OPUS.md](docs/handoffs/HANDOFF-TO-OPUS.md)**.

```
rung 0  transport + VBIOS POST from macOS + 11 GB VRAM   DONE
rung 1  our microcode on a falcon                         DONE
rung 2  ACR secure boot, FECS/GPCCS running               DONE
rung 3  MMU v2 + GPFIFO channel + copy engine             DONE
rung 4  compute dispatch: sm_61 vadd -> host memory        DONE
rung 5  model inference: MAX (SmolLM2, Gemma 4 E2B), llama.cpp    DONE (quantized weights next)
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

## Chat and the tool harness

`local-gpu chat` (or `bin/gpu-chat` for a new window) talks to the local endpoint; adding
`--tools` gives it the tool harness. Tools are a registry grouped into bundles, and **only the
active ones go into a request** -- the default is one tool wide (`use_tools`) and grows on demand:

| bundle | tools | loads when |
|---|---|---|
| `web` | `web_search` (your own SearXNG first, then keyless fallbacks), `fetch_url` | search / latest / url / news |
| `repo` | `list_files`, `read_file`, `grep_repo`, `git_cmd` (read-only git) | grep / git / diff / file / repo |
| `market` | `stock_quote` (Nasdaq for US tickers, Yahoo for foreign/index, CoinGecko for crypto), `fx_rate` (ECB) | stock / ticker / trading at / exchange rate |
| `compute` | `calc` (AST-checked arithmetic, no side effects) | how many / calculate / percent |
| `edit` | `write_file`, `run_command`, `python` | **never automatically**, and each call asks first |
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
| `llama/` | llama.cpp build for sm_61 (`--no-compress`, no VMM, no CUDA graphs) and its container |
| `maxpatch/` | the local MAX patches for sm_61 (and the `w8_linear` int8 custom-op kernel) + their installer |
| `tools/nvclass`, `tools/nvkm`, `tools/mesa` | vendored references (MIT): NVIDIA class/QMD headers, nouveau, Mesa |
| `docs/` | the chronological evidence trail and per-session handoffs |
| `docs/upstream/` | what was sent to Modular and tinygrad (issues, draft PRs, the exact diffs) and why the rest stays local |
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
