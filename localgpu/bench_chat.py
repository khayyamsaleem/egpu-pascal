"""Extended multi-turn chat benchmark for whatever `local-gpu` is serving (or any OpenAI endpoint).

Plays a fixed 12-turn conversation that grows its context the way real chat and agent sessions do:
explanations, code with follow-ups, a long pasted document and questions about it, a tool-call
round trip, translation, JSON output, and a final summary of the whole conversation. Greedy
decoding (temperature 0), so runs are comparable.

Per turn it records, client-side: time to first token (TTFT), wall time, tokens (usage); and from
llama-server's `timings` when present: prompt tokens actually processed vs reused from the prompt
cache, prefill tok/s, decode tok/s. Writes logs/bench/<time>-<model>.{json,md}.

Usage: python3 localgpu/bench_chat.py [--url http://127.0.0.1:11434/v1] [--max-tokens 200] [--label ...]
Standard library only.
"""
import argparse, json, os, statistics, sys, time, urllib.request

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SYSTEM = (
    "You are a careful, concise engineering assistant running locally on a GTX 1080 Ti. "
    "Answer directly, prefer short paragraphs and small code samples, state assumptions, and "
    "never invent facts about documents you were given: quote or cite them instead. When a tool "
    "is available and needed, call it rather than guessing, then use its result. Keep answers under "
    "about 200 words unless asked for more. Use Markdown for code.")

TOOLS = [{"type": "function", "function": {
    "name": "get_weather", "description": "Current weather for a city",
    "parameters": {"type": "object", "properties": {"city": {"type": "string", "description": "City name"}},
                   "required": ["city"]}}}]


def document():
    """~1.5-2k tokens of real technical text: this repo's own results write-up."""
    try:
        t = open(os.path.join(REPO, "RESULTS.md")).read()
    except OSError:
        t = "No document available. " * 300
    i = t.find("## 4. Blockers")
    return t[i:i + 7000] if i >= 0 else t[:7000]


TURNS = [
    ("explain", "In three short paragraphs: what does a GPU do differently from a CPU, and why does "
                "that matter for running language models?"),
    ("code", "Write a Python function moving_average(xs, window) that returns the moving averages of a "
             "list. Include a docstring and a tiny example."),
    ("code-followup", "Now make it work efficiently on NumPy arrays, and state its time complexity."),
    ("long-doc", "Here is a document about a GPU driver project:\n\n<document>\n%s\n</document>\n\n"
                 "Summarize the document in five bullet points." % document()),
    ("doc-question", "From that document: pick two of the blockers and explain their root causes and fixes "
                     "in plain language."),
    ("tool-call", "What's the weather right now in Tokyo and in Paris? Use the tool."),
    ("tool-result", None),                    # filled in with tool results for the previous turn's calls
    ("translate", "Translate your previous answer into French."),
    ("creative", "Write a haiku about a graphics card in a Thunderbolt enclosure."),
    ("json", "Describe yourself as a JSON object with exactly the keys \"name\", \"strengths\" (list) and "
             "\"weaknesses\" (list). Output only the JSON."),
    ("reasoning", "A model reads 5 GB of weights per generated token and the GPU's memory bandwidth is "
                  "30 GB/s. Ignoring everything else, how many tokens per second can it generate? "
                  "Show the arithmetic, then say what changes at 480 GB/s."),
    ("recap", "Summarize our whole conversation so far in three sentences."),
]


def chat(url, model, messages, max_tokens, tools=None):
    body = {"model": model, "messages": messages, "stream": True, "temperature": 0, "max_tokens": max_tokens,
            "stream_options": {"include_usage": True}}
    if tools:
        body["tools"] = tools
    req = urllib.request.Request(url + "/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    first = None
    text, reasoning, calls, usage, timings, reason = "", "", {}, None, None, None
    with urllib.request.urlopen(req, timeout=7200) as r:
        for line in r:
            line = line.strip()
            if not line.startswith(b"data:") or line[5:].strip() == b"[DONE]":
                continue
            j = json.loads(line[5:])
            usage = j.get("usage") or usage
            timings = j.get("timings") or timings
            for ch in j.get("choices") or []:
                reason = ch.get("finish_reason") or reason
                d = ch.get("delta") or {}
                if (d.get("content") or d.get("reasoning_content") or d.get("tool_calls")) and first is None:
                    first = time.perf_counter()
                text += d.get("content") or ""
                reasoning += d.get("reasoning_content") or ""
                for tc in d.get("tool_calls") or []:
                    e = calls.setdefault(tc.get("index", 0), {"id": "", "type": "function",
                                                              "function": {"name": "", "arguments": ""}})
                    e["id"] = tc.get("id") or e["id"]
                    f = tc.get("function") or {}
                    e["function"]["name"] += f.get("name") or ""
                    e["function"]["arguments"] += f.get("arguments") or ""
    t1 = time.perf_counter()
    return dict(text=text, reasoning=reasoning, tool_calls=[calls[i] for i in sorted(calls)], usage=usage or {},
                timings=timings or {}, finish=reason, ttft=(first or t1) - t0, wall=t1 - t0)


def fake_weather(city):
    return {"tokyo": {"temp_c": 14, "sky": "light rain"}, "paris": {"temp_c": 9, "sky": "overcast"}}.get(
        city.strip().lower(), {"temp_c": 20, "sky": "clear"})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:11434/v1")
    ap.add_argument("--model", default=None, help="default: the served model")
    ap.add_argument("--max-tokens", type=int, default=200)
    ap.add_argument("--label", default="")
    ap.add_argument("--out", default=os.path.join(REPO, "logs", "bench"))
    a = ap.parse_args()
    model = a.model
    if not model:
        with urllib.request.urlopen(a.url + "/models", timeout=10) as r:
            model = json.load(r)["data"][0]["id"]
    messages = [{"role": "system", "content": SYSTEM}]
    rows = []
    print("multi-turn benchmark: %s via %s, max %d tokens/turn, greedy" % (model, a.url, a.max_tokens), flush=True)
    print("%-3s %-14s %7s %7s %7s %8s %6s %8s %7s %6s" % ("#", "turn", "ctx", "new", "cached", "prefill", "gen",
                                                       "decode", "TTFT", "wall"), flush=True)
    t_start = time.perf_counter()
    pending_calls = []
    for i, (name, prompt) in enumerate(TURNS, 1):
        if name == "tool-result":
            if not pending_calls:
                continue                          # the model answered without calling the tool
            for tc in pending_calls:
                try:
                    city = json.loads(tc["function"]["arguments"] or "{}").get("city", "")
                except ValueError:
                    city = ""
                messages.append({"role": "tool", "tool_call_id": tc["id"] or "call_0",
                                 "content": json.dumps(fake_weather(city))})
        else:
            messages.append({"role": "user", "content": prompt})
        r = chat(a.url, model, messages, a.max_tokens, tools=TOOLS if name in ("tool-call", "tool-result") else None)
        msg = {"role": "assistant", "content": r["text"]}
        pending_calls = r["tool_calls"] if name == "tool-call" else []
        if r["tool_calls"]:
            for k, tc in enumerate(r["tool_calls"]):
                tc["id"] = tc["id"] or "call_%d_%d" % (i, k)
            msg["tool_calls"] = r["tool_calls"]
        messages.append(msg)
        tm, us = r["timings"], r["usage"]
        prompt_tok = us.get("prompt_tokens") or (tm.get("prompt_n", 0) + tm.get("cache_n", 0))
        gen = us.get("completion_tokens") or tm.get("predicted_n", 0)
        new = tm.get("prompt_n", prompt_tok)
        cached = tm.get("cache_n", prompt_tok - new if prompt_tok >= new else 0)
        pf = tm.get("prompt_per_second") or (new / r["ttft"] if r["ttft"] else 0)
        dec = tm.get("predicted_per_second") or ((gen - 1) / (r["wall"] - r["ttft"]) if gen > 1 and r["wall"] > r["ttft"] else 0)
        row = dict(turn=i, name=name, context=prompt_tok, new_prompt=new, cached=cached, prefill_tps=round(pf, 1),
                   generated=gen, decode_tps=round(dec, 2), ttft_s=round(r["ttft"], 2), wall_s=round(r["wall"], 2),
                   finish=r["finish"], tool_calls=[tc["function"]["name"] + tc["function"]["arguments"] for tc in r["tool_calls"]],
                   reply=(r["text"] or "")[:400], reasoning=(r["reasoning"] or "")[:400],
                   has_timings=bool(tm))
        rows.append(row)
        print("%-3d %-14s %7d %7d %7d %7.0f/s %6d %6.1f/s %6.1fs %5.1fs%s" % (
            i, name, prompt_tok, new, cached, pf, gen, dec, r["ttft"], r["wall"],
            ("  -> " + ", ".join(row["tool_calls"])) if row["tool_calls"]
            else ("  [think: %d chars, no content]" % len(row["reasoning"]) if not row["reply"] and row["reasoning"] else "")),
              flush=True)
    total = time.perf_counter() - t_start
    gen_total = sum(x["generated"] for x in rows)
    new_total = sum(x["new_prompt"] for x in rows)
    decode_time = sum(x["wall_s"] - x["ttft_s"] for x in rows)
    prefill_time = sum(x["ttft_s"] for x in rows)
    summary = dict(model=model, label=a.label, url=a.url, turns=len(rows), wall_s=round(total, 1),
                   final_context=rows[-1]["context"] if rows else 0, generated_tokens=gen_total,
                   prompt_tokens_processed=new_total, prompt_tokens_cached=sum(x["cached"] for x in rows),
                   decode_tps_overall=round(gen_total / decode_time, 2) if decode_time else 0,
                   prefill_tps_overall=round(new_total / prefill_time, 1) if prefill_time else 0,
                   ttft_median_s=round(statistics.median(x["ttft_s"] for x in rows), 2) if rows else 0,
                   ttft_max_s=max((x["ttft_s"] for x in rows), default=0),
                   tool_call_ok=any(x["tool_calls"] for x in rows if x["name"] == "tool-call"),
                   # a tool-call turn has an empty content field by design, not a spent budget
                   thinking_turns=sum(1 for x in rows
                                      if not x["reply"] and x["reasoning"] and x["finish"] != "tool_calls"),
                   timings_available=any(x["has_timings"] for x in rows))
    print("\n%d turns in %.0f s: %d tokens generated at %.1f tok/s overall; %d prompt tokens processed at %.0f tok/s "
          "(+%d reused from cache); TTFT median %.1f s, max %.1f s; final context %d tokens; tool call %s"
          % (summary["turns"], total, gen_total, summary["decode_tps_overall"], new_total,
             summary["prefill_tps_overall"], summary["prompt_tokens_cached"], summary["ttft_median_s"],
             summary["ttft_max_s"], summary["final_context"], "ok" if summary["tool_call_ok"] else "NOT made"), flush=True)
    os.makedirs(a.out, exist_ok=True)
    stem = os.path.join(a.out, time.strftime("%Y%m%d-%H%M%S") + "-" + model.replace("/", "_") + (("-" + a.label) if a.label else ""))
    json.dump(dict(summary=summary, turns=rows), open(stem + ".json", "w"), indent=1)
    with open(stem + ".md", "w") as f:
        f.write("# Multi-turn benchmark: %s %s\n\n" % (model, a.label))
        f.write("| # | turn | context | new | cached | prefill tok/s | generated | decode tok/s | TTFT s | wall s |\n|---|---|---|---|---|---|---|---|---|---|\n")
        for x in rows:
            f.write("| %d | %s | %d | %d | %d | %.0f | %d | %.1f | %.1f | %.1f |\n" % (
                x["turn"], x["name"], x["context"], x["new_prompt"], x["cached"], x["prefill_tps"], x["generated"],
                x["decode_tps"], x["ttft_s"], x["wall_s"]))
        if not summary["timings_available"]:
            f.write("\nThis backend returned no timing block: `cached` and `prefill tok/s` are derived only from "
                    "TTFT (new = whole prompt), so they are not comparable with llama.cpp numbers.\n")
        if summary["thinking_turns"]:
            f.write("\n%d turn(s) spent the whole token budget in the model's thinking channel (`reasoning` in the "
                    "JSON, no visible content).\n" % summary["thinking_turns"])
        f.write("\n" + json.dumps(summary, indent=1) + "\n")
    print("results: %s.{json,md}" % stem)


if __name__ == "__main__":
    main()
