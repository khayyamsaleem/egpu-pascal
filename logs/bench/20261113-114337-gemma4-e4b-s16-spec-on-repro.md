# Multi-turn benchmark: gemma4-e4b s16-spec-on-repro

| # | turn | context | new | cached | prefill tok/s | generated | decode tok/s | TTFT s | wall s |
|---|---|---|---|---|---|---|---|---|---|
| 1 | explain | 129 | 129 | 0 | 278 | 200 | 13.5 | 0.7 | 15.3 |
| 2 | code | 169 | 45 | 124 | 141 | 200 | 13.3 | 0.6 | 15.3 |
| 3 | code-followup | 334 | 170 | 164 | 294 | 200 | 13.4 | 0.9 | 15.5 |
| 4 | long-doc | 2801 | 2472 | 329 | 652 | 200 | 13.1 | 4.0 | 18.9 |
| 5 | doc-question | 2831 | 30 | 2801 | 70 | 200 | 12.8 | 0.7 | 15.9 |
| 6 | tool-call | 2907 | 2907 | 0 | 640 | 76 | 12.9 | 5.0 | 10.6 |
| 7 | tool-result | 2995 | 88 | 2907 | 197 | 74 | 12.8 | 0.5 | 6.2 |
| 8 | translate | 2989 | 163 | 2826 | 205 | 147 | 13.1 | 1.4 | 12.3 |
| 9 | creative | 3045 | 56 | 2989 | 119 | 20 | 13.2 | 0.5 | 1.9 |
| 10 | json | 3107 | 43 | 3064 | 97 | 200 | 13.1 | 0.7 | 15.7 |
| 11 | reasoning | 3232 | 125 | 3107 | 206 | 200 | 13.1 | 0.9 | 15.8 |
| 12 | recap | 3253 | 21 | 3232 | 48 | 200 | 12.8 | 0.7 | 16.0 |

6 turn(s) spent the whole token budget in the model's thinking channel (`reasoning` in the JSON, no visible content).

{
 "model": "gemma4-e4b",
 "label": "s16-spec-on-repro",
 "url": "http://127.0.0.1:11434/v1",
 "turns": 12,
 "wall_s": 159.5,
 "final_context": 3253,
 "generated_tokens": 1917,
 "prompt_tokens_processed": 6249,
 "prompt_tokens_cached": 21543,
 "decode_tps_overall": 13.41,
 "prefill_tps_overall": 377.1,
 "ttft_median_s": 0.71,
 "ttft_max_s": 5.02,
 "tool_call_ok": true,
 "thinking_turns": 6,
 "timings_available": true
}
