# Multi-turn benchmark: gemma4-e4b gemma4-e4b-mclk1080-bs-ngrammod

| # | turn | context | new | cached | prefill tok/s | generated | decode tok/s | TTFT s | wall s |
|---|---|---|---|---|---|---|---|---|---|
| 1 | explain | 129 | 129 | 0 | 305 | 200 | 14.1 | 0.7 | 14.6 |
| 2 | code | 169 | 45 | 124 | 148 | 200 | 13.9 | 0.6 | 14.7 |
| 3 | code-followup | 334 | 170 | 164 | 306 | 200 | 14.2 | 0.9 | 14.7 |
| 4 | long-doc | 2801 | 2472 | 329 | 641 | 200 | 13.5 | 4.1 | 18.6 |
| 5 | doc-question | 2831 | 30 | 2801 | 72 | 200 | 13.2 | 0.7 | 15.6 |
| 6 | tool-call | 2907 | 2907 | 0 | 634 | 76 | 13.7 | 5.0 | 10.3 |
| 7 | tool-result | 2995 | 88 | 2907 | 219 | 74 | 13.2 | 0.4 | 6.0 |
| 8 | translate | 2989 | 163 | 2826 | 240 | 147 | 14.1 | 1.2 | 11.4 |
| 9 | creative | 3045 | 56 | 2989 | 119 | 20 | 14.4 | 0.5 | 1.8 |
| 10 | json | 3107 | 43 | 3064 | 98 | 200 | 13.8 | 0.7 | 14.9 |
| 11 | reasoning | 3232 | 125 | 3107 | 219 | 200 | 13.6 | 0.8 | 15.2 |
| 12 | recap | 3253 | 21 | 3232 | 48 | 200 | 13.5 | 0.7 | 15.2 |

6 turn(s) spent the whole token budget in the model's thinking channel (`reasoning` in the JSON, no visible content).

{
 "model": "gemma4-e4b",
 "label": "gemma4-e4b-mclk1080-bs-ngrammod",
 "url": "http://127.0.0.1:11434/v1",
 "turns": 12,
 "wall_s": 152.9,
 "final_context": 3253,
 "generated_tokens": 1917,
 "prompt_tokens_processed": 6249,
 "prompt_tokens_cached": 21543,
 "decode_tps_overall": 14.03,
 "prefill_tps_overall": 385.3,
 "ttft_median_s": 0.69,
 "ttft_max_s": 5.02,
 "tool_call_ok": true,
 "thinking_turns": 6,
 "timings_available": true
}
