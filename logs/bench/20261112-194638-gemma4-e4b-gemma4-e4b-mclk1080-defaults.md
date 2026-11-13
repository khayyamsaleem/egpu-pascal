# Multi-turn benchmark: gemma4-e4b gemma4-e4b-mclk1080-defaults

| # | turn | context | new | cached | prefill tok/s | generated | decode tok/s | TTFT s | wall s |
|---|---|---|---|---|---|---|---|---|---|
| 1 | explain | 129 | 129 | 0 | 332 | 200 | 14.1 | 0.6 | 14.5 |
| 2 | code | 169 | 45 | 124 | 144 | 200 | 14.3 | 0.6 | 14.3 |
| 3 | code-followup | 334 | 170 | 164 | 320 | 200 | 13.8 | 0.8 | 15.0 |
| 4 | long-doc | 2801 | 2472 | 329 | 686 | 200 | 14.3 | 3.8 | 17.5 |
| 5 | doc-question | 2831 | 30 | 2801 | 72 | 200 | 13.0 | 0.6 | 15.8 |
| 6 | tool-call | 2907 | 2907 | 0 | 675 | 76 | 13.2 | 4.8 | 10.2 |
| 7 | tool-result | 2995 | 88 | 2907 | 222 | 74 | 13.3 | 0.4 | 5.9 |
| 8 | translate | 2989 | 163 | 2826 | 243 | 147 | 13.5 | 1.2 | 11.8 |
| 9 | creative | 3045 | 56 | 2989 | 120 | 20 | 13.5 | 0.5 | 1.9 |
| 10 | json | 3107 | 43 | 3064 | 98 | 200 | 13.5 | 0.7 | 15.2 |
| 11 | reasoning | 3232 | 125 | 3107 | 224 | 200 | 13.6 | 0.8 | 15.2 |
| 12 | recap | 3253 | 21 | 3232 | 49 | 200 | 13.4 | 0.7 | 15.3 |

6 turn(s) spent the whole token budget in the model's thinking channel (`reasoning` in the JSON, no visible content).

{
 "model": "gemma4-e4b",
 "label": "gemma4-e4b-mclk1080-defaults",
 "url": "http://127.0.0.1:11434/v1",
 "turns": 12,
 "wall_s": 152.7,
 "final_context": 3253,
 "generated_tokens": 1917,
 "prompt_tokens_processed": 6249,
 "prompt_tokens_cached": 21543,
 "decode_tps_overall": 13.98,
 "prefill_tps_overall": 401.3,
 "ttft_median_s": 0.68,
 "ttft_max_s": 4.76,
 "tool_call_ok": true,
 "thinking_turns": 6,
 "timings_available": true
}
