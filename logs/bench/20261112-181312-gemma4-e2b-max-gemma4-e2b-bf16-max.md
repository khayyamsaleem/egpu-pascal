# Multi-turn benchmark: gemma4-e2b-max gemma4-e2b-bf16-max

| # | turn | context | new | cached | prefill tok/s | generated | decode tok/s | TTFT s | wall s |
|---|---|---|---|---|---|---|---|---|---|
| 1 | explain | 127 | 127 | 0 | 14 | 150 | 5.1 | 9.3 | 38.3 |
| 2 | code | 316 | 316 | 0 | 73 | 200 | 5.0 | 4.3 | 44.5 |
| 3 | code-followup | 541 | 541 | 0 | 231 | 200 | 4.8 | 2.3 | 43.3 |
| 4 | long-doc | 3208 | 3208 | 0 | 15 | 185 | 3.7 | 216.2 | 266.3 |
| 5 | doc-question | 3422 | 3422 | 0 | 452 | 177 | 3.6 | 7.6 | 56.2 |
| 6 | tool-call | 3674 | 3674 | 0 | 9 | 29 | 6.7 | 390.4 | 394.6 |
| 7 | tool-result | 3759 | 3759 | 0 | 257 | 28 | 3.5 | 14.6 | 22.3 |
| 8 | translate | 3754 | 3754 | 0 | 271 | 33 | 3.5 | 13.9 | 22.9 |
| 9 | creative | 3809 | 3809 | 0 | 482 | 19 | 3.5 | 7.9 | 13.0 |
| 10 | json | 3870 | 3870 | 0 | 304 | 97 | 3.5 | 12.7 | 40.2 |
| 11 | reasoning | 4032 | 4032 | 0 | 257 | 144 | 3.5 | 15.7 | 57.0 |
| 12 | recap | 4196 | 4196 | 0 | 514 | 63 | 3.4 | 8.2 | 26.3 |

{
 "model": "gemma4-e2b-max",
 "label": "gemma4-e2b-bf16-max",
 "url": "http://127.0.0.1:11434/v1",
 "turns": 12,
 "wall_s": 1025.0,
 "final_context": 4196,
 "generated_tokens": 1325,
 "prompt_tokens_processed": 34708,
 "prompt_tokens_cached": 0,
 "decode_tps_overall": 4.12,
 "prefill_tps_overall": 49.4,
 "ttft_median_s": 11.0,
 "ttft_max_s": 390.41,
 "tool_call_ok": true
}
