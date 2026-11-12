# Multi-turn benchmark: smollm2 smollm2-q4_0-llama-cuda

| # | turn | context | new | cached | prefill tok/s | generated | decode tok/s | TTFT s | wall s |
|---|---|---|---|---|---|---|---|---|---|
| 1 | explain | 129 | 1 | 128 | 31 | 200 | 57.1 | 0.4 | 3.9 |
| 2 | code | 370 | 42 | 328 | 1153 | 200 | 56.0 | 0.0 | 3.6 |
| 3 | code-followup | 596 | 27 | 569 | 844 | 200 | 53.8 | 0.0 | 3.7 |
| 4 | long-doc | 3388 | 2593 | 795 | 2900 | 200 | 44.0 | 0.9 | 5.4 |
| 5 | doc-question | 3619 | 32 | 3587 | 812 | 200 | 43.5 | 0.1 | 4.6 |
| 6 | tool-call | 3846 | 28 | 3818 | 698 | 200 | 43.3 | 0.1 | 4.7 |
| 8 | translate | 4065 | 20 | 4045 | 512 | 200 | 43.9 | 0.1 | 4.6 |
| 9 | creative | 4290 | 26 | 4264 | 643 | 200 | 42.2 | 0.1 | 4.8 |
| 10 | json | 4535 | 46 | 4489 | 880 | 58 | 42.1 | 0.1 | 1.4 |
| 11 | reasoning | 4659 | 67 | 4592 | 1100 | 58 | 43.6 | 0.1 | 1.4 |
| 12 | recap | 4739 | 23 | 4716 | 555 | 61 | 42.5 | 0.1 | 1.5 |

{
 "model": "smollm2",
 "label": "smollm2-q4_0-llama-cuda",
 "url": "http://127.0.0.1:11434/v1",
 "turns": 11,
 "wall_s": 39.5,
 "final_context": 4739,
 "generated_tokens": 1777,
 "prompt_tokens_processed": 2905,
 "prompt_tokens_cached": 31331,
 "decode_tps_overall": 47.07,
 "prefill_tps_overall": 1622.9,
 "ttft_median_s": 0.05,
 "ttft_max_s": 0.9,
 "tool_call_ok": false
}
