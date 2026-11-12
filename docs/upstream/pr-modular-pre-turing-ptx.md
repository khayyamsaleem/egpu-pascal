Fixes #7573.

The four pre-Turing targets in this table ship `features = "+ptx50,+sm_XX"` while every Turing-or-newer entry uses `+ptx63` or newer. PTX ISA 5.0 predates `shfl.sync` (PTX 6.0), so **no kernel that contains a warp shuffle can be compiled for them** - reductions, matmul, softmax, i.e. most of the framework:

```
LLVM ERROR: Cannot select: intrinsic %llvm.nvvm.shfl.sync.idx.i32
```

The architecture is not the limitation: `sm_61` supports PTX 6.x, the table's own `sm_75`/`sm_86` entries already use `+ptx63`, and `ptxas` accepts `.version 6.3` for `sm_52`/`sm_60`/`sm_61`/`sm_70`:

```
PTX 5.0 + sm_52/sm_60/sm_61/sm_70 : FAIL  "Feature 'shfl.sync' requires PTX ISA .version 6.0 or later"
PTX 6.3 + sm_52/sm_60/sm_61/sm_70 : OK
```

(the full five-line repro is in #7573)

## What this changes

Four `+ptx50` -> `+ptx63` in `max/mojo/max/_plugin/cuda/targets.mojo`, for `_gtx970_target` (sm_52), `_teslap100_target` (sm_60), `_gtx1080ti_target` and `_gtx1060_target` (sm_61). Data only, same string length, no behaviour change for any other target.

## Testing

- `sm_61` on real hardware: with this exact change applied to the shipped package of modular 26.6.0 (and the 26.8 nightly), MAX runs SmolLM2-135M fp32 with output identical to MAX-on-CPU, and Gemma 4 E2B bf16 with output identical to `transformers` on the same prompt. Before the change, the first graph fails to compile.
- `sm_52`/`sm_60`: ptxas check only (above). We have no Maxwell or P100 hardware, so I can't claim more than "ptxas accepts the same version it accepts for sm_61".

`Assisted-by: Hermes Agent (AI); reviewed by the author`
