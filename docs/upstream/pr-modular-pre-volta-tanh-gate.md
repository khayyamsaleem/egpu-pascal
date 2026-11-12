`tanh.approx.{f16,f32,bf16}` are sm_75+ (PTX ISA 7.0). On Maxwell, Pascal and Volta `ptxas` rejects them:

```
ptxas error : Feature 'tanh' requires .target sm_75 or higher
```

so every GELU-tanh or logit-softcap model fails to compile for those targets - Gemma 4 among them.

## What this changes

At the top of `_tanh`, when the target is `sm_52`/`sm_60`/`sm_61`/`sm_70`, return std's generic rational `_tanh_impl` (which the file already has, in `std.math.math`) applied in fp32 and cast back to the input dtype. sm_75+ keeps the current intrinsics.

## Testing

- `_tanh_impl` accuracy on the card: rel err `6e-8` against the analytic tanh (measured with `shim/tests/exp_check.py`).
- With this change on a GTX 1080 Ti: `ops_check.py --dtype bfloat16` 12/12 (including GELU-tanh and tanh softcapping), and Gemma 4 E2B bf16 produces output identical to `transformers` on the same prompt.
- This file is byte-identical between `main` and the nightly revision the change was built and run with (`b786047492`), so this diff is exactly what was tested - not a rebase.

`Assisted-by: Hermes Agent (AI); reviewed by the author`

---

_Draft: Modular's AI Tool Use Policy asks the author to review AI-assisted changes before they are submitted for review. This one is a draft for exactly that reason; the diff is the whole change._
