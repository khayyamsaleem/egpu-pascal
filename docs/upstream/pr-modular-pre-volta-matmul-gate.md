The tensor-core block in `_matmul_gpu_impl` is guarded by `comptime if (... and use_tensor_core ...)`, but `use_tensor_core` is a compile-time flag rather than a property of the target, so on a pre-Volta NVIDIA target the block is elaborated anyway and codegen fails on instructions the target does not have:

```
LLVM ERROR: Cannot select: intrinsic %llvm.nvvm.ldmatrix...
```

(Pascal/Maxwell have no tensor cores and no `ldmatrix`/`mma.sync`/`cp.async`.) The runtime shape check that would have skipped the tensor-core path never gets a chance to run, and `has_static_NK`/the surrounding conditions can't express "this target has no tensor cores" either.

## What this changes

Add a comptime `nvidia_no_tensor_cores` (NVIDIA target, `compute < 7.0`) and add `and not nvidia_no_tensor_cores` to the tensor-core condition, so those targets elaborate the SIMT kernels that follow. Two hunks, no change for any sm_70+ target.

## Testing

- Verified on a GTX 1080 Ti (sm_61) with the 26.8 nightly: with this change, MAX graphs and ops run - `ops_check.py` 10/10 (fp32) and 12/12 (bf16), and Gemma 4 E2B bf16 generates text identical to `transformers`.
- Caveat: that run was against the pinned nightly revision (`2841b316`), where this file differs slightly from `main`; the patch here was rebased onto `main`'s file and only the two anchors above changed, so it has not been compiled on `main` itself. If you'd rather express the condition as "target is Maxwell/Pascal" than `compute < 7.0`, that also works - `compute < 7.0` is what was tested.

`Assisted-by: Hermes Agent (AI); reviewed by the author`

---

_Draft: Modular's AI Tool Use Policy asks the author to review AI-assisted changes before they are submitted for review. This one is a draft for exactly that reason; the diff is the whole change._
