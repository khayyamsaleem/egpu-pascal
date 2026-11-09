# maxpatch: local patches that make MAX 26.6.0 run on sm_61

MAX 26.6.0 / Mojo 1.1.0 (`modular==26.6.0`) has a GTX 1080 Ti target, but two things keep
real kernels from compiling for it. These patches are local to the container. They are
not upstreamed, and the stock install can be restored with `install.sh --revert`.

| # | Where | Problem | Fix |
|---|---|---|---|
| 1 | `std.mojoc` (`std/_gpu/host/info.mojo` `_get_gtx1080ti_target`) | features `"+ptx50,+sm_61"`: PTX ISA 5.0 has no `shfl.sync`, so every warp shuffle fails: `LLVM ERROR: Cannot select: intrinsic %llvm.nvvm.shfl.sync.idx.i32` | `"+ptx63,+sm_61"`, a same-length byte patch inside the package's zstd payload (`patch_std_target.py`) |
| 2 | `linalg` `matmul/gpu/__init__.mojo` `_matmul_gpu` | `comptime if use_tensor_core and has_static_NK` instantiates the Ampere multistage GEMM even when the runtime shape check would skip it: `Cannot select: intrinsic %llvm.nvvm.ldmatrix.sync.aligned.m8n8.x4.b16` | `nvidia_no_tensor_cores = has_nvidia_gpu_accelerator() and compute < 7.0` keeps the tensor-core block and the cuBLAS fallback out, so SIMT kernels run (`patches/0001-*.patch`) |

Kernel packages (`linalg.mojoc`, `nn.mojoc`, ...) hold non-elaborated IR, so a package
rebuilt from the matching source tag with `mojo precompile` is a drop-in replacement. An
unmodified rebuild of `linalg` comes out within 0.03% of the shipped size.

## Use

```sh
git clone --depth 1 --branch max/v26.6.0 https://github.com/modular/modular.git upstream/modular
docker exec maxlab bash /repo/maxpatch/install.sh          # patch, precompile, install, clear caches
docker exec maxlab bash /repo/maxpatch/install.sh --revert # restore the shipped packages
```

## Verified on the card

* `shim/tests/shfl.mojo`: `shuffle_idx` plus a 5-step `shuffle_xor` butterfly, 32 of 32 lanes correct.
* `shim/tests/graph_add.py matmul`: 64x64 fp32 `ops.matmul` against numpy with random B, max abs err 0.
* `shim/tests/ops_check.py`: 10/10 ops at SmolLM2 shapes (prefill matmul, decode gemv,
  softmax, RMSNorm, SiLU, ...).
* `max generate` SmolLM2-135M-Instruct fp32: GPU text identical to MAX-CPU over 64 greedy tokens.
* `install.sh --revert` then `install.sh` reproduces the above (35 s).

## Not patched yet

* `tanh` lowers to `tanh.approx.f32` (sm_75+, PTX 7.0), so ptxas rejects it for sm_61.
  GELU-tanh models need a fallback (e.g. via `exp`) in std's math lowering.
* Flash attention: `nn/attention/gpu/mha.mojo` `depth_supported_by_gpu` sends head depth 64
  to FA only on A100/H100/SM100, so SmolLM2 (and Llama-3.2-1B, also depth 64) fall back to
  the naive MHA kernels here. Depth 128 is "universally supported", so models with
  128-wide heads (Llama-3-8B, most Qwen) will route to the tensor-core FA2 path. That needs
  `flash_attention_hw_supported` gated on `compute >= 8.0`, like patch 0001.
