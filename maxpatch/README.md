# maxpatch: local patches that make MAX run on sm_61 (GTX 1080 Ti)

Two supported installs, one patch set each (`patches/<modular version>/`). `install.sh`
detects the installed `modular`, applies the matching set to a source checkout at the
matching commit, rebuilds the touched packages with `mojo precompile`, and copies patched
Python files. Every replaced file keeps a `.orig`, and `install.sh --revert` restores them all.

| modular | source checkout | what runs on the card |
|---|---|---|
| `26.6.0` (Mojo 1.1.0) | `upstream/modular` at tag `max/v26.6.0` | SmolLM2-135M (fp32) |
| `26.8.0.dev2026110805` nightly (Mojo 1.3) | `upstream/modular-main` at `2841b31631e7` (the lockfile pin of that nightly) | SmolLM2 (fp32/bf16), **Gemma 4 E2B (bf16)** |

## The patches

| Where | Problem on sm_61 | Fix |
|---|---|---|
| GTX 1080 Ti target string (`std.mojoc` in 26.6, `max.mojoc` in 26.8) | `"+ptx50,+sm_61"`: PTX 5.0 has no `shfl.sync`, so every warp shuffle fails (`Cannot select: intrinsic %llvm.nvvm.shfl.sync.idx.i32`) | `patch_target.py`: same-length byte patch to `+ptx63` inside the package's zstd payload |
| `linalg` `matmul/gpu/__init__.mojo` (both versions, `0001`) | `comptime if use_tensor_core` instantiates the Ampere multistage GEMM (`ldmatrix`) even when the runtime shape check would skip it | `nvidia_no_tensor_cores` (compute < 7.0) keeps the tensor-core block and the cuBLAS fallback out, so the SIMT kernels run |
| `std` `_plugin/cuda/cuda_math.mojo` `_tanh` (26.8, `0002`) | `tanh.approx.{f16,f32,bf16}` are sm_75+ (PTX 7.0); ptxas rejects them, so GELU-tanh and logit softcapping don't compile | sm_52/60/61/70 use std's generic rational `_tanh_impl` in fp32 (rel err 6e-8) |
| `gemma4` Python (26.8, `0003`) | E2B/E4B don't fit an 11 GB card as published, and the E-series checkpoints carry an audio tower MAX doesn't implement | see below |

Gemma 4 patch `0003` (Python only):
* **Per-layer embeddings (PLE) on the host.** E2B's PLE table is 2.35B params (4.7 GB in bf16),
  but each token reads one 8960-wide row. `_HostScaledEmbedding` keeps the table on the CPU,
  gathers there, and sends only the rows to the GPU. Device weights go from 9.3 GB to 4.6 GB.
  `GEMMA4_PLE_ON_HOST=0` restores upstream's GPU table.
* **Memory planner.** Device weights exclude the host table. `MAX_GEMMA4_ACTIVATION_MIB`
  overrides the fixed 15 GiB activation reservation (upstream's own FIXME); 1024 is plenty at
  batch 1 / 256 tokens.
* **Weight rules.** `model.audio_tower.*` is ignored with the other audio weights MAX
  doesn't implement.

Run E-series models with `--no-prefer-module-v3`. The nightly's default ModuleV3
`gemma4_modulev3` has no PLE or double-wide MLP, so E2B fails its weight shapes there.

Kernel packages hold non-elaborated IR, so a package rebuilt from the matching source with
`mojo precompile` is a drop-in replacement. Unmodified rebuilds come out within 0.03%
(26.6 linalg) and 0.007% (26.8 linalg) of the shipped sizes. `std` rebuilds the same way.

## Use

```sh
git clone --depth 1 --branch max/v26.6.0 https://github.com/modular/modular.git upstream/modular
git clone --filter=blob:none --no-checkout https://github.com/modular/modular.git upstream/modular-main
git -C upstream/modular-main sparse-checkout set max/kernels/src Mojo/stdlib/std max/python/max/pipelines/architectures/gemma4
git -C upstream/modular-main checkout 2841b31631e7d31ae74f0ca20335ed1541cddec1
docker exec <container> bash /repo/maxpatch/install.sh            # ~35 s (26.6) / ~55 s (26.8)
docker exec <container> bash /repo/maxpatch/install.sh --revert   # stock install back
```

## Verified on the card

* 26.6 and 26.8: `shim/tests/shfl.mojo` 32/32 lanes, `graph_add.py matmul` max abs err 0,
  `ops_check.py` 10/10 (fp32), SmolLM2-135M fp32 GPU text identical to MAX-CPU over 64 tokens.
* 26.8: `ops_check.py --dtype bfloat16` 12/12 (including GELU-tanh and tanh softcap),
  `exp_check.py` (tanh rel err 6e-8), upstream's Gemma 4 attention tests (sliding, global,
  KV sharing) at E2B dimensions, and `max generate` Gemma 4 E2B bf16:
  "The capital of France is **Paris**." (same as transformers), deterministic run to run.
* `install.sh --revert` then `install.sh` reproduces each set from stock.

## Not patched yet

* Flash attention for head depth 128 (Llama-3-8B, most Qwen) routes to the tensor-core FA2
  path; it needs `flash_attention_hw_supported` gated like patch 0001. Gemma 4 (256/512) and
  depth-64 models use the naive MHA kernels here and work.
* Gemma 4 E4B: 9.4 GB of device weights even with PLE on the host, over this card's ~8.8 GiB
  pool. It needs quantized weights, which MAX's gemma4 doesn't load (bf16/fp16/NVFP4 only).
