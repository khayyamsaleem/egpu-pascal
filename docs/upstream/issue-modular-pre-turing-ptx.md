### Summary

In `max/mojo/max/_plugin/cuda/targets.mojo`, every pre-Turing NVIDIA target ships `features = "+ptx50,+sm_XX"`:

| target | arch | line |
|---|---|---|
| `_gtx970_target` | `+ptx50,+sm_52` | ~859 |
| `_teslap100_target` | `+ptx50,+sm_60` | ~895 |
| `_gtx1080ti_target` | `+ptx50,+sm_61` | ~782 |
| `_gtx1060_target` | `+ptx50,+sm_61` | ~818 |

Every Turing-or-newer entry in the same table uses `+ptx63` or newer. PTX ISA 5.0 predates `shfl.sync` (added in PTX 6.0), so any kernel that contains a warp shuffle - reductions, matmul, softmax, i.e. most of the interesting ones - cannot be compiled for those targets:

```
LLVM ERROR: Cannot select: intrinsic %llvm.nvvm.shfl.sync.idx.i32
```

This is the first hard blocker we hit running MAX on a GTX 1080 Ti: the target exists, is selected for the card, and cannot compile MAX's own kernels.

### Description

Expected: a GPU listed in the target table can build the kernels the framework needs (a warp-shuffle reduction is not an exotic feature on sm_61, which the hardware supports natively).

Observed: with `modular 26.6.0` the compile of the first graph fails in codegen with the `shfl.sync` error above. The cause is the PTX version in the target string, not the architecture: `sm_61` supports PTX 6.x, the table's own `sm_75`/`sm_86` entries use `+ptx63`, and `ptxas` accepts `.version 6.3` for `sm_52`/`sm_60`/`sm_61`/`sm_70` (checked with CUDA 12.4 and 12.8 ptxas).

Minimal reproduction without a GPU - five lines of PTX, one `shfl.sync`:

```sh
cat > shfl.ptx <<'EOF'
.version 6.3
.target sm_61
.address_size 64
.visible .entry k() { .reg .b32 a,b,c,d; .reg .pred q; mov.b32 a,1; mov.b32 b,0; mov.b32 c,31;
                      shfl.sync.idx.b32 d|q, a, b, c, 0xffffffff; ret; }
EOF
for v in 5.0 6.3; do for a in sm_52 sm_60 sm_61 sm_70; do
  sed -e "s/^.version 6.3/.version $v/" -e "s/.target sm_61/.target $a/" shfl.ptx > x.ptx
  ptxas -arch=$a x.ptx -o x.cubin && echo "PTX $v + $a : OK" || echo "PTX $v + $a : FAIL"
done; done
```

```
PTX 5.0 + sm_52/sm_60/sm_61/sm_70 : FAIL  "Feature 'shfl.sync' requires PTX ISA .version 6.0 or later"
PTX 6.3 + sm_52/sm_60/sm_61/sm_70 : OK
```

What worked on the card: patching the target string in the shipped package in place (`std.mojoc` in 26.6, `max.mojoc` in the 26.8 nightly - the replacement string has the same length, so only the zstd frame changes). With `+ptx63` in place, MAX 26.6.0 runs SmolLM2-135M fp32 with text identical to MAX-on-CPU, and the 26.8 nightly runs Gemma 4 E2B bf16 with output identical to `transformers`. Notes and the patch script: https://github.com/khayyamsaleem/egpu-pascal (blocker 12 in `RESULTS.md`).

### Environment

- `modular 26.6.0` (Mojo 1.1.0) and `modular 26.8.0.dev2026110805` (Mojo 1.3.0.dev2026110805) - both ship the same PTX 5.0 entries
- Linux aarch64 container, macOS 26.6.2 host
- GPU: NVIDIA GTX 1080 Ti (GP102, sm_61) over Thunderbolt; the sm_52/sm_60 rows are from a ptxas check only, we have no Maxwell/P100 hardware here

### Severity

Blocker for anyone using a pre-Turing target (the target cannot compile the framework's own kernels), and a one-line-per-target data fix. Related: #6737 (Pascal support request), and `_get_gtx1060_target`/`_get_gtx1050_*` came in through #5280/#5075, so this looks like the PTX version never got bumped when those targets were added.

Happy to open the PR for the one-line-per-target change - it is a same-length data edit in work that already exists. Reported with AI assistance from our session notes; the hardware results above are ours.
