# Upstream contributions from this project

What has been sent to the upstream projects, and what is prepared but not yet submitted.
Drafts live here so the text can be reviewed before it goes anywhere.

## Sent

| Item | Venue | State |
|---|---|---|
| [Pre-Turing targets pin PTX ISA 5.0 (`+ptx50,+sm_52/60/61`), so nothing that uses `shfl.sync` compiles for them](issue-modular-pre-turing-ptx.md) | modular/modular | [issue #7573](https://github.com/modular/modular/issues/7573), open |
| [PCIe FLR issued to unsupported (pre-Ampere) chips: the WPR2 probe reads a Turing-only offset that returns PRI poison](issue-tinygrad-flr-on-unsupported-chips.md) | tinygrad/tinygrad | [PR #19323](https://github.com/tinygrad/tinygrad/pull/19323) - the repo has issues disabled, so the PR carries the write-up |
| [Kernel panic on wake from sleep while a TinyGPU client holds the card](discussion-tinygrad-sleep-panic.md) | tinygrad/tinygrad | [discussion #19322](https://github.com/tinygrad/tinygrad/discussions/19322); tinygrad/tinygpu_releases is the tracker for the app/dext |

The FLR fix in PR #19323 was verified with a simulation against tinygrad's own autogen register maps
(`nvsim.py` output in the PR): unpatched, an `architecture = 0x13` chip gets a FLR and then `KeyError: 19`;
patched, it gets `RuntimeError: unsupported chip architecture 0x13` and no reset; `0x17`/`0x19` are unchanged.
It has not been run against the card, because pointing that driver at the card is what triggers the FLR.

## Prepared, not submitted

Three patches to modular/modular, held for review before submission - Modular's
[AI Tool Use Policy](https://github.com/modular/modular/blob/main/AI_TOOL_POLICY.md) requires the human
author to review AI-assisted changes before a PR is opened, and labels them with `Assisted-by:`.

| Patch | Diff | PR body | Notes |
|---|---|---|---|
| Pre-Turing PTX version (`+ptx50` -> `+ptx63`, 4 targets) | `pr-modular-pre-turing-ptx.diff` | `pr-modular-pre-turing-ptx.md` | fixes issue #7573; sm_61 verified on the card, sm_52/sm_60 ptxas-only |
| Tensor-core block out of pre-Volta modules | `pr-modular-pre-volta-matmul-gate.diff` | `pr-modular-pre-volta-matmul-gate.md` | the local patch (0001) minus our shim-specific cuBLAS hunk; verified on the pinned nightly, rebased onto `main` |
| Generic `tanh` where `tanh.approx` is unavailable | `pr-modular-pre-volta-tanh-gate.diff` | `pr-modular-pre-volta-tanh-gate.md` | the file is identical on `main` and on the nightly it was verified against, so this diff is exactly what ran |

`make_diffs.py` regenerates the three diffs from current `main` (it fetches each file, asserts the
anchors, and writes the diff). All three apply cleanly to `main` as of 2026-11-12.

Not being sent: `maxpatch`'s in-place byte patch of the shipped `.mojoc` packages (the general fix is the
first PR), the libcuda/NVML/cudart/cublas shims, `sleepguard`, and `pascal/` itself - see
`../../README.md` for why each of those is local.
