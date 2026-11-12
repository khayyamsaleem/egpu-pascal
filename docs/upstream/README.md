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
(`nvsim.py` output is in the PR): unpatched, an `architecture = 0x13` chip gets a FLR and then `KeyError: 19`;
patched, it gets `RuntimeError: unsupported chip architecture 0x13` and no reset; `0x17`/`0x19` are unchanged.
It has not been run against the card, because pointing that driver at the card is what triggers the FLR.

## Modular patches: open as drafts

Three patches to modular/modular, opened as **draft** PRs - Modular's
[AI Tool Use Policy](https://github.com/modular/modular/blob/main/AI_TOOL_POLICY.md) requires the human
author to review AI-assisted changes before they are submitted for review, and labels them with `Assisted-by:`.

| Patch | Draft PR | Diff | PR body |
|---|---|---|---|
| Pre-Turing PTX version (`+ptx50` -> `+ptx63`, 4 targets) | [#7575](https://github.com/modular/modular/pull/7575) | `pr-modular-pre-turing-ptx.diff` | `pr-modular-pre-turing-ptx.md` |
| Tensor-core block out of pre-Volta modules | [#7576](https://github.com/modular/modular/pull/7576) | `pr-modular-pre-volta-matmul-gate.diff` | `pr-modular-pre-volta-matmul-gate.md` |
| Generic `tanh` where `tanh.approx` is unavailable | [#7577](https://github.com/modular/modular/pull/7577) | `pr-modular-pre-volta-tanh-gate.diff` | `pr-modular-pre-volta-tanh-gate.md` |

Each references #7573 and carries `Assisted-by: Hermes Agent (AI); reviewed by the author`. To move one
forward: read the diff, then "Ready for review".

Verified remote state (2026-11-12): all three `isDraft=true`, `fix-*` -> `main`, one file each - #7575
`+4/-4` (4 `+ptx63`, no `+ptx50` left), #7576 `+8/-0` (`nvidia_no_tensor_cores` in both the guard and the
condition), #7577 `+12/-0` (`_tanh_impl` import + call).

`make_diffs.py` regenerates the three diffs from current `main` (it fetches each file, asserts the anchors,
and writes the diff); `open_modular_prs.py` submits them (fork -> branch -> Contents API commit -> draft PR)
and is re-runnable.

Not being sent: `maxpatch`'s in-place byte patch of the shipped `.mojoc` packages (the general fix is #7575),
the libcuda/NVML/cudart/cublas shims, `sleepguard`, and `pascal/` itself - see `../../README.md` for why each
of those is local.
