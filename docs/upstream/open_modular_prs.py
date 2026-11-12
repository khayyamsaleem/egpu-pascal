"""Submit the three Modular patches as DRAFT pull requests.

For each patch: fork (once) -> create a branch off modular/main on the fork -> commit the
patched file through the Contents API -> open a draft PR against modular/modular.

Re-runnable: an existing branch is reset to main first. Idempotent-ish, so a partial run can
be repeated.
"""
import base64, json, os, subprocess, sys

OUT = os.path.expanduser("~/egpu-pascal/docs/upstream")
FORK = "khayyamsaleem/modular"
UPSTREAM = "modular/modular"
FILES = {
    "targets": "max/mojo/max/_plugin/cuda/targets.mojo",
    "linalg": "max/kernels/src/linalg/matmul/gpu/__init__.mojo",
    "cuda_math": "Mojo/stdlib/std/_plugin/cuda/cuda_math.mojo",
}

def sh(*args, **kw):
    return subprocess.run(args, capture_output=True, text=True, **kw)

def gh(*args):
    r = sh("gh", *args)
    if r.returncode:
        sys.exit("gh %s failed:\n%s%s" % (" ".join(args), r.stdout[-500:], r.stderr[-500:]))
    return r.stdout.strip()

def api(*args, input=None):
    cmd = ["gh", "api", *args]
    if input is not None:
        with open("/tmp/_payload.json", "w") as f:
            json.dump(input, f)
        cmd += ["--input", "/tmp/_payload.json"]
    r = sh(*cmd)
    if r.returncode:
        sys.exit("gh api %s failed:\n%s%s" % (" ".join(args), r.stdout[-800:], r.stderr[-800:]))
    return json.loads(r.stdout) if r.stdout.strip().startswith(("{", "[")) else r.stdout.strip()

# fork once
if sh("gh", "api", "repos/" + FORK).returncode:
    r = sh("gh", "repo", "fork", UPSTREAM, "--clone=false")
    print("fork: " + (r.stdout + r.stderr).strip().splitlines()[-1])
else:
    print("fork %s already exists" % FORK)
assert sh("gh", "api", "repos/" + FORK).returncode == 0, "fork was not created"

main_sha = api("repos/%s/commits/main" % UPSTREAM)["sha"]
print("modular/main =", main_sha)

# --- the three patches: (branch, key, title, commit message, pr body, patched file path) ---
def edits_targets(src):
    n = src.count("`features = \"+ptx50,+sm_61\", `,")
    assert n == 2, n
    src = src.replace("`features = \"+ptx50,+sm_61\", `,", "`features = \"+ptx63,+sm_61\", `,")
    for arch in ("sm_52", "sm_60"):
        old = "`features = \"+ptx50,+%s\", `," % arch
        assert src.count(old) == 1, (arch, src.count(old))
        src = src.replace(old, "`features = \"+ptx63,+%s\", `," % arch)
    assert "+ptx50" not in src
    return src

def edits_linalg(src):
    old = '    ), "use_tf32=False is only implemented for the SM100 matmul dispatch"\n'
    assert src.count(old) == 1
    src = src.replace(old, old + '''
    # Pre-Volta NVIDIA (Maxwell sm_5x, Pascal sm_6x) has no tensor cores and no
    # ldmatrix/mma.sync/cp.async, so the tensor-core kernels below cannot be elaborated for those
    # targets even though the runtime shape check would skip them. Keep them out of the module.
    comptime nvidia_no_tensor_cores = (
        ctx.target.is_nvidia_gpu() and ctx.default_device_info.compute < 7.0
    )
''')
    old = "        and use_tensor_core\n        and has_static_NK\n    ):"
    assert src.count(old) == 1
    return src.replace(old, "        and use_tensor_core\n        and has_static_NK\n        and not nvidia_no_tensor_cores\n    ):")

def edits_cuda_math(src):
    old = "    _llvm_unary_fn,\n"
    assert src.count(old) == 1
    src = src.replace(old, "    _llvm_unary_fn,\n    _tanh_impl,\n")
    old = "def _tanh(x: SIMD) -> type_of(x):\n"
    assert src.count(old) == 1
    return src.replace(old, old + '''    # tanh.approx.{f16,f32,bf16} need sm_75+ (PTX ISA 7.0): ptxas rejects them for
    # Maxwell/Pascal/Volta ("Feature 'tanh' requires .target sm_75 or higher"). Use the generic
    # rational approximation there, in fp32.
    comptime target = CompilationTarget.current()
    comptime if (
        target.is_nvidia_gpu["sm_52"]()
        or target.is_nvidia_gpu["sm_60"]()
        or target.is_nvidia_gpu["sm_61"]()
        or target.is_nvidia_gpu["sm_70"]()
    ):
        return _tanh_impl(x.cast[.float32]()).cast[x.dtype]()
''')

PATCHES = [
    ("fix-pre-turing-target-ptx", "targets", edits_targets,
     "cuda: fix the PTX ISA version of the pre-Turing GPU targets (+ptx50 -> +ptx63)",
     """cuda: fix the PTX ISA version of the pre-Turing GPU targets

The GTX 970 (sm_52), Tesla P100 (sm_60), GTX 1060 and GTX 1080 Ti (sm_61) targets ship
features = "+ptx50,+sm_XX", while every Turing-or-newer entry uses "+ptx63" or newer.
PTX ISA 5.0 predates shfl.sync, so no kernel containing a warp shuffle can be compiled
for them: LLVM cannot select llvm.nvvm.shfl.sync.*, and the failure is at codegen, so
the target cannot build the framework's own kernels.

sm_52/sm_60/sm_61 support PTX 6.x, and ptxas accepts ".version 6.3" for all four, so
bump them to the version the rest of the table already uses.

Fixes #7573

Verified on a GTX 1080 Ti with modular 26.6.0 and the 26.8 nightly, with the same change
applied to the shipped package: SmolLM2-135M fp32 and Gemma 4 E2B bf16 produce output
identical to the CPU/transformers reference. sm_52/sm_60 were checked with ptxas only.

Assisted-by: Hermes Agent (AI); reviewed by the author""",
     "pr-modular-pre-turing-ptx.md"),
    ("fix-pre-volta-matmul-tensor-cores", "linalg", edits_linalg,
     "linalg: keep the tensor-core matmul kernels out of pre-Volta modules",
     """linalg: keep the tensor-core matmul kernels out of pre-Volta modules

On Maxwell (sm_5x) and Pascal (sm_6x) the tensor-core block cannot be elaborated at
all: it uses ldmatrix/mma.sync/cp.async, which those targets do not have, so codegen
fails with "Cannot select: intrinsic %llvm.nvvm.ldmatrix..." even though the runtime
shape check would have skipped the tensor-core path. use_tensor_core is a compile-time
flag rather than a property of the target, so the block is instantiated regardless.

Gate it on a comptime nvidia_no_tensor_cores (NVIDIA target, compute < 7.0) so those
targets elaborate the SIMT kernels below.

Verified on a GTX 1080 Ti (26.8 nightly): ops_check.py 10/10 fp32 and 12/12 bf16, and
Gemma 4 E2B bf16 output identical to transformers. Rebased onto main - the anchors were
checked, but the verified build was the pinned nightly revision.

Assisted-by: Hermes Agent (AI); reviewed by the author""",
     "pr-modular-pre-volta-matmul-gate.md"),
    ("fix-pre-turing-tanh", "cuda_math", edits_cuda_math,
     "std: use the generic tanh on GPU targets that have no tanh.approx",
     """std: use the generic tanh on GPU targets that have no tanh.approx

tanh.approx.{f16,f32,bf16} are sm_75+ (PTX ISA 7.0); ptxas rejects them for Maxwell,
Pascal and Volta ("Feature 'tanh' requires .target sm_75 or higher"), so GELU-tanh and
logit-softcap models do not compile for those targets.

Use std's generic rational _tanh_impl in fp32 for sm_52/sm_60/sm_61/sm_70 and cast back
to the input dtype; sm_75+ keeps the intrinsics. rel err 6e-8 measured on the card, and
ops_check.py --dtype bfloat16 12/12 with Gemma 4 E2B output identical to transformers.

This file is identical on main and on the nightly revision the change was verified
against, so this diff is exactly what ran.

Assisted-by: Hermes Agent (AI); reviewed by the author""",
     "pr-modular-pre-volta-tanh-gate.md"),
]

urls = []
for branch, key, fn, title, message, body in PATCHES:
    path = FILES[key]
    src = subprocess.run(["curl", "-sL", "https://raw.githubusercontent.com/%s/main/%s" % (UPSTREAM, path)],
                         capture_output=True, text=True).stdout
    assert len(src) > 2000, "fetch failed for " + path
    new = fn(src)

    # branch on the fork, reset to main so re-runs are safe
    if sh("gh", "api", "repos/%s/git/ref/heads/%s" % (FORK, branch)).returncode:
        gh("api", "-X", "POST", "repos/%s/git/refs" % FORK, "-f", "ref=refs/heads/" + branch, "-f", "sha=" + main_sha)
    else:
        gh("api", "-X", "PATCH", "repos/%s/git/refs/heads/%s" % (FORK, branch), "-f", "sha=" + main_sha, "-F", "force=true")
        print("  (branch existed, reset to main)")
    blob = gh("api", "repos/%s/contents/%s?ref=%s" % (FORK, path, branch), "--jq", ".sha")
    res = api("-X", "PUT", "repos/%s/contents/%s" % (FORK, path),
              input={"message": message, "content": base64.b64encode(new.encode()).decode(),
                     "branch": branch, "sha": blob})
    print("  %-34s %s" % (branch, res["commit"]["sha"] if isinstance(res, dict) and "commit" in res else res))

    pr = gh("pr", "create", "--repo", UPSTREAM, "--head", "%s:%s" % (FORK.split("/")[0], branch),
            "--draft", "--title", title, "--body-file", os.path.join(OUT, body))
    print("  PR: %s" % pr)
    urls.append(pr)

print("\n%d draft PRs:" % len(urls))
for u in urls:
    print("  " + u)
