"""Build the three upstream patches against current modular/modular main.

Fetches the files fresh, applies the verified local fixes (minus our local-only hunks),
and writes unified diffs + a report. Nothing is committed or pushed.
"""
import subprocess, sys, os

OUT = os.path.expanduser("~/egpu-pascal/docs/upstream")
os.makedirs(OUT, exist_ok=True)

FILES = {
    "targets": "max/mojo/max/_plugin/cuda/targets.mojo",
    "linalg": "max/kernels/src/linalg/matmul/gpu/__init__.mojo",
    "cuda_math": "Mojo/stdlib/std/_plugin/cuda/cuda_math.mojo",
}

def fetch(key, path):
    dest = "/tmp/main_%s.mojo" % key
    url = "https://raw.githubusercontent.com/modular/modular/main/" + path
    r = subprocess.run(["curl", "-sL", "-o", dest, "-w", "%{http_code}", url], capture_output=True, text=True)
    src = open(dest).read()
    if r.stdout.strip() != "200" or len(src) < 2000:
        sys.exit("fetch failed for %s: %s" % (path, r.stdout))
    return src

def sub(src, old, new, count):
    n = src.count(old)
    assert n == count, "expected %d of %r, found %d" % (count, old[:60], n)
    return src.replace(old, new)

def diff(path, old, new):
    a, b = "/tmp/_a", "/tmp/_b"
    open(a, "w").write(old)
    open(b, "w").write(new)
    d = subprocess.run(["diff", "-u", a, b], capture_output=True, text=True).stdout
    body = [l for l in d.splitlines() if not l.startswith("--- ") and not l.startswith("+++ ")]
    return "diff --git a/%s b/%s\n--- a/%s\n+++ b/%s\n" % (path, path, path, path) + "\n".join(body) + "\n"

report = []

# --- 1. targets.mojo: pre-Turing PTX ISA version -------------------------------
old = fetch("targets", FILES["targets"])
new = sub(old, "`features = \"+ptx50,+sm_61\", `,", "`features = \"+ptx63,+sm_61\", `,", 2)
new = sub(new, "`features = \"+ptx50,+sm_52\", `,", "`features = \"+ptx63,+sm_52\", `,", 1)
new = sub(new, "`features = \"+ptx50,+sm_60\", `,", "`features = \"+ptx63,+sm_60\", `,", 1)
assert "+ptx50" not in new
open(OUT + "/pr-modular-pre-turing-ptx.diff", "w").write(diff(FILES["targets"], old, new))
report.append(("targets.mojo: +ptx50 -> +ptx63 on sm_52/sm_60/sm_61 x2", 4))

# --- 2. linalg: keep tensor-core kernels out of pre-Volta modules --------------
old = fetch("linalg", FILES["linalg"])
guard = '''    ), "use_tf32=False is only implemented for the SM100 matmul dispatch"

    # Pre-Volta NVIDIA (Maxwell sm_5x, Pascal sm_6x) has no tensor cores and no
    # ldmatrix/mma.sync/cp.async, so the tensor-core kernels below cannot be elaborated for those
    # targets even though the runtime shape check would skip them. Keep them out of the module.
    comptime nvidia_no_tensor_cores = (
        ctx.target.is_nvidia_gpu() and ctx.default_device_info.compute < 7.0
    )
'''
new = sub(old, '''    ), "use_tf32=False is only implemented for the SM100 matmul dispatch"
''', guard, 1)
new = sub(new, '''        and use_tensor_core
        and has_static_NK
    ):''', '''        and use_tensor_core
        and has_static_NK
        and not nvidia_no_tensor_cores
    ):''', 1)
open(OUT + "/pr-modular-pre-volta-matmul-gate.diff", "w").write(diff(FILES["linalg"], old, new))
report.append(("linalg matmul: nvidia_no_tensor_cores (compute < 7.0) gate", 2))

# --- 3. cuda_math: generic tanh where tanh.approx is unavailable ---------------
old = fetch("cuda_math", FILES["cuda_math"])
new = sub(old, "    _llvm_unary_fn,\n", "    _llvm_unary_fn,\n    _tanh_impl,\n", 1)
new = sub(new, "def _tanh(x: SIMD) -> type_of(x):\n", '''def _tanh(x: SIMD) -> type_of(x):
    # tanh.approx.{f16,f32,bf16} need sm_75+ (PTX ISA 7.0): ptxas rejects them for
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
''', 1)
open(OUT + "/pr-modular-pre-volta-tanh-gate.diff", "w").write(diff(FILES["cuda_math"], old, new))
report.append(("std cuda_math: generic _tanh_impl on sm_52/60/61/70", 2))

print("diffs written to %s:" % OUT)
for name, hunks in report:
    print("  %-60s (%d edits)" % (name, hunks))
for f in sorted(os.listdir(OUT)):
    if f.endswith(".diff"):
        p = os.path.join(OUT, f)
        print("  %-46s %5d bytes" % (f, os.path.getsize(p)))
