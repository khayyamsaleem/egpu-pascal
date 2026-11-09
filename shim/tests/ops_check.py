"""Op-level numerics on the GTX 1080 Ti through MAX: each op the SmolLM2 graph uses, at the
model's shapes (hidden 576, MLP 1536, 35-token prefill and 1-token decode), against numpy.
Usage (container, shim env set): python ops_check.py [name ...]
"""
import sys, time
import numpy as np
from max.driver import Accelerator, Buffer
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph, TensorType, ops

rng = np.random.default_rng(0)
dev = Accelerator()
sess = InferenceSession(devices=[dev])
GPU = DeviceRef.GPU()


def f32(*shape, scale=1.0):
    return (rng.standard_normal(shape) * scale).astype(np.float32)


def run(name, inputs, build, ref, tol=1e-4):
    types = [TensorType(DType.from_numpy(x.dtype), x.shape, device=GPU) for x in inputs]
    with Graph(name, input_types=types) as g:
        g.output(build(*g.inputs))
    t0 = time.time()
    model = sess.load(g)
    t1 = time.time()
    out = model.execute(*[Buffer.from_numpy(x).to(dev) for x in inputs])[0].to_numpy()
    want = ref(*inputs)
    if out.dtype.kind in "iu":
        err = float((out != want).mean())
        ok = err == 0
    else:
        err = float(np.abs(out.astype(np.float64) - want).max() / max(1e-6, np.abs(want).max()))
        ok = np.isfinite(out).all() and err < tol
    print("%-26s %-22s rel err %.2e  (compile %.1fs, run %.2fs)  %s"
          % (name, str(out.shape), err, t1 - t0, time.time() - t1, "ok" if ok else "WRONG"), flush=True)
    return ok


H, I, T = 576, 1536, 35
tests = {
    "gather": lambda: run("gather", [f32(4096, H), rng.integers(0, 4096, T).astype(np.int64)],
                          lambda w, i: ops.gather(w, i, axis=0), lambda w, i: w[i]),
    "matmul_nt_576": lambda: run("matmul_nt_576", [f32(T, H), f32(H, H)],
                                 lambda x, w: ops.matmul(x, ops.transpose(w, 0, 1)), lambda x, w: x @ w.T),
    "matmul_nt_1536": lambda: run("matmul_nt_1536", [f32(T, H), f32(I, H)],
                                  lambda x, w: ops.matmul(x, ops.transpose(w, 0, 1)), lambda x, w: x @ w.T),
    "matmul_nt_down": lambda: run("matmul_nt_down", [f32(T, I), f32(H, I)],
                                  lambda x, w: ops.matmul(x, ops.transpose(w, 0, 1)), lambda x, w: x @ w.T),
    "gemv_nt_576": lambda: run("gemv_nt_576", [f32(1, H), f32(H, H)],
                               lambda x, w: ops.matmul(x, ops.transpose(w, 0, 1)), lambda x, w: x @ w.T),
    "gemv_nt_4096": lambda: run("gemv_nt_4096", [f32(1, H), f32(4096, H)],
                                lambda x, w: ops.matmul(x, ops.transpose(w, 0, 1)), lambda x, w: x @ w.T),
    "rms_norm": lambda: run("rms_norm", [f32(T, H), f32(H)],
                            lambda x, w: ops.rms_norm(x, w, 1e-5),
                            lambda x, w: x / np.sqrt((x.astype(np.float64) ** 2).mean(-1, keepdims=True) + 1e-5) * w),
    "softmax": lambda: run("softmax", [f32(9, T, T, scale=3)], lambda x: ops.softmax(x),
                           lambda x: np.exp(x - x.max(-1, keepdims=True)) / np.exp(x - x.max(-1, keepdims=True)).sum(-1, keepdims=True)),
    "silu_mul": lambda: run("silu_mul", [f32(T, I), f32(T, I)], lambda a, b: ops.silu(a) * b,
                            lambda a, b: a / (1 + np.exp(-a.astype(np.float64))) * b),
    "argmax": lambda: run("argmax", [f32(1, 49152)], lambda x: ops.argmax(x, axis=-1),
                          lambda x: x.argmax(-1)[:, None]),
}
names = sys.argv[1:] or list(tests)
res = {n: tests[n]() for n in names}
print("RESULT:", "PASS" if all(res.values()) else "FAIL (%s)" % ", ".join(n for n, v in res.items() if not v))
