"""Check and time the w8_linear custom op (maxpatch/kernels/w8) on the card, in MAX graphs.

For each (M, N, K): out = ops.custom("w8_linear", x_bf16, w_int8, scale) against numpy
(x rounded to bf16, fp32 math), and the same product as MAX's own bf16 ops.matmul for timing.
Usage (maxnightly container, shim env set): python3 w8_check.py [--reps 20]
"""
import argparse, os, time
from pathlib import Path
import numpy as np
from max.driver import Accelerator, Buffer
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph, TensorType, ops

ap = argparse.ArgumentParser()
ap.add_argument("--reps", type=int, default=20)
ap.add_argument("--shapes", default="1x2560x2560,1x10240x2560,1x2560x10240,2x2048x1536,3x1536x2048,"
                                    "16x2560x2560,100x1536x2048,300x2048x1536")
args = ap.parse_args()
PKG = Path(__file__).resolve().parent / "w8"
dev, GPU = Accelerator(), DeviceRef.GPU()
sess = InferenceSession(devices=[dev])
rng = np.random.default_rng(0)


def bf16_round(a):
    u = a.astype(np.float32).view(np.uint32)
    return ((u + 0x7fff + ((u >> 16) & 1)) & 0xffff0000).view(np.float32)


def timed(model, inputs, reps):
    model.execute(*inputs)[0].to_numpy()                 # warm-up (module load)
    t0 = time.perf_counter()
    for _ in range(reps):
        out = model.execute(*inputs)[0]
    out = out.to_numpy()
    return out, (time.perf_counter() - t0) / reps


ok_all = True
for shp in args.shapes.split(","):
    M, N, K = map(int, shp.split("x"))
    x = rng.standard_normal((M, K), dtype=np.float32)
    wf = rng.standard_normal((N, K), dtype=np.float32) * 0.05
    scale = (np.abs(wf).max(axis=1) / 127).astype(np.float32)
    w = np.clip(np.round(wf / scale[:, None]), -127, 127).astype(np.int8)
    want = bf16_round(x) @ (w.astype(np.float32) * scale[:, None]).T

    tx = TensorType(DType.float32, [M, K], GPU)
    with Graph("w8_%s" % shp, input_types=[tx, TensorType(DType.int8, [N, K], GPU), TensorType(DType.float32, [N], GPU)],
               custom_extensions=[PKG]) as g:
        xi, wi, si = g.inputs
        y = ops.custom("w8_linear", device=GPU, values=[ops.cast(xi, DType.bfloat16), wi, si],
                       out_types=[TensorType(DType.bfloat16, [M, N], GPU)])[0].tensor
        g.output(ops.cast(y, DType.float32))
    with Graph("bf16_%s" % shp, input_types=[tx, TensorType(DType.bfloat16, [N, K], GPU)]) as g2:
        xi, wb = g2.inputs
        g2.output(ops.cast(ops.matmul(ops.cast(xi, DType.bfloat16), ops.transpose(wb, 0, 1)), DType.float32))
    with Graph("cvt_%s" % shp, input_types=[TensorType(DType.float32, [N, K], GPU)]) as g3:
        g3.output(ops.cast(g3.inputs[0], DType.bfloat16))

    m8, mb, mc = sess.load(g), sess.load(g2), sess.load(g3)
    xd, wd, sd = (Buffer.from_numpy(a).to(dev) for a in (x, w, scale))
    wbd = mc.execute(Buffer.from_numpy(w.astype(np.float32) * scale[:, None]).to(dev))[0]
    out, t8 = timed(m8, [xd, wd, sd], args.reps)
    ref_out, tb = timed(mb, [xd, wbd], args.reps)
    err = float(np.abs(out - want).max() / np.abs(want).max())
    err_b = float(np.abs(ref_out - want).max() / np.abs(want).max())
    ok = np.isfinite(out).all() and err < 2e-2
    ok_all &= ok
    print("M=%-4d N=%-6d K=%-6d  w8 %8.3f ms  bf16 matmul %8.3f ms  (%.2fx)  rel err %.1e (bf16 path %.1e)  %s"
          % (M, N, K, t8 * 1e3, tb * 1e3, tb / t8, err, err_b, "ok" if ok else "WRONG"), flush=True)
print("RESULT:", "PASS" if ok_all else "FAIL")
