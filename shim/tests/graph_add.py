"""MAX graph on the GTX 1080 Ti through the shim: elementwise first, then matmul."""
import sys, time
import numpy as np
from max.driver import Accelerator, Buffer
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph, TensorType, ops

which = sys.argv[1] if len(sys.argv) > 1 else "add"
M = 64
dev = Accelerator()
print("accelerator:", dev, flush=True)
t = TensorType(DType.float32, (M, M), device=DeviceRef.GPU())
with Graph("g_" + which, input_types=[t, t]) as g:
    a, b = g.inputs
    if which == "add":
        g.output(a * 2.0 + b)
    else:
        g.output(ops.matmul(a, b))
sess = InferenceSession(devices=[dev])
t0 = time.time(); model = sess.load(g); print("compiled+loaded in %.1f s" % (time.time() - t0), flush=True)
x = np.arange(M * M, dtype=np.float32).reshape(M, M) / 100
y = np.random.default_rng(0).standard_normal((M, M)).astype(np.float32)
out = model.execute(Buffer.from_numpy(x).to(dev), Buffer.from_numpy(y).to(dev))[0].to_numpy()
ref = x * 2.0 + y if which == "add" else x @ y
err = float(np.abs(out - ref).max())
print("max abs err vs numpy: %g" % err)
print("RESULT:", "PASS" if err < 1e-3 * max(1.0, float(np.abs(ref).max())) else "FAIL")
