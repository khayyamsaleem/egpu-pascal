"""The gemma4 host-PLE lookup pattern on the card: GPU indices -> CPU, gather from a CPU-resident
bf16 weight, transfer the rows to the GPU, scale there. Compared with numpy."""
import numpy as np
from max.driver import Accelerator, Buffer
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph, TensorType, Weight, ops

V, D, T = 4096, 8960, 14          # E2B: packed per-layer width 35 * 256
dev = Accelerator(); sess = InferenceSession(devices=[dev])
rng = np.random.default_rng(0)
table = rng.standard_normal((V, D)).astype(np.float32)
ids = rng.integers(0, V, T).astype(np.int64)
with Graph("host_embed", input_types=[TensorType(DType.int64, [T], device=DeviceRef.GPU())]) as g:
    w = Weight("weight", DType.float32, [V, D], device=DeviceRef.CPU())
    idx = ops.transfer_to(g.inputs[0].tensor, DeviceRef.CPU())
    wb = ops.cast(w, DType.bfloat16)
    rows = ops.gather(wb, idx, axis=0)
    out = ops.transfer_to(rows, DeviceRef.GPU()) * 16.0
    g.output(ops.cast(out, DType.float32))
model = sess.load(g, weights_registry={"weight": table})
got = model.execute(Buffer.from_numpy(ids).to(dev))[0].to_numpy()
want = table.astype(np.float32)[ids] * 16.0
bf = (want.view(np.uint32) & 0xffff0000).view(np.float32)   # bf16 truncation bound
err = float(np.abs(got - want).max() / np.abs(want).max())
print("host embed rows %s rel err %.2e" % (got.shape, err))
print("RESULT:", "PASS" if err < 1e-2 else "FAIL")
