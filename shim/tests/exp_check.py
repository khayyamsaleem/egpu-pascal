"""Transcendentals on sm_61 through MAX: values vs numpy, and which ops don't compile.
Known (MAX 26.6 + maxpatch): tanh emits tanh.approx.f32 (sm_75+, PTX 7.0) -> ptxas error."""
import numpy as np
from max.driver import Accelerator, Buffer
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph, TensorType, ops
dev = Accelerator(); sess = InferenceSession(devices=[dev])
x = np.array([-3, -1, -0.5, 0, 0.25, 0.5, 1, 2, 3, 10], dtype=np.float32)
t = TensorType(DType.float32, x.shape, device=DeviceRef.GPU())
bad = []
for name, op, ref in (("exp", ops.exp, np.exp), ("sigmoid", ops.sigmoid, lambda v: 1 / (1 + np.exp(-v))),
                      ("silu", ops.silu, lambda v: v / (1 + np.exp(-v))), ("tanh", ops.tanh, np.tanh),
                      ("sqrt_abs", lambda v: ops.sqrt(ops.abs(v)), lambda v: np.sqrt(np.abs(v))),
                      ("log1p_abs", lambda v: ops.log1p(ops.abs(v)), lambda v: np.log1p(np.abs(v)))):
    with Graph("t_" + name, input_types=[t]) as g:
        g.output(op(g.inputs[0]))
    try:
        out = sess.load(g).execute(Buffer.from_numpy(x).to(dev))[0].to_numpy()
    except Exception as e:
        msg = [l for l in str(e).splitlines() if "error" in l.lower()]
        print("%-10s does not compile: %s" % (name, (msg[1] if len(msg) > 1 else msg[0])[:110] if msg else e))
        bad.append(name); continue
    err = float(np.abs(out - ref(x)).max() / np.abs(ref(x)).max())
    print("%-10s rel err %.1e  %s" % (name, err, "ok" if err < 1e-5 else "WRONG"), flush=True)
    if err >= 1e-5: bad.append(name)
print("RESULT:", "PASS" if not bad else "FAIL (%s)" % ", ".join(bad))
