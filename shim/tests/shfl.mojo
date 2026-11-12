# Warp shuffle on sm_61: needs the PTX 6.3 target (maxpatch/patch_target.py).
# Every lane reads lane 7's value (7*3+1 = 22) and a butterfly sum of the lane ids (496).
from max.gpu import thread_idx
from max.gpu.host import DeviceContext
from max.gpu.primitives.warp import shuffle_idx, shuffle_xor

def k(o: Pointer[Int32, MutAnyOrigin]):
    var t = Int32(thread_idx.x)
    var s = t
    s += shuffle_xor(s, 16)
    s += shuffle_xor(s, 8)
    s += shuffle_xor(s, 4)
    s += shuffle_xor(s, 2)
    s += shuffle_xor(s, 1)
    o[2 * Int(thread_idx.x)] = shuffle_idx(t * 3 + 1, 7)
    o[2 * Int(thread_idx.x) + 1] = s

def main() raises:
    var ctx = DeviceContext()
    var o = ctx.enqueue_create_buffer[DType.int32](64)
    ctx.enqueue_function[k](o, grid_dim=1, block_dim=32)
    var ok = 0
    with o.map_to_host() as h:
        for i in range(32):
            if h[2 * i] == 22 and h[2 * i + 1] == 496:
                ok += 1
        print("lane 5:", h[10], h[11])
    print("correct lanes:", ok, "of 32")
    print("RESULT:", "PASS" if ok == 32 else "FAIL")
