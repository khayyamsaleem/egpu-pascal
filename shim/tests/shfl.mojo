from max.gpu import thread_idx
from max.gpu.host import DeviceContext
from max.gpu.primitives.warp import shuffle_idx

def k(o: UnsafePointer[Int32, MutAnyOrigin]):
    var v = Int32(thread_idx.x)
    o[thread_idx.x] = shuffle_idx(v, 0)

def main() raises:
    var ctx = DeviceContext()
    var o = ctx.enqueue_create_buffer[DType.int32](32)
    ctx.enqueue_function[k](o, grid_dim=1, block_dim=32)
    with o.map_to_host() as h:
        print("lane 5 got", h[5])
