from max.gpu import thread_idx, block_idx, block_dim
from max.gpu.host import DeviceContext

comptime N = 4096

def vadd(a: UnsafePointer[Float32, MutAnyOrigin], b: UnsafePointer[Float32, MutAnyOrigin],
        c: UnsafePointer[Float32, MutAnyOrigin], n: Int32):
    var i = Int(block_idx.x * block_dim.x + thread_idx.x)
    if i < Int(n):
        c[i] = a[i] + b[i]

def main() raises:
    var ctx = DeviceContext()
    var a = ctx.enqueue_create_buffer[DType.float32](N)
    var b = ctx.enqueue_create_buffer[DType.float32](N)
    var c = ctx.enqueue_create_buffer[DType.float32](N)
    with a.map_to_host() as ha, b.map_to_host() as hb:
        for i in range(N):
            ha[i] = Float32(i)
            hb[i] = Float32(2 * i)
    ctx.enqueue_function[vadd](a, b, c, Int32(N), grid_dim=N // 128, block_dim=128)
    with c.map_to_host() as hc:
        print("c0 c1 c2 c3:", hc[0], hc[1], hc[2], hc[3])
        print("c last:", hc[N - 1])
