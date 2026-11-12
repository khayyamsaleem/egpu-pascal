# Weight-only int8 linear layer for MAX on sm_61 (GTX 1080 Ti): dst = x @ dequant(w).T
#
#   x     [M, K] bfloat16      activations
#   w     [N, K] int8          weights, symmetric per output channel
#   scale [N]    float32       w_real[n, :] = w[n, :] * scale[n]
#   dst   [M, N] bfloat16
#
# Decode (M <= 4): one warp per 2 output rows; each lane streams 16 weight bytes per step (one
# 128-bit load) and the matching activations, fp32 accumulation, warp-shuffle reduction. The
# weights are read once at 1 byte per parameter, half of bf16, which is what decode is bound by.
# Prefill (M > 4): 64x64 output tiles, 32-deep K slices dequantized into shared memory, 4x4
# outputs per thread (MAX's own fallback on pre-Volta GPUs is the naive matmul).
# K must be a multiple of 32 (all transformer projection widths are).
import extensibility
from extensibility import InputTensor, OutputTensor
from max.gpu import block_idx, thread_idx
from max.gpu.host import DeviceContext
from max.gpu.primitives.warp import shuffle_xor
from max.gpu.sync import barrier
from std.math import ceildiv
from std.memory import unsafe_stack_allocation

comptime WARPS = 8
comptime ROWS = 2
comptime BM = 64
comptime BN = 64
comptime BK = 32


def w8_gemv[MT: Int](
    dst: Pointer[BFloat16, MutAnyOrigin], x: Pointer[BFloat16, MutAnyOrigin],
    w: Pointer[Int8, MutAnyOrigin], scale: Pointer[Float32, MutAnyOrigin], n_dev: Int32, k_dev: Int32,
):
    var n = Int(n_dev)
    var k = Int(k_dev)
    var lane = Int(thread_idx.x) % 32
    var row0 = (Int(block_idx.x) * WARPS + Int(thread_idx.x) // 32) * ROWS
    var acc = SIMD[DType.float32, ROWS * MT](0)
    for k0 in range(0, k, 512):
        var kk = k0 + lane * 16
        if kk < k:
            comptime for r in range(ROWS):
                if row0 + r < n:
                    var wv = (w + (row0 + r) * k + kk).unsafe_load[width=16, alignment=16]().cast[DType.float32]()
                    comptime for m in range(MT):
                        var xv = (x + m * k + kk).unsafe_load[width=16, alignment=32]().cast[DType.float32]()
                        acc[r * MT + m] += (xv * wv).reduce_add()
    comptime for i in range(ROWS * MT):
        var v = acc[i]
        v += shuffle_xor(v, 16)
        v += shuffle_xor(v, 8)
        v += shuffle_xor(v, 4)
        v += shuffle_xor(v, 2)
        v += shuffle_xor(v, 1)
        acc[i] = v
    if lane == 0:
        comptime for r in range(ROWS):
            if row0 + r < n:
                var sc = scale[row0 + r]
                comptime for m in range(MT):
                    dst[m * n + row0 + r] = (acc[r * MT + m] * sc).cast[DType.bfloat16]()


def w8_gemm(
    dst: Pointer[BFloat16, MutAnyOrigin], x: Pointer[BFloat16, MutAnyOrigin],
    w: Pointer[Int8, MutAnyOrigin], scale: Pointer[Float32, MutAnyOrigin],
    m_dev: Int32, n_dev: Int32, k_dev: Int32,
):
    var m_ = Int(m_dev)
    var n = Int(n_dev)
    var k = Int(k_dev)
    var xs = unsafe_stack_allocation[BK * BM, DType.float32, address_space=.SHARED]()   # [kk][m]
    var ws = unsafe_stack_allocation[BK * BN, DType.float32, address_space=.SHARED]()   # [kk][n]
    var t = Int(thread_idx.x)
    var tx = t % 16
    var ty = t // 16
    var m0 = Int(block_idx.y) * BM
    var n0 = Int(block_idx.x) * BN
    var lr = t // 4                     # tile row this thread loads (m for x, n for w)
    var lk = (t % 4) * 8                # its 8-wide K offset
    var acc = SIMD[DType.float32, 16](0)
    for k0 in range(0, k, BK):
        var xv = SIMD[DType.float32, 8](0)
        if m0 + lr < m_:
            xv = (x + (m0 + lr) * k + k0 + lk).unsafe_load[width=8, alignment=16]().cast[DType.float32]()
        var wv = SIMD[DType.float32, 8](0)
        if n0 + lr < n:
            wv = (w + (n0 + lr) * k + k0 + lk).unsafe_load[width=8, alignment=8]().cast[DType.float32]()
        comptime for j in range(8):
            xs[(lk + j) * BM + lr] = xv[j]
            ws[(lk + j) * BN + lr] = wv[j]
        barrier()
        for kk in range(BK):
            var a = SIMD[DType.float32, 4](0)
            var b = SIMD[DType.float32, 4](0)
            comptime for i in range(4):
                a[i] = xs[kk * BM + ty + 16 * i]
                b[i] = ws[kk * BN + tx + 16 * i]
            comptime for i in range(4):
                comptime for j in range(4):
                    acc[i * 4 + j] += a[i] * b[j]
        barrier()
    comptime for i in range(4):
        comptime for j in range(4):
            var r = m0 + ty + 16 * i
            var c = n0 + tx + 16 * j
            if r < m_ and c < n:
                dst[r * n + c] = (acc[i * 4 + j] * scale[c]).cast[DType.bfloat16]()


def enqueue_w8_linear(
    ctx: DeviceContext, dst: Pointer[BFloat16, MutAnyOrigin], x: Pointer[BFloat16, MutAnyOrigin],
    w: Pointer[Int8, MutAnyOrigin], scale: Pointer[Float32, MutAnyOrigin], m: Int, n: Int, k: Int,
) raises:
    if k % BK != 0:
        raise Error("w8_linear: K must be a multiple of 32, got ", k)
    var gv = ceildiv(n, WARPS * ROWS)
    if m == 1:
        ctx.enqueue_function[w8_gemv[1]](dst, x, w, scale, Int32(n), Int32(k), grid_dim=gv, block_dim=WARPS * 32)
    elif m == 2:
        ctx.enqueue_function[w8_gemv[2]](dst, x, w, scale, Int32(n), Int32(k), grid_dim=gv, block_dim=WARPS * 32)
    elif m <= 4:
        for m0 in range(0, m, 2):          # pairs of rows (an odd tail runs as M=1)
            var mt = 2 if m - m0 >= 2 else 1
            var xo = x + m0 * k
            var oo = dst + m0 * n
            if mt == 2:
                ctx.enqueue_function[w8_gemv[2]](oo, xo, w, scale, Int32(n), Int32(k), grid_dim=gv, block_dim=WARPS * 32)
            else:
                ctx.enqueue_function[w8_gemv[1]](oo, xo, w, scale, Int32(n), Int32(k), grid_dim=gv, block_dim=WARPS * 32)
    else:
        ctx.enqueue_function[w8_gemm](dst, x, w, scale, Int32(m), Int32(n), Int32(k),
                                      grid_dim=(ceildiv(n, BN), ceildiv(m, BM)), block_dim=256)


@extensibility.register("w8_linear")
struct W8Linear:
    @staticmethod
    def execute(
        output: OutputTensor[dtype=DType.bfloat16, rank=2, ...],
        x: InputTensor[dtype=DType.bfloat16, rank=2, ...],
        w: InputTensor[dtype=DType.int8, rank=2, ...],
        scale: InputTensor[dtype=DType.float32, rank=1, ...],
        ctx: DeviceContext,
    ) raises:
        comptime if ctx.target.is_gpu():
            enqueue_w8_linear(ctx, output.unsafe_ptr(), x.unsafe_ptr(), w.unsafe_ptr(), scale.unsafe_ptr(),
                              x.dim_size(0), w.dim_size(0), x.dim_size(1))
        else:
            raise Error("w8_linear: GPU only")
