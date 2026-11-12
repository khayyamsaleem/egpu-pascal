// libcublas.so.12 for the GP102 shim: the GEMM entry points ggml-cuda uses, on one tiled SIMT
// kernel (64x64 tiles, 256 threads, 4x4 outputs per thread, fp32 accumulation).
//
// ggml only calls cuBLAS for float-typed weights (f32/f16/bf16 matmuls; quantized types use its
// own dp4a MMQ/MMVQ kernels on Pascal), so this is a correctness backstop, not a fast path.
// NVIDIA's libcublas needs the CUDA runtime's private driver interfaces, which our libcuda
// doesn't have. Exported under version node libcublas.so.12 (cublas.map); unimplemented
// entry points come from gen_stubs.py and return CUBLAS_STATUS_NOT_SUPPORTED.
#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

template <typename T> __device__ __forceinline__ float ld(const T *p, size_t i);
template <> __device__ __forceinline__ float ld<float>(const float *p, size_t i) { return p[i]; }
template <> __device__ __forceinline__ float ld<__half>(const __half *p, size_t i) { return __half2float(p[i]); }
template <> __device__ __forceinline__ float ld<__nv_bfloat16>(const __nv_bfloat16 *p, size_t i) { return __bfloat162float(p[i]); }
template <typename T> __device__ __forceinline__ void st(T *p, size_t i, float v);
template <> __device__ __forceinline__ void st<float>(float *p, size_t i, float v) { p[i] = v; }
template <> __device__ __forceinline__ void st<__half>(__half *p, size_t i, float v) { p[i] = __float2half(v); }
template <> __device__ __forceinline__ void st<__nv_bfloat16>(__nv_bfloat16 *p, size_t i, float v) { p[i] = __float2bfloat16(v); }

// C = alpha * op(A) op(B) + beta * C, column-major; op(A) is m x k, op(B) is k x n.
// A batch is blockIdx.z: either pointer arrays (Aa/Ba/Ca) or strides (sA/sB/sC).
template <typename TA, typename TB, typename TC>
__global__ void __launch_bounds__(256) gemm(int m, int n, int k, float alpha,
        const TA *A, int lda, long long sA, const TA *const *Aa, int tA,
        const TB *B, int ldb, long long sB, const TB *const *Ba, int tB, float beta,
        TC *C, int ldc, long long sC, TC *const *Ca, int z0) {
    __shared__ float As[16][64 + 4], Bs[16][64 + 4];
    const int z = z0 + blockIdx.z;
    const TA *a = Aa ? Aa[z] : A + z * sA;
    const TB *b = Ba ? Ba[z] : B + z * sB;
    TC *c = Ca ? Ca[z] : C + z * sC;
    const int i0 = blockIdx.x * 64, j0 = blockIdx.y * 64, tx = threadIdx.x % 16, ty = threadIdx.x / 16;
    float acc[4][4] = {};
    for (int p0 = 0; p0 < k; p0 += 16) {
        for (int e = threadIdx.x; e < 1024; e += 256) {
            int ii = tA ? e / 16 : e % 64, pp = tA ? e % 16 : e / 64;      // walk the contiguous dim
            int gi = i0 + ii, gp = p0 + pp;
            As[pp][ii] = gi < m && gp < k ? ld(a, tA ? (size_t)gp + (size_t)gi * lda : (size_t)gi + (size_t)gp * lda) : 0.f;
            int jj = tB ? e % 64 : e / 16, qq = tB ? e / 64 : e % 16;
            int gj = j0 + jj, gq = p0 + qq;
            Bs[qq][jj] = gj < n && gq < k ? ld(b, tB ? (size_t)gj + (size_t)gq * ldb : (size_t)gq + (size_t)gj * ldb) : 0.f;
        }
        __syncthreads();
#pragma unroll
        for (int pp = 0; pp < 16; pp++) {
            float av[4], bv[4];
#pragma unroll
            for (int r = 0; r < 4; r++) { av[r] = As[pp][tx + 16 * r]; bv[r] = Bs[pp][ty + 16 * r]; }
#pragma unroll
            for (int r = 0; r < 4; r++)
#pragma unroll
                for (int q = 0; q < 4; q++) acc[r][q] += av[r] * bv[q];
        }
        __syncthreads();
    }
#pragma unroll
    for (int r = 0; r < 4; r++)
#pragma unroll
        for (int q = 0; q < 4; q++) {
            int i = i0 + tx + 16 * r, j = j0 + ty + 16 * q;
            if (i < m && j < n) {
                size_t o = (size_t)i + (size_t)j * ldc;
                float v = alpha * acc[r][q];
                if (beta != 0.f) v += beta * ld(c, o);
                st(c, o, v);
            }
        }
}

// Batched triangular solve (ggml SOLVE_TRI): op(A) X = alpha B (left) or X op(A) = alpha B (right),
// B overwritten by X. Rows (right) or columns (left) of X are independent: one thread each, plain
// substitution over the triangle (ggml's systems are <= a few hundred wide).
__global__ void trsm_batched(int left, int upper, int trans, int unit, int m, int n, float alpha,
                             const float *const *Aa, int lda, float *const *Ba, int ldb) {
    const float *A = Aa[blockIdx.y];
    float *B = Ba[blockIdx.y];
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    int dim = left ? m : n;                                     // order of the triangle
    if (t >= (left ? n : m)) return;
    bool up = upper != trans;                                  // op(A) = A^T swaps the triangle
#define T(r, c) (trans ? A[(c) + (size_t)(r) * lda] : A[(r) + (size_t)(c) * lda])
#define X(i) (left ? B[(i) + (size_t)t * ldb] : B[t + (size_t)(i) * ldb])
    // left: T x = b (x a column); right: x T = b (x a row, so the triangle is read by columns)
    bool forward = left ? !up : up;
    for (int s = 0; s < dim; s++) {
        int j = forward ? s : dim - 1 - s;
        float v = alpha * X(j);
        for (int r = 0; r < s; r++) {
            int k = forward ? r : dim - 1 - r;
            v -= left ? T(j, k) * X(k) : X(k) * T(k, j);
        }
        X(j) = unit ? v : v / T(j, j);
    }
#undef X
#undef T
}

struct handle { cudaStream_t stream; cublasMath_t math; };

static cublasStatus_t trsm(cublasSideMode_t side, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag,
                           int m, int n, float alpha, const float *const A[], int lda, float *const B[], int ldb, int batch) {
    if (m <= 0 || n <= 0 || batch <= 0) return CUBLAS_STATUS_SUCCESS;
    int indep = side == CUBLAS_SIDE_LEFT ? n : m;
    for (int z0 = 0; z0 < batch; z0 += 65535) {
        dim3 g((indep + 127) / 128, batch - z0 < 65535 ? batch - z0 : 65535);
        trsm_batched<<<g, 128>>>(side == CUBLAS_SIDE_LEFT, uplo == CUBLAS_FILL_MODE_UPPER, trans != CUBLAS_OP_N,
                                 diag == CUBLAS_DIAG_UNIT, m, n, alpha, A + z0, lda, B + z0, ldb);
    }
    return cudaGetLastError() == cudaSuccess ? CUBLAS_STATUS_SUCCESS : CUBLAS_STATUS_EXECUTION_FAILED;
}

template <typename TA, typename TB, typename TC>
static cublasStatus_t run(cublasOperation_t ta, cublasOperation_t tb, int m, int n, int k, float alpha,
                          const void *A, int lda, long long sA, const void *const *Aa,
                          const void *B, int ldb, long long sB, const void *const *Ba, float beta,
                          void *C, int ldc, long long sC, void *const *Ca, int batch) {
    if (m <= 0 || n <= 0 || batch <= 0) return CUBLAS_STATUS_SUCCESS;
    for (int z0 = 0; z0 < batch; z0 += 65535) {
        dim3 g((m + 63) / 64, (n + 63) / 64, batch - z0 < 65535 ? batch - z0 : 65535);
        gemm<TA, TB, TC><<<g, 256>>>(m, n, k, alpha, (const TA *)A, lda, sA, (const TA *const *)Aa, ta != CUBLAS_OP_N,
                                     (const TB *)B, ldb, sB, (const TB *const *)Ba, tb != CUBLAS_OP_N, beta,
                                     (TC *)C, ldc, sC, (TC *const *)Ca, z0);
    }
    return cudaGetLastError() == cudaSuccess ? CUBLAS_STATUS_SUCCESS : CUBLAS_STATUS_EXECUTION_FAILED;
}

static float scalar(const void *p, cublasComputeType_t ct) {
    return ct == CUBLAS_COMPUTE_16F || ct == CUBLAS_COMPUTE_16F_PEDANTIC ? __half2float(*(const __half *)p) : *(const float *)p;
}

static cublasStatus_t dispatch(cublasOperation_t ta, cublasOperation_t tb, int m, int n, int k, float alpha,
                               const void *A, cudaDataType at, int lda, long long sA, const void *const *Aa,
                               const void *B, cudaDataType bt, int ldb, long long sB, const void *const *Ba, float beta,
                               void *C, cudaDataType ct, int ldc, long long sC, void *const *Ca, int batch) {
#define R(TA, TB, TC) return run<TA, TB, TC>(ta, tb, m, n, k, alpha, A, lda, sA, Aa, B, ldb, sB, Ba, beta, C, ldc, sC, Ca, batch)
#define CT(TA, TB) \
    if (ct == CUDA_R_32F) R(TA, TB, float); \
    if (ct == CUDA_R_16F) R(TA, TB, __half); \
    if (ct == CUDA_R_16BF) R(TA, TB, __nv_bfloat16);
    if (at == bt) {
        if (at == CUDA_R_32F) { CT(float, float) }
        if (at == CUDA_R_16F) { CT(__half, __half) }
        if (at == CUDA_R_16BF) { CT(__nv_bfloat16, __nv_bfloat16) }
    }
#undef CT
#undef R
    fprintf(stderr, "[cublas-shim] GEMM types A=%d B=%d C=%d not supported\n", (int)at, (int)bt, (int)ct);
    return CUBLAS_STATUS_NOT_SUPPORTED;
}

extern "C" {
void cublas_log_stub(const char *name) { fprintf(stderr, "[cublas-shim] %s -> CUBLAS_STATUS_NOT_SUPPORTED (stub)\n", name); }

cublasStatus_t cublasCreate_v2(cublasHandle_t *h) { *h = (cublasHandle_t)calloc(1, sizeof(struct handle)); return CUBLAS_STATUS_SUCCESS; }
cublasStatus_t cublasDestroy_v2(cublasHandle_t h) { free(h); return CUBLAS_STATUS_SUCCESS; }
cublasStatus_t cublasSetStream_v2(cublasHandle_t h, cudaStream_t s) { ((struct handle *)h)->stream = s; return CUBLAS_STATUS_SUCCESS; }
cublasStatus_t cublasGetStream_v2(cublasHandle_t h, cudaStream_t *s) { *s = ((struct handle *)h)->stream; return CUBLAS_STATUS_SUCCESS; }
cublasStatus_t cublasSetMathMode(cublasHandle_t h, cublasMath_t m) { ((struct handle *)h)->math = m; return CUBLAS_STATUS_SUCCESS; }
cublasStatus_t cublasGetMathMode(cublasHandle_t h, cublasMath_t *m) { *m = ((struct handle *)h)->math; return CUBLAS_STATUS_SUCCESS; }
cublasStatus_t cublasSetWorkspace_v2(cublasHandle_t h, void *w, size_t n) { return CUBLAS_STATUS_SUCCESS; }
cublasStatus_t cublasLoggerConfigure(int on, int out, int err, const char *file) { return CUBLAS_STATUS_SUCCESS; }
const char *cublasGetStatusName(cublasStatus_t s) {
    switch (s) {
    case CUBLAS_STATUS_SUCCESS: return "CUBLAS_STATUS_SUCCESS";
    case CUBLAS_STATUS_NOT_SUPPORTED: return "CUBLAS_STATUS_NOT_SUPPORTED";
    case CUBLAS_STATUS_EXECUTION_FAILED: return "CUBLAS_STATUS_EXECUTION_FAILED";
    case CUBLAS_STATUS_INVALID_VALUE: return "CUBLAS_STATUS_INVALID_VALUE";
    default: return "CUBLAS_STATUS_UNKNOWN";
    }
}
const char *cublasGetStatusString(cublasStatus_t s) { return cublasGetStatusName(s); }

cublasStatus_t cublasSgemm_v2(cublasHandle_t h, cublasOperation_t ta, cublasOperation_t tb, int m, int n, int k,
                              const float *alpha, const float *A, int lda, const float *B, int ldb,
                              const float *beta, float *C, int ldc) {
    return run<float, float, float>(ta, tb, m, n, k, *alpha, A, lda, 0, NULL, B, ldb, 0, NULL, *beta, C, ldc, 0, NULL, 1);
}
cublasStatus_t cublasSgemmStridedBatched(cublasHandle_t h, cublasOperation_t ta, cublasOperation_t tb, int m, int n, int k,
                                         const float *alpha, const float *A, int lda, long long sA, const float *B, int ldb,
                                         long long sB, const float *beta, float *C, int ldc, long long sC, int batch) {
    return run<float, float, float>(ta, tb, m, n, k, *alpha, A, lda, sA, NULL, B, ldb, sB, NULL, *beta, C, ldc, sC, NULL, batch);
}
cublasStatus_t cublasSgemmBatched(cublasHandle_t h, cublasOperation_t ta, cublasOperation_t tb, int m, int n, int k,
                                  const float *alpha, const float *const Aa[], int lda, const float *const Ba[], int ldb,
                                  const float *beta, float *const Ca[], int ldc, int batch) {
    return run<float, float, float>(ta, tb, m, n, k, *alpha, NULL, lda, 0, (const void *const *)Aa, NULL, ldb, 0,
                                    (const void *const *)Ba, *beta, NULL, ldc, 0, (void *const *)Ca, batch);
}
cublasStatus_t cublasStrsmBatched(cublasHandle_t h, cublasSideMode_t side, cublasFillMode_t uplo,
                                  cublasOperation_t trans, cublasDiagType_t diag, int m, int n, const float *alpha,
                                  const float *const A[], int lda, float *const B[], int ldb, int batch) {
    return trsm(side, uplo, trans, diag, m, n, *alpha, A, lda, B, ldb, batch);
}

cublasStatus_t cublasGemmEx(cublasHandle_t h, cublasOperation_t ta, cublasOperation_t tb, int m, int n, int k,
                            const void *alpha, const void *A, cudaDataType at, int lda, const void *B, cudaDataType bt, int ldb,
                            const void *beta, void *C, cudaDataType ct, int ldc, cublasComputeType_t comp, cublasGemmAlgo_t algo) {
    return dispatch(ta, tb, m, n, k, scalar(alpha, comp), A, at, lda, 0, NULL, B, bt, ldb, 0, NULL, scalar(beta, comp),
                    C, ct, ldc, 0, NULL, 1);
}
cublasStatus_t cublasGemmStridedBatchedEx(cublasHandle_t h, cublasOperation_t ta, cublasOperation_t tb, int m, int n, int k,
                                          const void *alpha, const void *A, cudaDataType at, int lda, long long sA,
                                          const void *B, cudaDataType bt, int ldb, long long sB, const void *beta,
                                          void *C, cudaDataType ct, int ldc, long long sC, int batch,
                                          cublasComputeType_t comp, cublasGemmAlgo_t algo) {
    return dispatch(ta, tb, m, n, k, scalar(alpha, comp), A, at, lda, sA, NULL, B, bt, ldb, sB, NULL, scalar(beta, comp),
                    C, ct, ldc, sC, NULL, batch);
}
cublasStatus_t cublasGemmBatchedEx(cublasHandle_t h, cublasOperation_t ta, cublasOperation_t tb, int m, int n, int k,
                                   const void *alpha, const void *const Aa[], cudaDataType at, int lda,
                                   const void *const Ba[], cudaDataType bt, int ldb, const void *beta,
                                   void *const Ca[], cudaDataType ct, int ldc, int batch,
                                   cublasComputeType_t comp, cublasGemmAlgo_t algo) {
    return dispatch(ta, tb, m, n, k, scalar(alpha, comp), NULL, at, lda, 0, Aa, NULL, bt, ldb, 0, Ba, scalar(beta, comp),
                    NULL, ct, ldc, 0, Ca, batch);
}
}
