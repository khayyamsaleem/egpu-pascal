// End-to-end check of shim/cudart + shim/cublas on the card: build with
//   nvcc -arch=sm_61 --no-compress -cudart shared -o test_runtime test_runtime.cu -lcublas
// and run with LD_LIBRARY_PATH=/repo/shim/build-cuda12 SHIM_MODE=remote.
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>

__global__ void affine(float *x, int n, float a, float b) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] = x[i] * a + b + i;
}
__global__ void smem_rev(const int *in, int *out, int n) {        // dynamic shared memory + barrier
    extern __shared__ int s[];
    int t = threadIdx.x;
    s[t] = in[blockIdx.x * blockDim.x + t];
    __syncthreads();
    out[blockIdx.x * blockDim.x + t] = s[blockDim.x - 1 - t];
}

static int fails = 0;
#define CHECK(c, ...) do { bool ok_ = (c); printf("%-52s %s\n", __VA_ARGS__, ok_ ? "ok" : "FAIL"); fails += !ok_; } while (0)

static bool gemm_check(cublasHandle_t h, cublasOperation_t ta, cublasOperation_t tb, int m, int n, int k, bool half_in) {
    int ar = ta == CUBLAS_OP_N ? m : k, ac = ta == CUBLAS_OP_N ? k : m;
    int br = tb == CUBLAS_OP_N ? k : n, bc = tb == CUBLAS_OP_N ? n : k;
    std::vector<float> A(ar * ac), B(br * bc), C(m * n, 0.5f), R(m * n);
    for (auto &v : A) v = (rand() % 200 - 100) / 64.f;
    for (auto &v : B) v = (rand() % 200 - 100) / 64.f;
    for (int i = 0; i < m; i++)
        for (int j = 0; j < n; j++) {
            double s = 0;
            for (int p = 0; p < k; p++) {
                float a = ta == CUBLAS_OP_N ? A[i + p * ar] : A[p + i * ar];
                float b = tb == CUBLAS_OP_N ? B[p + j * br] : B[j + p * br];
                s += (double)a * b;
            }
            R[i + j * m] = (float)(1.5 * s + 0.25 * 0.5);
        }
    float alpha = 1.5f, beta = 0.25f;
    void *dA, *dB; float *dC;
    size_t es = half_in ? 2 : 4;
    cudaMalloc(&dA, A.size() * es); cudaMalloc(&dB, B.size() * es); cudaMalloc((void **)&dC, C.size() * 4);
    if (half_in) {
        std::vector<__half> Ah(A.size()), Bh(B.size());
        for (size_t i = 0; i < A.size(); i++) Ah[i] = __float2half(A[i]);
        for (size_t i = 0; i < B.size(); i++) Bh[i] = __float2half(B[i]);
        cudaMemcpy(dA, Ah.data(), A.size() * 2, cudaMemcpyHostToDevice);
        cudaMemcpy(dB, Bh.data(), B.size() * 2, cudaMemcpyHostToDevice);
    } else {
        cudaMemcpy(dA, A.data(), A.size() * 4, cudaMemcpyHostToDevice);
        cudaMemcpy(dB, B.data(), B.size() * 4, cudaMemcpyHostToDevice);
    }
    cudaMemcpy(dC, C.data(), C.size() * 4, cudaMemcpyHostToDevice);
    cublasStatus_t st = half_in
        ? cublasGemmEx(h, ta, tb, m, n, k, &alpha, dA, CUDA_R_16F, ar, dB, CUDA_R_16F, br, &beta, dC, CUDA_R_32F, m,
                       CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT)
        : cublasSgemm(h, ta, tb, m, n, k, &alpha, (float *)dA, ar, (float *)dB, br, &beta, dC, m);
    cudaMemcpy(C.data(), dC, C.size() * 4, cudaMemcpyDeviceToHost);
    double err = 0;
    for (int i = 0; i < m * n; i++) err = fmax(err, fabs(C[i] - R[i]) / (fabs(R[i]) + 1));
    cudaFree(dA); cudaFree(dB); cudaFree(dC);
    return st == CUBLAS_STATUS_SUCCESS && err < (half_in ? 1e-2 : 1e-4);
}

int main() {
    int n = 0; cudaGetDeviceCount(&n);
    cudaDeviceProp p; cudaGetDeviceProperties(&p, 0);
    printf("device: %s, CC %d.%d, %d SMs, %.1f GiB\n", p.name, p.major, p.minor, p.multiProcessorCount, p.totalGlobalMem / 1073741824.0);
    CHECK(n == 1 && p.major == 6 && p.minor == 1, "cudaGetDeviceCount / cudaGetDeviceProperties");

    const int N = 100000;
    std::vector<float> h(N), g(N);
    for (int i = 0; i < N; i++) h[i] = i * 0.5f;
    float *d; cudaMalloc((void **)&d, N * 4);
    cudaMemcpy(d, h.data(), N * 4, cudaMemcpyHostToDevice);
    affine<<<(N + 255) / 256, 256>>>(d, N, 2.f, 1.f);
    CHECK(cudaGetLastError() == cudaSuccess, "kernel launch (<<<>>>)");
    cudaMemcpy(g.data(), d, N * 4, cudaMemcpyDefault);
    bool ok = true;
    for (int i = 0; i < N; i++) ok &= g[i] == h[i] * 2.f + 1.f + i;
    CHECK(ok, "affine kernel result, cudaMemcpyDefault both ways");

    int *a, *b; std::vector<int> in(1024), out(1024);
    for (int i = 0; i < 1024; i++) in[i] = i * 7;
    cudaMalloc((void **)&a, 4096); cudaMalloc((void **)&b, 4096);
    cudaMemcpyAsync(a, in.data(), 4096, cudaMemcpyHostToDevice, 0);
    smem_rev<<<4, 256, 256 * 4>>>(a, b, 1024);
    cudaMemcpyAsync(out.data(), b, 4096, cudaMemcpyDeviceToHost, 0);
    cudaStreamSynchronize(0);
    ok = true;
    for (int i = 0; i < 1024; i++) ok &= out[i] == in[(i / 256) * 256 + 255 - i % 256];
    CHECK(ok, "dynamic shared memory + __syncthreads");

    cudaMemsetAsync(b, 0x5a, 4096, 0);
    cudaMemcpy2DAsync(b, 64, a, 32, 16, 8, cudaMemcpyDeviceToDevice, 0);
    cudaMemcpy(out.data(), b, 4096, cudaMemcpyDeviceToHost);
    ok = true;
    const unsigned char *ob = (const unsigned char *)out.data(), *ib = (const unsigned char *)in.data();
    for (int r = 0; r < 8; r++)
        for (int c = 0; c < 64; c++) ok &= ob[r * 64 + c] == (c < 16 ? ib[r * 32 + c] : 0x5a);
    CHECK(ok, "cudaMemsetAsync + cudaMemcpy2DAsync (DtoD)");

    cudaFuncAttributes fa; cudaFuncGetAttributes(&fa, affine);
    int nb = 0; cudaOccupancyMaxActiveBlocksPerMultiprocessor(&nb, affine, 256, 0);
    printf("affine: %d regs, occupancy %d blocks/SM of 256\n", fa.numRegs, nb);
    CHECK(fa.numRegs > 0 && nb >= 1 && nb <= 8, "cudaFuncGetAttributes / occupancy");

    cublasHandle_t hb; cublasCreate(&hb);
    CHECK(gemm_check(hb, CUBLAS_OP_N, CUBLAS_OP_N, 67, 45, 33, false), "cublasSgemm N,N 67x45x33");
    CHECK(gemm_check(hb, CUBLAS_OP_T, CUBLAS_OP_N, 128, 70, 100, false), "cublasSgemm T,N 128x70x100");
    CHECK(gemm_check(hb, CUBLAS_OP_N, CUBLAS_OP_T, 33, 129, 65, false), "cublasSgemm N,T 33x129x65");
    CHECK(gemm_check(hb, CUBLAS_OP_T, CUBLAS_OP_N, 96, 17, 256, true), "cublasGemmEx f16 in, f32 out, T,N");
    cublasDestroy(hb);
    printf("RESULT: %s\n", fails ? "FAIL" : "PASS");
    return fails != 0;
}
