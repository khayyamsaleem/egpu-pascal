// Stream-ordered copies and fills for the shim runtime (nvpascal async path).
// Any alignment, any length; vectorized when source and destination share their misalignment.
// Built with CUDA 12.8 nvcc -arch=sm_61 -cubin (kernels/build.sh).
#include <stdint.h>

#define GRID_STRIDE(i, n) for (uint64_t i = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x; i < (n); \
                               i += (uint64_t)gridDim.x * blockDim.x)

extern "C" __global__ void xcopy(const uint8_t *__restrict__ src, uint8_t *__restrict__ dst, uint64_t n) {
    uint64_t s = (uint64_t)src, d = (uint64_t)dst;
    if (((s ^ d) & 15) == 0) {                      // same phase mod 16: bytes, 16-B body, bytes
        uint64_t head = (16 - (s & 15)) & 15;
        if (head > n) head = n;
        uint64_t n16 = (n - head) >> 4, tail0 = head + (n16 << 4);
        GRID_STRIDE(i, head) dst[i] = src[i];
        const uint4 *s4 = (const uint4 *)(src + head);
        uint4 *d4 = (uint4 *)(dst + head);
        GRID_STRIDE(i, n16) d4[i] = s4[i];
        GRID_STRIDE(i, n - tail0) dst[tail0 + i] = src[tail0 + i];
    } else if (((s ^ d) & 3) == 0) {                // same phase mod 4: 4-B body
        uint64_t head = (4 - (s & 3)) & 3;
        if (head > n) head = n;
        uint64_t n4 = (n - head) >> 2, tail0 = head + (n4 << 2);
        GRID_STRIDE(i, head) dst[i] = src[i];
        const uint32_t *s1 = (const uint32_t *)(src + head);
        uint32_t *d1 = (uint32_t *)(dst + head);
        GRID_STRIDE(i, n4) d1[i] = s1[i];
        GRID_STRIDE(i, n - tail0) dst[tail0 + i] = src[tail0 + i];
    } else {
        GRID_STRIDE(i, n) dst[i] = src[i];
    }
}

// Byte k of the fill (k = offset from dst) is byte (k % 4) of pat; pat is the 1/2/4-byte value
// replicated to 32 bits, so 16-bit fills only need 2-byte alignment (cuMemsetD16 rules).
extern "C" __global__ void xfill(uint8_t *__restrict__ dst, uint32_t pat, uint64_t n) {
    uint64_t d = (uint64_t)dst;
    uint64_t head = (16 - (d & 15)) & 15;
    if (head > n) head = n;
    uint64_t n16 = (n - head) >> 4, tail0 = head + (n16 << 4);
    GRID_STRIDE(i, head) dst[i] = (uint8_t)(pat >> (8 * (i & 3)));
    uint32_t r = head & 3, w = r ? (pat >> (8 * r)) | (pat << (32 - 8 * r)) : pat;   // pattern at dst+head
    uint4 v = make_uint4(w, w, w, w);
    uint4 *d4 = (uint4 *)(dst + head);
    GRID_STRIDE(i, n16) d4[i] = v;
    GRID_STRIDE(i, n - tail0) dst[tail0 + i] = (uint8_t)(pat >> (8 * ((tail0 + i) & 3)));
}
