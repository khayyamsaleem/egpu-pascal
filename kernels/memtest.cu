// VRAM integrity test for memory reclocking (pascal/memtest.py).
// mt_fill writes an address-derived 32-bit pattern; mt_check re-derives it and counts
// mismatches, so a stuck bit, a bad timing or an aliased address all show as errors.
// out[0] = mismatching words, out[1] = first bad word index, out[2] = OR of flipped bits,
// out[3] = the word read at out[1].

__device__ __forceinline__ unsigned int pat(unsigned long long i, unsigned int seed) {
    unsigned int x = (unsigned int)i * 2654435761u ^ (unsigned int)(i >> 32) * 40503u ^ seed;
    x ^= x >> 15; x *= 0x2c1b3c6dU; x ^= x >> 12;
    return x;
}

extern "C" __global__ void mt_fill(uint4 *p, unsigned long long n4, unsigned int seed) {
    for (unsigned long long i = blockIdx.x * (unsigned long long)blockDim.x + threadIdx.x; i < n4;
         i += (unsigned long long)gridDim.x * blockDim.x) {
        unsigned long long w = i * 4;
        p[i] = make_uint4(pat(w, seed), pat(w + 1, seed), pat(w + 2, seed), pat(w + 3, seed));
    }
}

extern "C" __global__ void mt_check(const uint4 *p, unsigned long long n4, unsigned int seed,
                                    unsigned long long *out) {
    unsigned long long bad = 0, first = ~0ull;
    unsigned int flips = 0, firstval = 0;
    for (unsigned long long i = blockIdx.x * (unsigned long long)blockDim.x + threadIdx.x; i < n4;
         i += (unsigned long long)gridDim.x * blockDim.x) {
        uint4 v = p[i];
        unsigned long long w = i * 4;
        unsigned int g[4] = {v.x, v.y, v.z, v.w};
        for (int k = 0; k < 4; k++) {
            unsigned int e = pat(w + k, seed), d = g[k] ^ e;
            if (d) {
                bad++; flips |= d;
                if (w + k < first) { first = w + k; firstval = g[k]; }
            }
        }
    }
    if (bad) {
        atomicAdd(&out[0], bad);
        atomicOr(&out[2], (unsigned long long)flips);
        unsigned long long prev = atomicMin(&out[1], first);
        if (first < prev) out[3] = firstval;   // best effort: racing writers may overwrite
    }
}
