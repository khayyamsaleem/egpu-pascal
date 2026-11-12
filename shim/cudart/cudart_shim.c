/*
 * libcudart.so.12 for the GP102 shim: the CUDA runtime API over our libcuda.so.1 (shim/src).
 *
 * NVIDIA's libcudart talks to the driver through undocumented export tables (cuGetExportTable)
 * that our libcuda does not implement, so programs built with nvcc (llama.cpp's ggml-cuda) load
 * this library instead (LD_LIBRARY_PATH). It covers what nvcc-generated code and ggml-cuda use:
 *
 *  - Kernel registration: __cudaRegisterFatBinary/Function/Var. A fatbin is loaded lazily, on the
 *    first launch of one of its kernels: the sm_61 ELF is picked out of the fatbin (built with
 *    nvcc --no-compress) and handed to cuModuleLoadDataEx. ggml-cuda registers thousands of kernels
 *    in ~150 translation units, and most of them are never launched.
 *  - Launch: __cudaPush/PopCallConfiguration + cudaLaunchKernel(-Ex) -> cuLaunchKernel.
 *  - Memory, copies, memsets -> cuMem*. Streams and events are views of the shim's single in-order
 *    queue: synchronize = cuCtxSynchronize, waits between streams are no-ops.
 *  - Device properties for a GTX 1080 Ti (CC 6.1, 28 SMs), occupancy, errors.
 *
 * Built against the CUDA 12.8 headers, so struct layouts (cudaDeviceProp, cudaFuncAttributes)
 * match the programs; exported under version node libcudart.so.12 (cudart.map).
 * Unimplemented entry points come from gen_stubs.py and return cudaErrorNotSupported.
 */
#define _GNU_SOURCE
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

int shim_is_device_pointer(unsigned long long p);   /* exported by our libcuda.so.1 */

static int verbose = -1;
#define LOG(...) do { if (verbose < 0) verbose = getenv("CUDART_SHIM_VERBOSE") != NULL; \
                      if (verbose) { fprintf(stderr, "[cudart-shim] " __VA_ARGS__); fputc('\n', stderr); } } while (0)
#define ERR(...) do { fprintf(stderr, "[cudart-shim] " __VA_ARGS__); fputc('\n', stderr); } while (0)
void cudart_log_stub(const char *name) { ERR("%s -> cudaErrorNotSupported (stub)", name); }

static __thread cudaError_t last_error = cudaSuccess;
static cudaError_t ret(cudaError_t e) { if (e != cudaSuccess) last_error = e; return e; }
static cudaError_t from_cu(CUresult r) {
    switch (r) {
    case CUDA_SUCCESS: return cudaSuccess;
    case CUDA_ERROR_INVALID_VALUE: return cudaErrorInvalidValue;
    case CUDA_ERROR_OUT_OF_MEMORY: return cudaErrorMemoryAllocation;
    case CUDA_ERROR_NOT_INITIALIZED: return cudaErrorInitializationError;
    case CUDA_ERROR_INVALID_PTX: return cudaErrorInvalidPtx;
    case CUDA_ERROR_NOT_FOUND: return cudaErrorInvalidDeviceFunction;
    case CUDA_ERROR_LAUNCH_FAILED: return cudaErrorLaunchFailure;
    case CUDA_ERROR_NOT_SUPPORTED: return cudaErrorNotSupported;
    case CUDA_ERROR_NOT_READY: return cudaErrorNotReady;
    default: return cudaErrorUnknown;
    }
}
#define CU(x) ret(from_cu(x))

/* ------------------------------------------------------------------ init */
static pthread_once_t once = PTHREAD_ONCE_INIT;
static CUcontext ctx;
static CUresult init_status = CUDA_ERROR_NOT_INITIALIZED;
static void do_init(void) {
    CUdevice d;
    if ((init_status = cuInit(0)) || (init_status = cuDeviceGet(&d, 0)) ||
        (init_status = cuDevicePrimaryCtxRetain(&ctx, d)) || (init_status = cuCtxSetCurrent(ctx)))
        ERR("driver init failed: %d", init_status);
}
static cudaError_t init(void) {
    pthread_once(&once, do_init);
    if (init_status) return ret(cudaErrorInitializationError);
    cuCtxSetCurrent(ctx);
    return cudaSuccess;
}

/* ------------------------------------------------------------------ registration */
typedef struct { int magic, version; const unsigned char *data; void *filename_or_fatbins; } fatbin_wrapper;
struct fatmod { const unsigned char *fatbin; CUmodule mod; int state; /* 0 not loaded, 1 ok, -1 failed */ };
struct kfunc { const void *host; struct fatmod *fm; const char *name; CUfunction f; };

static pthread_mutex_t reg_lk = PTHREAD_MUTEX_INITIALIZER;
static struct kfunc *ktab; static size_t kcap, kcount;      /* open addressing on the host stub */

static size_t khash(const void *p) { uint64_t x = (uint64_t)(uintptr_t)p; x ^= x >> 33; x *= 0xff51afd7ed558ccdull; x ^= x >> 33; return (size_t)x; }
static void kput(struct kfunc k) {
    if ((kcount + 1) * 2 > kcap) {
        size_t ncap = kcap ? kcap * 2 : 4096; struct kfunc *n = calloc(ncap, sizeof *n), *o = ktab; size_t ocap = kcap;
        ktab = n; kcap = ncap; kcount = 0;
        for (size_t i = 0; i < ocap; i++) if (o[i].host) kput(o[i]);
        free(o);
    }
    size_t i = khash(k.host) & (kcap - 1);
    while (ktab[i].host && ktab[i].host != k.host) i = (i + 1) & (kcap - 1);
    if (!ktab[i].host) kcount++;
    ktab[i] = k;
}
static struct kfunc *kget(const void *host) {
    if (!kcap) return NULL;
    size_t i = khash(host) & (kcap - 1);
    while (ktab[i].host) { if (ktab[i].host == host) return &ktab[i]; i = (i + 1) & (kcap - 1); }
    return NULL;
}

void **__cudaRegisterFatBinary(void *fatCubin) {
    fatbin_wrapper *w = fatCubin;
    struct fatmod *m = calloc(1, sizeof *m);
    m->fatbin = w->data;
    return (void **)m;
}
void __cudaRegisterFatBinaryEnd(void **h) {}
void __cudaUnregisterFatBinary(void **h) {}
char __cudaInitModule(void **h) { return 1; }
void __cudaRegisterFunction(void **h, const char *hostFun, char *deviceFun, const char *deviceName, int tl,
                            uint3 *tid, uint3 *bid, dim3 *bDim, dim3 *gDim, int *wSize) {
    pthread_mutex_lock(&reg_lk);
    kput((struct kfunc){hostFun, (struct fatmod *)h, deviceName, NULL});
    pthread_mutex_unlock(&reg_lk);
}
void __cudaRegisterVar(void **h, char *hostVar, char *deviceAddress, const char *deviceName, int ext,
                       size_t size, int constant, int global) {
    LOG("__cudaRegisterVar(%s, %zu bytes) -- host access to device variables is not supported", deviceName, size);
}
void __cudaRegisterManagedVar(void **h, void **hostVarPtrAddress, char *deviceAddress, const char *deviceName,
                              int ext, size_t size, int constant, int global) {}
void __cudaRegisterTexture(void **h, const void *hostVar, const void **deviceAddress, const char *deviceName,
                           int dim, int norm, int ext) {}
void __cudaRegisterSurface(void **h, const void *hostVar, const void **deviceAddress, const char *deviceName,
                           int dim, int ext) {}

/* The sm_61 ELF inside a fatbin: header {u32 magic 0xba55ed50, u16 version, u16 header size,
 * u64 size}, then entries {u16 kind (1 PTX, 2 ELF), u16, u32 header size, u64 size, ...,
 * u32 arch @28, u64 flags @40 (0x2000 = compressed)} followed by their payload. */
static const unsigned char *fatbin_elf(const unsigned char *fb, size_t *len, int *compressed) {
    *compressed = 0;
    if (*(const uint32_t *)fb != 0xba55ed50u) return NULL;
    uint16_t hsz = *(const uint16_t *)(fb + 6);
    uint64_t fsz = *(const uint64_t *)(fb + 8);
    const unsigned char *p = fb + hsz, *end = p + fsz;
    while (p + 48 <= end) {
        uint16_t kind = *(const uint16_t *)p;
        uint32_t ehsz = *(const uint32_t *)(p + 4), arch = *(const uint32_t *)(p + 28);
        uint64_t size = *(const uint64_t *)(p + 8), flags = *(const uint64_t *)(p + 40);
        if (kind == 2 && arch == 61) {
            if (flags & 0x2000) { *compressed = 1; return NULL; }
            *len = size; return p + ehsz;
        }
        p += ehsz + size;
    }
    return NULL;
}

static cudaError_t resolve(struct kfunc *k) {
    if (k->f) return cudaSuccess;
    cudaError_t e = init();
    if (e) return e;
    pthread_mutex_lock(&reg_lk);
    struct fatmod *m = k->fm;
    if (m->state == 0) {
        size_t n = 0; int comp;
        const unsigned char *elf = fatbin_elf(m->fatbin, &n, &comp);
        if (!elf) {
            ERR("no %s sm_61 image for %s (rebuild with -arch=sm_61 --no-compress)", comp ? "uncompressed" : "", k->name);
            m->state = -1;
        } else {
            CUresult r = cuModuleLoadDataEx(&m->mod, elf, 0, NULL, NULL);
            m->state = r ? -1 : 1;
            LOG("module for %s: %zu bytes -> %d", k->name, n, r);
            if (r) ERR("cuModuleLoadDataEx failed (%d) for the module of %s", r, k->name);
        }
    }
    CUresult r = m->state == 1 ? cuModuleGetFunction(&k->f, m->mod, k->name) : CUDA_ERROR_NOT_FOUND;
    pthread_mutex_unlock(&reg_lk);
    if (r) { ERR("kernel %s unavailable (%d)", k->name, r); return ret(cudaErrorInvalidDeviceFunction); }
    return cudaSuccess;
}

/* ------------------------------------------------------------------ launch */
struct callcfg { dim3 g, b; size_t smem; void *stream; };
static __thread struct callcfg cfg_stack[16];
static __thread int cfg_n;
unsigned __cudaPushCallConfiguration(dim3 g, dim3 b, size_t smem, void *stream) {
    if (cfg_n < 16) cfg_stack[cfg_n++] = (struct callcfg){g, b, smem, stream};
    return 0;
}
cudaError_t __cudaPopCallConfiguration(dim3 *g, dim3 *b, size_t *smem, void *stream) {
    if (!cfg_n) return cudaErrorInvalidConfiguration;
    struct callcfg c = cfg_stack[--cfg_n];
    *g = c.g; *b = c.b; *smem = c.smem; *(void **)stream = c.stream;
    return cudaSuccess;
}
static cudaError_t launch(const void *func, dim3 g, dim3 b, void **args, size_t smem) {
    pthread_mutex_lock(&reg_lk);
    struct kfunc *k = kget(func);
    pthread_mutex_unlock(&reg_lk);
    if (!k) { ERR("launch of an unregistered function %p", func); return ret(cudaErrorInvalidDeviceFunction); }
    cudaError_t e = resolve(k);
    if (e) return e;
    if (smem > 49152) { ERR("%s: %zu bytes of dynamic shared memory (sm_61 max 48 KiB)", k->name, smem); return ret(cudaErrorInvalidValue); }
    return CU(cuLaunchKernel(k->f, g.x, g.y, g.z, b.x, b.y, b.z, (unsigned)smem, NULL, args, NULL));
}
cudaError_t cudaLaunchKernel(const void *func, dim3 g, dim3 b, void **args, size_t smem, cudaStream_t s) {
    return launch(func, g, b, args, smem);
}
cudaError_t cudaLaunchKernelExC(const cudaLaunchConfig_t *c, const void *func, void **args) {
    return launch(func, c->gridDim, c->blockDim, args, c->dynamicSmemBytes);
}
cudaError_t cudaLaunchCooperativeKernel(const void *func, dim3 g, dim3 b, void **args, size_t smem, cudaStream_t s) {
    return launch(func, g, b, args, smem);
}

/* ------------------------------------------------------------------ device */
cudaError_t cudaGetDeviceCount(int *n) { if (init()) { *n = 0; return ret(cudaErrorNoDevice); } *n = 1; return cudaSuccess; }
cudaError_t cudaGetDevice(int *d) { *d = 0; return cudaSuccess; }
cudaError_t cudaSetDevice(int d) { return d ? ret(cudaErrorInvalidDevice) : init(); }
cudaError_t cudaSetDeviceFlags(unsigned f) { return cudaSuccess; }
cudaError_t cudaGetDeviceFlags(unsigned *f) { *f = 0; return cudaSuccess; }
cudaError_t cudaInitDevice(int d, unsigned df, unsigned f) { return init(); }
cudaError_t cudaDeviceSynchronize(void) { cudaError_t e = init(); return e ? e : CU(cuCtxSynchronize()); }
cudaError_t cudaDeviceReset(void) { return cudaDeviceSynchronize(); }
cudaError_t cudaThreadSynchronize(void) { return cudaDeviceSynchronize(); }
cudaError_t cudaDriverGetVersion(int *v) { *v = 12080; return cudaSuccess; }
cudaError_t cudaRuntimeGetVersion(int *v) { *v = 12080; return cudaSuccess; }
cudaError_t cudaDeviceGetAttribute(int *v, enum cudaDeviceAttr a, int d) {
    cudaError_t e = init(); if (e) return e;
    return CU(cuDeviceGetAttribute(v, (CUdevice_attribute)a, 0));
}
static int attr(int a) { int v = 0; cuDeviceGetAttribute(&v, (CUdevice_attribute)a, 0); return v; }
cudaError_t cudaGetDeviceProperties_v2(struct cudaDeviceProp *p, int d) {
    cudaError_t e = init(); if (e) return e;
    memset(p, 0, sizeof *p);
    cuDeviceGetName(p->name, sizeof p->name, 0);
    cuDeviceGetUuid((CUuuid *)&p->uuid, 0);
    size_t fr = 0, tot = 0; cuMemGetInfo(&fr, &tot);
    p->totalGlobalMem = tot;
    p->sharedMemPerBlock = 49152; p->regsPerBlock = 65536; p->warpSize = 32; p->memPitch = 2147483647;
    p->maxThreadsPerBlock = 1024; p->maxThreadsDim[0] = 1024; p->maxThreadsDim[1] = 1024; p->maxThreadsDim[2] = 64;
    p->maxGridSize[0] = 2147483647; p->maxGridSize[1] = 65535; p->maxGridSize[2] = 65535;
    p->clockRate = attr(CU_DEVICE_ATTRIBUTE_CLOCK_RATE); p->totalConstMem = 65536;
    p->major = 6; p->minor = 1; p->textureAlignment = 512; p->texturePitchAlignment = 32; p->deviceOverlap = 1;
    p->multiProcessorCount = 28; p->canMapHostMemory = 0; p->concurrentKernels = 1;
    p->pciBusID = 1; p->asyncEngineCount = 2; p->unifiedAddressing = 1;
    p->memoryClockRate = attr(CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE); p->memoryBusWidth = 352;
    p->l2CacheSize = 2883584; p->maxThreadsPerMultiProcessor = 2048; p->streamPrioritiesSupported = 1;
    p->globalL1CacheSupported = 1; p->localL1CacheSupported = 1; p->sharedMemPerMultiprocessor = 98304;
    p->regsPerMultiprocessor = 65536; p->singleToDoublePrecisionPerfRatio = 32; p->computePreemptionSupported = 1;
    p->sharedMemPerBlockOptin = 49152; p->maxBlocksPerMultiProcessor = 32; p->hostRegisterSupported = 0;
    return cudaSuccess;
}
#undef cudaGetDeviceProperties
cudaError_t cudaGetDeviceProperties(struct cudaDeviceProp *p, int d) { return cudaGetDeviceProperties_v2(p, d); }
cudaError_t cudaDeviceGetPCIBusId(char *id, int len, int d) { snprintf(id, len, "0000:01:00.0"); return cudaSuccess; }
cudaError_t cudaDeviceCanAccessPeer(int *c, int d, int p) { *c = 0; return cudaSuccess; }
cudaError_t cudaDeviceEnablePeerAccess(int p, unsigned f) { return ret(cudaErrorInvalidDevice); }
cudaError_t cudaDeviceDisablePeerAccess(int p) { return ret(cudaErrorInvalidDevice); }
cudaError_t cudaDeviceGetStreamPriorityRange(int *lo, int *hi) { if (lo) *lo = 0; if (hi) *hi = -1; return cudaSuccess; }
cudaError_t cudaDeviceSetLimit(enum cudaLimit l, size_t v) { return cudaSuccess; }
cudaError_t cudaDeviceGetLimit(size_t *v, enum cudaLimit l) { *v = l == cudaLimitStackSize ? 1024 : 0; return cudaSuccess; }
cudaError_t cudaDeviceSetCacheConfig(enum cudaFuncCache c) { return cudaSuccess; }
cudaError_t cudaDeviceSetSharedMemConfig(enum cudaSharedMemConfig c) { return cudaSuccess; }
cudaError_t cudaProfilerStart(void) { return cudaSuccess; }
cudaError_t cudaProfilerStop(void) { return cudaSuccess; }

/* ------------------------------------------------------------------ errors */
cudaError_t cudaGetLastError(void) { cudaError_t e = last_error; last_error = cudaSuccess; return e; }
cudaError_t cudaPeekAtLastError(void) { return last_error; }
const char *cudaGetErrorName(cudaError_t e) {
    switch (e) {
#define N(x) case x: return #x;
    N(cudaSuccess) N(cudaErrorInvalidValue) N(cudaErrorMemoryAllocation) N(cudaErrorInitializationError)
    N(cudaErrorInvalidConfiguration) N(cudaErrorInvalidDevice) N(cudaErrorInvalidDeviceFunction)
    N(cudaErrorNoDevice) N(cudaErrorInvalidPtx) N(cudaErrorLaunchFailure) N(cudaErrorNotSupported)
    N(cudaErrorNotReady) N(cudaErrorUnknown) N(cudaErrorNoKernelImageForDevice)
    N(cudaErrorStreamCaptureUnsupported) N(cudaErrorInvalidResourceHandle)
#undef N
    default: return "cudaErrorUnknown";
    }
}
const char *cudaGetErrorString(cudaError_t e) {
    if (e == cudaErrorNotSupported) return "operation not supported by the GP102 shim (yet)";
    return cudaGetErrorName(e);
}

/* ------------------------------------------------------------------ memory */
cudaError_t cudaMemGetInfo(size_t *fr, size_t *tot) { cudaError_t e = init(); return e ? e : CU(cuMemGetInfo(fr, tot)); }
cudaError_t cudaMalloc(void **p, size_t n) {
    cudaError_t e = init(); if (e) return e;
    CUdeviceptr d = 0; CUresult r = cuMemAlloc(&d, n ? n : 1);
    *p = (void *)(uintptr_t)d; return CU(r);
}
cudaError_t cudaMallocAsync(void **p, size_t n, cudaStream_t s) { return cudaMalloc(p, n); }
cudaError_t cudaFree(void *p) { return p ? CU(cuMemFree((CUdeviceptr)(uintptr_t)p)) : cudaSuccess; }
cudaError_t cudaFreeAsync(void *p, cudaStream_t s) { return cudaFree(p); }
/* "Pinned" host memory is plain container memory: it is not GPU-visible over this transport. */
cudaError_t cudaMallocHost(void **p, size_t n) { *p = aligned_alloc(4096, (n + 4095) & ~(size_t)4095); return *p ? cudaSuccess : ret(cudaErrorMemoryAllocation); }
cudaError_t cudaHostAlloc(void **p, size_t n, unsigned f) { return cudaMallocHost(p, n); }
cudaError_t cudaFreeHost(void *p) { free(p); return cudaSuccess; }
cudaError_t cudaHostRegister(void *p, size_t n, unsigned f) { return ret(cudaErrorNotSupported); }
cudaError_t cudaHostUnregister(void *p) { return cudaSuccess; }
cudaError_t cudaHostGetDevicePointer(void **d, void *h, unsigned f) { return ret(cudaErrorNotSupported); }
cudaError_t cudaMallocManaged(void **p, size_t n, unsigned f) { return ret(cudaErrorNotSupported); }
cudaError_t cudaMemAdvise(const void *p, size_t n, enum cudaMemoryAdvise a, int d) { return cudaSuccess; }
cudaError_t cudaPointerGetAttributes(struct cudaPointerAttributes *a, const void *p) {
    memset(a, 0, sizeof *a);
    if (shim_is_device_pointer((uintptr_t)p)) { a->type = cudaMemoryTypeDevice; a->device = 0; a->devicePointer = (void *)p; }
    else { a->type = cudaMemoryTypeUnregistered; a->device = -1; }
    return cudaSuccess;
}

static enum cudaMemcpyKind kind_of(void *dst, const void *src, enum cudaMemcpyKind k) {
    if (k != cudaMemcpyDefault) return k;
    int d = shim_is_device_pointer((uintptr_t)dst), s = shim_is_device_pointer((uintptr_t)src);
    return d && s ? cudaMemcpyDeviceToDevice : d ? cudaMemcpyHostToDevice : s ? cudaMemcpyDeviceToHost : cudaMemcpyHostToHost;
}
static cudaError_t copy(void *dst, const void *src, size_t n, enum cudaMemcpyKind k) {
    if (!n) return cudaSuccess;
    cudaError_t e = init(); if (e) return e;
    switch (kind_of(dst, src, k)) {
    case cudaMemcpyHostToHost: memmove(dst, src, n); return cudaSuccess;
    case cudaMemcpyHostToDevice: return CU(cuMemcpyHtoD((CUdeviceptr)(uintptr_t)dst, src, n));
    case cudaMemcpyDeviceToHost: return CU(cuMemcpyDtoH(dst, (CUdeviceptr)(uintptr_t)src, n));
    case cudaMemcpyDeviceToDevice: return CU(cuMemcpyDtoD((CUdeviceptr)(uintptr_t)dst, (CUdeviceptr)(uintptr_t)src, n));
    default: return ret(cudaErrorInvalidValue);
    }
}
cudaError_t cudaMemcpy(void *d, const void *s, size_t n, enum cudaMemcpyKind k) { return copy(d, s, n, k); }
cudaError_t cudaMemcpyAsync(void *d, const void *s, size_t n, enum cudaMemcpyKind k, cudaStream_t st) { return copy(d, s, n, k); }
cudaError_t cudaMemcpyPeer(void *d, int dd, const void *s, int sd, size_t n) { return copy(d, s, n, cudaMemcpyDeviceToDevice); }
cudaError_t cudaMemcpyPeerAsync(void *d, int dd, const void *s, int sd, size_t n, cudaStream_t st) { return copy(d, s, n, cudaMemcpyDeviceToDevice); }
cudaError_t cudaMemcpy2D(void *dst, size_t dp, const void *src, size_t sp, size_t w, size_t h, enum cudaMemcpyKind k) {
    if (dp == w && sp == w) return copy(dst, src, w * h, k);
    k = kind_of(dst, src, k);
    if (k == cudaMemcpyDeviceToHost) {               /* one round trip: fetch the whole span, then scatter */
        size_t span = sp * (h - 1) + w; char *tmp = malloc(span);
        cudaError_t e = copy(tmp, src, span, k);
        for (size_t r = 0; !e && r < h; r++) memcpy((char *)dst + r * dp, tmp + r * sp, w);
        free(tmp); return e;
    }
    for (size_t r = 0; r < h; r++) {
        cudaError_t e = copy((char *)dst + r * dp, (const char *)src + r * sp, w, k);
        if (e) return e;
    }
    return cudaSuccess;
}
cudaError_t cudaMemcpy2DAsync(void *dst, size_t dp, const void *src, size_t sp, size_t w, size_t h,
                              enum cudaMemcpyKind k, cudaStream_t st) { return cudaMemcpy2D(dst, dp, src, sp, w, h, k); }
cudaError_t cudaMemset(void *p, int v, size_t n) {
    cudaError_t e = init(); if (e) return e;
    return n ? CU(cuMemsetD8((CUdeviceptr)(uintptr_t)p, (unsigned char)v, n)) : cudaSuccess;
}
cudaError_t cudaMemsetAsync(void *p, int v, size_t n, cudaStream_t s) { return cudaMemset(p, v, n); }

/* ------------------------------------------------------------------ streams / events */
struct stream { int dummy; };
cudaError_t cudaStreamCreate(cudaStream_t *s) { *s = calloc(1, sizeof(struct stream)); return init(); }
cudaError_t cudaStreamCreateWithFlags(cudaStream_t *s, unsigned f) { return cudaStreamCreate(s); }
cudaError_t cudaStreamCreateWithPriority(cudaStream_t *s, unsigned f, int p) { return cudaStreamCreate(s); }
cudaError_t cudaStreamDestroy(cudaStream_t s) { free(s); return cudaSuccess; }
cudaError_t cudaStreamSynchronize(cudaStream_t s) { return cudaDeviceSynchronize(); }
cudaError_t cudaStreamQuery(cudaStream_t s) { return cudaDeviceSynchronize(); }
cudaError_t cudaStreamWaitEvent(cudaStream_t s, cudaEvent_t e, unsigned f) { return cudaSuccess; }
cudaError_t cudaStreamIsCapturing(cudaStream_t s, enum cudaStreamCaptureStatus *st) { *st = cudaStreamCaptureStatusNone; return cudaSuccess; }
cudaError_t cudaStreamGetCaptureInfo_v2(cudaStream_t s, enum cudaStreamCaptureStatus *st, unsigned long long *id,
                                        cudaGraph_t *g, const cudaGraphNode_t **deps, size_t *n) {
    *st = cudaStreamCaptureStatusNone; return cudaSuccess;
}
cudaError_t cudaThreadExchangeStreamCaptureMode(enum cudaStreamCaptureMode *m) { return cudaSuccess; }
cudaError_t cudaLaunchHostFunc(cudaStream_t s, cudaHostFn_t fn, void *data) {
    cudaError_t e = cudaDeviceSynchronize();
    if (!e) fn(data);
    return e;
}
static double now_ms(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec * 1e3 + t.tv_nsec / 1e6; }
struct event { double t; };
cudaError_t cudaEventCreate(cudaEvent_t *e) { *e = calloc(1, sizeof(struct event)); return cudaSuccess; }
cudaError_t cudaEventCreateWithFlags(cudaEvent_t *e, unsigned f) { return cudaEventCreate(e); }
cudaError_t cudaEventDestroy(cudaEvent_t e) { free(e); return cudaSuccess; }
cudaError_t cudaEventRecord(cudaEvent_t e, cudaStream_t s) { ((struct event *)e)->t = now_ms(); return cudaSuccess; }
cudaError_t cudaEventRecordWithFlags(cudaEvent_t e, cudaStream_t s, unsigned f) { return cudaEventRecord(e, s); }
cudaError_t cudaEventSynchronize(cudaEvent_t e) { cudaError_t r = cudaDeviceSynchronize(); ((struct event *)e)->t = now_ms(); return r; }
cudaError_t cudaEventQuery(cudaEvent_t e) { return cudaEventSynchronize(e); }
cudaError_t cudaEventElapsedTime(float *ms, cudaEvent_t a, cudaEvent_t b) {
    *ms = (float)(((struct event *)b)->t - ((struct event *)a)->t); return cudaSuccess;
}

/* ------------------------------------------------------------------ functions / occupancy */
cudaError_t cudaFuncSetAttribute(const void *f, enum cudaFuncAttribute a, int v) { return cudaSuccess; }
cudaError_t cudaFuncSetCacheConfig(const void *f, enum cudaFuncCache c) { return cudaSuccess; }
cudaError_t cudaFuncSetSharedMemConfig(const void *f, enum cudaSharedMemConfig c) { return cudaSuccess; }
static int fattr(CUfunction f, CUfunction_attribute a) { int v = 0; cuFuncGetAttribute(&v, a, f); return v; }
cudaError_t cudaFuncGetAttributes(struct cudaFuncAttributes *a, const void *func) {
    pthread_mutex_lock(&reg_lk); struct kfunc *k = kget(func); pthread_mutex_unlock(&reg_lk);
    if (!k) return ret(cudaErrorInvalidDeviceFunction);
    cudaError_t e = resolve(k); if (e) return e;
    memset(a, 0, sizeof *a);
    a->sharedSizeBytes = fattr(k->f, CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES);
    a->constSizeBytes = fattr(k->f, CU_FUNC_ATTRIBUTE_CONST_SIZE_BYTES);
    a->localSizeBytes = fattr(k->f, CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES);
    a->maxThreadsPerBlock = 1024;
    a->numRegs = fattr(k->f, CU_FUNC_ATTRIBUTE_NUM_REGS);
    a->ptxVersion = 61; a->binaryVersion = 61;
    a->maxDynamicSharedSizeBytes = 49152 - (int)a->sharedSizeBytes;
    return cudaSuccess;
}
cudaError_t cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(int *n, const void *func, int bs, size_t smem, unsigned fl) {
    struct cudaFuncAttributes a;
    cudaError_t e = cudaFuncGetAttributes(&a, func); if (e) return e;
    if (bs <= 0) { *n = 0; return cudaSuccess; }
    int warps = (bs + 31) / 32, regs = a.numRegs ? a.numRegs : 32;
    int regs_per_warp = ((regs * 32 + 255) / 256) * 256;
    int by_threads = 64 / warps, by_regs = 65536 / (regs_per_warp * warps), by_blocks = 32;
    size_t sh = ((a.sharedSizeBytes + smem + 255) / 256) * 256;
    int by_smem = sh ? (int)(98304 / sh) : 32;
    int m = by_threads < by_regs ? by_threads : by_regs;
    if (by_smem < m) m = by_smem;
    if (by_blocks < m) m = by_blocks;
    *n = m; return cudaSuccess;
}
cudaError_t cudaOccupancyMaxActiveBlocksPerMultiprocessor(int *n, const void *func, int bs, size_t smem) {
    return cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(n, func, bs, smem, 0);
}

/* ------------------------------------------------------------------ graphs: not supported */
cudaError_t cudaStreamBeginCapture(cudaStream_t s, enum cudaStreamCaptureMode m) { return ret(cudaErrorStreamCaptureUnsupported); }
