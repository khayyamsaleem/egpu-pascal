/*
 * libcuda.so.1 shim for Modular MAX -> GTX 1080 Ti (GP102) over TinyGPU.
 *
 * SHIM_MODE unset / "sim"  DISCOVERY: one simulated GP102 (CC 6.1, 28 SMs, driver 580 /
 *                          CUDA 13.0); "device" memory is host memory; launches are logged,
 *                          not executed; module images saved to $SHIM_CAPTURE_DIR.
 * SHIM_MODE=remote         REAL: memory, copies, memsets, modules and launches are forwarded
 *                          to pascal/shimd.py on the Mac ($SHIM_SERVER, default
 *                          host.docker.internal:7461), which drives the card through
 *                          pascal/nvpascal.py.  Device pointers are real GPU VAs.
 *
 * Contexts/streams/events are bookkeeping in both modes: the server executes every
 * operation synchronously and in order, so stream semantics hold trivially.
 * Every symbol MAX resolves (src/cuda_symbols.txt) is exported; the rest come from
 * gen_stubs.py and return CUDA_ERROR_NOT_SUPPORTED after logging.
 * Logging: stderr, or $SHIM_LOG; $SHIM_QUIET=1 logs only errors/stubs/launches.
 */
#define _GNU_SOURCE
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <pthread.h>
#include "shim_client.h"

typedef int CUresult;
typedef int CUdevice;
typedef unsigned long long CUdeviceptr;
typedef void *CUcontext, *CUstream, *CUevent, *CUmodule, *CUfunction;

enum { CUDA_SUCCESS = 0, CUDA_ERROR_INVALID_VALUE = 1, CUDA_ERROR_OUT_OF_MEMORY = 2,
       CUDA_ERROR_NOT_INITIALIZED = 3, CUDA_ERROR_INVALID_DEVICE = 101,
       CUDA_ERROR_INVALID_HANDLE = 400, CUDA_ERROR_NOT_FOUND = 500, CUDA_ERROR_NOT_READY = 600,
       CUDA_ERROR_NOT_SUPPORTED = 801, CUDA_ERROR_UNKNOWN = 999 };

/* ------------------------------------------------------------------ logging */
static FILE *logf_;
static int quiet = -1;
static pthread_mutex_t lk = PTHREAD_MUTEX_INITIALIZER;
static void shim_vlog(int important, const char *fmt, va_list ap) {
    if (quiet < 0) quiet = getenv("SHIM_QUIET") != NULL;
    if (quiet && !important) return;
    pthread_mutex_lock(&lk);
    if (!logf_) {
        const char *p = getenv("SHIM_LOG");
        logf_ = p ? fopen(p, "a") : stderr;
        if (!logf_) logf_ = stderr;
    }
    fprintf(logf_, "[cuda-shim] "); vfprintf(logf_, fmt, ap); fputc('\n', logf_); fflush(logf_);
    pthread_mutex_unlock(&lk);
}
static void LOG(const char *fmt, ...) { va_list ap; va_start(ap, fmt); shim_vlog(0, fmt, ap); va_end(ap); }
static void LOGI(const char *fmt, ...) { va_list ap; va_start(ap, fmt); shim_vlog(1, fmt, ap); va_end(ap); }
void shim_log_stub(const char *name) { LOGI("%s -> NOT_SUPPORTED (stub)", name); }

static int remote = -1;
static int is_remote(void) { if (remote < 0) remote = shim_remote(); return remote; }

/* ------------------------------------------------------------------ the device */
#define GP102_NAME "NVIDIA GeForce GTX 1080 Ti"
static unsigned long long vram_total = 11ull << 30, vram_free = 11ull << 30;
static char ctx_obj, stream_objs[256];
static __thread CUcontext cur_ctx;
static int n_streams;
static unsigned long long allocated;

static void refresh_info(void) {
    if (!is_remote()) return;
    void *out = NULL; uint32_t n = 0;
    if (shim_rpc(OP_INFO, NULL, 0, NULL, 0, &out, &n) == 0 && n >= 16) {
        memcpy(&vram_free, out, 8); memcpy(&vram_total, (char *)out + 8, 8);
    }
    free(out);
}

static int attr_value(int a, int *ok) {
    *ok = 1;
    switch (a) {
    case 1: return 1024;            /* MAX_THREADS_PER_BLOCK */
    case 2: return 1024; case 3: return 1024; case 4: return 64;          /* MAX_BLOCK_DIM_X/Y/Z */
    case 5: return 2147483647; case 6: return 65535; case 7: return 65535;/* MAX_GRID_DIM_X/Y/Z */
    case 8: return 49152;           /* MAX_SHARED_MEMORY_PER_BLOCK */
    case 9: return 65536;           /* TOTAL_CONSTANT_MEMORY */
    case 10: return 32;             /* WARP_SIZE */
    case 11: return 2147483647;     /* MAX_PITCH */
    case 12: return 65536;          /* MAX_REGISTERS_PER_BLOCK */
    case 13: return 607000;         /* CLOCK_RATE kHz (measured boot clock) */
    case 14: return 512;            /* TEXTURE_ALIGNMENT */
    case 15: return 1;              /* GPU_OVERLAP */
    case 16: return 28;             /* MULTIPROCESSOR_COUNT */
    case 17: return 0;              /* KERNEL_EXEC_TIMEOUT */
    case 18: return 0;              /* INTEGRATED */
    case 19: return 1;              /* CAN_MAP_HOST_MEMORY */
    case 20: return 0;              /* COMPUTE_MODE default */
    case 30: return 512;            /* SURFACE_ALIGNMENT */
    case 31: return 1;              /* CONCURRENT_KERNELS */
    case 32: return 0;              /* ECC_ENABLED */
    case 33: return 1; case 34: return 0; /* PCI_BUS_ID, PCI_DEVICE_ID */
    case 35: return 0;              /* TCC_DRIVER */
    case 36: return 405000;         /* MEMORY_CLOCK_RATE kHz (boot) */
    case 37: return 352;            /* GLOBAL_MEMORY_BUS_WIDTH */
    case 38: return 2883584;        /* L2_CACHE_SIZE */
    case 39: return 2048;           /* MAX_THREADS_PER_MULTIPROCESSOR */
    case 40: return 2;              /* ASYNC_ENGINE_COUNT */
    case 41: return 1;              /* UNIFIED_ADDRESSING */
    case 50: return 0;              /* PCI_DOMAIN_ID */
    case 75: return 6; case 76: return 1;  /* COMPUTE_CAPABILITY_MAJOR/MINOR */
    case 78: return 1;              /* STREAM_PRIORITIES_SUPPORTED */
    case 79: return 1; case 80: return 1;  /* GLOBAL/LOCAL_L1_CACHE_SUPPORTED */
    case 81: return 98304;          /* MAX_SHARED_MEMORY_PER_MULTIPROCESSOR */
    case 82: return 65536;          /* MAX_REGISTERS_PER_MULTIPROCESSOR */
    case 83: return 0;              /* MANAGED_MEMORY */
    case 84: return 0;              /* MULTI_GPU_BOARD */
    case 86: return 32;             /* SINGLE_TO_DOUBLE_PRECISION_PERF_RATIO */
    case 87: return 0; case 88: return 0;  /* PAGEABLE / CONCURRENT_MANAGED access */
    case 89: return 1;              /* COMPUTE_PREEMPTION_SUPPORTED */
    case 90: return 0;              /* CAN_USE_HOST_POINTER_FOR_REGISTERED_MEM */
    case 95: return 0;              /* COOPERATIVE_LAUNCH */
    case 97: return 49152;          /* MAX_SHARED_MEMORY_PER_BLOCK_OPTIN */
    case 102: return 0;             /* VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED */
    case 106: return 32;            /* MAX_BLOCKS_PER_MULTIPROCESSOR */
    case 110: return 0;
    case 115: return 0;             /* MEMORY_POOLS_SUPPORTED */
    default: *ok = 0; return 0;
    }
}

/* ------------------------------------------------------------------ init / device */
CUresult cuInit(unsigned flags) {
    LOGI("cuInit(%u) mode=%s", flags, is_remote() ? "remote" : "sim");
    if (is_remote()) {
        if (shim_connect()) return CUDA_ERROR_NOT_INITIALIZED;
        refresh_info();
        LOGI("  server: %llu MiB free / %llu MiB", vram_free >> 20, vram_total >> 20);
    }
    return CUDA_SUCCESS;
}
CUresult cuDriverGetVersion(int *v) { *v = 13000; LOG("cuDriverGetVersion -> %d", *v); return CUDA_SUCCESS; }
CUresult cuDeviceGetCount(int *n) { *n = 1; LOG("cuDeviceGetCount -> 1"); return CUDA_SUCCESS; }
CUresult cuDeviceGet(CUdevice *d, int ord) {
    LOG("cuDeviceGet(%d)", ord);
    if (ord != 0) return CUDA_ERROR_INVALID_DEVICE;
    *d = 0; return CUDA_SUCCESS;
}
CUresult cuDeviceGetName(char *name, int len, CUdevice d) { snprintf(name, len, "%s", GP102_NAME); return CUDA_SUCCESS; }
CUresult cuDeviceGetAttribute(int *pi, int attrib, CUdevice d) {
    int ok; *pi = attr_value(attrib, &ok);
    if (ok) LOG("cuDeviceGetAttribute(%d) -> %d", attrib, *pi);
    else LOGI("cuDeviceGetAttribute(%d) -> 0 [unknown attribute]", attrib);
    return CUDA_SUCCESS;
}
CUresult cuDeviceCanAccessPeer(int *can, CUdevice a, CUdevice b) { *can = 0; return CUDA_SUCCESS; }
CUresult cuDeviceGetP2PAttribute(int *v, int attr, CUdevice a, CUdevice b) { *v = 0; return CUDA_SUCCESS; }

CUresult cuGetErrorName(CUresult e, const char **s) {
    switch (e) {
    case 0: *s = "CUDA_SUCCESS"; break;
    case 1: *s = "CUDA_ERROR_INVALID_VALUE"; break;
    case 2: *s = "CUDA_ERROR_OUT_OF_MEMORY"; break;
    case 218: *s = "CUDA_ERROR_INVALID_PTX"; break;
    case 719: *s = "CUDA_ERROR_LAUNCH_FAILED"; break;
    case 801: *s = "CUDA_ERROR_NOT_SUPPORTED"; break;
    default: *s = "CUDA_ERROR_UNKNOWN"; break;
    }
    return CUDA_SUCCESS;
}
CUresult cuGetErrorString(CUresult e, const char **s) {
    if (e == 801) { *s = "operation not supported by the GP102 shim (yet)"; return CUDA_SUCCESS; }
    return cuGetErrorName(e, s);
}

/* ------------------------------------------------------------------ contexts */
CUresult cuDevicePrimaryCtxRetain(CUcontext *c, CUdevice d) { *c = &ctx_obj; LOG("cuDevicePrimaryCtxRetain"); return CUDA_SUCCESS; }
CUresult cuDevicePrimaryCtxRelease_v2(CUdevice d) { return CUDA_SUCCESS; }
CUresult cuCtxCreate_v2(CUcontext *c, unsigned flags, CUdevice d) { *c = &ctx_obj; cur_ctx = *c; return CUDA_SUCCESS; }
CUresult cuCtxDestroy(CUcontext c) { return CUDA_SUCCESS; }
CUresult cuCtxGetCurrent(CUcontext *c) { *c = cur_ctx; return CUDA_SUCCESS; }
CUresult cuCtxSetCurrent(CUcontext c) { cur_ctx = c; return CUDA_SUCCESS; }
CUresult cuCtxPushCurrent(CUcontext c) { cur_ctx = c; return CUDA_SUCCESS; }
CUresult cuCtxPopCurrent(CUcontext *c) { if (c) *c = cur_ctx; return CUDA_SUCCESS; }
CUresult cuCtxGetStreamPriorityRange(int *least, int *greatest) {
    if (least) *least = 0;
    if (greatest) *greatest = -1;
    return CUDA_SUCCESS;
}

/* ------------------------------------------------------------------ streams / events */
static double now_ms(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec * 1e3 + t.tv_nsec / 1e6; }
struct event { double t; };
CUresult cuStreamCreate(CUstream *s, unsigned flags) { *s = &stream_objs[n_streams++ & 255]; return CUDA_SUCCESS; }
CUresult cuStreamCreateWithPriority(CUstream *s, unsigned flags, int prio) { *s = &stream_objs[n_streams++ & 255]; return CUDA_SUCCESS; }
CUresult cuStreamDestroy(CUstream s) { return CUDA_SUCCESS; }
CUresult cuStreamSynchronize(CUstream s) { LOG("cuStreamSynchronize"); return CUDA_SUCCESS; }
CUresult cuStreamWaitEvent(CUstream s, CUevent e, unsigned f) { return CUDA_SUCCESS; }
CUresult cuStreamIsCapturing(CUstream s, int *st) { *st = 0; return CUDA_SUCCESS; }
CUresult cuEventCreate(CUevent *e, unsigned flags) { *e = calloc(1, sizeof(struct event)); return CUDA_SUCCESS; }
CUresult cuEventDestroy(CUevent e) { free(e); return CUDA_SUCCESS; }
CUresult cuEventRecord(CUevent e, CUstream s) { ((struct event *)e)->t = now_ms(); return CUDA_SUCCESS; }
CUresult cuEventQuery(CUevent e) { return CUDA_SUCCESS; }
CUresult cuEventSynchronize(CUevent e) { return CUDA_SUCCESS; }
CUresult cuEventElapsedTime(float *ms, CUevent a, CUevent b) { *ms = (float)(((struct event *)b)->t - ((struct event *)a)->t); return CUDA_SUCCESS; }

/* ------------------------------------------------------------------ memory */
CUresult cuMemGetInfo_v2(size_t *fr, size_t *tot) {
    refresh_info();
    *tot = vram_total; *fr = is_remote() ? vram_free : vram_total - allocated;
    LOG("cuMemGetInfo -> free %zu total %zu", *fr, *tot); return CUDA_SUCCESS;
}
static CUresult dev_alloc(CUdeviceptr *p, size_t n, const char *who) {
    if (is_remote()) {
        uint64_t sz = n, va = 0; void *out = &va; uint32_t got;
        int st = shim_rpc(OP_ALLOC, &sz, 8, NULL, 0, &out, &got);
        LOGI("%s(%zu) -> %#llx (status %d)", who, n, (unsigned long long)va, st);
        if (st) return st;
        *p = va; return CUDA_SUCCESS;
    }
    void *m = aligned_alloc(4096, (n + 4095) & ~4095ull);
    if (!m) return CUDA_ERROR_OUT_OF_MEMORY;
    *p = (CUdeviceptr)(uintptr_t)m; allocated += n;
    LOG("%s(%zu) -> %#llx [sim]", who, n, *p); return CUDA_SUCCESS;
}
static CUresult dev_free(CUdeviceptr p) {
    if (is_remote()) { uint64_t va = p; return shim_rpc(OP_FREE, &va, 8, NULL, 0, NULL, NULL); }
    free((void *)(uintptr_t)p); return CUDA_SUCCESS;
}
CUresult cuMemAlloc_v2(CUdeviceptr *p, size_t n) { return dev_alloc(p, n, "cuMemAlloc"); }
CUresult cuMemAllocAsync(CUdeviceptr *p, size_t n, CUstream s) { return dev_alloc(p, n, "cuMemAllocAsync"); }
CUresult cuMemFree_v2(CUdeviceptr p) { LOG("cuMemFree(%#llx)", p); return dev_free(p); }
CUresult cuMemFreeAsync(CUdeviceptr p, CUstream s) { LOG("cuMemFreeAsync(%#llx)", p); return dev_free(p); }
/* pinned host memory stays local: the container's memory is not GPU-visible */
CUresult cuMemAllocHost_v2(void **p, size_t n) { *p = aligned_alloc(4096, (n + 4095) & ~4095ull); LOG("cuMemAllocHost(%zu)", n); return *p ? 0 : 2; }
CUresult cuMemHostAlloc(void **p, size_t n, unsigned f) { *p = aligned_alloc(4096, (n + 4095) & ~4095ull); LOG("cuMemHostAlloc(%zu, %#x)", n, f); return *p ? 0 : 2; }
CUresult cuMemFreeHost(void *p) { free(p); return CUDA_SUCCESS; }
CUresult cuMemHostGetDevicePointer_v2(CUdeviceptr *d, void *h, unsigned f) {
    if (is_remote()) { LOGI("cuMemHostGetDevicePointer -> NOT_SUPPORTED (remote)"); return CUDA_ERROR_NOT_SUPPORTED; }
    *d = (CUdeviceptr)(uintptr_t)h; return CUDA_SUCCESS;
}
CUresult cuMemHostRegister_v2(void *p, size_t n, unsigned f) { return CUDA_SUCCESS; }
CUresult cuMemHostUnregister(void *p) { return CUDA_SUCCESS; }

#define CHUNK (8u << 20)
static CUresult htod(CUdeviceptr d, const void *s, size_t n) {
    LOG("HtoD(%#llx, %zu)", d, n);
    if (!is_remote()) { memcpy((void *)(uintptr_t)d, s, n); return 0; }
    for (size_t off = 0; off < n; off += CHUNK) {
        uint32_t k = (uint32_t)(n - off < CHUNK ? n - off : CHUNK);
        uint64_t va = d + off;
        int st = shim_rpc(OP_HTOD, &va, 8, (const char *)s + off, k, NULL, NULL);
        if (st) return st;
    }
    return 0;
}
static CUresult dtoh(void *d, CUdeviceptr s, size_t n) {
    LOG("DtoH(%#llx, %zu)", s, n);
    if (!is_remote()) { memcpy(d, (void *)(uintptr_t)s, n); return 0; }
    for (size_t off = 0; off < n; off += CHUNK) {
        uint64_t req[2] = {s + off, n - off < CHUNK ? n - off : CHUNK};
        void *out = (char *)d + off; uint32_t got;
        int st = shim_rpc(OP_DTOH, req, 16, NULL, 0, &out, &got);
        if (st) return st;
    }
    return 0;
}
static CUresult dtod(CUdeviceptr d, CUdeviceptr s, size_t n) {
    LOG("DtoD(%#llx <- %#llx, %zu)", d, s, n);
    if (!is_remote()) { memmove((void *)(uintptr_t)d, (void *)(uintptr_t)s, n); return 0; }
    uint64_t req[3] = {d, s, n};
    return shim_rpc(OP_DTOD, req, 24, NULL, 0, NULL, NULL);
}
static CUresult dmemset(CUdeviceptr d, unsigned v, unsigned width, size_t count) {
    LOG("Memset(%#llx, %#x, w%u, %zu)", d, v, width, count);
    if (!is_remote()) {
        for (size_t i = 0; i < count; i++) memcpy((char *)(uintptr_t)d + i * width, &v, width);
        return 0;
    }
    struct __attribute__((packed)) { uint64_t va; uint32_t v, w; uint64_t n; } r = {d, v, width, count};
    return shim_rpc(OP_MEMSET, &r, sizeof r, NULL, 0, NULL, NULL);
}
CUresult cuMemcpyHtoD_v2(CUdeviceptr d, const void *s, size_t n) { return htod(d, s, n); }
CUresult cuMemcpyHtoD(CUdeviceptr d, const void *s, size_t n) { return htod(d, s, n); }
CUresult cuMemcpyHtoDAsync_v2(CUdeviceptr d, const void *s, size_t n, CUstream st) { return htod(d, s, n); }
CUresult cuMemcpyDtoH_v2(void *d, CUdeviceptr s, size_t n) { return dtoh(d, s, n); }
CUresult cuMemcpyDtoH(void *d, CUdeviceptr s, size_t n) { return dtoh(d, s, n); }
CUresult cuMemcpyDtoHAsync_v2(void *d, CUdeviceptr s, size_t n, CUstream st) { return dtoh(d, s, n); }
CUresult cuMemcpyDtoD_v2(CUdeviceptr d, CUdeviceptr s, size_t n) { return dtod(d, s, n); }
CUresult cuMemcpyDtoD(CUdeviceptr d, CUdeviceptr s, size_t n) { return dtod(d, s, n); }
CUresult cuMemcpyDtoDAsync_v2(CUdeviceptr d, CUdeviceptr s, size_t n, CUstream st) { return dtod(d, s, n); }
CUresult cuMemsetD8_v2(CUdeviceptr d, unsigned char v, size_t n) { return dmemset(d, v, 1, n); }
CUresult cuMemsetD8Async(CUdeviceptr d, unsigned char v, size_t n, CUstream s) { return dmemset(d, v, 1, n); }
CUresult cuMemsetD16_v2(CUdeviceptr d, unsigned short v, size_t n) { return dmemset(d, v, 2, n); }
CUresult cuMemsetD16Async(CUdeviceptr d, unsigned short v, size_t n, CUstream s) { return dmemset(d, v, 2, n); }
CUresult cuMemsetD32_v2(CUdeviceptr d, unsigned v, size_t n) { return dmemset(d, v, 4, n); }
CUresult cuMemsetD32Async(CUdeviceptr d, unsigned v, size_t n, CUstream s) { return dmemset(d, v, 4, n); }

/* ------------------------------------------------------------------ modules / launch */
#define MAXP 64
struct fn { char name[512]; int mod, idx, regs, shared, nparam; uint16_t off[MAXP], size[MAXP]; };
struct mod { int id, nfn; struct fn *fns; };
static int n_modules;

static size_t image_size(const unsigned char *b) {
    if (b[0] == 0x7f && b[1] == 'E' && b[2] == 'L' && b[3] == 'F') {
        uint64_t shoff = *(const uint64_t *)(b + 0x28);
        uint16_t shentsize = *(const uint16_t *)(b + 0x3a), shnum = *(const uint16_t *)(b + 0x3c);
        return shoff + (size_t)shentsize * shnum;
    }
    return strlen((const char *)b) + 1;
}

CUresult cuModuleLoadDataEx(CUmodule *m, const void *image, unsigned nopt, int *opts, void **vals) {
    const unsigned char *b = image;
    size_t sz = image_size(b);
    int id = n_modules++;
    const char *dir = getenv("SHIM_CAPTURE_DIR");
    if (dir) {
        char path[512]; snprintf(path, sizeof path, "%s/module%d.bin", dir, id);
        FILE *f = fopen(path, "wb"); if (f) { fwrite(image, 1, sz, f); fclose(f); }
    }
    struct mod *md = calloc(1, sizeof *md);
    md->id = id;
    if (is_remote()) {
        void *out = NULL; uint32_t n = 0;
        int st = shim_rpc(OP_MODULE, image, (uint32_t)sz, NULL, 0, &out, &n);
        LOGI("cuModuleLoadDataEx(%s, %zu bytes) -> status %d", b[0] == 0x7f ? "cubin" : "PTX", sz, st);
        if (st) { free(out); free(md); return st; }
        const unsigned char *p = out;
        uint32_t sid, nfn; memcpy(&sid, p, 4); memcpy(&nfn, p + 4, 4); p += 8;
        md->id = (int)sid; md->nfn = (int)nfn; md->fns = calloc(nfn, sizeof(struct fn));
        for (uint32_t i = 0; i < nfn; i++) {
            struct fn *f = &md->fns[i];
            uint16_t nl; memcpy(&nl, p, 2); p += 2;
            snprintf(f->name, sizeof f->name, "%.*s", nl, p); p += nl;
            uint32_t r3[3]; memcpy(r3, p, 12); p += 12;
            f->regs = r3[0]; f->shared = r3[1]; f->nparam = r3[2] > MAXP ? MAXP : r3[2];
            f->mod = md->id; f->idx = (int)i;
            for (uint32_t j = 0; j < r3[2]; j++, p += 4)
                if (j < MAXP) { memcpy(&f->off[j], p, 2); memcpy(&f->size[j], p + 2, 2); }
            LOGI("  function %s: %d regs, %d params", f->name, f->regs, f->nparam);
        }
        free(out);
    } else {
        LOGI("cuModuleLoadDataEx(%s, %zu bytes) -> module %d [sim]", b[0] == 0x7f ? "cubin" : "PTX", sz, id);
    }
    *m = md;
    return CUDA_SUCCESS;
}
CUresult cuModuleUnload(CUmodule m) { return CUDA_SUCCESS; }
CUresult cuModuleGetFunction(CUfunction *f, CUmodule m, const char *name) {
    struct mod *md = m;
    if (is_remote()) {
        for (int i = 0; i < md->nfn; i++)
            if (!strcmp(md->fns[i].name, name)) { *f = &md->fns[i]; return CUDA_SUCCESS; }
        LOGI("cuModuleGetFunction(\"%s\") NOT_FOUND", name);
        return CUDA_ERROR_NOT_FOUND;
    }
    struct fn *p = calloc(1, sizeof *p);
    snprintf(p->name, sizeof p->name, "%s", name); p->mod = md->id;
    *f = p; return CUDA_SUCCESS;
}
CUresult cuModuleGetGlobal_v2(CUdeviceptr *d, size_t *sz, CUmodule m, const char *name) {
    LOGI("cuModuleGetGlobal(\"%s\") NOT_FOUND", name); return CUDA_ERROR_NOT_FOUND;
}
CUresult cuFuncGetAttribute(int *v, int attr, CUfunction fp) {
    struct fn *f = fp;
    switch (attr) {
    case 0: *v = 1024; break;            /* MAX_THREADS_PER_BLOCK */
    case 1: *v = f->shared; break;       /* SHARED_SIZE_BYTES */
    case 2: *v = 0; break;               /* CONST_SIZE_BYTES */
    case 3: *v = 0; break;               /* LOCAL_SIZE_BYTES */
    case 4: *v = f->regs; break;         /* NUM_REGS */
    case 5: case 6: *v = 61; break;      /* PTX_VERSION / BINARY_VERSION */
    case 8: *v = 49152 - f->shared; break; /* MAX_DYNAMIC_SHARED_SIZE_BYTES */
    default: *v = 0; break;
    }
    return CUDA_SUCCESS;
}
CUresult cuFuncSetAttribute(CUfunction f, int attr, int v) { return CUDA_SUCCESS; }
CUresult cuOccupancyMaxActiveBlocksPerMultiprocessor(int *n, CUfunction f, int bs, size_t smem) {
    *n = 2048 / (bs ? bs : 1); if (*n > 32) *n = 32; return 0;
}

static CUresult launch(struct fn *f, unsigned gx, unsigned gy, unsigned gz, unsigned bx, unsigned by, unsigned bz,
                       unsigned smem, void **params) {
    if (!is_remote()) {
        LOGI("launch %s grid %u,%u,%u block %u,%u,%u smem %u [sim: not executed]", f->name, gx, gy, gz, bx, by, bz, smem);
        return CUDA_SUCCESS;
    }
    unsigned char blob[4096]; uint32_t plen = 0;
    memset(blob, 0, sizeof blob);
    for (int i = 0; i < f->nparam; i++) {
        uint32_t o = f->off[i], s = f->size[i];
        if (o + s > sizeof blob) return CUDA_ERROR_INVALID_VALUE;
        memcpy(blob + o, params[i], s);
        if (o + s > plen) plen = o + s;
    }
    uint32_t hdr[10] = {(uint32_t)f->mod, (uint32_t)f->idx, gx, gy, gz, bx, by, bz, smem, plen};
    uint64_t ns = 0; void *out = &ns; uint32_t got;
    int st = shim_rpc(OP_LAUNCH, hdr, sizeof hdr, blob, plen, &out, &got);
    LOGI("launch %s grid %u,%u,%u block %u,%u,%u smem %u -> status %d, %.1f us on GP102",
         f->name, gx, gy, gz, bx, by, bz, smem, st, ns / 1e3);
    return st;
}
CUresult cuLaunchKernel(CUfunction f, unsigned gx, unsigned gy, unsigned gz, unsigned bx, unsigned by, unsigned bz,
                        unsigned smem, CUstream s, void **params, void **extra) {
    return launch(f, gx, gy, gz, bx, by, bz, smem, params);
}
typedef struct { unsigned gx, gy, gz, bx, by, bz, smem; CUstream stream; void *attrs; unsigned nattrs; } CUlaunchConfig_;
CUresult cuLaunchKernelEx(const CUlaunchConfig_ *c, CUfunction f, void **params, void **extra) {
    return launch(f, c->gx, c->gy, c->gz, c->bx, c->by, c->bz, c->smem, params);
}
CUresult cuPointerGetAttributes(unsigned n, int *attrs, void **data, CUdeviceptr p) {
    LOGI("cuPointerGetAttributes(%u attrs, %#llx) NOT_SUPPORTED", n, p); return CUDA_ERROR_NOT_SUPPORTED;
}
