/* libnvidia-ml.so.1 shim: presents the GP102 to MAX's device discovery. */
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "shim_client.h"
typedef int nvmlReturn_t;
typedef void *nvmlDevice_t;
typedef struct { unsigned long long total, free, used; } nvmlMemory_t;
typedef struct { char busIdLegacy[16]; unsigned domain, bus, device, pciDeviceId, pciSubSystemId; char busId[32]; } nvmlPciInfo_t;
static char dev_obj;
static void L(const char *s) { const char *p = getenv("SHIM_LOG"); FILE *f = p ? fopen(p, "a") : stderr; if (!f) f = stderr; fprintf(f, "[nvml-shim] %s\n", s); if (f != stderr) fclose(f); }
nvmlReturn_t nvmlInit_v2(void) { L("nvmlInit_v2"); return 0; }
nvmlReturn_t nvmlShutdown(void) { L("nvmlShutdown"); return 0; }
nvmlReturn_t nvmlDeviceGetCount_v2(unsigned *n) { *n = 1; L("nvmlDeviceGetCount -> 1"); return 0; }
nvmlReturn_t nvmlDeviceGetHandleByIndex(unsigned i, nvmlDevice_t *d) { L("nvmlDeviceGetHandleByIndex"); if (i) return 2; *d = &dev_obj; return 0; }
nvmlReturn_t nvmlDeviceGetName(nvmlDevice_t d, char *n, unsigned len) { L("nvmlDeviceGetName"); snprintf(n, len, "NVIDIA GeForce GTX 1080 Ti"); return 0; }
nvmlReturn_t nvmlDeviceGetCudaComputeCapability(nvmlDevice_t d, int *ma, int *mi) { *ma = 6; *mi = 1; L("nvmlDeviceGetCudaComputeCapability -> 6.1"); return 0; }
nvmlReturn_t nvmlDeviceGetMemoryInfo(nvmlDevice_t d, nvmlMemory_t *m) {
    m->total = 11ull << 30; m->used = 0; m->free = m->total;
    if (shim_remote()) {          /* MAX sizes its VRAM pool from this: report the server's view */
        void *out = NULL; uint32_t n = 0;
        if (shim_rpc(OP_INFO, NULL, 0, NULL, 0, &out, &n) == 0 && n >= 16) {
            memcpy(&m->free, out, 8); memcpy(&m->total, (char *)out + 8, 8); m->used = m->total - m->free;
        }
        free(out);
    }
    char b[128]; snprintf(b, sizeof b, "nvmlDeviceGetMemoryInfo -> free %llu MiB / %llu MiB", m->free >> 20, m->total >> 20); L(b);
    return 0;
}
nvmlReturn_t nvmlSystemGetDriverVersion(char *v, unsigned len) { snprintf(v, len, "580.95.05"); L("nvmlSystemGetDriverVersion -> 580.95.05"); return 0; }
nvmlReturn_t nvmlDeviceGetPciInfo_v3(nvmlDevice_t d, nvmlPciInfo_t *p) {
    memset(p, 0, sizeof *p); p->bus = 1; p->pciDeviceId = 0x1b0610de; p->pciSubSystemId = 0x66963842;
    snprintf(p->busId, sizeof p->busId, "00000000:01:00.0"); snprintf(p->busIdLegacy, sizeof p->busIdLegacy, "0000:01:00.0");
    L("nvmlDeviceGetPciInfo_v3"); return 0;
}
nvmlReturn_t nvmlDeviceGetMigMode(nvmlDevice_t d, unsigned *c, unsigned *p) { L("nvmlDeviceGetMigMode -> NOT_SUPPORTED"); return 3; }
nvmlReturn_t nvmlDeviceGetTotalEccErrors(nvmlDevice_t d, int t, int c, unsigned long long *n) { L("nvmlDeviceGetTotalEccErrors -> NOT_SUPPORTED"); return 3; }
nvmlReturn_t nvmlDeviceGetCurrentClocksThrottleReasons(nvmlDevice_t d, unsigned long long *r) { *r = 0; L("nvmlDeviceGetCurrentClocksThrottleReasons"); return 0; }
nvmlReturn_t nvmlDeviceGetComputeRunningProcesses_v3(nvmlDevice_t d, unsigned *n, void *info) { *n = 0; L("nvmlDeviceGetComputeRunningProcesses_v3"); return 0; }
const char *nvmlErrorString(nvmlReturn_t r) { return r == 0 ? "Success" : r == 3 ? "Not Supported" : "Unknown Error"; }
void shim_log_stub(const char *name) { char b[256]; snprintf(b, sizeof b, "%s -> NOT_SUPPORTED (stub)", name); L(b); }
