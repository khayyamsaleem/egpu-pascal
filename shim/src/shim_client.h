/* Shared by the libcuda / libnvidia-ml shims: one TCP connection to pascal/shimd.py.
 * Protocol: request <u32 op><u32 len>payload, reply <i32 status><u32 len>payload.
 * SHIM_MODE=remote enables it; SHIM_SERVER=host:port (default host.docker.internal:7461). */
#ifndef SHIM_CLIENT_H
#define SHIM_CLIENT_H
#include <netdb.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <unistd.h>

enum { OP_INFO = 1, OP_ALLOC, OP_FREE, OP_HTOD, OP_DTOH, OP_DTOD, OP_MEMSET, OP_MODULE, OP_LAUNCH };

static int shim_fd = -1;
static pthread_mutex_t shim_rpc_lk = PTHREAD_MUTEX_INITIALIZER;

static int shim_remote(void) {
    const char *m = getenv("SHIM_MODE");
    return m && !strcmp(m, "remote");
}

static int shim_connect(void) {
    if (shim_fd >= 0) return 0;
    const char *srv = getenv("SHIM_SERVER");
    char host[256] = "host.docker.internal", port[16] = "7461";
    if (srv) {
        const char *c = strrchr(srv, ':');
        if (c) { snprintf(host, sizeof host, "%.*s", (int)(c - srv), srv); snprintf(port, sizeof port, "%s", c + 1); }
        else snprintf(host, sizeof host, "%s", srv);
    }
    struct addrinfo hints = {0}, *res;
    hints.ai_socktype = SOCK_STREAM;
    if (getaddrinfo(host, port, &hints, &res)) { fprintf(stderr, "[shim] cannot resolve %s\n", host); return -1; }
    int fd = socket(res->ai_family, SOCK_STREAM, 0);
    if (connect(fd, res->ai_addr, res->ai_addrlen)) { fprintf(stderr, "[shim] cannot connect to %s:%s\n", host, port); freeaddrinfo(res); close(fd); return -1; }
    freeaddrinfo(res);
    int one = 1; setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof one);
    shim_fd = fd;
    return 0;
}

static int shim_io(int send_, void *buf, size_t n) {
    char *p = (char *)buf;
    while (n) {
        ssize_t k = send_ ? send(shim_fd, p, n, 0) : recv(shim_fd, p, n, 0);
        if (k <= 0) return -1;
        p += k; n -= (size_t)k;
    }
    return 0;
}

/* Send op with up to two payload pieces; reply payload goes to out (malloc'd if *out == NULL). */
static int shim_rpc(uint32_t op, const void *a, uint32_t na, const void *b, uint32_t nb,
                    void **out, uint32_t *nout) {
    int status = 999;
    uint32_t hdr[2] = {op, na + nb};
    int32_t rh[2] = {0, 0};
    uint32_t n = 0;
    void *dst = NULL;
    pthread_mutex_lock(&shim_rpc_lk);
    if (shim_connect()) goto done;
    if (shim_io(1, hdr, 8) || (na && shim_io(1, (void *)a, na)) || (nb && shim_io(1, (void *)b, nb))) goto done;
    if (shim_io(0, rh, 8)) goto done;
    n = (uint32_t)rh[1];
    dst = out && *out ? *out : (n ? malloc(n) : NULL);
    if (n && shim_io(0, dst, n)) goto done;
    if (out) *out = dst; else free(dst);
    if (nout) *nout = n;
    status = rh[0];
done:
    pthread_mutex_unlock(&shim_rpc_lk);
    return status;
}
#endif
