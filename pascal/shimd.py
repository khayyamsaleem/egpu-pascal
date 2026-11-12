"""shimd: the macOS-side server behind the libcuda/NVML shim (shim/, SHIM_MODE=remote).

Owns one nvpascal.Device and serves a tiny binary protocol over TCP:

  request  = <u32 op> <u32 len> payload        reply = <i32 status> <u32 len> payload
  status 0 = ok, otherwise a CUresult the shim returns to its caller.

  op 1 INFO                      -> <u64 free> <u64 total> <u32 sm_count> <u32 sm_khz> <u32 mem_khz>
  op 2 ALLOC   <u64 size>        -> <u64 va>
  op 3 FREE    <u64 va>
  op 4 HTOD    <u64 va> data
  op 5 DTOH    <u64 va> <u64 n>  -> data
  op 6 DTOD    <u64 dst> <u64 src> <u64 n>
  op 7 MEMSET  <u64 va> <u32 value> <u32 width 1|2|4> <u64 count>
  op 8 MODULE  image (PTX text or cubin ELF)
               -> <u32 mod> <u32 nfunc> then per function:
                  <u16 namelen> name <u32 regs> <u32 shared> <u32 nparam> nparam x (<u16 off> <u16 size>)
  op 9 LAUNCH  <u32 mod> <u32 fn> <3 x u32 grid> <3 x u32 block> <u32 smem> <u32 plen> params
  op 10 SYNC                     wait for everything queued

op | 0x80000000 = no reply (the shim sends LAUNCH, HTOD, DTOD and MEMSET this way): the work is
queued on the GPU in order and the client does not wait. A failure is kept and returned by the
connection's next replying op (CUDA's "error from an earlier async call" semantics).
DTOH and SYNC order after everything queued before them, so stream semantics hold.

PTX is JIT-compiled with CUDA 12.4 ptxas -arch=sm_61 (Docker image egpu-ptxas:12.4).

Usage: ~/.venvs/tinygpu/bin/python pascal/shimd.py [--port 7461] [--vram-limit-gib 2]
(after reset_engines.py + gr.py)
"""
import argparse, os, socket, struct, subprocess, sys, tempfile, threading, time, traceback
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import nvpascal, cubin as cubinmod

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
PTXAS_IMAGE = "egpu-ptxas:12.4"
CUDA_ERROR_INVALID_VALUE, CUDA_ERROR_OUT_OF_MEMORY, CUDA_ERROR_INVALID_PTX = 1, 2, 218
CUDA_ERROR_LAUNCH_FAILED, CUDA_ERROR_NOT_SUPPORTED = 719, 801
TRACE = bool(os.environ.get("SHIMD_TRACE"))       # log copies and kernel pointer params


def kparams(k):
    """[(offset, size)] by ordinal from KPARAM_INFO (u32 index, u16 ordinal, u16 offset, u32 flags|size<<18)."""
    out = {}
    for a, v in k["info"]:
        if a == "KPARAM_INFO" and isinstance(v, bytes) and len(v) >= 12:
            _, ordinal, off, w = struct.unpack_from("<IHHI", v)
            out[ordinal] = (off, (w >> 18) & 0x3fff)
    return [out[i] for i in sorted(out)]


class Server:
    def __init__(self, limit):
        t0 = time.time()
        self.dev = nvpascal.Device()
        self.limit = limit
        self.used = 0
        self.modules = []
        self.mark = self.dev.module_mark()   # code/module-data watermark after the builtin kernels
        self.active = 0                      # open client connections
        self.lock = threading.Lock()          # one GPU, one op at a time (each shim .so has its own socket)
        print("shimd: GP102 up in %.1f s; VRAM limit %.1f GiB" % (time.time() - t0, limit / 2**30), flush=True)

    # -------------------------------------------------------------- ops
    def info(self, p):
        return struct.pack("<QQIII", self.limit - self.used, self.limit, 28, 607000, 405000)

    def alloc(self, p, owner=None):
        (n,) = struct.unpack("<Q", p)
        if self.used + n > self.limit:
            raise Cu(CUDA_ERROR_OUT_OF_MEMORY)
        t0 = time.time()
        buf = self.dev.alloc(n)
        self.used += buf.size
        if owner is not None:
            owner.append(buf.va)
        print("  alloc %d MiB -> va %#x (%.2f s)" % (n >> 20, buf.va, time.time() - t0), flush=True)
        return struct.pack("<Q", buf.va)

    def free(self, p, owner=None):
        (va,) = struct.unpack("<Q", p)
        if owner is not None and va in owner:
            owner.remove(va)
        buf, off = self.dev.find(va)
        self.used -= buf.size
        self.dev.free(va)
        return b""

    def _pa(self, va, n):
        buf, off = self.dev.find(va)
        if off + n > buf.size:
            raise Cu(CUDA_ERROR_INVALID_VALUE)
        return buf.pa + off

    def htod(self, p):
        (va,) = struct.unpack_from("<Q", p)
        data = p[8:]
        if TRACE:
            print("T htod %x +%d" % (va, len(data)), flush=True)
        self._pa(va, len(data))
        self.dev.htod(va, data)
        return b""

    def dtoh(self, p):
        va, n = struct.unpack("<QQ", p)
        self._pa(va, n)
        return self.dev.dtoh(va, n)

    def dtod(self, p):
        dst, src, n = struct.unpack("<QQQ", p)
        if TRACE:
            print("T dtod %x <- %x +%d" % (dst, src, n), flush=True)
        self._pa(dst, n); self._pa(src, n)
        self.dev.copy_async(dst, src, n)
        return b""

    def memset(self, p):
        va, value, width, count = struct.unpack("<QIIQ", p)
        n = count * width
        self._pa(va, n)
        pat = {1: (value & 0xff) * 0x01010101, 2: (value & 0xffff) * 0x00010001, 4: value}[width]
        self.dev.fill_async(va, pat, n)
        return b""

    def module(self, p):
        if p[:4] == b"\x7fELF":
            image = p
        else:
            image = self._ptxas(p.rstrip(b"\0"))
        mid = len(self.modules)
        dump = os.environ.get("SHIMD_DUMP")          # keep every module for offline inspection
        if dump:
            os.makedirs(dump, exist_ok=True)
            if p[:4] != b"\x7fELF":
                open(os.path.join(dump, "mod%03d.ptx" % mid), "wb").write(p.rstrip(b"\0"))
            open(os.path.join(dump, "mod%03d.cubin" % mid), "wb").write(image)
        ks = self.dev.load(image)
        order = list(ks.values())
        self.modules.append(order)
        out = struct.pack("<II", mid, len(order))
        for k in order:
            name = k["name"].encode()
            ps = kparams(k)
            k["params"] = ps
            out += struct.pack("<H", len(name)) + name + struct.pack("<III", k["regs"], k["shared"], len(ps))
            out += b"".join(struct.pack("<HH", o, s) for o, s in ps)
            print("  module %d: %s (%d regs, %d params / %d bytes, frame %d, stack %d, smem %d)"
                  % (mid, k["name"][:60], k["regs"], len(ps), max([o + n for o, n in ps] or [0]),
                     k["frame"], k["max_stack"], k["shared"]), flush=True)
        return out

    def _ptxas(self, ptx):
        d = tempfile.mkdtemp(prefix="shimd-")
        open(os.path.join(d, "k.ptx"), "wb").write(ptx)
        r = subprocess.run(["docker", "run", "--rm", "-v", "%s:/k" % d, PTXAS_IMAGE,
                            "ptxas", "-arch=sm_61", "-o", "k.cubin", "k.ptx"], capture_output=True, text=True)
        if r.returncode:
            print("  ptxas failed:\n" + r.stderr, flush=True)
            raise Cu(CUDA_ERROR_INVALID_PTX)
        return open(os.path.join(d, "k.cubin"), "rb").read()

    def launch(self, p):
        mid, fi, gx, gy, gz, bx, by, bz, smem, plen = struct.unpack_from("<10I", p)
        params = p[40:40 + plen]
        k = self.modules[mid][fi]
        if TRACE:                                   # device pointers among the 8-byte params
            ptrs = sorted({v for (v,) in struct.iter_unpack("<Q", params[:len(params) // 8 * 8])
                           if nvpascal.DATA_VA0 <= v < nvpascal.DATA_VA0 + (16 << 30)})
            print("T launch %s grid %d,%d,%d ptrs %s" % (k["name"][:48], gx, gy, gz,
                  " ".join("%x" % v for v in ptrs)), flush=True)
        self.dev.launch_async(k, (gx, gy, gz), (bx, by, bz), params, smem=smem)
        return b""

    def sync(self, p):
        self.dev.drain()
        return b""

    def prof(self):
        if not self.dev.prof:
            return
        tot = sum(v[1] for v in self.dev.prof.values())
        print("shimd: GPU time by kernel (NVPASCAL_PROF), %.1f ms total:" % (tot / 1e6), flush=True)
        for name, (n, ns) in sorted(self.dev.prof.items(), key=lambda kv: -kv[1][1])[:25]:
            print("  %8.1f ms %6.1f%% %7d x %7.1f us  %s" % (ns / 1e6, 100 * ns / tot, n, ns / n / 1e3, name[:90]), flush=True)
        self.dev.prof.clear()

    OPS = {1: "info", 2: "alloc", 3: "free", 4: "htod", 5: "dtoh", 6: "dtod", 7: "memset", 8: "module", 9: "launch",
           10: "sync"}
    NOREPLY = 0x80000000

    def serve(self, port):
        ls = socket.socket(); ls.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        ls.bind(("127.0.0.1", port)); ls.listen(4)
        print("shimd: listening on 127.0.0.1:%d" % port, flush=True)
        while True:
            conn, addr = ls.accept()
            conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            print("shimd: client %s" % (addr,), flush=True)
            threading.Thread(target=self._client_thread, args=(conn, addr), daemon=True).start()

    def _client_thread(self, conn, addr):
        owned = []
        with self.lock:
            self.active += 1
        try:
            self._client(conn, owned)
        except (ConnectionError, EOFError):
            pass
        except nvpascal.SystemSlept as e:       # cleanup below would itself be MMIO: stop now
            print("shimd: %s -- exiting without touching the card" % e, flush=True)
            os._exit(3)
        finally:
            conn.close()
            n_owned = len(owned)
            with self.lock:
                try:
                    self.dev.drain()                      # its queued work, before freeing anything
                except Exception:
                    traceback.print_exc()
                self.prof()
                for va in list(owned):
                    self.free(struct.pack("<Q", va), owned)
                if not self.dev.allocs:                  # nothing live: rewind the bump heaps
                    self.dev.data_heap = nvpascal.nvmmu.VramHeap(base=nvpascal.DATA_PA0)
                    self.dev.va_next = nvpascal.DATA_VA0
                    self.used = 0
                self.active -= 1
                if not self.active:                      # no process left: drop its modules
                    self.dev.module_rewind(self.mark)
                    self.modules = []
            print("shimd: client %s gone (released %d allocations, %d MiB in use)"
                  % (addr, n_owned, self.used >> 20), flush=True)

    def _client(self, conn, owned):
        def recvn(n):
            b = bytearray()
            while len(b) < n:
                ch = conn.recv(min(n - len(b), 1 << 20))
                if not ch:
                    raise EOFError
                b += ch
            return bytes(b)
        sticky = 0                                   # first failure of a no-reply op
        while True:
            op, ln = struct.unpack("<II", recvn(8))
            payload = recvn(ln) if ln else b""
            noreply, op = op & self.NOREPLY, op & ~self.NOREPLY
            status, out = 0, b""
            try:
                with self.lock:
                    fn = getattr(self, self.OPS[op])
                    out = fn(payload, owned) if op in (2, 3) else fn(payload)
            except Cu as e:
                status = e.code
            except KeyError as e:
                print("  op %d: %s" % (op, e), flush=True); status = CUDA_ERROR_INVALID_VALUE
            except Exception:
                traceback.print_exc(); status = CUDA_ERROR_LAUNCH_FAILED
            if noreply:
                sticky = sticky or status
                continue
            if sticky and not status:
                status, out, sticky = sticky, b"", 0
            conn.sendall(struct.pack("<iI", status, len(out)) + out)


class Cu(Exception):
    def __init__(self, code):
        self.code = code


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=7461)
    ap.add_argument("--vram-limit-gib", type=float, default=2.0)
    a = ap.parse_args()
    Server(int(a.vram_limit_gib * 2**30)).serve(a.port)
