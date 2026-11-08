"""Boot the SEC2 RTOS (LS firmware) after the HS ACR, and ask it to bootstrap FECS/GPCCS.

On GP102 nouveau's ACR has no `bootstrap_falcons`, so FECS/GPCCS are booted by
the SEC2 RTOS, never by the HS ACR alone (nvkm_acr_bootstrap_falcons ->
gp102_sec2_acr_bootstrap_falcon).  The sequence reproduced here:

  1. WPR = FECS (lazy), GPCCS (lazy), SEC2 RTOS (non-lazy)   [acr.acr_load(sec2_rtos=1)]
     The HS ACR verifies all three, copies them into the WPR, and leaves SEC2
     in LS mode with the RTOS bootloader loaded.
  2. nvkm_sec2_init: 0x014 = ~0, nvkm_falcon_v1_start (0x130 if CPUCTL alias).
  3. gp102_sec2_initmsg: the RTOS posts nv_sec2_init_msg (32 B, unit 0x01) on its
     message queue; registers msgq head/tail = 0xa30/0xa34 (stride 8), cmdq
     head/tail = 0xa00/0xa04 (stride 8).  Queue offsets are DMEM-space
     addresses; >= 0x01000000 means EMEM (PIO via 0xac0/0xac4).
  4. NV_SEC2_ACR_CMD_BOOTSTRAP_FALCON (unit 0x08, 16 B, flags RESET_YES) per
     falcon over the cmdq; reply nv_sec2_acr_bootstrap_falcon_msg (16 B,
     error_code at +8, falcon_id at +12).

No interrupts on this transport, so every wait is a poll on the queue
registers (nouveau's handler is driven by SEC2 interrupt bit 6).
"""
import os, struct, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from card import Card
import acr

EMEM_ADDR = 0x01000000
CMDQ_HEAD, CMDQ_TAIL, CMDQ_STRIDE = 0xa00, 0xa04, 8
MSGQ_HEAD, MSGQ_TAIL, MSGQ_STRIDE = 0xa30, 0xa34, 8
UNIT_REWIND, UNIT_INIT, UNIT_UNLOAD, UNIT_ACR = 0x00, 0x01, 0x06, 0x08
HDR, ALIGNQ, MSG_BUF = 4, 4, 128
CMD_FLAGS = 0x3      # CMD_FLAGS_STATUS | CMD_FLAGS_INTR


def AL(x): return (x + ALIGNQ - 1) & ~(ALIGNQ - 1)


class Mem:
    """nvkm_falcon_pio_{rd,wr} with gp102's EMEM routing (emem_addr = 0x01000000)."""
    def __init__(self, f): self.f = f

    def rd(self, addr, n):
        n4 = AL(n)
        if addr >= EMEM_ADDR:
            self.f.w(0xac0, (1 << 25) | (addr - EMEM_ADDR))
            raw = struct.pack("<%dI" % (n4 // 4), *[self.f.r(0xac4) for _ in range(n4 // 4)])
        else:
            raw = self.f.dmem_read(addr, n4)
        return raw[:n]

    def wr(self, addr, data):
        data = bytes(data) + b"\0" * (-len(data) % 4)
        if addr >= EMEM_ADDR:
            self.f.w(0xac0, (1 << 24) | (addr - EMEM_ADDR))
            for v in struct.unpack("<%dI" % (len(data) // 4), data): self.f.w(0xac4, v)
        else:
            self.f.dmem_pio(data, addr)


class MsgQ:
    def __init__(self, f, index, offset, size):
        self.f, self.m = f, Mem(f)
        self.head_reg, self.tail_reg = MSGQ_HEAD + index * MSGQ_STRIDE, MSGQ_TAIL + index * MSGQ_STRIDE
        self.offset, self.size = offset, size

    def empty(self): return self.f.r(self.head_reg) == self.f.r(self.tail_reg)

    def _pop(self, n):
        head = self.f.r(self.head_reg)
        if head < self.pos: self.pos = self.offset          # the buffer looped
        if n > head - self.pos:
            raise IOError("msgq: want %d B, %d available (head %#x pos %#x)" % (n, head - self.pos, head, self.pos))
        d = self.m.rd(self.pos, n); self.pos += AL(n); return d

    def recv(self):
        """nvkm_falcon_msgq_read: one message, or None if empty."""
        self.pos = self.f.r(self.tail_reg)
        if self.empty(): return None
        hdr = self._pop(HDR)
        unit, size, ctrl, seq = hdr
        if size > MSG_BUF: raise IOError("msgq: message too big (%d)" % size)
        body = self._pop(size - HDR) if size > HDR else b""
        self.f.w(self.tail_reg, self.pos)
        return dict(unit=unit, size=size, ctrl=ctrl, seq=seq, raw=hdr + body)


class CmdQ:
    def __init__(self, f, index, offset, size):
        self.f, self.m = f, Mem(f)
        self.head_reg, self.tail_reg = CMDQ_HEAD + index * CMDQ_STRIDE, CMDQ_TAIL + index * CMDQ_STRIDE
        self.offset, self.size = offset, size

    def _has_room(self, size):
        head, tail = self.f.r(self.head_reg), self.f.r(self.tail_reg)
        size, rewind, free = AL(size), False, 0
        if head >= tail:
            free = self.offset + self.size - head - HDR
            if size > free: rewind, head = True, self.offset
        if head < tail: free = tail - head - 1
        return size <= free, rewind

    def send(self, cmd):
        ok, rewind = self._has_room(len(cmd))
        if not ok: raise IOError("cmdq full")
        pos = self.f.r(self.head_reg)
        if rewind:
            self.m.wr(pos, bytes([UNIT_REWIND, HDR, 0, 0])); pos = self.offset
        self.m.wr(pos, cmd); pos += AL(len(cmd))
        self.f.w(self.head_reg, pos)


def sec2_state(f):
    return "cpuctl=%08x sctl=%08x mbox0=%08x mbox1=%08x pc=%08x irqstat=%08x bootvec=%08x" % (
        f.r(0x100), f.r(0x240), f.r(0x040), f.r(0x044), f.trace_pc(), f.r(0x008), f.r(0x104))


def queue_regs(f):
    return " ".join("%03x=%08x" % (o, f.r(o)) for o in (0xa00, 0xa04, 0xa08, 0xa0c, 0xa30, 0xa34, 0xa38, 0xa3c))


def start_rtos(c, f, timeout=2.0, verbose=True):
    """nvkm_sec2_init + gp102_sec2_initmsg (polled)."""
    if verbose:
        print("SEC2 after HS ACR: %s" % sec2_state(f))
        print("  queues before start: %s" % queue_regs(f))
    f.w(0x014, 0xffffffff)
    f.start()
    t0 = time.time()
    got = c.wait(lambda: f.r(MSGQ_HEAD) != f.r(MSGQ_TAIL), timeout)
    dt = (time.time() - t0) * 1000
    if verbose:
        print("SEC2 RTOS start: init message %s after %.0f ms; %s" % ("POSTED" if got else "NOT posted", dt, sec2_state(f)))
        print("  queues: %s" % queue_regs(f))
    if not got:
        return None
    # nvkm_falcon_msgq_recv_initmsg: offset = tail, pop sizeof(nv_sec2_init_msg)
    tail = f.r(MSGQ_TAIL)
    raw = Mem(f).rd(tail, 32)
    unit, size, ctrl, seq, mtype, nq, dbg = struct.unpack_from("<BBBBBBH", raw, 0)
    qs = [struct.unpack_from("<IHBB", raw, 8 + 8 * i) for i in range(2)]
    swoff, swsize = struct.unpack_from("<IH", raw, 24)
    if verbose:
        print("  init msg @%#x: %s" % (tail, raw.hex()))
        print("  unit=%#x size=%d type=%d num_queues=%d os_debug_entry=%#x sw_managed=%#x+%#x"
              % (unit, size, mtype, nq, dbg, swoff, swsize))
        for off, sz, idx, qid in qs:
            print("    queue id=%d (%s) index=%d offset=%#x size=%#x" % (qid, "MSGQ" if qid == 1 else "CMDQ", idx, off, sz))
    if unit != UNIT_INIT or mtype != 0 or size != 32:
        print("  -> not a valid init message"); return None
    f.w(MSGQ_TAIL, tail + AL(size))           # close(commit)
    f.mask(0x004, 0x40, 0x40)                 # what the irq handler acks
    cmdq = msgq = None
    for off, sz, idx, qid in qs:
        if qid == 1: msgq = MsgQ(f, idx, off, sz)
        else: cmdq = CmdQ(f, idx, off, sz)
    return cmdq, msgq


# gp102 RTOS v1 reports these in its init message (sec2-rtos.log); used to talk
# to an RTOS that an earlier process started.
DEFAULT_CMDQ = (0, 0x01000000, 0x80)
DEFAULT_MSGQ = (0, 0x01000080, 0x80)


def rtos_running(f):
    """LS mode (sctl bit 0) and not halted: an RTOS from an earlier run is alive."""
    return bool(f.r(0x240) & 1) and not (f.r(0x100) & 0x10)


def rtos_unload(c, f, cmdq=None, msgq=None, verbose=True):
    """nvkm_sec2_fini: NV_SEC2_UNIT_UNLOAD (4-byte command), then wait for SEC2 to halt."""
    cmdq = cmdq or CmdQ(f, *DEFAULT_CMDQ)
    msgq = msgq or MsgQ(f, *DEFAULT_MSGQ)
    while msgq.recv() is not None:            # drain anything stale
        pass
    cmdq.send(struct.pack("<BBBB", UNIT_UNLOAD, 4, CMD_FLAGS, 0xf))
    t0 = time.time(); reply = None
    while time.time() - t0 < 1.0 and reply is None:
        reply = msgq.recv(); time.sleep(0.001)
    halted = c.wait(lambda: f.r(0x100) & 0x10, 2.0)
    if verbose:
        print("SEC2 RTOS unload: reply=%s halted=%s  %s" % (None if reply is None else reply["raw"].hex(), halted, sec2_state(f)))
    return halted


def bootstrap_falcon(c, f, cmdq, msgq, falcon_id, seq, timeout=1.0, verbose=True):
    """gp102_sec2_acr_bootstrap_falcon, polled."""
    cmd = struct.pack("<BBBBB3xII", UNIT_ACR, 16, CMD_FLAGS, seq, 0, 0, falcon_id)
    cmdq.send(cmd)
    t0 = time.time(); msg = None
    while time.time() - t0 < timeout:
        msg = msgq.recv()
        if msg is not None: break
        time.sleep(0.001)
    dt = (time.time() - t0) * 1000
    if msg is None:
        if verbose: print("  BOOTSTRAP_FALCON %d: no reply after %.0f ms; %s | %s" % (falcon_id, dt, sec2_state(f), queue_regs(f)))
        return None
    f.mask(0x004, 0x40, 0x40)
    err = fid = None
    if msg["unit"] == UNIT_ACR and len(msg["raw"]) >= 16:
        err, fid = struct.unpack_from("<II", msg["raw"], 8)
    if verbose:
        print("  BOOTSTRAP_FALCON %d (%s): reply after %.0f ms unit=%#x seq=%d size=%d raw=%s -> error_code=%s falcon_id=%s"
              % (falcon_id, {2: "FECS", 3: "GPCCS"}.get(falcon_id, "?"), dt, msg["unit"], msg["seq"], msg["size"],
                 msg["raw"].hex(), None if err is None else "%#x" % err, fid))
    return dict(msg=msg, error=err, falcon_id=fid)


if __name__ == "__main__":
    import fb, faults, acr_unload
    ver = 0 if "--v0" in sys.argv else 1
    c = Card()
    mem, iovas = c.dev.alloc_sysmem(0x1000, contiguous=True)
    fb.fb_init(c, iovas[0])
    if fb.vpr_scrub_required(c):
        nv = next(d for d in fb.top_parse(c) if d["name"].startswith("NVDEC"))
        print("VPR scrub:", fb.vpr_scrub(c, nv["addr"], nv["reset"])[:2])
    c.mask(0x200, 1 << 12, 1 << 12)            # PGRAPH on so FECS/GPCCS exist
    faults.dump(c, clear=True)
    if acr.acr_claim(c):
        u = acr_unload.acr_unload(c, verbose=False)
        print("ACR unload first: ok=%s mbox0=%#x claim now %d" % (u["ok"], u["mbox0"], acr.acr_claim(c)))
    r = acr.acr_load(c, sec2_rtos=ver)
    print("ACR load with SEC2 RTOS v%d: %s" % (ver, "PASS" if r["ok"] else "FAIL mbox0=%08x" % r["mbox0"]))
    if not r["ok"]:
        faults.dump(c); sys.exit(1)
    sec2 = acr.Falcon(c, acr.SEC2, pmc_bit=14)
    q = start_rtos(c, sec2)
    if q is None:
        faults.dump(c); print("SEC2 RTOS RESULT: FAIL (no init message)"); sys.exit(1)
    cmdq, msgq = q
    fecs, gpccs = acr.Falcon(c, acr.FECS), acr.Falcon(c, acr.GPCCS)
    print("before bootstrap: FECS %s\n                  GPCCS %s" % (sec2_state(fecs), sec2_state(gpccs)))
    c.wr(0x260, 0)                             # nvkm_mc_unk260(device, 0)
    res = [bootstrap_falcon(c, sec2, cmdq, msgq, acr.LSF_FECS, 0),
           bootstrap_falcon(c, sec2, cmdq, msgq, acr.LSF_GPCCS, 1)]
    c.wr(0x260, 1)                             # nvkm_mc_unk260(device, 1)
    print("after bootstrap:  FECS %s\n                  GPCCS %s" % (sec2_state(fecs), sec2_state(gpccs)))
    faults.dump(c)
    ok = all(x is not None and x["error"] == 0 for x in res)
    print("SEC2 RTOS RESULT:", "PASS (FECS + GPCCS bootstrapped)" if ok else "FAIL")
