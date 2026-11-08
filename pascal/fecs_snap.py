"""Read-only FECS / PFIFO snapshot for the rung-4 channel-switch stall.

Nothing here changes engine state: the only writes are to falcon *debug ports*
(ICD_CMD for RREG, DMEMC to select a DMEM read address), which do not touch the
falcon's execution or its interrupt/mailbox state.

Falcon register map (envytools rnndb falcon.xml / nvgpu hw_falcon):
  0x008 IRQSTAT (R)   0x00c IRQMODE   0x018 IRQMASK (R)   0x01c IRQDEST
  0x000/0x004 IRQSSET/IRQSCLR and 0x010/0x014 IRQMSET/IRQMCLR are WRITE-ONLY:
  never write them from a probe (session 5 wrote 0x409014 = IRQMCLR by mistake).
  0x120 XFER_STATUS   0x1c0/0x1c4 DMEMC(0)/DMEMD(0)   0x200 ICD_CMD   0x20c ICD_RDATA
FECS-specific (nvgpu hw_gr_gp10b): 0x800+4i CTXSW_MAILBOX(i) (i<16),
  0x840+4i MAILBOX_CLEAR(i), 0x8c0+4i MAILBOX_SET(i) (write-only),
  0xb00 CURRENT_CTX, 0xb04 NEW_CTX, 0xb0c ctxsw ack (fw writes 1).
fecs_inst.bin DMEM layout (from the disassembly, session 6):
  queue A (CHSW + default methods): entries 0x7e4, put/get 0x824/0x828
  queue B (ext 9/15, methods 0x4,0x5,0x7,0x38..0x3a): entries 0x82c, 0x86c/0x870
  queue C (ext 10, method 0x10002): entries 0x874, 0x8b4/0x8b8
  flags 0x704, 0x740, 0x748, 0x750, 0x758 (method handlers set/clear them)
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

FECS = 0x409000
# nvgpu enum falcon_reg
REG = {"SP": 20, "PC": 21, "CSW": 24, "IV0": 16, "EV": 19, "EXCI": 28}
DMEM_WORDS = (0x700, 0x704, 0x740, 0x748, 0x750, 0x758, 0x824, 0x828, 0x86c, 0x870, 0x8b4, 0x8b8)


def icd_rreg(c, base, idx):
    c.wr(base + 0x200, 0xe | ((idx & 0x1f) << 8))      # ICD_CMD: OPC_RREG | IDX
    time.sleep(0.001)
    return c.rd(base + 0x20c)


def trace_pcs(c, base, n=8):
    """Falcon branch-trace buffer (nvgpu falcon_dump_stats): TRACEIDX 0x148 bits 16:23
    = entries, write the index, read TRACEPC 0x14c.  ICD/DMEM are locked in LS mode."""
    ti = c.rd(base + 0x148)
    cnt = (ti >> 16) & 0xff
    out = []
    for i in range(min(cnt, n)):
        c.wr(base + 0x148, i)
        out.append("%05x" % c.rd(base + 0x14c))
    return "traceidx=%08x [%s]" % (ti, " ".join(out))


def dmem_r32(c, base, addr):
    c.wr(base + 0x1c0, (addr & 0xfffc))                  # DMEMC(0): offset, no auto-inc
    return c.rd(base + 0x1c4)


def snap(c, tag="", chid=None, base=FECS, pcs=4):
    r = c.rd
    print("== FECS/PFIFO snapshot %s" % tag)
    print("  falcon IRQSTAT=%08x IRQMASK=%08x IRQMODE=%08x IRQDEST=%08x CPUCTL=%08x SCTL=%08x XFER=%08x DMACTL=%08x"
          % tuple(r(base + o) for o in (0x008, 0x018, 0x00c, 0x01c, 0x100, 0x240, 0x120, 0x10c)))
    print("  mailbox 0..15: " + " ".join("%08x" % r(base + 0x800 + 4 * i) for i in range(16)))
    print("  CURRENT_CTX b00=%08x NEW_CTX b04=%08x b08=%08x b0c=%08x  a20=%08x a24=%08x"
          % tuple(r(base + o) for o in (0xb00, 0xb04, 0xb08, 0xb0c, 0xa20, 0xa24)))
    print("  TRACEPC: %s" % trace_pcs(c, base))
    if "--dmem" in sys.argv:   # host DMEM port reads 0xbadf548a while FECS is in LS mode
        print("  DMEM: " + " ".join("%03x=%08x" % (a, dmem_r32(c, base, a)) for a in DMEM_WORDS))
    print("  PFIFO: FIFO_INTR=%08x sched_dis 2630=%08x preempt 2634=%08x eng_fault 262c=%08x"
          " eng0 status 2640=%08x 2644=%08x  runl0 2284=%08x"
          % (r(0x2100), r(0x2630), r(0x2634), r(0x262c), r(0x2640), r(0x2644), r(0x2284)))
    print("  PBDMA0: 40100=%08x intr 40108=%08x intr1 40148=%08x chan 40120=%08x ib_get=%08x pb_pos=%08x"
          % (r(0x40100), r(0x40108), r(0x40148), r(0x40120), r(0x40014), r(0x40054)))
    if chid is not None:
        st = r(0x800004 + chid * 8)
        print("  PCCSR ch%d 800004=%08x: enabled=%d next=%d status=%d busy=%d"
              % (chid, st, st & 1, (st >> 1) & 1, (st >> 24) & 0xf, (st >> 28) & 1))
    print("  PGRAPH intr 400100=%08x 400700=%08x 400704=%08x 409c18=%08x"
          % (r(0x400100), r(0x400700), r(0x400704), r(0x409c18)))


if __name__ == "__main__":
    from card import Card
    snap(Card(), "idle")
