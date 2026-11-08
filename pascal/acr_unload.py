"""Port of nouveau's gp102 ACR *unload* (gp102_acr_unload_fwif), run on the PMU.

Why this exists
===============
The ACR load app (acr/ucode_load.bin, run on SEC2) and the unload app
(acr/ucode_unload.bin, run on the PMU) share one piece of cross-engine state:
bits[23:20] of 0x10e5f8, guarded by falcon hardware mutex #2.  Disassembly
(envydis -m fuc -V fuc6):

  load,   main @0x100: chip check (0x16af) -> 0x220 -> ... -> WPR setup (0xffe)
          0x220   gets the chip's state value from PMC_BOOT_42 (0xa00) bits[28:20]
                  in 0x166f: 0x132 (GP102) / 0x134 -> 4, 0x136 -> 5,
                  0x137 / 0x138 -> 2, else 0
          0x15fd  takes mutex 2, reads 0x10e5f8, fails 0x29 unless bits[23:20]
                  == 0, writes bits[23:20] = 4, reads back (0x1b on mismatch)
  unload, main @0x100: chip check (0x3cd) -> 0x1b7 -> ... -> WPR teardown (0x904)
          0x322   takes mutex 2, fails 0x29 unless bits[23:20] == that same
                  value (4), clears them to 0, reads back (0x1b on mismatch)

So the field means "an ACR load has claimed this card", the load writes it
*before* the WPR range check (error 0x25), and nothing but the unload (or a cold
power cycle) clears it.  Once one load passes 0x220 and then fails later, every
further load fails at 0x29.

nouveau sequence reproduced here (subdev/acr/gp102.c, acr/gm200.c,
falcon/gm200.c, pmu/gm200.c, pmu/gp102.c):
  gp102_acr_unload_fwif = { gm200_acr_hsfw_ctor, &gm200_acr_unload_0,
                            NVKM_ACR_HSF_PMU, boot_mbox0 = 0x1d, intr_clear = 0x10 }
  firmware: acr/unload_bl.bin (start_tag 0x100 -> boot_addr 0x10000)
            acr/ucode_unload.bin
  falcon:   PMU @0x10a000, gp102_pmu_flcn (reset_eng only, debug = 0xc08)
  load:     gm200_flcn_fw_load "fw->inst" path with the PMU's own bind:
            gm200_pmu_flcn_bind_inst (FBIF 0xe00..0xe10, 0x090 bit16,
            0x480 = inst) and gm200_pmu_flcn_bind_stat (0x200 = 0x30e, then
            0x20c bits[14:12])
  bld:      gm200_acr_hsfw_load_bld, ctx_dma = VIRT, code_dma_base = the VA
  boot:     gm200_flcn_fw_boot, success = mbox0 == 0x1d
"""
import os, struct, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from card import Card
import acr
import mmu as nvmmu

PMU = 0x10a000
PMU_DEBUG = 0xc08
UNLOAD_OK = 0x1d
UNLOAD_IMG_ADDR = 0x01100000     # phys VRAM, clear of the load image at 0x01000000
UNLOAD_VA = 0x00010000
STATE_REG, STATE_FIELD = 0x10e5f8, 0x00f00000


def state(c):
    v = c.rd(STATE_REG)
    return v, (v & STATE_FIELD) >> 20


def temp_c(c):
    t = c.rd(0x20460)
    return ((t & 0x1fff8) >> 8) if t & 0x20000000 else None


def pmu_bind_inst(c, f, inst, target=0):
    """gm200_flcn_fw_load's fw->inst path with the PMU's bind_inst/bind_stat."""
    f.mask(0x048, 1, 1)
    f.w(0xe00, 4); f.w(0xe04, 0); f.w(0xe08, 4); f.w(0xe0c, 5); f.w(0xe10, 6)
    f.mask(0x090, 0x00010000, 0x00010000)
    f.w(0x480, (1 << 30) | (target << 28) | (inst >> 12))

    def stat():
        f.w(0x200, 0x0000030e)
        return (f.r(0x20c) & 0x00007000) >> 12

    bound = c.wait(lambda: stat() == 5, 0.010)
    s5 = stat()
    f.mask(0x004, 0x00000008, 0x00000008)
    f.mask(0x058, 0x00000002, 0x00000002)
    idle = c.wait(lambda: stat() == 0, 0.010)
    return bound, idle, s5, stat()


def acr_unload(c, heap_base=0x10100000, verbose=True):
    pmu = acr.Falcon(c, PMU, pmc_bit=None, reset_eng=True)
    hs = acr.HSFW("acr/unload_bl.bin", "acr/ucode_unload.bin")
    before, fld0 = state(c)
    c.wr(PMU + 0x148, 0)
    snap = dict(cpuctl=pmu.r(0x100), sctl=pmu.r(0x240), mbox0=pmu.r(0x040), mbox1=pmu.r(0x044), pc=pmu.r(0x14c))
    if verbose:
        print("0x10e5f8 before: %08x  (bits[23:20] = %d)   temp %s C" % (before, fld0, temp_c(c)))
        print("PMU before (VBIOS PRE_OS app is about to be stopped): " +
              " ".join("%s=%08x" % kv for kv in snap.items()))
        print("HS unload: boot_addr=%#x boot=%d B nmem=%#x+%#x imem=%#x+%#x dmem=%#x+%#x sig@%#x(%d)"
              % (hs.boot_addr, len(hs.boot), hs.nmem_base, hs.nmem_size, hs.imem_base, hs.imem_size,
                 hs.dmem_base_img, hs.dmem_size, hs.sig_loc, hs.sig_size))

    # gm200_flcn_fw_signature: enable, then the debug fuse bit
    pmu.enable()
    debug = bool(pmu.r(PMU_DEBUG) & 0x00100000)
    hs.patch_sig(debug)
    if verbose:
        print("PMU debug-fused=%s -> %s signature" % (debug, "debug" if debug else "production"))

    # the image in VRAM, mapped into a 4 KiB-page VMM, published via an instance block
    c.vram_write(UNLOAD_IMG_ADDR, bytes(hs.img))
    assert c.vram_read(UNLOAD_IMG_ADDR, len(hs.img)) == bytes(hs.img), "unload image VRAM readback mismatch"
    heap = nvmmu.VramHeap(base=heap_base)
    v = nvmmu.VMM(c, heap)
    inst = nvmmu.inst_alloc(c, heap)
    v.map(UNLOAD_VA, UNLOAD_IMG_ADDR, acr.ALIGN(len(hs.img), 0x1000))
    v.join(inst)
    flushed = v.flush()
    walked = v.walk(UNLOAD_VA)
    assert walked == ("vram", UNLOAD_IMG_ADDR), "page table walk mismatch: %s" % (walked,)

    # gm200_flcn_fw_reset -> nvkm_falcon_reset (disable + enable), then load
    pmu.reset()
    bound, idle, s5, s_end = pmu_bind_inst(c, pmu, inst)
    if verbose:
        print("MMU: inst=%#x pd3=%#x va=%#x -> %s flush=%s | PMU bind: saw5=%s idle=%s (stat %d -> %d) 0x480=%08x"
              % (inst, v.pd3, UNLOAD_VA, walked, flushed, bound, idle, s5, s_end, pmu.r(0x480)))
    pmu.imem_pio(hs.boot, pmu.code_limit - len(hs.boot), hs.boot_addr >> 8)
    pmu.dmem_pio(hs.bl_desc(UNLOAD_VA, acr.DMAIDX_VIRT), 0)

    # gm200_flcn_fw_boot
    pmu.w(0x040, 0xcafebeef)
    pmu.w(0x104, hs.boot_addr)
    pmu.w(0x100, 2)
    t0 = time.time()
    stopped = c.wait(lambda: pmu.r(0x100) & 0x10, 2.0)
    dt = (time.time() - t0) * 1000
    mb0, mb1 = pmu.r(0x040), pmu.r(0x044)
    pmu.mask(0x004, 0xffffffff, 0x10)
    after, fld1 = state(c)
    ok = stopped and mb0 == UNLOAD_OK
    if verbose:
        print("ACR unload: stopped=%s after %.0f ms  mbox0=%08x (want %08x) mbox1=%08x  PMU %s"
              % (stopped, dt, mb0, UNLOAD_OK, mb1, pmu.state()))
        print("0x10e5f8 after:  %08x  (bits[23:20] = %d)   temp %s C" % (after, fld1, temp_c(c)))
    return dict(ok=ok, stopped=stopped, mbox0=mb0, mbox1=mb1, before=before, after=after, cleared=(fld0 != 0 and fld1 == 0))


if __name__ == "__main__":
    import faults
    c = Card()
    faults.dump(c, clear=True)
    r = acr_unload(c)
    print("MMU faults after:")
    faults.dump(c)
    print("ACR UNLOAD RESULT:", "PASS" if r["ok"] else "FAIL",
          "| state field", "CLEARED 4 -> 0" if r["cleared"] else "%08x -> %08x" % (r["before"], r["after"]))
