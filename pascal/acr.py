"""Port of nouveau's gp102 ACR (secure boot) for GP102, run on SEC2.

The falcon's DMA goes through Pascal MMU v2 page tables (pascal/mmu.py), which
is what nouveau does: acr_oneinit allocates a 4 KiB VRAM instance block, creates
the ACR VMM, maps the HS firmware image into it, joins, and then
gm200_acr_hsfw_load_bld hands the bootloader `ctx_dma = FALCON_DMAIDX_VIRT`
with `code_dma_base = the image's virtual address`.

The first attempt at this DMA'd from a *physical* VRAM address instead (a
deliberate shortcut, no page tables), and the card refused it: SEC /
PHYSICAL / PRIV_VIOLATION -- the FB hub will not do physical DMA for a falcon
that is not yet in secure mode.  Setting `vmm=False` reproduces that path.

WPR contains FECS + GPCCS only (no SEC2 RTOS), non-lazy, so the HS ACR itself
bootstraps them.  If that refuses, plan B is the SEC2 RTOS + cmdq path.
"""
import os, struct, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from card import Card
import mmu as nvmmu

FW = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "firmware", "gp102")
SEC2, FECS, GPCCS = 0x87000, 0x409000, 0x41a000
LSF_FECS, LSF_GPCCS, LSF_SEC2 = 2, 3, 7
DMAIDX_UCODE, DMAIDX_VIRT, DMAIDX_PHYS_VID, DMAIDX_PHYS_SYS_COH, DMAIDX_PHYS_SYS_NCOH = 0, 1, 2, 3, 4
ALIGN = lambda x, a: (x + a - 1) & ~(a - 1)
u32s = lambda b, o, n=1: struct.unpack_from("<%dI" % n, b, o)

# VRAM physical layout (everything inside the proven-good 0..3.5 GB range).
#
# The WPR must sit above 0x18000000: the ACR app's own range check (disassembled
# at imem 0x13d4/0x1444) returns error 0x25 when the region start is
# <= 0x17ffffff, and it then refuses to program WPR2 (0x100cd8) from the
# descriptor -- which is what gm200_acr_wpr_check verifies afterwards.
HS_IMG_ADDR = 0x01000000
WPR_ALLOC_ADDR = 0x18000000
# Virtual address the HS image is mapped at.  nouveau's first-fit allocator
# hands out the lowest free VA, i.e. 0; 64 KiB is used here so that a
# zero-address base can never be confused with an unmapped one in a fault dump.
ACR_VA = 0x00010000

def rdfw(p): return open(os.path.join(FW, p), "rb").read()


# ---------------------------------------------------------------- ACR claim field
# 0x10e5f8 bits[23:20] is the ACR "claim" state shared by the HS load (SEC2) and
# the HS unload (PMU), guarded by falcon HW mutex #2.  The load's 0x15fd fails
# with 0x29 unless the field is 0, then writes the chip's value (GP102 -> 4,
# from PMC_BOOT_42 in 0x166f).  The unload's 0x322 requires that value and
# clears it.  The load claims it *before* the WPR range check, so a load that
# fails later leaves it at 4, and every later load fails 0x29 until the unload
# runs (or the enclosure is power-cycled).  Neither host nor non-HS falcon
# writes stick.  See pascal/acr_unload.py.
ACR_CLAIM_REG = 0x10e5f8
ACR_CLAIM_FIELD = 0x00f00000     # bits 20-23


def acr_claim(c):
    return (c.rd(ACR_CLAIM_REG) & ACR_CLAIM_FIELD) >> 20


# ---------------------------------------------------------------- falcon helpers
class Falcon:
    def __init__(self, c, base, pmc_bit=None, debug_reg=None, reset_eng=True):
        self.c, self.base, self.pmc_bit, self.debug_reg, self.has_reset_eng = c, base, pmc_bit, debug_reg, reset_eng
    def r(self, o): return self.c.rd(self.base + o)
    def w(self, o, v): self.c.wr(self.base + o, v)
    def mask(self, o, m, v): return self.c.mask(self.base + o, m, v)
    @property
    def code_limit(self): return (self.r(0x108) & 0x1ff) << 8
    def disable(self):
        self.mask(0x048, 3, 0); self.w(0x014, 0xffffffff)
        if self.pmc_bit is not None: self.c.mask(0x200, 1 << self.pmc_bit, 0)
    def enable(self):
        if self.has_reset_eng:
            self.mask(0x3c0, 1, 1); time.sleep(0.0001); self.mask(0x3c0, 1, 0)   # gp102_flcn_reset_eng
            if not self.c.wait(lambda: not (self.r(0x10c) & 6), 0.05): raise TimeoutError("scrub (reset_eng)")
        if self.pmc_bit is not None: self.c.mask(0x200, 1 << self.pmc_bit, 1 << self.pmc_bit)
        if not self.c.wait(lambda: not (self.r(0x10c) & 6), 0.05): raise TimeoutError("scrub")
        self.w(0x084, self.c.rd(0))
    def reset(self): self.disable(); self.enable()
    def imem_pio(self, data, addr, tag, sec=False, port=0):
        data = bytes(data) + b"\0" * (-len(data) % 4)
        self.w(0x180 + port * 16, (1 << 24) | ((1 << 28) if sec else 0) | addr)
        words = struct.unpack("<%dI" % (len(data) // 4), data); i = 0
        for i, v in enumerate(words):
            if (i * 4) % 256 == 0: self.w(0x188 + port * 16, tag); tag += 1
            self.w(0x184 + port * 16, v)
        n = len(words) * 4
        while n % 256: self.w(0x184 + port * 16, 0); n += 4
    def dmem_pio(self, data, addr, port=0):
        data = bytes(data) + b"\0" * (-len(data) % 4)
        self.w(0x1c0 + port * 8, (1 << 24) | addr)
        for v in struct.unpack("<%dI" % (len(data) // 4), data): self.w(0x1c4 + port * 8, v)
    def dmem_read(self, addr, n, port=0):
        self.w(0x1c0 + port * 8, (1 << 25) | addr)
        return struct.pack("<%dI" % ((n + 3) // 4), *[self.r(0x1c4 + port * 8) for _ in range((n + 3) // 4)])[:n]
    def start(self):   # nvkm_falcon_v1_start
        if self.r(0x100) & 0x40: self.w(0x130, 2)
        else: self.w(0x100, 2)
    def state(self):
        return "cpuctl=%08x sctl=%08x mbox0=%08x mbox1=%08x pc=%08x 0x10c=%08x" % (
            self.r(0x100), self.r(0x240), self.r(0x040), self.r(0x044), self.r(0x14c) if False else self.trace_pc(), self.r(0x10c))
    def trace_pc(self):
        self.w(0x148, 0); return self.r(0x14c)


# ---------------------------------------------------------------- LS firmware (FECS/GPCCS)
class LSFW:
    """nvkm_acr_lsfw_load_bl_inst_data_sig()"""
    def __init__(self, name, lsf_id, flags):
        self.name, self.id, self.flags = name, lsf_id, flags
        bl = rdfw("gr/%s_bl.bin" % name); inst = rdfw("gr/%s_inst.bin" % name); data = rdfw("gr/%s_data.bin" % name)
        self.sig = rdfw("gr/%s_sig.bin" % name); assert len(self.sig) == 192, len(self.sig)
        _, _, _, hoff, doff, dsz = u32s(bl, 0, 6)
        start_tag, _, code_off, code_size, _, _ = u32s(bl, hoff, 6)
        self.bootloader_size = ALIGN(code_size, 256)
        self.bootloader_imem_offset = start_tag << 8
        self.app_start_offset = self.bootloader_size
        self.app_imem_entry = 0
        self.app_resident_code_offset = 0
        self.app_resident_code_size = ALIGN(len(inst), 256)
        self.app_resident_data_offset = self.app_resident_code_size
        self.app_resident_data_size = ALIGN(len(data), 256)
        self.app_size = self.app_resident_code_size + self.app_resident_data_size
        img = bytearray(self.bootloader_size + self.app_size)
        bld = bl[doff:doff + self.bootloader_size]; img[:len(bld)] = bld
        o = self.app_start_offset + self.app_resident_code_offset; img[o:o + len(inst)] = inst
        o = self.app_start_offset + self.app_resident_data_offset; img[o:o + len(data)] = data
        self.img = bytes(img)
        self.ucode_size = ALIGN(self.app_resident_data_offset, 256) + self.bootloader_size
        self.data_size = self.app_size + self.bootloader_size - self.ucode_size
        self.bin_version = u32s(self.sig, 80)[0]
        self.bld_size = 76   # sizeof(flcn_bl_dmem_desc_v1), packed

    def bld(self, wpr_start):   # gm200_gr_acr_bld_write + gp102 wpr_patch(adjust = wpr_start)
        base = self.off_img + self.app_start_offset
        code, data = base + self.app_resident_code_offset, base + self.app_resident_data_offset
        return struct.pack("<4I4IIQIIIIIQI", 0, 0, 0, 0, 0, 0, 0, 0, DMAIDX_UCODE, code + wpr_start,
                           self.app_resident_code_offset, self.app_resident_code_size, 0, 0,
                           self.app_imem_entry, data + wpr_start, self.app_resident_data_size)

    def lsb(self):   # lsb_header_v1 = lsf_signature_v1 (192) + lsb_header_tail (48)
        tail = struct.pack("<12I", self.off_img, self.ucode_size, self.data_size, self.bootloader_size,
                           self.bootloader_imem_offset, self.off_bld, self.bl_data_size,
                           self.app_start_offset + self.app_resident_code_offset, self.app_resident_code_size,
                           self.app_start_offset + self.app_resident_data_offset, self.app_resident_data_size,
                           self.flags)
        return self.sig + tail


FALCON_SEC2_DMAIDX_UCODE = 6
SEC2_EMEM_ADDR = 0x01000000


class LSFW_SEC2:
    """nvkm_acr_lsfw_load_sig_image_desc_v1 + gp102_sec2_acr_{0,1} (the SEC2 RTOS LSF).

    ver 1 (sec2/*-1.bin, nouveau's first choice) uses flcn_bl_dmem_desc_v2 as
    its BL data; ver 0 uses loader_config_v1.
    """
    def __init__(self, ver=1):
        sfx = "-%d" % ver if ver else ""
        self.name, self.id, self.flags, self.ver = "sec2", LSF_SEC2, 0, ver
        self.sig = rdfw("sec2/sig%s.bin" % sfx); assert len(self.sig) == 192, len(self.sig)
        self.img = rdfw("sec2/image%s.bin" % sfx)
        d = rdfw("sec2/desc%s.bin" % sfx)
        (bl_start, bl_size, bl_imem_off, bl_entry, app_start, app_size, app_imem_off, app_imem_entry,
         app_dmem_off, rc_off, rc_size, rd_off, rd_size) = u32s(d, 80, 13)
        # nvkm_acr_lsfw_from_desc
        self.bootloader_size = ALIGN(bl_size, 256)
        self.bootloader_imem_offset = bl_imem_off
        self.app_size = ALIGN(app_size, 256)
        self.app_start_offset = app_start
        self.app_imem_entry = app_imem_entry
        self.app_resident_code_offset, self.app_resident_code_size = rc_off, rc_size
        self.app_resident_data_offset, self.app_resident_data_size = rd_off, rd_size
        self.ucode_size = ALIGN(self.app_resident_data_offset, 256) + self.bootloader_size
        self.data_size = self.app_size + self.bootloader_size - self.ucode_size
        self.bin_version = u32s(self.sig, 80)[0]
        self.bld_size = 84 if ver else 56   # sizeof(flcn_bl_dmem_desc_v2) / sizeof(loader_config_v1), packed

    def bld(self, wpr_start):   # gp102_sec2_acr_bld_write{_1,} + bld_patch(adjust = wpr_start)
        base = wpr_start + self.off_img + self.app_start_offset
        if self.ver:
            return struct.pack("<4I4IIQIIIIIQIII", 0, 0, 0, 0, 0, 0, 0, 0, FALCON_SEC2_DMAIDX_UCODE, base,
                               self.app_resident_code_offset, self.app_resident_code_size, 0, 0,
                               self.app_imem_entry, base + self.app_resident_data_offset,
                               self.app_resident_data_size, 1, SEC2_EMEM_ADDR)
        return struct.pack("<IIQIIIQIQII", 0, FALCON_SEC2_DMAIDX_UCODE, base, self.app_size,
                           self.app_resident_code_size, self.app_imem_entry,
                           base + self.app_resident_data_offset, self.app_resident_data_size,
                           base, 1, SEC2_EMEM_ADDR)

    lsb = LSFW.lsb


def build_wpr(lsfws, rtos_id=None):
    """gp102_acr_wpr_layout + gp102_acr_wpr_alloc + gp102_acr_wpr_build (+patch)."""
    wpr = ALIGN(11 * 24, 256) + 0x100
    for l in lsfws:
        wpr = ALIGN(wpr, 256); l.off_lsb = wpr; wpr += 240
        wpr = ALIGN(wpr, 4096); l.off_img = wpr; wpr += len(l.img)
        wpr = ALIGN(wpr, 256); l.off_bld = wpr; l.bl_data_size = ALIGN(l.bld_size, 256); wpr += l.bl_data_size
    size = ALIGN(wpr, 0x40000) << 1
    shadow_start = WPR_ALLOC_ADDR
    wpr_start = shadow_start + (size >> 1)
    wpr_end = wpr_start + (size >> 1)
    blob = bytearray(size >> 1)
    off = 0
    for l in lsfws:
        lazy = 1 if (rtos_id is not None and l.id != rtos_id) else 0
        blob[off:off + 24] = struct.pack("<6I", l.id, l.off_lsb, LSF_SEC2, lazy, l.bin_version, 1)
        off += 24
        blob[l.off_lsb:l.off_lsb + 240] = l.lsb()
        blob[l.off_img:l.off_img + len(l.img)] = l.img
        b = l.bld(wpr_start); blob[l.off_bld:l.off_bld + len(b)] = b
    blob[off:off + 4] = struct.pack("<I", 0xffffffff)
    return dict(blob=bytes(blob), shadow_start=shadow_start, wpr_start=wpr_start, wpr_end=wpr_end, used=wpr)


# ---------------------------------------------------------------- HS firmware (ACR load)
class HSFW:
    """nvkm_falcon_fw_ctor_hs()"""
    def __init__(self, bl_path, img_path):
        blob = rdfw(img_path)
        self._parse(blob)
        if bl_path is None:
            self.boot_addr, self.boot = self.nmem_base, None; return
        self._parse_bl(bl_path)

    def _parse(self, blob):
        magic, _, _, hoff, doff, dsz = u32s(blob, 0, 6)
        sig_dbg_off, sig_dbg_sz, sig_prod_off, sig_prod_sz, patch_loc, patch_sig, lhoff, _ = u32s(blob, hoff, 8)
        if magic == 0x10de: loc, sig = u32s(blob, patch_loc)[0], u32s(blob, patch_sig)[0]
        elif magic == 0x3b1d14f0: loc, sig = patch_loc, patch_sig
        else: raise ValueError("bin_magic %#x" % magic)
        self.img = bytearray(blob[doff:doff + dsz])
        self.sig_loc, self.sig_size = loc, sig_prod_sz
        self.sig_prod = blob[sig_prod_off + sig:sig_prod_off + sig + sig_prod_sz]
        self.sig_dbg = blob[sig_dbg_off + sig:sig_dbg_off + sig + sig_prod_sz]
        nsc_off, nsc_size, data_dma_base, data_size, num_apps = u32s(blob, lhoff, 5)
        apps = u32s(blob, lhoff + 20, 2 * num_apps)
        self.nmem_base, self.nmem_size = nsc_off, nsc_size
        self.imem_base_img, self.imem_base, self.imem_size = apps[0], ALIGN(apps[0], 0x100), apps[num_apps]
        self.dmem_base_img, self.dmem_size = data_dma_base, data_size

    def _parse_bl(self, bl_path):
        bl = rdfw(bl_path)
        _, _, _, bhoff, bdoff, _ = u32s(bl, 0, 6)
        start_tag, _, code_off, code_size, _, _ = u32s(bl, bhoff, 6)
        self.boot_addr, self.boot = start_tag << 8, bl[bdoff + code_off:bdoff + code_off + code_size]

    def patch_sig(self, debug):
        s = self.sig_dbg if debug else self.sig_prod
        self.img[self.sig_loc:self.sig_loc + self.sig_size] = s

    def setup_acr_desc(self, wpr):   # gp102_acr_load_setup -> flcn_acr_desc_v1 at dmem_base_img
        d = self.dmem_base_img
        struct.pack_into("<I", self.img, d + 0x210, 1)           # wpr_region_id
        struct.pack_into("<I", self.img, d + 0x21c, 2)           # regions.no_regions
        struct.pack_into("<7I", self.img, d + 0x220, wpr["wpr_start"] >> 8, wpr["wpr_end"] >> 8, 1,
                         0xf, 0xc, 0x2, wpr["shadow_start"] >> 8)

    def bl_desc(self, img_addr, ctx_dma):   # gm200_acr_hsfw_load_bld
        return struct.pack("<4I4IIQIIIIIQI", 0, 0, 0, 0, 0, 0, 0, 0, ctx_dma, img_addr,
                           self.nmem_base, self.nmem_size, self.imem_base, self.imem_size, 0,
                           img_addr + self.dmem_base_img, self.dmem_size)


def HSFW_nobl(img_path): return HSFW(None, img_path)


def wpr_check(c):
    c.wr(0x100cd4, 2); start = (c.rd(0x100cd4) & 0xffffff00) << 8
    c.wr(0x100cd4, 3); limit = ((c.rd(0x100cd4) & 0xffffff00) << 8) + 0x20000
    return start, limit


def acr_load(c, vmm=True, verbose=True, sec2_rtos=None):
    """sec2_rtos=None: WPR = FECS + GPCCS (non-lazy; the HS ACR does not boot them
    on GP102).  sec2_rtos=1 or 0: also the SEC2 RTOS LSF (that firmware version),
    non-lazy, with FECS/GPCCS lazy -- nouveau's layout when the RTOS is present."""
    sec2 = Falcon(c, SEC2, pmc_bit=14, debug_reg=0x408)
    lsfws = [LSFW("fecs", LSF_FECS, 0), LSFW("gpccs", LSF_GPCCS, 0x8)]
    if sec2_rtos is not None:
        lsfws.append(LSFW_SEC2(sec2_rtos))
    wpr = build_wpr(lsfws, rtos_id=LSF_SEC2 if sec2_rtos is not None else None)
    hs = HSFW("acr/bl.bin", "acr/ucode_load.bin")
    if verbose:
        print("WPR: shadow=%#x wpr=%#x-%#x used=%#x" % (wpr["shadow_start"], wpr["wpr_start"], wpr["wpr_end"], wpr["used"]))
        for l in lsfws: print("  LSF %-5s id=%d lsb=%#x img=%#x(+%#x) bld=%#x ucode=%#x data=%#x ver=%d"
                              % (l.name, l.id, l.off_lsb, l.off_img, len(l.img), l.off_bld, l.ucode_size, l.data_size, l.bin_version))
        print("HS: boot_addr=%#x boot=%d B nmem=%#x+%#x imem=%#x+%#x dmem=%#x+%#x sig@%#x(%d)"
              % (hs.boot_addr, len(hs.boot), hs.nmem_base, hs.nmem_size, hs.imem_base, hs.imem_size,
                 hs.dmem_base_img, hs.dmem_size, hs.sig_loc, hs.sig_size))

    # signature (gm200_flcn_fw_signature: enable, then debug bit 20)
    sec2.enable()
    debug = bool(sec2.r(0x408) & 0x00100000)
    hs.patch_sig(debug)
    print("SEC2 debug-fused=%s -> %s signature" % (debug, "debug" if debug else "production"))
    if verbose:
        print("ACR claim 0x10e5f8 = %08x (bits[23:20] = %d; load needs 0)" % (c.rd(ACR_CLAIM_REG), acr_claim(c)))

    # write the WPR blob into the shadow region; zero the WPR proper
    c.vram_write(wpr["shadow_start"], wpr["blob"])
    c.vram_write(wpr["wpr_start"], bytes(len(wpr["blob"])))
    assert c.vram_read(wpr["shadow_start"], 256) == wpr["blob"][:256]

    # reset + setup + place HS image in VRAM
    sec2.reset()
    hs.setup_acr_desc(wpr)
    c.vram_write(HS_IMG_ADDR, bytes(hs.img))
    assert c.vram_read(HS_IMG_ADDR, len(hs.img)) == bytes(hs.img), "HS image VRAM readback mismatch"

    # Host side of the MMU (nouveau's nvkm_acr_oneinit): a 4 KiB instance block
    # in VRAM, a 4 KiB-page VMM, the HS image mapped into it, joined.  The
    # bootloader then DMAs from a virtual address -- which is the whole point,
    # because the hub denies a *physical* DMA from a non-secure falcon.
    if vmm:
        heap = nvmmu.VramHeap()
        v = nvmmu.VMM(c, heap)
        inst = nvmmu.inst_alloc(c, heap)
        v.map(ACR_VA, HS_IMG_ADDR, ALIGN(len(hs.img), 0x1000))
        base = v.join(inst)
        print("MMU: inst=%#x pd3=%#x join=%#x va=%#x+%#x flush=%s" % (
            inst, v.pd3, base, ACR_VA, ALIGN(len(hs.img), 0x1000), v.flush()))
        walked = v.walk(ACR_VA)
        print("  walk va %#x -> %s (read back out of the VRAM page tables)" % (ACR_VA, walked))
        assert walked == ("vram", HS_IMG_ADDR), "page table walk mismatch: %s" % (walked,)
        img_base, ctx_dma = ACR_VA, DMAIDX_VIRT
        bound, status = nvmmu.bind_inst(c, sec2, inst, target=0)
        print("SEC2 instance block bound=%s  bind status=%d  0x054=%08x  0x604=%08x"
              % (bound, status, sec2.r(0x054), sec2.r(0x604)))
    else:
        # the old, refuted path: physical DMA through an FBIF slot
        sec2.mask(0x624, 0x80, 0x80)
        sec2.w(0x10c, 0)
        sec2.mask(0x600 + 4 * DMAIDX_PHYS_VID, 0x7, 0x4)      # FBIF: physical, LOCAL_FB
        img_base, ctx_dma = HS_IMG_ADDR, DMAIDX_PHYS_VID

    sec2.imem_pio(hs.boot, sec2.code_limit - len(hs.boot), hs.boot_addr >> 8)
    sec2.dmem_pio(hs.bl_desc(img_base, ctx_dma), 0)

    # gm200_flcn_fw_boot
    sec2.w(0x040, 0xcafebeef)
    sec2.w(0x104, hs.boot_addr)
    sec2.w(0x100, 2)
    t0 = time.time()
    stopped = c.wait(lambda: sec2.r(0x100) & 0x10, 2.0)
    dt = (time.time() - t0) * 1000
    mb0, mb1 = sec2.r(0x040), sec2.r(0x044)
    sec2.mask(0x004, 0xffffffff, 0x10)
    print("ACR HS: stopped=%s after %.0f ms  mbox0=%08x mbox1=%08x  SEC2 %s" % (stopped, dt, mb0, mb1, sec2.state()))
    s, l = wpr_check(c)
    print("WPR2 regs: start=%#x limit=%#x  (expected %#x-%#x)" % (s, l, wpr["wpr_start"], wpr["wpr_end"]))
    # Negative control: once WPR2 is up, a host (PRAMIN) read inside it must be
    # refused -- 0xbad0acXX data plus a PHYSICAL/HOST_CPU/REGION_VIOL fault --
    # whereas before the load the same address read back what we wrote.
    probe = c.vram_r32(wpr["wpr_start"])
    fid = c.rd(0x259c)
    print("  host read inside WPR @%#x -> %08x  (%s); fault mask %08x"
          % (wpr["wpr_start"], probe, "REFUSED, WPR enforced" if (probe >> 8) == 0xbad0ac else "readable", fid))
    if fid & 0x80000000:
        t = c.rd(0x2800 + 31 * 0x10 + 12)
        if (t & 0x1f) == 13:          # REGION_VIOL: the expected refusal, clear it
            c.wr(0x259c, 1 << 31); c.wr(0x2100, 0x18000000)
    return dict(ok=stopped and mb0 == 0 and s == wpr["wpr_start"], mbox0=mb0, wpr=wpr,
                wpr_enforced=(probe >> 8) == 0xbad0ac)


if __name__ == "__main__":
    import fb, faults
    c = Card()
    mem, iovas = c.dev.alloc_sysmem(0x1000, contiguous=True)   # flush page must outlive every DMA
    fb.fb_init(c, iovas[0])
    if fb.vpr_scrub_required(c):
        nv = next(d for d in fb.top_parse(c) if d["name"].startswith("NVDEC"))
        print("VPR scrub:", fb.vpr_scrub(c, nv["addr"], nv["reset"])[:2])
    c.mask(0x200, 1 << 12, 1 << 12)   # PGRAPH on so FECS/GPCCS exist
    faults.dump(c, clear=True)
    if acr_claim(c) and "--no-unload" not in sys.argv:
        # nvkm_acr_reload: unload, then load
        import acr_unload
        u = acr_unload.acr_unload(c)
        print("ACR unload first (claim was set): %s" % ("ok" if u["ok"] else "FAILED mbox0=%08x" % u["mbox0"]))
    r = acr_load(c, vmm=("--phys" not in sys.argv))
    for n, b in (("FECS", FECS), ("GPCCS", GPCCS)):
        f = Falcon(c, b); print("%-5s %s" % (n, f.state()))
    s, l = wpr_check(c)
    print("WPR2 regs: start=%#x limit=%#x  (expected %#x-%#x)" % (s, l, r["wpr"]["wpr_start"], r["wpr"]["wpr_end"]))
    if not r["ok"]:
        print("MMU faults (what the card objected to):")
        faults.dump(c)
    print("ACR RESULT:", "PASS" if r["ok"] else "FAIL")
