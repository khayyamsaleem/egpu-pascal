"""Device topology, FB-hub init, and VPR unlock (nouveau top/gk104, fb/{base,gf100,gm200,gp100,gp102}.c)."""
import os, sys, struct
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from card import Card

TYPES = {0: "GR", 1: "CE0", 2: "CE1", 3: "CE2", 8: "MSPDEC", 9: "MSPPP", 0xa: "MSVLD", 0xb: "MSENC",
         0xc: "VIC", 0xd: "SEC2", 0xe: "NVENC", 0xf: "NVENC1", 0x10: "NVDEC", 0x13: "CE", 0x14: "GSP"}

# VRAM physical layout owned by the driver (inside the low, non-VPR range)
MMU_RD_ADDR = 0x00400000   # 128 KB each; MMU debug read/write buffers
MMU_WR_ADDR = 0x00420000


def top_parse(c):
    devs, info, typ, inst = [], {}, None, 0
    for i in range(64):
        d = c.rd(0x22700 + 4 * i)
        k = d & 3
        if k == 0: continue
        if k == 1:
            inst = (d >> 26) & 0xf; info["addr"] = d & 0x00fff000
            if d & 4: info["fault"] = (d >> 3) & 0x7f
        elif k == 2:
            if d & 0x20: info["engine"] = (d >> 26) & 0xf
            if d & 0x10: info["runlist"] = (d >> 21) & 0xf
            if d & 0x08: info["intr"] = (d >> 15) & 0x1f
            if d & 0x04: info["reset"] = (d >> 9) & 0x1f
        else:
            typ = (d & 0x7ffffffc) >> 2
        if d & 0x80000000: continue
        info["type"] = typ; info["name"] = TYPES.get(typ, "?%x" % typ) + ("" if not inst else str(inst)); info["inst"] = inst
        devs.append(info); info, typ, inst = {}, None, 0
    return devs


def fb_init(c, flush_page_iova):
    """nvkm_fb_init for gp102: flush page, gm200_fb_init, gp100 remapper, gm200 init_page(0)."""
    c.wr(0x100c10, flush_page_iova >> 8)
    c.vram_write(MMU_RD_ADDR, bytes(0x20000)); c.vram_write(MMU_WR_ADDR, bytes(0x20000))
    c.wr(0x100cc8, MMU_WR_ADDR >> 8)
    c.wr(0x100ccc, MMU_RD_ADDR >> 8)
    c.mask(0x100cc4, 0x00060000, min(0x20000 >> 16, 2) << 17)
    c.mask(0x100c14, 0x00040000, 0)                  # disable address remapper
    c.mask(0x100c80, 0x00001800, 0x00001800)         # gm200_fb_init_page, page == 0 (gp102 default)


def vpr_scrub_required(c):
    c.wr(0x100cd0, 2); return bool(c.rd(0x100cd0) & 0x10)


def vpr_range(c):
    """0x100cd0/0x100cd4 indexed reads, as for WPR: idx 2/3 = VPR lo/hi on gp10x (best effort)."""
    out = {}
    for idx in range(8):
        c.wr(0x100cd4, idx); out[idx] = c.rd(0x100cd4)
    return out


def vpr_scrub(c, nvdec_base, nvdec_reset_bit):
    """gp102_fb_vpr_scrub: boot nvdec/scrubber.bin (HS, no separate BL) on NVDEC via gm200_flcn_fw."""
    from acr import Falcon, HSFW_nobl
    f = Falcon(c, nvdec_base, pmc_bit=nvdec_reset_bit, reset_eng=False)
    fw = HSFW_nobl("nvdec/scrubber.bin")
    f.enable()
    debug = bool(f.r(0xd00) & 0x00100000)   # gm200_flcn_fw_signature (gm107_nvdec_flcn.debug = 0xd00)
    fw.patch_sig(debug)
    f.reset()
    f.mask(0x624, 0x80, 0x80); f.w(0x10c, 0)
    f.imem_pio(fw.img[fw.nmem_base:fw.nmem_base + fw.nmem_size], fw.nmem_base, fw.nmem_base >> 8, sec=False)
    f.imem_pio(fw.img[fw.imem_base_img:fw.imem_base_img + fw.imem_size], fw.imem_base, fw.imem_base >> 8, sec=True)
    f.dmem_pio(fw.img[fw.dmem_base_img:fw.dmem_base_img + fw.dmem_size], 0)
    f.w(0x040, 0xcafebeef); f.w(0x104, fw.nmem_base); f.w(0x100, 2)
    stopped = c.wait(lambda: f.r(0x100) & 0x10, 2.0)
    return stopped, f.r(0x040), f.r(0x044), debug


if __name__ == "__main__":
    c = Card()
    devs = top_parse(c)
    for d in devs: print("  %-7s addr=%06x reset=%s engine=%s runlist=%s intr=%s fault=%s" % (
        d["name"], d.get("addr", 0), d.get("reset"), d.get("engine"), d.get("runlist"), d.get("intr"), d.get("fault")))
    mem, iovas = c.dev.alloc_sysmem(0x1000, contiguous=True)
    print("flush page IOVA %#x" % iovas[0])
    fb_init(c, iovas[0])
    print("VPR scrub required:", vpr_scrub_required(c), " cd4 idx:", {k: "%08x" % v for k, v in vpr_range(c).items()})
    nvdec = next(d for d in devs if d["name"].startswith("NVDEC"))
    ok, mb0, mb1, dbg = vpr_scrub(c, nvdec["addr"], nvdec["reset"])
    print("scrubber: stopped=%s mbox0=%08x mbox1=%08x debug=%s" % (ok, mb0, mb1, dbg))
    print("VPR scrub required after:", vpr_scrub_required(c), {k: "%08x" % v for k, v in vpr_range(c).items()})
    for mb in (1, 3600, 4096, 8192, 11000):
        a = mb << 20; c.vram_w32(a, 0x5a5a0000 | mb); v = c.vram_r32(a)
        print("  VRAM @%5d MB: %s (%08x)" % (mb, "OK" if v == (0x5a5a0000 | mb) else "BAD", v))
