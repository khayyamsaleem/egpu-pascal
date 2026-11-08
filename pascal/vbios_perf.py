"""Read-only: performance levels and memory-frequency bands from the VBIOS (offline).

Formats as nouveau nvkm/subdev/bios/perf.c and rammap.c (BIT 'P' version 2):
  BIT 'P' + 0 -> perf table   (ver 0x40: hdr@1, entry len@2, subentry size@3, subentry count@4, count@5)
     entry +0 = pstate id; each subentry u16 & 0x3fff = a clock-domain frequency in MHz
  BIT 'P' + 4 -> rammap table (ver 0x10/0x11: same header layout)
     entry +0 = rammap_min MHz, +2 = rammap_max MHz (memory-clock band), then per-strap ramcfg subentries
Usage: ~/.venvs/tinygpu/bin/python pascal/vbios_perf.py firmware/gp102_vbios.rom
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from vbios import VBIOS


def table(v, ptr):
    ver, hdr, ln, ssz, snr, cnt = (v.rd8(ptr + i) for i in range(6))
    return dict(ver=ver, hdr=hdr, len=ln, ssz=ssz, snr=snr, cnt=cnt, ptr=ptr)


def entries(v, t):
    stride = t["len"] + t["snr"] * t["ssz"]
    for i in range(t["cnt"]):
        e = t["ptr"] + t["hdr"] + i * stride
        yield i, e, [e + t["len"] + j * t["ssz"] for j in range(t["snr"])]


def main(path):
    v = VBIOS(open(path, "rb").read())
    P = v.bit("P")
    print("BIT 'P': %s" % P)
    perf_ptr, rammap_ptr = v.rd32(P["offset"] + 0), v.rd32(P["offset"] + 4)
    pt = table(v, perf_ptr)
    print("perf table @%#x: %s" % (perf_ptr, pt))
    if pt["ver"] == 0x40:
        for i, e, subs in entries(v, pt):
            ps = v.rd8(e)
            if ps == 0xff:
                continue
            freqs = [v.rd16(s) & 0x3fff for s in subs]
            print("  entry %d: pstate %#04x  raw %s  domain MHz %s"
                  % (i, ps, v.raw(e, pt["len"]).hex(), freqs))
    elif pt["ver"] == 0x50:
        # Pascal perf 5.0, decoded from this VBIOS: header bytes are
        # ver, hdr, entry_len, entry_cnt, sub_size, sub_cnt (NOT the 0x40 order);
        # sub = u16 nominal MHz, u16 (flags<<14 | MHz), u16 aux. Domain 2 = memory clock
        # (aux = MHz/4); domains 0/1 are 2x ("gpc2clk"-style) clocks.
        ln, cnt, ssz, snr = (v.rd8(perf_ptr + i) for i in (2, 3, 4, 5))
        names = {0x0f: "P0", 0x0d: "P2", 0x0a: "P5", 0x07: "P8"}
        for i in range(cnt):
            e = perf_ptr + pt["hdr"] + i * (ln + ssz * snr)
            subs = [e + ln + j * ssz for j in range(snr)]
            dom = [(v.rd16(s), v.rd16(s + 2) & 0x3fff, v.rd16(s + 4)) for s in subs]
            ps = v.rd8(e)
            print("  %s (id %#04x): memory %4d MHz | dom0 %d / dom1 %d MHz (2x clocks -> /2 = %d / %d) | all %s"
                  % (names.get(ps, "?"), ps, dom[2][0], dom[0][0], dom[1][0], dom[0][0] // 2, dom[1][0] // 2,
                     [d[0] for d in dom]))
    else:
        print("  (perf table version %#x not decoded)" % pt["ver"])
    rt = table(v, rammap_ptr)
    print("rammap table @%#x: %s" % (rammap_ptr, rt))
    if rt["ver"] in (0x10, 0x11):
        for i, e, subs in entries(v, rt):
            print("  band %d: %4d .. %4d MHz  (%d strap configs)" % (i, v.rd16(e), v.rd16(e + 2), len(subs)))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else
         os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "firmware", "gp102_vbios.rom"))
