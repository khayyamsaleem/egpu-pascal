"""Port of nouveau's nvkm/subdev/bios {base,image,pcir,npde,bit,pmu}.c."""
import struct

class VBIOS:
    def __init__(self, data):
        self.data, self.image0_size, self.imaged_addr = bytes(data), 0, 0
        imgs = self.images()
        self.image0_size = imgs[0][1]
        for base, size, typ, last in imgs[1:]:
            if typ == 0xe0: self.imaged_addr = base; break
        self.bit_offset = self.data.find(b"\xff\xb8BIT")  # nouveau: findstr "\xff\xb8""BIT"
    def _a(self, a):
        if a >= self.image0_size and self.imaged_addr: a = a - self.image0_size + self.imaged_addr
        return a
    def rd8(self, a): return self.data[self._a(a)]
    def rd16(self, a): a = self._a(a); return struct.unpack_from("<H", self.data, a)[0]
    def rd32(self, a): a = self._a(a); return struct.unpack_from("<I", self.data, a)[0]
    def raw(self, a, n): a = self._a(a); return self.data[a:a + n]
    # image chain (raw addressing, as nouveau does with imaged_addr=0)
    def images(self):
        d, out, base, size, last = self.data, [], 0, 0, False
        u16 = lambda a: struct.unpack_from("<H", d, a)[0]
        u32 = lambda a: struct.unpack_from("<I", d, a)[0]
        while not last:
            base += size
            if base + 0x1a > len(d) or u16(base) not in (0xaa55, 0xbb77, 0x4e56): break
            p = base + u16(base + 0x18)
            if u32(p) not in (0x52494350, 0x53494752, 0x5344504e): break
            hdr = u16(p + 0x0a); size = u16(p + 0x10) * 512; typ = d[p + 0x14]; last = bool(d[p + 0x15] & 0x80)
            if typ != 0x70:
                n = (p + hdr + 0x0f) & ~0x0f
                if u32(n) == 0x4544504e: size = u16(n + 8) * 512; last = bool(d[n + 0x0a] & 0x80)
            else: last = True
            out.append((base, size, typ, last))
        return out
    def bit(self, id_):
        b = self.bit_offset; n = self.rd8(b + 10); e = b + 12; step = self.rd8(b + 9)
        for _ in range(n):
            if self.rd8(e) == ord(id_):
                return dict(version=self.rd8(e + 1), length=self.rd16(e + 2), offset=self.rd16(e + 4))
            e += step
        return None
    def pmu_entries(self):
        p = self.bit('p'); assert p and p["version"] == 2, p
        t = self.rd32(p["offset"]); hdr, ln, cnt = self.rd8(t + 1), self.rd8(t + 2), self.rd8(t + 3)
        return [(self.rd8(t + hdr + i * ln), self.rd32(t + hdr + i * ln + 2)) for i in range(cnt)]
    def pmuR(self, typ):
        for t, data in self.pmu_entries():
            if t == typ and data:
                r = self.rd32
                i = dict(init_addr_pmu=r(data + 8), args_addr_pmu=r(data + 0xc), boot_addr=data + 0x30,
                         boot_addr_pmu=r(data + 0x10) + r(data + 0x18), boot_size=r(data + 0x1c) - r(data + 0x18),
                         code_size=r(data + 0x20), data_addr=data + 0x30 + r(data + 0x24),
                         data_addr_pmu=r(data + 0x28), data_size=r(data + 0x2c), desc=data)
                i["code_addr"] = i["boot_addr"] + i["boot_size"]; i["code_addr_pmu"] = i["boot_addr_pmu"] + i["boot_size"]
                return i
        return None

if __name__ == "__main__":
    import sys
    v = VBIOS(open(sys.argv[1], "rb").read())
    for im in v.images(): print("image base=0x%x size=0x%x type=0x%x last=%s" % im)
    print("image0_size=0x%x imaged_addr=0x%x bit@0x%x" % (v.image0_size, v.imaged_addr, v.bit_offset))
    for k in "IpPMB": print(k, v.bit(k))
    print("pmu entries:", [(hex(t), hex(d)) for t, d in v.pmu_entries()])
    for t in (0x04, 0x01):
        r = v.pmuR(t); print("type %#x:" % t, {k: hex(x) for k, x in r.items()} if r else None)
    I = v.bit('I'); o = I["offset"]
    print("BIT_I tables @0x%x len 0x%x ; scripts @0x%x len 0x%x" % (v.rd16(o + 0x14), v.rd16(o + 0x16), v.rd16(o + 0x18), v.rd16(o + 0x1a)))
