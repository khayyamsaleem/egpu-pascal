"""Minimal CUDA cubin (ELF64) reader: what a launch needs from a ptxas output.

Per kernel <name>:
  .text.<name>          SASS; section sh_info bits 24..31 = register count
  .nv.constant0.<name>  constant bank 0 image size (driver area 0x0..0x13f + params)
  .nv.info.<name>       EIATTR records (param cbank offset/size, ...)
  .nv.shared.<name>     static shared memory (sh_size, NOBITS)

EIATTR record: u8 format, u8 attr, then for format 0x04 (EIFMT_SVAL) u16 size + payload;
format 0x03 (EIFMT_HVAL) a u16 value; 0x02 (EIFMT_BVAL) u8; 0x01 (EIFMT_NVAL) nothing.
"""
import struct, sys

EIATTR = {0x0a: "PARAM_CBANK", 0x17: "KPARAM_INFO", 0x19: "CBANK_PARAM_SIZE", 0x1b: "MAXREG_COUNT",
          0x1c: "EXIT_INSTR_OFFSETS", 0x2f: "REGCOUNT", 0x12: "FRAME_SIZE", 0x11: "MIN_STACK_SIZE",
          0x36: "SW_WAR", 0x37: "CUDA_API_VERSION", 0x50: "SW1850030_WAR", 0x04: "CTAIDZ_USED",
          0x05: "MAX_THREADS", 0x23: "MAX_STACK_SIZE", 0x1e: "S2RCTAID_INSTR_OFFSETS"}


class Cubin:
    def __init__(self, data: bytes):
        self.data = data
        assert data[:4] == b"\x7fELF" and data[4] == 2, "not an ELF64 cubin"
        (self.e_flags,) = struct.unpack_from("<I", data, 0x30)
        shoff, = struct.unpack_from("<Q", data, 0x28)
        shentsize, shnum, shstrndx = struct.unpack_from("<HHH", data, 0x3a)
        secs = []
        for i in range(shnum):
            name, typ, flags, addr, off, size, link, info, align, entsize = struct.unpack_from(
                "<IIQQQQIIQQ", data, shoff + i * shentsize)
            secs.append(dict(name_off=name, type=typ, flags=flags, off=off, size=size, link=link, info=info, align=align))
        strtab = secs[shstrndx]
        for s in secs:
            o = strtab["off"] + s["name_off"]
            s["name"] = data[o:data.index(b"\0", o)].decode()
        self.sections = {s["name"]: s for s in secs}

    @property
    def sm(self):
        return self.e_flags & 0xff                       # EF_CUDA_SM: 61 for sm_61

    def kernels(self):
        return [n[len(".text."):] for n in self.sections if n.startswith(".text.")]

    def sec(self, name):
        s = self.sections.get(name)
        return None if s is None else (self.data[s["off"]:s["off"] + s["size"]] if s["type"] != 8 else bytes(s["size"]))

    def info(self, kernel):
        raw = self.sec(".nv.info." + kernel) or b""
        out, i = [], 0
        while i < len(raw):
            fmt, attr = raw[i], raw[i + 1]; i += 2
            if fmt == 0x04:
                (n,) = struct.unpack_from("<H", raw, i); val = raw[i + 2:i + 2 + n]; i += 2 + n
            elif fmt == 0x03:
                (val,) = struct.unpack_from("<H", raw, i); i += 2
            elif fmt == 0x02:
                val = raw[i]; i += 1
            else:
                val = None
            out.append((EIATTR.get(attr, "0x%02x" % attr), val))
        return out

    def kernel(self, name):
        text = self.sections[".text." + name]
        info = self.info(name)
        cb = dict(info).get("PARAM_CBANK")
        param_off, param_size = (struct.unpack_from("<HH", cb, 4) if cb else (0x140, 0))
        shared = self.sections.get(".nv.shared." + name)
        return dict(name=name, code=self.sec(".text." + name), regs=text["info"] >> 24,
                    cbuf0_size=len(self.sec(".nv.constant0." + name) or b""),
                    param_off=param_off, param_size=param_size,
                    shared=shared["size"] if shared else 0, info=info)


if __name__ == "__main__":
    cb = Cubin(open(sys.argv[1], "rb").read())
    print("sm_%d, sections: %s" % (cb.sm, ", ".join(cb.sections)))
    for k in cb.kernels():
        d = cb.kernel(k)
        print("kernel %s: %d bytes SASS, %d regs, cbuf0 %#x bytes, params at c[0][%#x] size %#x, shared %#x"
              % (k, len(d["code"]), d["regs"], d["cbuf0_size"], d["param_off"], d["param_size"], d["shared"]))
        for a, v in d["info"]:
            print("   %-22s %s" % (a, v.hex() if isinstance(v, bytes) else v))
        if len(sys.argv) > 2:
            open(sys.argv[2], "wb").write(d["code"])
