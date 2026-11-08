import sys; sys.path.insert(0, __file__.rsplit("/",1)[0])
from card import Card
c = Card()
old = c.rd(0x88050); c.wr(0x88050, old & ~1)   # nouveau shadow/prom: disable ROM shadow
data = bytearray()
for off in range(0, 0x100000, 4):
    data += c.rd(0x300000 + off).to_bytes(4, "little")
c.wr(0x88050, old)
# trim to the chained PCI images
pos, end = 0, 0
while pos < len(data) and data[pos:pos+2] == b"\x55\xaa":
    pcir = pos + int.from_bytes(data[pos+0x18:pos+0x1a], "little")
    ln = int.from_bytes(data[pcir+0x10:pcir+0x12], "little") * 512
    ind = data[pcir+0x15]; print("image @0x%x len 0x%x type %d last=%d" % (pos, ln, data[pcir+0x14], ind & 0x80 != 0))
    end = pos + ln; pos += ln
    if ind & 0x80: break
open(sys.argv[1], "wb").write(data)
print("wrote", max(end, 0x20000), "bytes")
