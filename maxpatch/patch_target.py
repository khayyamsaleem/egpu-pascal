"""Patch MAX/Mojo's GTX 1080 Ti (sm_61) GPU target from PTX ISA 5.0 to 6.3, in place.

Works on any modular release: the target table is in std.mojoc in 26.6.0 and in max.mojoc in
the 26.8 nightlies; every package is searched for the string.

Why: modular 26.6.0 ships `_get_gtx1080ti_target()` with features "+ptx50,+sm_61". PTX 5.0
predates `shfl.sync`, so LLVM cannot select llvm.nvvm.shfl.sync.* and every kernel that uses a
warp shuffle (reductions, matmul, softmax, ...) fails with
    LLVM ERROR: Cannot select: intrinsic %llvm.nvvm.shfl.sync.idx.i32
sm_61 supports PTX 6.x (the Turing entry already uses "+ptx63"), and CUDA 12.4 ptxas accepts it.

A package (.mojoc) is an MPKG header followed by one zstd frame whose payload holds a
NUL-terminated string table. The replacement string has the
same length, so no offsets move. Run inside the MAX container:

    python3 /repo/maxpatch/patch_target.py            # patch (keeps <pkg>.mojoc.orig)
    python3 /repo/maxpatch/patch_target.py --revert   # restore the original
Needs `pip install zstandard`. Clear the MAX kernel cache afterwards (~/.cache/modular).
"""
import argparse, glob, os, shutil, struct, sys
import zstandard as zstd

OLD, NEW = b"\0+ptx50,+sm_61\0", b"\0+ptx63,+sm_61\0"
MAGIC = b"\x28\xb5\x2f\xfd"


def lib_dir():
    import importlib.util
    spec = importlib.util.find_spec("modular")
    roots = list(spec.submodule_search_locations) if spec else []
    for r in roots + glob.glob("/usr/local/lib/python3*/site-packages/modular"):
        d = os.path.join(r, "lib", "mojo")
        if os.path.isdir(d):
            return d
    sys.exit("modular/lib/mojo not found")


def payload(path):
    d = open(path, "rb").read()
    i = d.find(MAGIC)
    if d[:4] != b"MPKG" or i < 0:
        return None
    return d[:i], d[i:], zstd.ZstdDecompressor().decompress(d[i:], max_output_size=1 << 31)


def target_pkg():
    """The package whose string table holds the sm_61 target (patched or not)."""
    d = lib_dir()
    for name in ["std.mojoc", "max.mojoc"] + sorted(os.listdir(d)):
        p = os.path.join(d, name)
        if name.endswith(".mojoc") and os.path.exists(p):
            pl = payload(p)
            if pl and (OLD in pl[2] or NEW in pl[2]):
                return p
    sys.exit("no package contains the sm_61 target string")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--revert", action="store_true")
    ap.add_argument("--path")
    a = ap.parse_args()
    path = a.path or target_pkg()
    orig = path + ".orig"
    if a.revert:
        shutil.copy2(orig, path); print("restored", path); return
    pl = payload(path)
    assert pl, "unexpected package layout"
    hdr, frame, raw = pl
    assert struct.pack("<Q", len(raw)) in hdr, "header does not record the payload size"
    if NEW in raw and OLD not in raw:
        print("already patched:", path); return
    assert raw.count(OLD) == 1, "expected exactly one %r, found %d" % (OLD, raw.count(OLD))
    raw2 = raw.replace(OLD, NEW)
    assert len(raw2) == len(raw)
    frame2 = zstd.ZstdCompressor(level=19, write_content_size=True, write_checksum=False).compress(raw2)
    if not os.path.exists(orig):
        shutil.copy2(path, orig)
    with open(path + ".tmp", "wb") as f:
        f.write(hdr + frame2)
    os.replace(path + ".tmp", path)
    assert zstd.ZstdDecompressor().decompress(open(path, "rb").read()[len(hdr):]) == raw2
    print("patched %s: %r -> %r (frame %d -> %d bytes; backup %s)"
          % (path, OLD.strip(b"\0"), NEW.strip(b"\0"), len(frame), len(frame2), orig))


if __name__ == "__main__":
    main()
