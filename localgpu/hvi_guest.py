"""Boot a guest under hvi, mount a host directory over virtio-fs, run a command, read the result.

This is the io protocol for the hvi sandbox backend: the host writes `run.sh` into a shared
directory, the guest's /init loads the virtio drivers, mounts the share at /io, runs the script,
writes stdout/stderr to out.txt plus the exit code to rc.txt, and powers off.  Because each call
gets a fresh VM, the guest is disposable and nothing it does outlives the call.

Run it directly to see the whole path work end to end.
"""
import io
import os
import shutil
import struct
import subprocess
import sys
import tarfile
import time

HVI_DIR = os.path.expanduser("~/.local/state/local-gpu/sandbox/hvi")
HVI = os.path.expanduser("~/.local/bin/hvi")
KERNEL = os.path.join(HVI_DIR, "Image.alpine.raw")
APK = os.path.join(HVI_DIR, "linux-virt.apk")
MINIROOTFS = os.path.join(HVI_DIR, "alpine-minirootfs-3.21.0-aarch64.tar.gz")
MODVER = "6.12.117-0-virt"
# virtio core is built into Alpine's kernel; only the transport and the fs drivers are modules.
# insmod order matters: fuse before virtiofs.
MODULES = [
    "kernel/drivers/virtio/virtio_mmio.ko.gz",
    "kernel/fs/fuse/fuse.ko.gz",
    "kernel/fs/fuse/virtiofs.ko.gz",
]

INIT = r"""#!/bin/sh
# hvi sandbox init: load virtio, mount the share, run the payload, report, power off.
mount -t proc proc /proc 2>/dev/null
mount -t sysfs sysfs /sys 2>/dev/null
mount -t devtmpfs devtmpfs /dev 2>/dev/null
export PATH=/usr/bin:/bin:/usr/sbin:/sbin
echo "GUEST-UP $(uname -r)"
for m in virtio_mmio fuse virtiofs; do
  insmod /modules/$m.ko 2>/dev/null || echo "insmod $m: failed"
done
mkdir -p /work
if mount -t virtiofs work /work; then
  echo "SHARE-MOUNTED"
else
  echo "SHARE-MOUNT-FAILED"
  ls /sys/bus/virtio/devices/ 2>/dev/null | tr '\n' ' '
  echo
fi
mkdir -p /deps
if mount -t virtiofs deps /deps 2>/dev/null; then
  export PYTHONPATH=/deps
  echo "DEPS-MOUNTED"
fi
for tag in $(cat /proc/cmdline | tr ' ' '\n' | grep -o 'egpu.mount=[a-z0-9_]*' | cut -d= -f2); do
  mkdir -p /mnt/$tag
  mount -t virtiofs $tag /mnt/$tag 2>/dev/null && echo "MOUNTED /mnt/$tag"
done
cd /work
if [ -f /work/run.sh ]; then
  sh /work/run.sh > /work/out.txt 2>&1
  echo $? > /work/rc.txt
  echo "RUN-DONE rc=$(cat /work/rc.txt)"
else
  echo "NO-RUN-SH"
fi
sync
echo "GUEST-DOWN"
poweroff -f
"""


class Cpio:
    """newc (070701) cpio, built in memory: no cpio binary, no root, works on macOS."""

    def __init__(self):
        self.buf = io.BytesIO()
        self.ino = 1

    def _hdr(self, name, mode, size, nlink=1, rdev=(0, 0)):
        self.ino += 1
        # newc: 6-byte magic + 13 fields of 8 hex digits = 110 bytes exactly.  One extra field here
        # shifts every entry by 8 bytes and the kernel resets while unpacking.
        hdr = "070701%08X%08X%08X%08X%08X%08X%08X%08X%08X%08X%08X%08X%08X" % (
            self.ino, mode, 0, 0, nlink, 0, size, 0, 0, rdev[0], rdev[1], len(name) + 1, 0)
        assert len(hdr) == 110, len(hdr)
        self.buf.write(hdr.encode())
        self.buf.write(name.encode() + b"\0")
        pad = (-(len(hdr) + len(name) + 1)) % 4
        self.buf.write(b"\0" * pad)

    def add(self, name, mode, data=b"", rdev=(0, 0)):
        self._hdr(name, mode, len(data), rdev=rdev)
        self.buf.write(data)
        self.buf.write(b"\0" * ((-len(data)) % 4))

    def add_tree(self, src, prefix=""):
        for dirpath, dirnames, filenames in os.walk(src):
            rel = os.path.relpath(dirpath, src)
            rel = "" if rel == "." else rel
            if rel:
                self.add(prefix + rel, 0o040755 | 0o40000)
            for d in dirnames:
                dpath = os.path.join(rel, d)
                st = os.lstat(os.path.join(src, dpath))
                self.add(prefix + dpath, 0o040755 | 0o40000)
                _ = st
            for f in filenames:
                p = os.path.join(dirpath, f)
                st = os.lstat(p)
                name = prefix + os.path.join(rel, f)
                if os.path.islink(p):
                    self.add(name, 0o120777, os.readlink(p).encode())
                else:
                    with open(p, "rb") as fh:
                        self.add(name, (st.st_mode & 0o7777) | 0o100000, fh.read())

    def finish(self):
        self.add("TRAILER!!!", 0)
        return self.buf.getvalue()


def build_initramfs(out, rootfs_tar=None, rootfs_dir=None, work=None):
    """rootfs tree (directory or tarball) + the virtio modules + /init -> a bootable cpio."""
    work = work or os.path.join(HVI_DIR, "initramfs-build")
    shutil.rmtree(work, ignore_errors=True)
    os.makedirs(work)
    root = os.path.join(work, "root")
    os.makedirs(root)
    if rootfs_dir:
        subprocess.run(["cp", "-a", rootfs_dir.rstrip("/") + "/.", root + "/"], check=True)
    else:
        tar = rootfs_tar or MINIROOTFS
        if not os.path.exists(tar):
            raise SystemExit("missing %s (fetch an alpine minirootfs)" % tar)
        with tarfile.open(tar) as t:
            if sys.version_info >= (3, 12):      # `filter=` only exists from 3.11.4/3.12
                t.extractall(root, filter="tar")
            else:
                t.extractall(root)
    moddir = os.path.join(root, "modules")
    os.makedirs(moddir)
    with tarfile.open(APK) as apk:
        for m in MODULES:
            member = "lib/modules/%s/%s" % (MODVER, m)
            fh = apk.extractfile(member)
            if fh is None:
                raise SystemExit("missing module %s in the apk" % member)
            with open(os.path.join(moddir, os.path.basename(m).replace(".gz", "")), "wb") as o:
                import gzip
                o.write(gzip.decompress(fh.read()))
    with open(os.path.join(root, "init"), "w") as f:
        f.write(INIT)
    os.chmod(os.path.join(root, "init"), 0o755)
    for d in ("dev", "proc", "sys", "io", "tmp"):
        os.makedirs(os.path.join(root, d), exist_ok=True)
    c = Cpio()
    c.add("dev", 0o040755 | 0o40000)
    c.add("dev/console", 0o020600, rdev=(5, 1))
    c.add("dev/null", 0o020666, rdev=(1, 3))
    c.add_tree(root, prefix="")
    data = c.finish()
    with open(out, "wb") as f:
        f.write(data)
    return out, len(data)


def guest_stamp():
    """Fingerprint of everything baked into the guest image, so a changed init rebuilds it."""
    import hashlib
    h = hashlib.sha256()
    h.update(INIT.encode())
    h.update("|".join(MODULES).encode())
    for p in (APK, os.path.join(HVI_DIR, "rootfs")):
        try:
            h.update(b"%s:%d" % (p.encode(), os.path.getmtime(p)))
        except OSError:
            pass
    return h.hexdigest()[:16]


def ensure_initramfs(path=None, rootfs_dir=None, force=False):
    """Build the guest image if it is missing or was built from a different init."""
    path = path or os.path.join(HVI_DIR, "initramfs-hvi.cpio")
    stamp = path + ".stamp"
    want = guest_stamp()
    have = open(stamp).read().strip() if os.path.exists(stamp) else ""
    if force or not os.path.exists(path) or have != want:
        build_initramfs(path, rootfs_dir=rootfs_dir or os.path.join(HVI_DIR, "rootfs"))
        with open(stamp, "w") as f:
            f.write(want)
    return path


def run_in_guest(code, share=None, mem=2048, cpus=1, timeout=180, kernel=KERNEL, initramfs=None,
                 deps_dir=None, extra_mounts=()):
    """Run `code` (a shell fragment) in a guest with `share` mounted at /io.  -> (rc, output, console)

    extra_mounts: (host_dir, tag) pairs, mounted read-only at /<tag>.  The share itself is
    read-write, which is how results come back.
    """
    share = share or os.path.join(HVI_DIR, "io")
    os.makedirs(share, exist_ok=True)
    for f in ("out.txt", "rc.txt"):
        try:
            os.unlink(os.path.join(share, f))
        except OSError:
            pass
    with open(os.path.join(share, "run.sh"), "w") as f:
        f.write(code if code.startswith("#!") else "#!/bin/sh\n" + code + "\n")
    initramfs = initramfs or ensure_initramfs()
    t0 = time.time()
    cmd = [HVI, "boot", "--kernel", kernel, "--initramfs", initramfs,
           "--mem-mib", str(mem), "--cpus", str(cpus),
           "--fs", "tag=work,path=%s" % share]
    extra = ""
    if deps_dir:
        os.makedirs(deps_dir, exist_ok=True)
        cmd += ["--fs", "tag=deps,path=%s" % deps_dir]
        extra = " egpu.mount=deps"
    for item in extra_mounts:
        host_dir, tag = item[0], item[1]
        ro = item[2] if len(item) > 2 else True
        cmd += ["--fs", "tag=%s,path=%s%s" % (tag, host_dir, ",readonly=on" if ro else "")]
        extra += " egpu.mount=%s" % tag
    # hvi passes the cmdline through; the guest's init reads the tags back off it
    cmd += ["--cmdline", "earlycon console=ttyAMA0 panic=-1" + extra]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        console = (r.stdout or "") + (r.stderr or "")
    except subprocess.TimeoutExpired as e:
        console = ((e.stdout or b"").decode("utf-8", "replace") if isinstance(e.stdout, bytes)
                   else (e.stdout or "")) + "\n[timed out after %ds]" % timeout
    out = rc = None
    try:
        with open(os.path.join(share, "out.txt")) as f:
            out = f.read()
        with open(os.path.join(share, "rc.txt")) as f:
            rc = int(f.read().strip())
    except (OSError, ValueError):
        pass
    return dict(rc=rc, out=out, console=console, wall=time.time() - t0)


if __name__ == "__main__":
    code = sys.argv[1] if len(sys.argv) > 1 else "uname -a; python3 -V; python3 -c 'import numpy; print(\"numpy\", numpy.__version__)'"
    print("building initramfs...")
    rf = os.path.join(HVI_DIR, "rootfs")
    path, size = build_initramfs(os.path.join(HVI_DIR, "initramfs-hvi.cpio"),
                                 rootfs_dir=rf if os.path.isdir(rf) else None)
    print("  %s (%.1f MB)" % (path, size / 1e6))
    print("running in the guest: %r" % code[:80])
    r = run_in_guest(code)
    print("\n--- console (tail) ---")
    print("\n".join(r["console"].splitlines()[-12:]))
    print("\n--- rc: %s, wall %.1fs ---" % (r["rc"], r["wall"]))
    print("--- guest stdout ---")
    print(r["out"] if r["out"] is not None else "(no out.txt: the share never mounted)")
