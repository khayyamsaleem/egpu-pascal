#!/usr/bin/env python3
"""Run model-written code in a sandbox: an hvi microVM by default, a container as fallback.

Two backends, picked automatically (`--backend`, or LOCAL_GPU_SANDBOX_BACKEND to force one):

* **hvi** (preferred) -- the microVMM from brig-sh/hvi-vmm, which talks to Hypervisor.framework
  directly, so each call boots its own Linux guest.  The boundary is a VM with its own kernel, not a
  shared kernel: the guest gets a writable virtio-fs share at /work (how results come back), a
  persistent /deps on PYTHONPATH, and read-only mounts for whatever the user declared.  Measured:
  0.7 s for boot + Python + numpy + pandas + a file written back.
* **docker** (fallback) -- egpu-sandbox:3.12 with no network, a read-only rootfs and dropped caps.
  Used when hvi or its guest image is missing, and always for `deps` installs, because that step
  needs egress and the guest has none.

What holds in both: no host filesystem beyond the declared mounts, no docker socket, no host env,
capped memory and CPU, a wall-clock timeout, an unprivileged user.  Host folders are mounted only
from the *user's* declaration -- LOCAL_GPU_SANDBOX_MOUNTS, `local-gpu chat --mount`, or `/mount` in a
session.  There is deliberately no tool argument for it: the model must not be able to ask for ~/.ssh.

Dependencies: both backends see the same set.  Preinstalled: numpy pandas matplotlib scipy sympy
tabulate requests beautifulsoup4 lxml pyyaml python-dateutil.  Anything else is `deps=[...]`, which
pip-installs into the shared deps dir (network for that step only, and it asks first).

  bin/local-gpu sandbox --status            what is installed, and which backend would run
  bin/local-gpu sandbox --build             build the container image (the fallback)
  bin/local-gpu sandbox --guest             build the hvi guest image (kernel + rootfs + initramfs)
  bin/local-gpu sandbox --reset             wipe the deps dir and the work dir
  bin/local-gpu sandbox --shell             a shell in the container backend
  bin/local-gpu sandbox --run 'print(1+1)'  one run, printed
  bin/local-gpu sandbox --mount ~/data:/data:ro --run 'print(open("/mnt/data/x").read())'
"""
import os
import re
import subprocess
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMAGE = "egpu-sandbox:3.12"
HVI = os.path.expanduser("~/.local/bin/hvi")
STATE = os.path.expanduser("~/.local/state/local-gpu/sandbox")
WORK = os.path.join(STATE, "work")
DEPS = os.path.join(STATE, "deps")
GUEST_KERNEL = os.path.join(STATE, "hvi", "Image.alpine.raw")
GUEST_INITRAMFS = os.path.join(STATE, "hvi", "initramfs-hvi.cpio")
MAX_OUT = 6000
LIMITS = ["--memory", "512m", "--cpus", "1", "--pids-limit", "128", "--cap-drop", "ALL",
          "--security-opt", "no-new-privileges"]


# ------------------------------------------------------------------ mounts, backend choice
def parse_mounts(spec):
    """'~/data:/data:ro,/tmp/x' -> [('/Users/…/data', 'data', True), …]  (host, tag, readonly)."""
    out = []
    for item in [s for s in str(spec or "").split(",") if s.strip()]:
        parts = item.strip().split(":")
        host = os.path.abspath(os.path.expanduser(parts[0]))
        raw = parts[1] if len(parts) > 1 and parts[1] else os.path.basename(host)
        tag = os.path.basename(str(raw).rstrip("/")) or "share"     # "/data" and "data" both -> data
        ro = not (len(parts) > 2 and parts[2].lower() in ("rw", "readwrite"))
        out.append((host, re.sub(r"[^A-Za-z0-9_]", "_", tag), ro))
    return out


def declared_mounts(extra=()):
    return parse_mounts(os.environ.get("LOCAL_GPU_SANDBOX_MOUNTS", "")) + [tuple(m) for m in extra]


def _docker(args, timeout=180):
    return subprocess.run(["docker"] + args, stdin=subprocess.DEVNULL, capture_output=True,
                          text=True, timeout=timeout)


def image_present():
    return _docker(["image", "inspect", IMAGE], timeout=30).returncode == 0


def guest_ready():
    return (os.path.exists(HVI) and os.path.exists(GUEST_KERNEL)
            and os.path.exists(GUEST_INITRAMFS))


def backend(prefer=None):
    want = prefer or os.environ.get("LOCAL_GPU_SANDBOX_BACKEND", "")
    if want in ("hvi", "docker"):
        return want
    return "hvi" if guest_ready() else "docker"


# ------------------------------------------------------------------ building
def build(quiet=False):
    """Build the container image (the fallback backend)."""
    d = os.path.join(REPO, "sandbox")
    r = _docker(["build", "-t", IMAGE, "-f", os.path.join(d, "Dockerfile"), d], timeout=1800)
    if r.returncode and not quiet:
        print((r.stdout or "") + (r.stderr or ""))
    return r.returncode == 0


def build_guest(verbose=True):
    """Build the hvi guest image: an initramfs carrying Python, the virtio drivers and /init."""
    sys.path.insert(0, os.path.join(REPO, "localgpu"))
    import hvi_guest
    rootfs = os.path.join(hvi_guest.HVI_DIR, "rootfs")
    if not os.path.isdir(rootfs):
        return ("no guest rootfs at %s -- build one with:\n"
                "  docker run --rm -v %s:/out alpine:3.21 sh -c 'apk add --no-cache --root /out "
                "--initdb --arch aarch64 --keys-dir /etc/apk/keys "
                "-X https://dl-cdn.alpinelinux.org/alpine/v3.21/main "
                "-X https://dl-cdn.alpinelinux.org/alpine/v3.21/community "
                "python3 py3-numpy py3-pandas py3-requests py3-yaml py3-tabulate py3-sympy busybox'"
                % (rootfs, rootfs))
    path = hvi_guest.ensure_initramfs(GUEST_INITRAMFS, rootfs_dir=rootfs, force=True)
    return path if not verbose else "%s (%s)" % (path, _size(path))


def _size(path):
    try:
        return "%.1f MB" % (os.path.getsize(path) / 1e6)
    except OSError:
        return "missing"


def ensure_image():
    if image_present():
        return None
    if not build(quiet=True):
        return ("no sandbox image and the build failed -- run `bin/local-gpu sandbox --build` "
                "(needs network once, ~200 MB)")
    return None


# ------------------------------------------------------------------ deps
def install(deps, timeout=900):
    """pip-install extras into the shared deps dir.  Container backend: this is the one step that
    needs a network, and the guest has none."""
    if not deps:
        return ""
    os.makedirs(DEPS, exist_ok=True)
    r = _docker(["run", "--rm", "--network", "bridge"] + LIMITS +
                ["-v", "%s:/deps" % DEPS, "-e", "HOME=/tmp", IMAGE,
                 "pip", "install", "--target", "/deps", "--no-cache-dir"]
                + [str(d) for d in deps], timeout=timeout)
    out = ((r.stdout or "") + (r.stderr or "")).strip()
    return "%s: %s\n%s" % ("installed" if r.returncode == 0 else "pip FAILED",
                           ", ".join(map(str, deps)), "\n".join(out.splitlines()[-6:]))


# ------------------------------------------------------------------ running
def run(code=None, command=None, files=None, deps=None, timeout=30, net=False, root=None,
        mounts=(), backend_pref=None):
    """Run `code` (python) or `command` (sh) in the sandbox.  Returns text, never raises."""
    if not code and not command:
        return "nothing to run (pass `code` or `command`)"
    notes = []
    if files:
        os.makedirs(WORK, exist_ok=True)
        for name, body in (files or {}).items():
            p = os.path.join(WORK, os.path.basename(str(name)))
            with open(p, "w") as f:
                f.write(str(body))
            notes.append("wrote /work/%s (%d bytes)" % (os.path.basename(str(name)),
                                                        len(str(body))))
    if deps:
        notes.append(install(deps))
    back = backend(backend_pref)
    all_mounts = declared_mounts(mounts)          # env-declared + this session's
    if back == "hvi" and not net:
        body = _run_hvi(code=code, command=command, timeout=timeout, root=root, mounts=all_mounts)
    else:
        body = _run_docker(code=code, command=command, timeout=timeout, net=net, root=root,
                           mounts=all_mounts)
    return "\n".join([n for n in notes if n] + [body])


def _shq(s):
    return "'" + str(s).replace("'", "'\"'\"'") + "'"


def _run_hvi(code=None, command=None, timeout=30, root=None, mounts=()):
    sys.path.insert(0, os.path.join(REPO, "localgpu"))
    import hvi_guest
    payload = ("python3 -c " + _shq(code)) if code is not None else str(command)
    extra = list(mounts)
    if root and os.path.isdir(root):
        extra.append((os.path.abspath(root), "repo", True))
    r = hvi_guest.run_in_guest("cd /work\n" + payload, share=WORK, deps_dir=DEPS,
                               timeout=timeout + 5,
                               extra_mounts=[(h, t, ro) for h, t, ro in extra], mem=2048, cpus=1)
    out = (r["out"] or "").strip()
    rc = r["rc"]
    if rc is None:                       # the guest never reported: show why
        tail = "\n".join(str(r["console"]).splitlines()[-6:])
        out = (out + "\n[guest did not report; console tail]\n" + tail).strip()
        rc = 1
    if len(out) > MAX_OUT:
        out = out[:MAX_OUT] + "\n[truncated at %d bytes]" % MAX_OUT
    return "[sandbox hvi microVM, %.1fs, net off] exit %s\n%s" % (r["wall"], rc,
                                                                 out or "(no output)")


def _run_docker(code=None, command=None, timeout=30, net=False, root=None, mounts=()):
    msg = ensure_image()
    if msg:
        return msg
    os.makedirs(WORK, exist_ok=True)
    os.makedirs(DEPS, exist_ok=True)
    m = ["-v", "%s:/work" % WORK, "-v", "%s:/deps" % DEPS]
    for host, tag, ro in list(mounts):
        m += ["-v", "%s:/mnt/%s%s" % (host, tag, ":ro" if ro else "")]
    if root and os.path.isdir(root):
        m += ["-v", "%s:/repo:ro" % os.path.abspath(root)]
    cmd = ["python", "-c", str(code)] if code is not None else ["sh", "-c", str(command)]
    name = "egpu-sandbox-%d" % (time.time() * 1000 % 10 ** 9)
    args = (["run", "--rm", "--name", name, "--network", "bridge" if net else "none",
             "--read-only", "--tmpfs", "/tmp:size=256m,mode=1777"] + LIMITS + m + [IMAGE] + cmd)
    t0 = time.time()
    try:
        r = _docker(args, timeout=timeout + 3)
        out, rc = ((r.stdout or "") + (r.stderr or "")), r.returncode
    except subprocess.TimeoutExpired:
        _docker(["rm", "-f", name], timeout=60)          # a hung run must not hold a slot
        out, rc = "killed after %ds (timeout)" % timeout, 124
    out = out.strip()
    if len(out) > MAX_OUT:
        out = out[:MAX_OUT] + "\n[truncated at %d bytes]" % MAX_OUT
    return "[sandbox docker, %s, net %s] exit %d\n%s" % (
        "%ds limit" % timeout if rc == 124 else "%.1fs" % (time.time() - t0),
        "on" if net else "off", rc, out or "(no output)")


# ------------------------------------------------------------------ status / reset / CLI
def status():
    b = backend()
    lines = ["backend  %s%s" % (b, "" if b == "hvi" else "   (hvi guest not built)"),
             "hvi      %s" % ("%s  guest %s  kernel %s" % (HVI, os.path.basename(GUEST_INITRAMFS),
                                                           os.path.basename(GUEST_KERNEL))
                              if guest_ready() else
                              "not ready: %s" % ("guest image missing -- bin/local-gpu sandbox "
                                                 "--guest" if os.path.exists(HVI)
                                                 else "~/.local/bin/hvi missing")),
             "docker   %s%s" % (IMAGE, "" if image_present() else
                                "   (not built: bin/local-gpu sandbox --build)"),
             "mounts   %s" % (", ".join("%s -> /mnt/%s%s" % (h, t, ":ro" if ro else ":rw")
                                        for h, t, ro in declared_mounts())
                              or "none declared (LOCAL_GPU_SANDBOX_MOUNTS or --mount)"),
             "deps dir %s" % DEPS,
             "work dir %s" % WORK]
    if image_present():
        r = _docker(["run", "--rm", "--network", "none"] + LIMITS + ["-v", "%s:/deps" % DEPS,
                     IMAGE, "sh", "-c",
                     "python -c 'import sys; print(\"container python\", sys.version.split()[0])'"],
                     timeout=180)
        lines.append((r.stdout or r.stderr or "").strip())
    return "\n".join(lines)


def reset():
    subprocess.run(["rm", "-rf", WORK, DEPS], check=False)
    return "removed %s and %s" % (WORK, DEPS)


def main(argv):
    what = argv[0] if argv else "--status"
    cli_mounts = parse_mounts(",".join(a.split("=", 1)[1] for a in argv if a.startswith("--mount=")))
    if what == "--build":
        print("building %s ..." % IMAGE)
        print("built" if build() else "build FAILED")
    elif what == "--guest":
        print(build_guest())
    elif what == "--reset":
        print(reset())
    elif what == "--shell":
        problem = ensure_image()
        if problem:
            print(problem)
            return 1
        os.makedirs(WORK, exist_ok=True)
        m = ["-v", "%s:/work" % WORK, "-v", "%s:/deps" % DEPS]
        for host, tag, ro in declared_mounts(cli_mounts):
            m += ["-v", "%s:/mnt/%s%s" % (host, tag, ":ro" if ro else "")]
        subprocess.run(["docker", "run", "--rm", "-it", "--network", "bridge"] + LIMITS + m +
                       [IMAGE, "bash"], check=False)
    elif what == "--run":
        print(run(code=argv[1] if len(argv) > 1 else "print(1+1)",
                  mounts=declared_mounts(cli_mounts), timeout=60))
    else:
        print(status())
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
