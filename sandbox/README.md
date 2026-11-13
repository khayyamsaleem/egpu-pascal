# The code sandbox

The chat's `python` and `shell` tools run in here, not on your Mac. It exists so the model can write
and run code without the answer to "may I run this?" having to be "only if I read it first".

Two backends; `bin/local-gpu sandbox --status` says which one would run.

## 1. hvi microVM (default)

[hvi](https://github.com/brig-sh/hvi-vmm) is a small Apache-2.0 microVMM that talks to
Hypervisor.framework directly — the substrate the Brig agent-sandbox project runs on. Each call
boots **its own Linux guest with its own kernel**, so the boundary is a VM rather than a shared
kernel, and it is quick: a measured 0.3–0.7 s per call for boot + Python + numpy + pandas + reading
a file, because the guest image is small and lives in RAM.

| piece | where |
|---|---|
| `hvi` binary | `~/.local/bin/hvi` (signed release; `hvi smoke` proves the host will run guests) |
| guest kernel | `~/.local/state/local-gpu/sandbox/hvi/Image.alpine.raw` — Alpine 6.12.117, raw arm64 `Image` |
| guest rootfs | `…/hvi/rootfs` — Alpine 3.21 + python3 + the curated packages, built with `apk --root` |
| guest image | `…/hvi/initramfs-hvi.cpio`, built by `localgpu/hvi_guest.py` (pure-Python `newc` cpio) |

Rebuild the guest after changing the init or the package set: `bin/local-gpu sandbox --guest`.
`hvi_guest.py` stamps the image with a fingerprint of the init and modules and rebuilds when they
change — it will not silently run a stale guest (which it did once, and that cost an hour).

**What the guest sees**: `/work` (writable, where results come back), `/deps` (on `PYTHONPATH`),
`/repo` (the chat's `--root`, read-only) and `/mnt/<tag>` for anything you mounted. Nothing else —
verified: no host home, no docker socket, no network (`--fs` shares only; hvi's default network is a
stub that forwards nothing).

The guest's devices are virtio-mmio as described in the devicetree hvi generates, so the guest kernel
needs `virtio_mmio`, `fuse` and `virtiofs` — held as modules, dropped into the image and `insmod`ed
by the guest's `/init` in that order. `run_in_guest` writes `run.sh` into the share; the guest runs
it, writes `out.txt` and `rc.txt` back, then powers off.

Two gotchas worth remembering: a `newc` header is exactly 110 bytes (6 magic + 13 fields of 8 hex
digits) and one extra field shifts every entry, which the kernel answers with an instant reset; and
Alpine's `boot/vmlinuz-virt` is a PE32+ EFI stub, not the raw `Image` hvi wants — the gzip payload
inside it decompresses to one (`…/Image.efi-stub` is kept beside `Image.alpine.raw`).

## 2. docker (fallback)

`egpu-sandbox:3.12`, used when hvi or its guest image is missing, and **always** for `deps` installs
(that step needs egress; the guest has none).

| measure | how |
|---|---|
| no egress | `--network none` (only `net=true` or a deps install gets a network, and both ask first) |
| no host filesystem | only the declared mounts; no docker socket, no host env, no SSH agent |
| read-only rootfs | `--read-only` plus a 256 MB `tmpfs` on `/tmp` |
| unprivileged | uid 1000 from the image, `--cap-drop ALL`, `--security-opt no-new-privileges` |
| bounded | `--memory 512m --cpus 1 --pids-limit 128`, plus a wall-clock timeout and `docker rm -f` |

Build it with `bin/local-gpu sandbox --build` (needs network once, ~650 MB).

## Sharing a host folder

Read-only by default, and **only you can declare it** — there is deliberately no tool argument for
mounts, so the model cannot ask for `~/.ssh`. Three equivalent ways:

```sh
bin/local-gpu chat --tools --mount ~/data:/data:ro --mount ~/scratch:/scratch:rw
LOCAL_GPU_SANDBOX_MOUNTS=~/data:/data:ro bin/local-gpu chat --tools
```
```
> /mount ~/data:/data         # in a session: read-only by default
> /mount                      # list what is shared
```

The folder appears at `/mnt/<tag>` (`/mnt/data` above) in both backends. Verified: a read-only mount
answers `Read-only file system` to a write, and a read-write mount's file shows up on the host
afterwards.

One tag is special: `repo` mounts at **`/repo`** in both backends, not `/mnt/repo`. It did differ —
the guest's `/init` put every extra tag under `/mnt` — and a skill note that promised
`/repo/sandbox/tools/candles.py` then failed inside the guest with *"No such file or directory"*.
Anything you want a model to be able to run belongs in the repo (read-only, so the model cannot
change it): `sandbox/tools/candles.py` is there for exactly that reason.

The guest also carries `python` as well as `python3` (Alpine ships only the latter, the container
image has both, and a model that types `python` otherwise gets a bare `command not found`).

## Getting dependencies

Both backends are built from **one list** (`sandbox_exec.APK_SEED` / `PIP_SEED`), so they cannot
drift apart.  They did once — the guest was built without matplotlib while the tool description
promised it, and the model spent three round trips discovering that — which is why the list now
lives in exactly one place and each build stamps itself with it.

    numpy  pandas  matplotlib  scipy  sympy  tabulate  requests  beautifulsoup4  lxml  pyyaml
    python-dateutil

For anything else the tool takes `deps`, and the model is told to use it:

```python
python(code="import humanize; print(humanize.intcomma(1234567))", deps=["humanize"])
```

**Each backend has its own deps dir**, because they are different libcs: the container is Debian
(glibc, manylinux wheels), the guest is Alpine (musl).  A glibc numpy sitting on the musl guest's
`PYTHONPATH` shadows its working numpy and every import dies with *"No module named
'numpy._core._multiarray_umath'"* — so `deps` installs run in a container matching the libc of the
backend that will use them:

    ~/.local/state/local-gpu/sandbox/deps/docker   glibc, for the container
    ~/.local/state/local-gpu/sandbox/deps/guest    musl,  for the microVM

Installing needs egress, so that step runs in a container even when the run itself is the guest; it
asks you first, because pip runs arbitrary code from the index.

### Pre-seeding

One lean image per backend plus a deps dir is the right split for this machine: the guest is a
~325 MB initramfs (it was 195 MB before matplotlib, scipy, bs4 and lxml were added to the list) that
*every* call loads into RAM, so heavy stacks belong in the deps dir, not in the image.  `--seed`
pre-loads a named set into both dirs so a session does not spend its first calls installing:

    bin/local-gpu sandbox --seed plot     # mplfinance, seaborn, plotly, pillow
    bin/local-gpu sandbox --seed data     # scikit-learn, statsmodels, duckdb, pyarrow, openpyxl
    bin/local-gpu sandbox --seed web      # httpx, html5lib, feedparser, trafilatura
    bin/local-gpu sandbox --seed text | math

If you need *system* libraries no pip wheel can bring (libpq, ffmpeg, a compiler), that is the point
at which a second guest image earns its keep — add it to `APK_SEED` and rebuild
(`sandbox --guest`), rather than bolting it onto every call.

Prepare it by hand when you know what a session will need:

    bin/local-gpu sandbox --guest                 # rebuild the guest image
    bin/local-gpu sandbox --status                # backend, mounts, deps dir, work dir
    bin/local-gpu sandbox --run 'print(1+1)'      # one run
    bin/local-gpu sandbox --shell                 # interactive shell (container backend)
    bin/local-gpu sandbox --reset                 # wipe the work and deps dirs

## Notes

- The repo is mounted read-only as `/repo`; edits go through `write_file`, which asks.
- `/work` and the deps dir persist across calls and sessions; `--reset` clears them.
- hvi's docs note that its disk and share paths have no live-boot coverage in its CI, so the numbers
  and the read-only/read-write behaviour above were measured here, on this host, rather than taken
  from its documentation.
