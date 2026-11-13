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

## Getting dependencies

Both backends ship the same curated set, so most questions need nothing:

    numpy  pandas  matplotlib  scipy  sympy  tabulate  requests  beautifulsoup4  lxml  pyyaml
    python-dateutil

For anything else the tool takes `deps`, and the model is told to use it:

```python
python(code="import humanize; print(humanize.intcomma(1234567))", deps=["humanize"])
```

That is `pip install --target` into `~/.local/state/local-gpu/sandbox/deps`, which both backends
mount at `/deps` on `PYTHONPATH` — one install serves the container *and* every later guest. It is
the only step that gets a network and it asks you first, because pip runs arbitrary code from the
index.

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
