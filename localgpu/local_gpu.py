#!/usr/bin/env python3
"""local-gpu: serve a model on the GTX 1080 Ti eGPU with one command.

  local-gpu                         serve the default model (gemma4-e4b) and print the endpoints
  local-gpu serve --model llama3.1-8b [--ctx 16384]
  local-gpu chat [--model M]        terminal chat against the running server (starts it if needed)
  local-gpu run M "prompt"          one-shot answer
  local-gpu status | models | pull M | logs [server|shimd|proxy] | bench [M] | stop [--all]

What `serve` does, skipping whatever is already up:
  1. the card: devinit + fb if cold; the memory reclocked to --mclk (default 1080 MHz, boot is
     405; pascal/mclk.py) and proven with a 9 GiB VRAM pattern test, stepping down to 810 and
     then 405 on failure; reset_engines + gr; then pascal/shimd.py (the driver server; it holds
     the card and keeps the Mac awake)
  2. the runtime container: `llamacpp` (llama.cpp built for sm_61 + shim/build-cuda12), or
     `maxnightly` for MAX models (*-max)
  3. the model: downloaded once into the container's /models volume (resumable)
  4. the server: llama-server (or `max serve`), all layers on the GPU
  5. an Ollama-compatible endpoint on http://127.0.0.1:11434 (localgpu/ollama_proxy.py) that
     also passes OpenAI requests through at /v1 -- the one address clients need.

Standard library only; card scripts run with ~/.venvs/tinygpu/bin/python.
"""
import argparse, json, os, signal, socket, struct, subprocess, sys, time, urllib.request

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STATE = os.path.expanduser("~/.local/state/local-gpu")
VENV_PY = os.path.expanduser("~/.venvs/tinygpu/bin/python")
LLAMA_CT, LLAMA_IMAGE, MAX_CT = "llamacpp", "egpu-llamacpp:12.8", "maxnightly"
LLAMA_PORT, MAX_PORT, PROXY_PORT, SHIMD_PORT = 8080, 8000, 11434, 7461
SHIM_ENV = "LD_LIBRARY_PATH=/repo/shim/build-cuda12 SHIM_MODE=remote SHIM_QUIET=1"
HF = "https://huggingface.co/%s/resolve/main/%s"

MODELS = {
    "gemma4-e4b": dict(repo="google/gemma-4-E4B-it-qat-q4_0-gguf", file="gemma-4-E4B_q4_0-it.gguf", size=5.15,
                       family="gemma4", params="8B (E4B)", quant="Q4_0", ctx=16384,
                       desc="Google Gemma 4 E4B, quantization-aware 4-bit: fast, tools, the default",
                       # its 2.7 GB of per-layer/input embeddings on the card: +10% prompt speed (pp512 764 -> 840)
                       args="-ot per_layer_token_embd.weight=CUDA0,token_embd.weight=CUDA0"),
    "gemma4-12b": dict(repo="google/gemma-4-12B-it-qat-q4_0-gguf", file="gemma-4-12b-it-qat-q4_0.gguf", size=6.98,
                       family="gemma4", params="12B", quant="Q4_0", ctx=8192,
                       desc="Google Gemma 4 12B, QAT 4-bit: strongest that fits 11 GB, slower"),
    "llama3.1-8b": dict(repo="bartowski/Meta-Llama-3.1-8B-Instruct-GGUF", file="Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf",
                        size=4.92, family="llama", params="8B", quant="Q4_K_M", ctx=16384,
                        desc="Meta Llama 3.1 8B Instruct, Q4_K_M"),
    "qwen3-8b": dict(repo="Qwen/Qwen3-8B-GGUF", file="Qwen3-8B-Q4_K_M.gguf", size=5.03, family="qwen3",
                     params="8B", quant="Q4_K_M", ctx=16384, thinking=True,
                     desc="Qwen3 8B, Q4_K_M: reasoning (thinking) model, good tool use"),
    "smollm2": dict(repo="bartowski/SmolLM2-135M-Instruct-GGUF", file="SmolLM2-135M-Instruct-Q4_0.gguf", size=0.09,
                    family="llama", params="135M", quant="Q4_0", ctx=8192, desc="SmolLM2 135M: tiny, for testing"),
    # MAX (Modular) runtime: Hugging Face checkpoints, bf16/fp32 only (see maxpatch/README.md)
    "gemma4-e2b-max": dict(runtime="max", hf="google/gemma-4-E2B-it", encoding="bfloat16", size=9.6, family="gemma4",
                           params="5B (E2B)", quant="BF16", ctx=8192, extra="--no-prefer-module-v3",
                           env="MAX_GEMMA4_ACTIVATION_MIB=1024", desc="Gemma 4 E2B on Modular MAX (nightly), bf16"),
    "smollm2-max": dict(runtime="max", hf="HuggingFaceTB/SmolLM2-135M-Instruct", encoding="float32", size=0.5,
                        family="llama", params="135M", quant="F32", ctx=8192, desc="SmolLM2 135M on Modular MAX, fp32"),
}
DEFAULT = "gemma4-e4b"


def say(msg):
    print("local-gpu: " + msg, flush=True)


def die(msg):
    print("local-gpu: error: " + msg, file=sys.stderr, flush=True)
    sys.exit(1)


def sh(cmd, check=True, capture=True, **kw):
    r = subprocess.run(cmd, shell=isinstance(cmd, str), text=True, capture_output=capture, **kw)
    if check and r.returncode:
        die("command failed (%d): %s\n%s" % (r.returncode, cmd if isinstance(cmd, str) else " ".join(cmd),
                                            (r.stderr or r.stdout or "")[-2000:] if capture else ""))
    return r


def dexec(ct, cmd, check=True, capture=True, detach=False):
    return sh(["docker", "exec"] + (["-d"] if detach else []) + [ct, "bash", "-lc", cmd], check=check, capture=capture)


def state_file(name):
    os.makedirs(STATE, exist_ok=True)
    return os.path.join(STATE, name)


def pid_alive(name):
    try:
        pid = int(open(state_file(name + ".pid")).read())
        os.kill(pid, 0)
        return pid
    except (OSError, ValueError):
        return None


def spawn(name, argv, cwd=None, env=None):
    log = open(state_file(name + ".log"), "a")
    log.write("\n=== %s %s\n" % (time.strftime("%F %T"), " ".join(argv))); log.flush()
    p = subprocess.Popen(argv, cwd=cwd, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                         start_new_session=True, env=env)
    open(state_file(name + ".pid"), "w").write(str(p.pid))
    return p


def kill(name):
    pid = pid_alive(name)
    if pid:
        os.kill(pid, signal.SIGTERM)
        for _ in range(50):
            if not pid_alive(name):
                break
            time.sleep(0.1)


def http_ok(url, timeout=2):
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return r.status == 200
    except Exception:
        return False


# ------------------------------------------------------------------ the card + shimd
def shimd_info():
    try:
        s = socket.create_connection(("127.0.0.1", SHIMD_PORT), timeout=2)
        s.sendall(struct.pack("<II", 1, 0))
        st, n = struct.unpack("<iI", s.recv(8))
        d = s.recv(n); s.close()
        free, total, sms, sm_khz, mem_khz = struct.unpack("<QQIII", d[:28])
        return dict(free=free, total=total, sm_mhz=sm_khz // 1000, mem_mhz=mem_khz // 1000)
    except (OSError, struct.error):
        return None


def shimd_temp():
    try:
        s = socket.create_connection(("127.0.0.1", SHIMD_PORT), timeout=2)
        s.sendall(struct.pack("<II", 11, 0))
        st, n = struct.unpack("<iI", s.recv(8))
        d = s.recv(n); s.close()
        return struct.unpack("<iI", d[:8])[0] if st == 0 and n >= 8 else None
    except (OSError, struct.error):
        return None


def card_scripts(*names_args):
    for args in names_args:
        say("  %s" % " ".join(args))
        r = sh([VENV_PY] + list(args), cwd=REPO, check=False)
        out = (r.stdout or "") + (r.stderr or "")
        if r.returncode or "FAIL" in out.split("RESULT:")[-1][:40]:
            die("%s failed:\n%s" % (args[0], out[-2500:]))


MCLK_DEFAULT = int(os.environ.get("LOCAL_GPU_MCLK", "1080"))
MCLK_FALLBACK = 810


def card_out(*args):
    r = sh([VENV_PY] + list(args), cwd=REPO, check=False)
    return r.returncode, (r.stdout or "") + (r.stderr or "")


def set_mclk(mhz, gr_up=False):
    """Reclock the memory (pascal/mclk.py), reload GR, and prove the VRAM with a 9 GiB pattern
    test. Returns the clock actually in use; on any failure re-POSTs and steps down."""
    for target in [mhz] + [f for f in (MCLK_FALLBACK,) if f < mhz]:
        rc, cur = card_out("pascal/mclk.py", "get")
        cur = int(cur.split()[-1]) if rc == 0 and cur.split() else 0
        if cur != target:
            say("memory clock %d -> %d MHz (pascal/mclk.py)" % (cur or 405, target))
            rc, out = card_out("pascal/mclk.py", "set", str(target), "--live")
            if rc:
                say("  reclock to %d MHz failed: %s" % (target, out.strip().splitlines()[-1][:200]))
                repost(); gr_up = False; continue
            gr_up = False
        if not gr_up:
            card_scripts(("pascal/reset_engines.py",), ("pascal/gr.py",))
        # 20 passes (~5 s): a marginal lane shows as rare single-bit flips (1134 MHz warm: 17 words
        # in 100 passes), so a 1-pass check is not enough
        rc, out = card_out("pascal/memtest.py", "--gib", "9", "--seeds", "5", "--loops", "4")
        res = [l for l in out.splitlines() if "bandwidth" in l or "RESULT" in l]
        if rc == 0 and "PASS" in out:
            say("  VRAM verified at %d MHz: %s" % (target, res[0].strip() if res else ""))
            return target
        say("  VRAM test FAILED at %d MHz (%s); re-POST and step down" % (target, " ".join(res)[-200:] or out.strip()[-200:]))
        repost(); gr_up = False
    say("staying at the boot memory clock (405 MHz)")
    card_scripts(("pascal/reset_engines.py",), ("pascal/gr.py",))
    return 405


def repost():
    card_scripts(("pascal/devinit.py", "firmware/gp102_vbios.rom"), ("pascal/fb.py",))


def ensure_card(vram_gib, mclk=None):
    mclk = MCLK_DEFAULT if mclk is None else mclk
    if shimd_info():
        return
    if not os.path.exists(VENV_PY):
        die("missing %s (see README.md: toolchain)" % VENV_PY)
    say("bringing up the GPU (keep the lid open while it runs)")
    probe = ("import sys; sys.path.insert(0, 'pascal'); from card import Card; import grchan; c = Card(); "
             "print(c.rd(0x100ce0), int(bool(grchan.gr_ready(c))))")
    r = sh([VENV_PY, "-c", probe], cwd=REPO, check=False)
    try:
        fb, gr = map(int, r.stdout.split()[-2:])
    except ValueError:
        die("cannot reach the card through TinyGPU (is the enclosure on and connected?)\n" + (r.stderr or "")[-1500:])
    if fb == 0:
        say("card is cold: running devinit + fb (~20 s)")
        card_scripts(("pascal/devinit.py", "firmware/gp102_vbios.rom"), ("pascal/fb.py",))
        gr = 0
    if mclk and mclk != 405:
        set_mclk(mclk, gr_up=bool(gr))     # reloads GR when it reclocks
    elif not gr:
        say("starting the graphics engine (reset_engines + gr)")
        card_scripts(("pascal/reset_engines.py",), ("pascal/gr.py",))
    spawn("shimd", [VENV_PY, "-u", "pascal/shimd.py", "--vram-limit-gib", str(vram_gib)], cwd=REPO)
    for _ in range(120):
        if shimd_info():
            say("driver server up (pascal/shimd.py)")
            return
        if not pid_alive("shimd"):
            break
        time.sleep(0.5)
    die("shimd did not start; see %s" % state_file("shimd.log"))


# ------------------------------------------------------------------ containers
def ct_state(ct):
    r = sh(["docker", "inspect", "-f", "{{.State.Running}}", ct], check=False)
    return None if r.returncode else r.stdout.strip() == "true"


def ensure_llama_container():
    st = ct_state(LLAMA_CT)
    if st is None:
        if sh(["docker", "image", "inspect", LLAMA_IMAGE], check=False).returncode:
            say("building the llama.cpp image (once)")
            sh(["docker", "build", "-t", LLAMA_IMAGE, "-f", os.path.join(REPO, "llama", "Dockerfile"),
                os.path.join(REPO, "llama")], capture=False)
        sh(["docker", "run", "-d", "--name", LLAMA_CT, "-v", "%s:/repo" % REPO, "-v", "llamacpp-models:/models",
            "-p", "127.0.0.1:%d:8080" % LLAMA_PORT, LLAMA_IMAGE, "sleep", "infinity"])
    elif not st:
        sh(["docker", "start", LLAMA_CT])
    if dexec(LLAMA_CT, "test -x /opt/llama-build/bin/llama-server", check=False).returncode:
        say("building llama.cpp for sm_61 (once, ~20 min)")
        if not os.path.isdir(os.path.join(REPO, "upstream", "llama.cpp")):
            sh(["git", "clone", "--depth", "1", "https://github.com/ggml-org/llama.cpp",
                os.path.join(REPO, "upstream", "llama.cpp")], capture=False)
        dexec(LLAMA_CT, "sh /repo/llama/build.sh", capture=False)
    if dexec(LLAMA_CT, "test -f /repo/shim/build-cuda12/libcudart.so.12", check=False).returncode:
        say("building the CUDA runtime shims")
        dexec(LLAMA_CT, "sh /repo/shim/build-cuda12.sh", capture=False)


def ensure_max_container():
    st = ct_state(MAX_CT)
    if st is None:
        die("container %s is missing (see maxpatch/README.md to set up the MAX nightly)" % MAX_CT)
    if not st:
        sh(["docker", "start", MAX_CT])


def max_url():
    ip = sh(["docker", "inspect", "-f", "{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}", MAX_CT]).stdout.strip()
    return "http://%s:%d" % (ip, MAX_PORT)


# ------------------------------------------------------------------ models
def resolve(name):
    if name in MODELS:
        return name, dict(MODELS[name])
    if name.endswith(".gguf") and "/" in name:          # hf:<repo>/<file>.gguf or <repo>/<file>.gguf
        repo, _, f = name.removeprefix("hf:").rpartition("/")
        return f.removesuffix(".gguf").lower(), dict(repo=repo, file=f, size=0, family="llama", params="", quant="",
                                                     ctx=8192, desc="custom")
    die("unknown model %r; try `local-gpu models`, or pass <hf-repo>/<file>.gguf" % name)


def ensure_model(name, m):
    if m.get("runtime") == "max":
        return
    if not dexec(LLAMA_CT, "test -f /models/%s" % m["file"], check=False).returncode:
        return
    say("downloading %s (%.1f GB) from huggingface.co/%s -- resumable, safe to Ctrl-C and rerun"
        % (m["file"], m["size"], m["repo"]))
    url = HF % (m["repo"], m["file"])
    r = dexec(LLAMA_CT, "cd /models && curl -fL -C - --retry 50 --retry-all-errors --retry-delay 3 -# -o '%s.part' '%s' "
              "&& mv '%s.part' '%s'" % (m["file"], url, m["file"], m["file"]), check=False, capture=False)
    if r.returncode:
        die("download failed; rerun the same command to resume")


def current():
    try:
        return json.load(open(state_file("current.json")))
    except (OSError, ValueError):
        return None


def stop_servers():
    kill("proxy")
    if ct_state(LLAMA_CT):
        dexec(LLAMA_CT, "pkill -f llama-server; for i in $(seq 50); do pgrep -f llama-server >/dev/null || break; sleep 0.2; done",
              check=False)
    if ct_state(MAX_CT):
        dexec(MAX_CT, "pkill -f 'max serve'; for i in $(seq 100); do pgrep -f 'max serve' >/dev/null || break; sleep 0.2; done",
              check=False)
    try:
        os.remove(state_file("current.json"))
    except OSError:
        pass


# Measured on the card (logs/bench/, RESULTS.md section 11): backend sampling +1.5-2% decode with identical
# text; ngram-mod speculation helps text the model has seen (code edits, quoting), 11/12 bench turns
# byte-identical (batched verification rounds differently). LOCAL_GPU_PLAIN=1 drops both.
LLAMA_FAST = "" if os.environ.get("LOCAL_GPU_PLAIN") == "1" else "-bs --spec-type ngram-mod"


def llama_args(a, m):
    """Extra llama-server flags: LLAMA_FAST, the model's own (MODELS[..]["args"]), then --llama-args /
    LOCAL_GPU_LLAMA_ARGS (shell text, appended last so it can override)."""
    plain = os.environ.get("LOCAL_GPU_PLAIN") == "1"
    parts = [LLAMA_FAST, "" if plain else m.get("args", ""), getattr(a, "llama_args", "") or ""]
    return " ".join(x for x in parts if x)


def serve(a):
    name, m = resolve(a.model)
    ctx = a.ctx or m["ctx"]
    cur = current()
    if cur and cur.get("model") == name and cur.get("ctx") == ctx and http_ok("http://127.0.0.1:%d/" % PROXY_PORT) \
            and http_ok(cur["backend"] + ("/health" if cur["runtime"] == "llama" else "/v1/models")):
        if not a.quiet:
            print_endpoints(name, m, cur)
        return cur
    runtime = m.get("runtime", "llama")
    ensure_llama_container() if runtime == "llama" else ensure_max_container()
    ensure_model(name, m)
    ensure_card(a.vram_gib, getattr(a, "mclk", None))
    stop_servers()
    t0 = time.time()
    if runtime == "llama":
        say("starting llama-server: %s, context %d" % (name, ctx))
        dexec(LLAMA_CT, "%s SHIM_LOG=/tmp/shim-server.log exec /opt/llama-build/bin/llama-server -m /models/%s "
              "--alias %s -ngl 99 -fa on -c %d -np 1 -cram 2048 --jinja --host 0.0.0.0 --port 8080 %s "
              "> /tmp/llama-server.log 2>&1" % (SHIM_ENV, m["file"], name, ctx, llama_args(a, m)), detach=True)
        backend, health = "http://127.0.0.1:%d" % LLAMA_PORT, "/health"
        alive = lambda: not dexec(LLAMA_CT, "pgrep -f llama-server >/dev/null", check=False).returncode
        logcmd = (LLAMA_CT, "tail -25 /tmp/llama-server.log")
    else:
        say("starting max serve: %s (%s), context %d -- compiles kernels on first start" % (name, m["hf"], ctx))
        dexec(MAX_CT, "export LD_LIBRARY_PATH=/repo/shim/build SHIM_MODE=remote SHIM_QUIET=1 SHIM_LOG=/tmp/shim-serve.log "
              "MODULAR_NVPTX_COMPILER_PATH=/usr/local/lib/python3.12/site-packages/nvidia/cuda_nvcc/bin/ptxas %s; "
              "exec max serve --model %s --served-model-name %s --quantization-encoding %s --devices gpu "
              "--no-device-graph-capture --max-length %d --max-batch-size 1 --host 0.0.0.0 --port %d %s "
              "> /tmp/serve.log 2>&1"
              % (m.get("env", ""), m["hf"], name, m["encoding"], ctx, MAX_PORT, m.get("extra", "")), detach=True)
        backend, health = max_url(), "/v1/models"
        alive = lambda: not dexec(MAX_CT, "pgrep -f 'max serve' >/dev/null", check=False).returncode
        logcmd = (MAX_CT, "grep -v UserWarning /tmp/serve.log | tail -25")
    last = 0
    while not http_ok(backend + health, timeout=3):
        if time.time() - t0 > 1800:
            die("server did not come up in 30 min:\n" + dexec(*logcmd, check=False).stdout)
        if time.time() - t0 > 5 and not alive():
            die("server exited:\n" + dexec(*logcmd, check=False).stdout)
        if time.time() - last > 15:
            say("  loading... %.0f s" % (time.time() - t0)); last = time.time()
        time.sleep(1)
    say("model loaded in %.0f s" % (time.time() - t0))
    args = [sys.executable, os.path.join(REPO, "localgpu", "ollama_proxy.py"), "--backend", backend, "--model", name,
            "--port", str(PROXY_PORT), "--size", str(int(m["size"] * 1e9)), "--ctx", str(ctx),
            "--family", m["family"], "--params", m["params"], "--quant", m["quant"], "--rewrite-model",
            "--format", "gguf" if runtime == "llama" else "safetensors"] + (["--thinking"] if m.get("thinking") else [])
    spawn("proxy", args)
    for _ in range(50):
        if http_ok("http://127.0.0.1:%d/" % PROXY_PORT):
            break
        time.sleep(0.1)
    cur = dict(model=name, runtime=runtime, ctx=ctx, backend=backend, started=time.time())
    json.dump(cur, open(state_file("current.json"), "w"))
    if not a.quiet:
        print_endpoints(name, m, cur)
    return cur


def print_endpoints(name, m, cur):
    print("""
  serving  %s  (%s, %s, context %d) on the GTX 1080 Ti
  Ollama   http://127.0.0.1:%d          e.g. OLLAMA_HOST=127.0.0.1:%d <ollama client>
  OpenAI   http://127.0.0.1:%d/v1       model "%s", any API key
  chat     local-gpu chat               stop: local-gpu stop [--all releases the card]
""" % (name, m["desc"], cur["runtime"], cur["ctx"], PROXY_PORT, PROXY_PORT, PROXY_PORT, name), flush=True)


# ------------------------------------------------------------------ commands
def cmd_status(a):
    info = shimd_info()
    t = shimd_temp() if info else None
    print("card     " + ("up: %.1f of %.1f GiB free, SM %d MHz, memory %d MHz%s" % (
        info["free"] / 2**30, info["total"] / 2**30, info["sm_mhz"], info["mem_mhz"],
        ", %d C" % t if t is not None and t >= 0 else "") if info else "driver server not running"))
    cur = current()
    if cur:
        ok = http_ok(cur["backend"] + ("/health" if cur["runtime"] == "llama" else "/v1/models"))
        print("server   %s on %s (context %d) %s, up %.0f min" % (cur["model"], cur["runtime"], cur["ctx"],
              "healthy" if ok else "NOT RESPONDING", (time.time() - cur["started"]) / 60))
    else:
        print("server   none")
    print("endpoint http://127.0.0.1:%d %s" % (PROXY_PORT, "up" if http_ok("http://127.0.0.1:%d/" % PROXY_PORT) else "down"))


def cmd_models(a):
    have = set()
    if ct_state(LLAMA_CT):
        have = set(dexec(LLAMA_CT, "ls /models", check=False).stdout.split())
    cur = (current() or {}).get("model")
    for n, m in MODELS.items():
        mark = "*" if n == cur else " "
        got = "downloaded" if m.get("file") in have else ("" if m.get("runtime") != "max" else "MAX")
        print("%s %-15s %5.1f GB  %-10s %s%s" % (mark, n, m["size"], got, m["desc"], "  [default]" if n == DEFAULT else ""))


def cmd_pull(a):
    name, m = resolve(a.model)
    if m.get("runtime") == "max":
        die("MAX models download on first `serve`")
    ensure_llama_container(); ensure_model(name, m); say("%s ready" % name)


def cmd_stop(a):
    stop_servers(); say("server stopped")
    if a.all:
        kill("shimd")
        sh(["pkill", "-f", "pascal/shimd.py"], check=False)       # also one started by hand
        for _ in range(50):
            if not shimd_info():
                break
            time.sleep(0.1)
        say("driver server stopped; the card is released (still warm for next time)")


def cmd_logs(a):
    if a.which == "server":
        cur = current() or {}
        if cur.get("runtime") == "max":
            print(dexec(MAX_CT, "grep -v UserWarning /tmp/serve.log | tail -%d" % a.n, check=False).stdout)
        else:
            print(dexec(LLAMA_CT, "tail -%d /tmp/llama-server.log" % a.n, check=False).stdout)
    else:
        sh("tail -%d '%s'" % (a.n, state_file(a.which + ".log")), check=False, capture=False)


def cmd_bench(a):
    name, m = resolve(a.model)
    if m.get("runtime") == "max":
        die("bench runs llama.cpp models")
    ensure_llama_container(); ensure_model(name, m); ensure_card(a.vram_gib, getattr(a, "mclk", None))
    stop_servers()
    say("llama-bench %s (prompt 512, generate 128); the server is stopped while it runs" % name)
    dexec(LLAMA_CT, "%s /opt/llama-build/bin/llama-bench -m /models/%s -ngl 99 -fa 1 -p 512 -n 128 -r 2"
          % (SHIM_ENV, m["file"]), capture=False)


def cmd_chat(a, prompt=None):
    a.quiet = True
    cur = serve(a)
    sys.path.insert(0, os.path.join(REPO, "localgpu"))
    import chat
    chat.run("http://127.0.0.1:%d/v1" % PROXY_PORT, cur["model"], prompt=prompt, system=a.system)


def main():
    ap = argparse.ArgumentParser(prog="local-gpu", description="Serve LLMs on the GTX 1080 Ti eGPU.")
    sub = ap.add_subparsers(dest="cmd")

    def common(p, model=True):
        if model:
            p.add_argument("--model", "-m", default=os.environ.get("LOCAL_GPU_MODEL", DEFAULT))
        p.add_argument("--ctx", type=int, default=0, help="context length (default: per model)")
        p.add_argument("--vram-gib", type=float, default=10.0)
        p.add_argument("--llama-args", default=os.environ.get("LOCAL_GPU_LLAMA_ARGS", ""),
                       help="extra llama-server flags, appended to the model's own (env LOCAL_GPU_LLAMA_ARGS)")
        p.add_argument("--mclk", type=int, default=MCLK_DEFAULT,
                       help="memory clock in MHz to run at (default %(default)s; 405 = boot clock, no reclock). "
                            "1080 passes a 200-pass soak warm; 1134 passes cold but flips bits warm. "
                            "Env LOCAL_GPU_MCLK.")
        p.add_argument("--quiet", action="store_true")
        return p
    common(sub.add_parser("serve", help="start serving (default command)"))
    p = common(sub.add_parser("chat", help="interactive terminal chat")); p.add_argument("--system", default=None)
    p = common(sub.add_parser("run", help="one-shot: local-gpu run MODEL PROMPT"), model=False)
    p.add_argument("model"); p.add_argument("prompt", nargs="*"); p.add_argument("--system", default=None)
    sub.add_parser("status"); sub.add_parser("models")
    p = sub.add_parser("pull"); p.add_argument("model")
    p = sub.add_parser("stop"); p.add_argument("--all", action="store_true", help="also stop the driver server")
    p = sub.add_parser("logs"); p.add_argument("which", nargs="?", default="server", choices=["server", "shimd", "proxy"])
    p.add_argument("-n", type=int, default=60)
    p = common(sub.add_parser("bench", help="llama-bench a model"), model=False)
    p.add_argument("model", nargs="?", default=DEFAULT)
    a = ap.parse_args(sys.argv[1:] or ["serve"])
    if a.cmd in (None, "serve"):
        if a.cmd is None:
            a = ap.parse_args(["serve"] + sys.argv[1:])
        serve(a)
    elif a.cmd == "chat":
        cmd_chat(a)
    elif a.cmd == "run":
        cmd_chat(a, prompt=" ".join(a.prompt) or None)
    else:
        {"status": cmd_status, "models": cmd_models, "pull": cmd_pull, "stop": cmd_stop, "logs": cmd_logs,
         "bench": cmd_bench}[a.cmd](a)


if __name__ == "__main__":
    main()
