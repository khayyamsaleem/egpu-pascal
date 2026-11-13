"""Streaming terminal chat over an OpenAI-compatible endpoint (used by `local-gpu chat`).

Commands: /exit, /clear, /system <text>, /tools, /load, /unload, /auto, /yes, \"\"\" for multi-line.
Thinking (reasoning_content) is shown dimmed.  Standard library only.

Tools are a **registry, not a fixed list**: every tool belongs to a bundle (web, repo, compute, edit,
memory, time, demo) and only the active ones are sent in a request, so the prompt stays small.
Activation is conditional -- the bundle whose keywords match the user's message is loaded for that
turn -- and summonable -- the model can call `use_tools` to load more, and calling a tool that
isn't loaded yet loads it and retries.  `edit` (write_file, run_command, python) never auto-loads
and every call in it asks first unless `--yes`.  `--tools` turns the harness on (conditional
loading only), `--tools=web,repo` pins those bundles, `--tools=all` pins everything; `/load` and
`/unload` do the same inside the session.  File paths are confined to `--root`.
"""
import ast, datetime, html, json, os, re, subprocess, sys, time, urllib.parse, urllib.request

DIM, RESET = ("\033[2m", "\033[0m") if sys.stdout.isatty() else ("", "")
MAX_ITER = 8                                   # tool round trips per turn
MAX_OUT = 4000                                 # bytes of tool output kept (context is 16k)
NOTES = os.path.expanduser("~/.local/state/local-gpu/chat-notes.md")

SUMMONER = "use_tools"
BUNDLES = {
    "web": "search the web and read pages",
    "repo": "files, grep and read-only git in the working root",
    "compute": "arithmetic (no side effects)",
    "edit": "write files, run shell and Python -- each one asks first, and nothing here auto-loads",
    "memory": "notes that outlive /clear",
    "time": "the clock (models hallucinate dates)",
    "demo": "the offline weather tool",
}
# Which bundle a message pulls in without anyone asking.  Deliberately short and distinctive:
# substring matching makes short generic words worthless ("test" fires inside "latest", "log"
# inside "blog"), and a missed bundle costs only one use_tools call, while a false positive
# widens every request.  The model can always summon what is missing.
BUNDLE_KEYWORDS = {
    "web": ("search", "google", "look up", "on the web", "http://", "https://", "www.",
            "latest", "news", "url", "documentation", "release notes"),
    "repo": ("grep", "git ", "diff", "commit", "repo", "file", "directory", "branch",
             "source tree"),
    "compute": ("calculat", "compute", "how many", "average", "percent", "convert",
                "arithmetic", "sum of"),
    "memory": ("remember", "recall", "note that", "forget"),
    "time": ("what time", "the date", "today's date", "what day", "utc"),
}

S = {"type": "string"}
I = {"type": "integer"}
_registry = {}


def register(name, bundle, desc, props, required, gated=False):
    """One entry point for new tools: definition, bundle and (in call_tool) handler."""
    _registry[name] = dict(
        bundle=bundle, gated=gated,
        definition={"type": "function", "function": {"name": name, "description": desc,
                    "parameters": {"type": "object", "properties": props, "required": required}}})
    return name


def tool_defs(names):
    return [_registry[n]["definition"] for n in sorted(names) if n in _registry]


def bundle_of(name):
    return _registry.get(name, {}).get("bundle")


def names_in_bundles(bundles):
    return {n for n, t in _registry.items() if t["bundle"] in bundles}


def resolve(word):
    """A bundle name, a tool name or "all" -> the set of tools it stands for (empty if unknown)."""
    w = str(word or "").strip()
    if w == "all":
        return set(_registry)
    if w in BUNDLES:
        return names_in_bundles({w})
    return {w} if w in _registry else set()


def active_names(state, text=""):
    """The tool set for one request: pinned + the summoner, plus keyword-matched bundles."""
    names = {SUMMONER} | set(state.get("pinned", ()))
    if state.get("auto") and text:
        low = text.lower()
        hit = {b for b, kws in BUNDLE_KEYWORDS.items() if any(k in low for k in kws)}
        names |= names_in_bundles(hit)
    return names


SYSTEM_BASE = (
    "You are a careful engineering assistant running locally on a GTX 1080 Ti. Use a tool when it "
    "answers the question better than guessing, and use its result verbatim instead of inventing "
    "one. Keep answers short and state assumptions.")
SYSTEM_TOOLS = (
    " Tools are loaded on demand. Call use_tools to load more; bundles: %s. A tool you need but "
    "have not loaded can also just be called -- it will be loaded and you can call it again."
    % "; ".join("%s (%s)" % (b, d) for b, d in BUNDLES.items()))

WEATHER = {"tokyo": "14 C, light rain", "paris": "9 C, overcast", "san francisco": "17 C, fog",
           "new york": "11 C, clear"}

# ---------------------------------------------------------------- the registry
register(SUMMONER, None,
         "Load more tools (or whole bundles) for this session. Bundles: %s."
         % ", ".join(BUNDLES),
         {"tools": {"type": "array", "items": S}, "why": S}, ["tools"])
# web
register("web_search", "web", "Search the web (the self-hosted SearXNG, then Bing RSS, DuckDuckGo, Wikipedia)",
         {"query": S, "n": I}, ["query"])
register("fetch_url", "web", "Fetch a URL as text; a blocked/JS page falls back to the Wayback copy",
         {"url": S, "max_chars": I}, ["url"])
# repo
register("list_files", "repo", "List files in a directory under the working root", {"path": S}, ["path"])
register("read_file", "repo", "Read a text file under the working root",
         {"path": S, "max_bytes": I}, ["path"])
register("grep_repo", "repo", "Search file contents under the working root (ripgrep if present)",
         {"pattern": S, "path": S, "glob": S, "max_hits": I}, ["pattern"])
register("git_cmd", "repo", "Read-only git in the working root: status, diff, log, show, branch, rev-parse, ls-files, blame",
         {"subcommand": S}, ["subcommand"])
register("write_file", "edit", "Write a text file under the working root (asks first)",
         {"path": S, "content": S}, ["path", "content"], gated=True)
register("run_command", "edit", "Run a shell command in the working root (asks first)",
         {"command": S}, ["command"], gated=True)
# compute
register("calc", "compute", "Evaluate an arithmetic expression exactly (no side effects)",
         {"expression": S}, ["expression"])
register("python", "edit", "Run a short Python snippet and return its output (asks first)",
         {"code": S}, ["code"], gated=True)
# memory
register("remember", "memory", "Append a durable note (survives /clear and later sessions)",
         {"text": S}, ["text"])
register("recall", "memory", "Read back notes saved with remember (optionally filtered)", {"query": S}, [])
# time / demo
register("now", "time", "Current date and time", {}, [])
register("get_weather", "demo", "Current weather for a city (offline demo tool)", {"city": S}, ["city"])


def _under(root, path):
    """Absolute path inside `root`, or None if the request escapes it."""
    root, p = os.path.realpath(root), os.path.realpath(os.path.join(root, path or "."))
    return p if p == root or p.startswith(root + os.sep) else None


def _confirm(what, auto_yes):
    if auto_yes:
        return True
    try:
        return input("%s    run it? [y/N] " % what).strip().lower() in ("y", "yes")
    except (EOFError, KeyboardInterrupt):
        print()
        return False


def _refused(what, auto_yes):
    """Gated action -> None to go ahead, else the reason to hand back to the model.  In a pipe
    there is nobody to ask, and silently refusing just makes the model flail at the wall."""
    if auto_yes:
        return None
    if not sys.stdin.isatty():
        return "refused: %s needs a yes and there is no terminal to ask (pass --yes for a pipe)" % what
    return None if _confirm(what, auto_yes) else "refused by the user"


UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) "
      "Chrome/124.0 Safari/537.36")
# The self-hosted SearXNG behind `web_search`: either a directly reachable base URL, or
# "ssh-host|docker-ip:port" for the one-hop ssh route (cherryblossom's instance publishes no port).
SEARXNG_URL = os.environ.get("LOCAL_GPU_SEARXNG_URL", "")
SEARXNG_SSH = os.environ.get("LOCAL_GPU_SEARXNG_SSH", "root@cherryblossom|172.17.0.19:8080")


def _get(url, timeout=25):
    req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept-Language": "en-US,en"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read().decode("utf-8", "replace")


def _text(h):
    """HTML -> readable text (drop scripts/styles/head, tags, entities, runs of blank lines)."""
    h = re.sub(r"(?is)<(script|style|noscript|svg|head|template)[^>]*>.*?</\1>", " ", h)
    h = html.unescape(re.sub(r"(?s)<[^>]+>", " ", h))
    return re.sub(r"\n\s*\n\s*\n+", "\n\n", re.sub(r"[ \t\r\f\v]+", " ", h)).strip()


def _searxng(q, n=5):
    """The user's own SearXNG (cherryblossom-searxng-1), which is on a docker network with no
    published port, so the default route is a one-hop ssh + curl.  Point LOCAL_GPU_SEARXNG_URL at
    a directly reachable instance instead (published port, tailscale serve, ...) when there is one."""
    path = "/search?q=%s&format=json" % urllib.parse.quote(q, safe="")
    if SEARXNG_URL:
        j = json.loads(_get(SEARXNG_URL.rstrip("/") + path))
    else:
        host, _, addr = SEARXNG_SSH.partition("|")
        r = subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", host,
                            "curl -s -m 20 'http://%s%s'" % (addr, path)],
                           capture_output=True, text=True, timeout=60)
        if r.returncode or not (r.stdout or "").strip().startswith("{"):
            raise RuntimeError("searxng via %s: rc=%s %s" % (host, r.returncode, (r.stderr or "")[:120]))
        j = json.loads(r.stdout)
    return [(x.get("title", ""), x.get("url", ""), re.sub(r"\s+", " ", x.get("content") or "")[:300])
            for x in (j.get("results") or [])]


def web_search(query, n=5):
    """Search: the user's SearXNG first (meta-search, self-hosted), then keyless fallbacks --
    Bing's RSS endpoint, DuckDuckGo lite, Wikipedia's API.  Says which backend answered."""
    def bing(q):
        out = []
        for it in re.findall(r"<item>(.*?)</item>",
                             _get("https://www.bing.com/search?format=rss&q=" + urllib.parse.quote(q)), re.S):
            t = re.search(r"<title>(.*?)</title>", it, re.S)
            l = re.search(r"<link>(.*?)</link>", it, re.S)
            d = re.search(r"<description>(.*?)</description>", it, re.S)
            if t and l:
                out.append((_text(t.group(1)), html.unescape(l.group(1)).strip(),
                            _text(d.group(1)) if d else ""))
        return out

    def ddg(q):
        h = _get("https://lite.duckduckgo.com/lite/?q=" + urllib.parse.quote(q))
        snips = re.findall(r"class=['\"]result-snippet['\"]>(.*?)</td>", h, re.S)
        hits = re.findall(r'href="(//duckduckgo\.com/l/\?uddg=[^"]+)"', h)
        return [("", urllib.parse.unquote(x.split("uddg=")[1].split("&")[0]),
                 _text(snips[i]) if i < len(snips) else "") for i, x in enumerate(hits)]

    def wiki(q):
        j = json.loads(_get("https://en.wikipedia.org/w/api.php?action=query&list=search&srsearch=%s"
                            "&format=json" % urllib.parse.quote(q)))
        return [(s["title"], "https://en.wikipedia.org/wiki/" + urllib.parse.quote(s["title"].replace(" ", "_")),
                 _text(s.get("snippet", ""))) for s in j.get("query", {}).get("search", [])]

    n = max(1, min(int(n or 5), 10))
    tried = []
    for name, fn in (("searxng", _searxng), ("bing", bing), ("duckduckgo", ddg), ("wikipedia", wiki)):
        try:
            hits = fn(query)[:n]
            tried.append(name)
        except Exception as e:
            tried.append("%s(%s)" % (name, type(e).__name__))
            continue
        if hits:
            return "[%s] %d results for %r:\n%s" % (name, len(hits), query, "\n".join(
                "%d. %s\n   %s%s" % (i + 1, h[0] or h[1], h[1], ("\n   " + h[2][:220]) if h[2] else "")
                for i, h in enumerate(hits)))
    return "no results for %r (tried %s)" % (query, ", ".join(tried))


def fetch_url(url, max_chars=6000):
    """Fetch a URL as text.  A blocked or JS-only page falls back to the Wayback copy, and the
    result says which one it is (the blocked-page-recovery ladder: never pass a snapshot off as
    the live page)."""
    if not re.match(r"^https?://", url or ""):
        url = "https://" + str(url or "")
    try:
        live = _text(_get(url))
        if len(live) < 200:
            raise ValueError("only %d chars of text (JS-only page or a block interstitial)" % len(live))
        return "live fetch of %s (%d chars of text, %d shown)\n\n%s" % (
            url, len(live), min(len(live), max_chars), live[:max_chars])
    except Exception as e:
        try:
            j = json.loads(_get("https://archive.org/wayback/available?url=" + urllib.parse.quote(url, "")))
            c = (j.get("archived_snapshots") or {}).get("closest") or {}
        except Exception:
            c = {}
        if c.get("url"):
            snap = _text(_get(c["url"]))
            return ("live fetch failed (%s).  ARCHIVED COPY (as archived %s) -- a snapshot, not the "
                    "live page:\n%s\n\n%s" % (e, c.get("timestamp"), c["url"], snap[:max_chars]))
        return "could not fetch %s: %s (no Wayback copy either)" % (url, e)


def calc(expression):
    """Arithmetic only -- AST-checked, so no side effects and no imports."""
    import math
    env = {"sqrt": math.sqrt, "log": math.log, "log2": math.log2, "log10": math.log10,
           "exp": math.exp, "sin": math.sin, "cos": math.cos, "tan": math.tan, "floor": math.floor,
           "ceil": math.ceil, "fabs": math.fabs, "pi": math.pi, "e": math.e, "tau": math.tau,
           "abs": abs, "round": round, "min": min, "max": max, "sum": sum, "int": int, "float": float}
    tree = ast.parse(str(expression), mode="eval")
    ok = (ast.Expression, ast.BinOp, ast.UnaryOp, ast.Constant, ast.Call, ast.Name, ast.Attribute,
          ast.List, ast.Tuple, ast.Load, ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod,
          ast.Pow, ast.USub, ast.UAdd)
    for n in ast.walk(tree):
        if isinstance(n, ast.BinOp) and isinstance(n.op, (ast.BitXor, ast.LShift, ast.RShift)):
            raise ValueError("^ is XOR in Python, not a power -- write 2**32 instead")
        if not isinstance(n, ok):
            raise ValueError("only arithmetic is allowed here (%s)" % type(n).__name__)
        if isinstance(n, ast.Call) and not (isinstance(n.func, ast.Name) and n.func.id in env):
            raise ValueError("only these helpers: %s" % ", ".join(sorted(env)))
        if isinstance(n, ast.Name) and n.id not in env:
            raise ValueError("unknown name %r" % n.id)
    return repr(eval(compile(tree, "<calc>", "eval"), {"__builtins__": {}}, env))


def grep_repo(pattern, root, path=".", glob=None, max_hits=40):
    """ripgrep when available, else a stdlib walk (so the tool works on a bare machine)."""
    p = _under(root, path) or root
    hits = []
    try:
        args = ["rg", "--no-heading", "--line-number", "--color=never", "-m", "1", str(pattern)]
        if glob:
            args += ["-g", str(glob)]
        args += ["."]          # always name a path: with a piped stdin, rg would search stdin
        r = subprocess.run(args, cwd=p, capture_output=True, text=True, timeout=30)
        for line in (r.stdout or "").splitlines():
            if line.strip():
                hits.append(line[:300])
            if len(hits) >= max_hits:
                break
        if r.returncode not in (0, 1):
            raise RuntimeError((r.stderr or "").strip()[:200] or "rg failed")
    except (FileNotFoundError, RuntimeError):
        rx = re.compile(str(pattern))
        for dirpath, dirnames, filenames in os.walk(p):
            dirnames[:] = [d for d in dirnames if d not in (".git", "__pycache__", "node_modules")]
            for fn in filenames:
                if glob and not fn.endswith(glob.lstrip("*")):
                    continue
                f = os.path.join(dirpath, fn)
                if os.path.getsize(f) > 2_000_000:
                    continue
                try:
                    with open(f, errors="replace") as fh:
                        for i, line in enumerate(fh, 1):
                            if rx.search(line):
                                hits.append("%s:%d:%s" % (os.path.relpath(f, p), i, line.strip()[:240]))
                                break
                except OSError:
                    continue
                if len(hits) >= max_hits:
                    break
            if len(hits) >= max_hits:
                break
    return "%d hit(s) for %r under %s:\n%s" % (len(hits), pattern, path or ".",
                                              "\n".join(hits) if hits else "(none)")


GIT_READONLY = ("status", "diff", "log", "show", "rev-parse", "ls-files", "blame", "describe",
                "shortlog", "tag", "remote", "whatchanged", "cat-file")
GIT_BLOCKED = ("-d", "-D", "-m", "-M", "--delete", "--move", "--force", "-f", "--amend", "add",
               "commit", "push", "pull", "checkout", "reset", "clean", "rm", "stash")


def git_cmd(subcommand, root):
    words = str(subcommand).split()
    if not words or words[0] not in GIT_READONLY or any(w in GIT_BLOCKED for w in words):
        return "refused: read-only subcommands only (%s)" % ", ".join(GIT_READONLY)
    r = subprocess.run(["git"] + words, cwd=root, capture_output=True, text=True, timeout=60)
    out = ((r.stdout or "") + (r.stderr or ""))[:MAX_OUT]
    return "git %s -> exit %d\n%s" % (" ".join(words), r.returncode, out or "(no output)")


def now():
    d = datetime.datetime.now().astimezone()
    return "%s (%s, UTC%s)" % (d.strftime("%Y-%m-%d %H:%M:%S"), d.tzname(),
                               d.strftime("%z")[:3] + ":" + d.strftime("%z")[3:])


def remember(text):
    os.makedirs(os.path.dirname(NOTES), exist_ok=True)
    with open(NOTES, "a") as f:
        f.write("- %s  %s\n" % (now(), str(text).replace("\n", " ")))
    return "noted in %s" % NOTES


def recall(query="", n=30):
    try:
        lines = [l for l in open(NOTES, errors="replace").read().splitlines() if l.startswith("- ")]
    except OSError:
        return "no notes yet (%s)" % NOTES
    if query:
        lines = [l for l in lines if str(query).lower() in l.lower()]
    return "%d note(s)%s:\n%s" % (len(lines), (" matching %r" % query) if query else "",
                                  "\n".join(lines[-n:]) or "(none)")


def call_tool(name, args, ctx):
    """Execute one tool call and return its result as a string (never raises)."""
    root, auto_yes, state = ctx["root"], ctx["auto_yes"], ctx["state"]
    try:
        if name == SUMMONER:                       # self-summoning: widen the session's tool set
            want = [str(x) for x in (args.get("tools") or [])]
            add = set()
            for w in want:
                add |= resolve(w)
            state["pinned"] = sorted(set(state["pinned"]) | add)
            return "loaded: %s" % (", ".join(sorted(add)) or "nothing (unknown name?)")
        if name == "web_search":
            return web_search(str(args.get("query", "")), args.get("n") or 5)
        if name == "fetch_url":
            return fetch_url(str(args.get("url", "")), args.get("max_chars") or 6000)
        if name == "get_weather":
            return WEATHER.get(str(args.get("city", "")).strip().lower(), "no data for that city")
        if name == "now":
            return now()
        if name == "calc":
            return calc(args.get("expression", ""))
        if name == "python":
            code = str(args.get("code", ""))
            why = _refused("python -c %r" % (code[:120],), auto_yes)
            if why:
                return why
            r = subprocess.run([sys.executable, "-c", code], cwd=root, capture_output=True,
                               text=True, timeout=120)
            out = ((r.stdout or "") + (r.stderr or ""))[:MAX_OUT]
            return "exit %d\n%s" % (r.returncode, out or "(no output)")
        if name == "remember":
            return remember(args.get("text", ""))
        if name == "recall":
            return recall(args.get("query", ""))
        if name == "git_cmd":
            return git_cmd(args.get("subcommand", ""), root)
        if name == "grep_repo":
            return grep_repo(args.get("pattern", ""), root, args.get("path") or ".",
                             args.get("glob"), int(args.get("max_hits") or 40))
        p = _under(root, args.get("path"))
        if p is None:
            return "refused: path is outside the working root (%s)" % root
        if name == "list_files":
            names = sorted(os.listdir(p))
            return "%d entries in %s:\n%s" % (len(names), args.get("path") or ".",
                                              "\n".join(names[:200]))
        if name == "read_file":
            n = int(args.get("max_bytes") or 8000)
            with open(p, "rb") as f:
                data = f.read(min(n, MAX_OUT * 4))
            return data.decode("utf-8", "replace")[:n]
        if name == "write_file":
            body = str(args.get("content", ""))
            why = _refused("write %s (%d bytes)" % (p, len(body)), auto_yes)
            if why:
                return why
            os.makedirs(os.path.dirname(p) or ".", exist_ok=True)
            with open(p, "w") as f:
                f.write(body)
            return "wrote %s (%d bytes)" % (p, len(body))
        if name == "run_command":
            cmd = str(args.get("command", ""))
            why = _refused("$ %s" % cmd, auto_yes)
            if why:
                return why
            r = subprocess.run(cmd, shell=True, cwd=root, capture_output=True, text=True, timeout=120)
            out = ((r.stdout or "") + (r.stderr or ""))[:MAX_OUT]
            return "exit %d\n%s" % (r.returncode, out or "(no output)")
        return "unknown tool %r" % name
    except Exception as e:                          # a tool must never kill the session
        return "error running %s: %s: %s" % (name, type(e).__name__, e)


def stream(base, model, messages, tools=None):
    req_body = {"model": model, "messages": messages, "stream": True,
                "stream_options": {"include_usage": True}}
    if tools:
        req_body["tools"] = tools
        req_body["tool_choice"] = "auto"
    body = json.dumps(req_body).encode()
    req = urllib.request.Request(base + "/chat/completions", data=body, headers={"Content-Type": "application/json"})
    text, think, timings, t0, first, calls = "", False, None, time.time(), None, {}
    with urllib.request.urlopen(req, timeout=3600) as r:
        for line in r:
            line = line.strip()
            if not line.startswith(b"data:") or line[5:].strip() == b"[DONE]":
                continue
            j = json.loads(line[5:])
            timings = j.get("timings") or timings
            for ch in j.get("choices") or []:
                d = ch.get("delta") or {}
                for tc in d.get("tool_calls") or []:       # streamed in fragments (OpenAI format)
                    i = tc.get("index", 0)
                    slot = calls.setdefault(i, {"id": None, "name": None, "args": ""})
                    if tc.get("id"):
                        slot["id"] = tc["id"]
                    fn = tc.get("function") or {}
                    if fn.get("name"):
                        slot["name"] = fn["name"]
                    if fn.get("arguments"):
                        slot["args"] += fn["arguments"]
                if d.get("reasoning_content"):
                    if not think:
                        sys.stdout.write(DIM); think = True
                    sys.stdout.write(d["reasoning_content"])
                if d.get("content"):
                    if think:
                        sys.stdout.write(RESET + "\n"); think = False
                    first = first or time.time()
                    sys.stdout.write(d["content"]); text += d["content"]
                sys.stdout.flush()
    if think:
        sys.stdout.write(RESET)
    if timings:
        sys.stdout.write("%s\n[%d prompt tokens at %.0f t/s, %d generated at %.1f t/s]%s\n" % (
            DIM, timings.get("prompt_n", 0), timings.get("prompt_per_second", 0),
            timings.get("predicted_n", 0), timings.get("predicted_per_second", 0), RESET))
    else:
        sys.stdout.write("%s\n[%.1f s]%s\n" % (DIM, time.time() - t0, RESET))
    return text, [calls[i] for i in sorted(calls)]


def run(base, model, prompt=None, system=None, tools=False, yes=False, root=None,
        pin="", auto=True):
    """Interactive chat.

    tools: turn the tool harness on.  pin: "all" or a comma list of bundles/tool names to load up
    front ("" = nothing, conditional loading only).  auto: let the user's wording load bundles.
    """
    root = root or os.getcwd()
    harness = bool(tools)
    auto_yes = bool(yes)
    pins = set()
    for w in str(pin or "").split(","):
        pins |= resolve(w)
    state = {"pinned": sorted(pins), "auto": bool(auto)}
    sys_prompt = system or (SYSTEM_BASE + (SYSTEM_TOOLS if harness else ""))
    messages = [{"role": "system", "content": sys_prompt}] if sys_prompt else []

    def turn(text=""):
        """One user turn: stream, run any tool calls, repeat until the model answers."""
        ctx = {"root": root, "auto_yes": auto_yes, "state": state}
        for _ in range(MAX_ITER):
            names = active_names(state, text) if harness else set()
            reply, calls = stream(base, model, messages, tools=tool_defs(names) if harness else None)
            if not calls:
                messages.append({"role": "assistant", "content": reply})
                return
            messages.append({"role": "assistant", "content": reply,
                             "tool_calls": [{"id": c["id"] or ("call_%d" % i), "type": "function",
                                             "function": {"name": c["name"], "arguments": c["args"]}}
                                            for i, c in enumerate(calls)]})
            for i, c in enumerate(calls):
                try:
                    args = json.loads(c["args"] or "{}")
                except ValueError:
                    args = {}
                name = c["name"]
                if harness and name in _registry and name not in names:
                    # called something it had not loaded: load it (and its bundle) for the retry
                    state["pinned"] = sorted(set(state["pinned"]) | {name})
                    print("%s    (loaded %s on demand)%s" % (DIM, name, RESET), flush=True)
                print("%s    -> %s(%s)%s" % (DIM, name, json.dumps(args)[:300], RESET), flush=True)
                out = call_tool(name, args, ctx)
                shown = out if len(out) <= 600 else out[:600] + " ..."
                print("%s    <- %s%s" % (DIM, shown.replace("\n", "\n       "), RESET), flush=True)
                messages.append({"role": "tool", "tool_call_id": c["id"] or ("call_%d" % i),
                                 "content": out})
        print("%s(tool loop hit its %d-step limit)%s" % (DIM, MAX_ITER, RESET))
        messages.append({"role": "assistant", "content": "(stopped: too many tool calls)"})

    def tool_report():
        marks = "pinned: %s | auto-load: %s" % (", ".join(state["pinned"]) or "none",
                                                "on" if state["auto"] else "off")
        print("%s%s%s" % (DIM, marks, RESET))
        for b in list(BUNDLES) + [None]:
            ts = [n for n, t in _registry.items() if t["bundle"] == b]
            what = BUNDLES[b] if b else "always available"
            print("  %-8s %-32s %s" % (b or "(core)", ",".join(sorted(ts)), what))

    if prompt:
        messages.append({"role": "user", "content": prompt})
        turn(prompt)
        return
    print("%schatting with %s on the GTX 1080 Ti -- /exit, /clear, /system <text>, /tools, /load, "
          "/unload, /auto, /yes, \"\"\" for multi-line%s" % (DIM, model, RESET))
    print("%stool harness %s -- %s; root %s%s"
          % (DIM, ("on, %s" % ("pinned: " + (", ".join(state["pinned"]) or "none")))
             if harness else "off (/tools on)",
             "no confirmations" if auto_yes else "write/run ask first (/yes)", root, RESET))
    while True:
        try:
            line = input("\n> ")
        except (EOFError, KeyboardInterrupt):
            print(); return
        if line.strip() == '"""':
            lines = []
            while True:
                l = input()
                if l.strip() == '"""':
                    break
                lines.append(l)
            line = "\n".join(lines)
        cmd = line.strip()
        if cmd in ("/exit", "/quit", "/bye"):
            return
        if cmd == "/clear":
            messages = [m for m in messages if m["role"] == "system"]; print("(cleared)"); continue
        if cmd.startswith("/system "):
            messages = [{"role": "system", "content": cmd[8:]}] + [m for m in messages if m["role"] != "system"]
            print("(system prompt set)"); continue
        if cmd == "/tools" or cmd.startswith("/tools "):
            arg = cmd[6:].strip().lower()
            if arg in ("on", "off"):
                harness = arg == "on"
            elif arg:
                tool_report()
                continue
            else:
                harness = not harness
            print("(tool harness %s)" % ("ON" if harness else "off")); continue
        if cmd.startswith("/load ") or cmd.startswith("/unload "):
            load = cmd.startswith("/load ")
            words = [w for w in cmd.split()[1].split(",") if w]
            for w in words:
                got = resolve(w)
                if not got:
                    print("(unknown bundle or tool %r)" % w); continue
                state["pinned"] = sorted((set(state["pinned"]) | got) if load
                                         else (set(state["pinned"]) - got))
                print("(%s %s: %s)" % ("loaded" if load else "unloaded", w, ", ".join(sorted(got))))
            continue
        if cmd == "/auto":
            state["auto"] = not state["auto"]
            print("(keyword auto-load %s)" % ("on" if state["auto"] else "off")); continue
        if cmd == "/yes":
            auto_yes = not auto_yes
            print("(confirmations %s)" % ("off: write/run execute immediately" if auto_yes else "on")); continue
        if not cmd:
            continue
        messages.append({"role": "user", "content": line})
        try:
            turn(line)
        except KeyboardInterrupt:
            print("\n(interrupted)"); messages.pop(); continue
