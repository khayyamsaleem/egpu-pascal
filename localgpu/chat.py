"""Streaming terminal chat over an OpenAI-compatible endpoint (used by `local-gpu chat`).

Commands: /exit, /clear (forget the conversation), /system <text>, /tools [on|off|list], /yes,
\"\"\" starts and ends a multi-line message.  Thinking (reasoning_content) is shown dimmed.
Standard library only.

With tools on (`local-gpu chat --tools`) the model gets a small, explicit tool set it can call in
a loop: read-only tools run immediately, anything that writes or runs a command asks first unless
`--yes` was given.  File paths are confined to `--root` (default: the working directory).

Tools: web_search (the self-hosted SearXNG first -- LOCAL_GPU_SEARXNG_URL, or
LOCAL_GPU_SEARXNG_SSH for cherryblossom's unpublished instance -- then Bing RSS, DuckDuckGo lite,
Wikipedia), fetch_url (live, with a Wayback fallback that says it is a snapshot), get_weather,
list_files, read_file, write_file, run_command.
"""
import html, json, os, re, subprocess, sys, time, urllib.parse, urllib.request

DIM, RESET = ("\033[2m", "\033[0m") if sys.stdout.isatty() else ("", "")
MAX_ITER = 8                                   # tool round trips per turn
MAX_OUT = 4000                                 # bytes of tool output kept (context is 16k)

SYSTEM_WITH_TOOLS = (
    "You are a careful engineering assistant running locally on a GTX 1080 Ti. You have tools: "
    "call one when it answers the question better than guessing, then use its result verbatim "
    "instead of inventing one. Keep answers short and state assumptions.")

TOOLS = [
    {"type": "function", "function": {
        "name": "web_search", "description": "Search the web; returns titles, URLs and snippets",
        "parameters": {"type": "object", "properties": {
            "query": {"type": "string"}, "n": {"type": "integer"}}, "required": ["query"]}}},
    {"type": "function", "function": {
        "name": "fetch_url", "description": "Fetch a URL and return its text (falls back to the Wayback copy)",
        "parameters": {"type": "object", "properties": {
            "url": {"type": "string"}, "max_chars": {"type": "integer"}}, "required": ["url"]}}},
    {"type": "function", "function": {
        "name": "get_weather", "description": "Current weather for a city (offline demo tool)",
        "parameters": {"type": "object", "properties": {"city": {"type": "string"}},
                       "required": ["city"]}}},
    {"type": "function", "function": {
        "name": "list_files", "description": "List files in a directory under the working root",
        "parameters": {"type": "object", "properties": {"path": {"type": "string"}},
                       "required": ["path"]}}},
    {"type": "function", "function": {
        "name": "read_file", "description": "Read a text file under the working root",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string"}, "max_bytes": {"type": "integer"}}, "required": ["path"]}}},
    {"type": "function", "function": {
        "name": "write_file", "description": "Write a text file under the working root (asks first)",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string"}, "content": {"type": "string"}},
            "required": ["path", "content"]}}},
    {"type": "function", "function": {
        "name": "run_command", "description": "Run a shell command in the working root (asks first)",
        "parameters": {"type": "object", "properties": {"command": {"type": "string"}},
                       "required": ["command"]}}},
]
WEATHER = {"tokyo": "14 C, light rain", "paris": "9 C, overcast", "san francisco": "17 C, fog",
           "new york": "11 C, clear"}


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


def call_tool(name, args, root, auto_yes):
    """Execute one tool call and return its result as a string (never raises)."""
    try:
        if name == "web_search":
            return web_search(str(args.get("query", "")), args.get("n") or 5)
        if name == "fetch_url":
            return fetch_url(str(args.get("url", "")), args.get("max_chars") or 6000)
        if name == "get_weather":
            return WEATHER.get(str(args.get("city", "")).strip().lower(), "no data for that city")
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
            if not _confirm("write %s (%d bytes)" % (p, len(body)), auto_yes):
                return "refused by the user"
            os.makedirs(os.path.dirname(p) or ".", exist_ok=True)
            with open(p, "w") as f:
                f.write(body)
            return "wrote %s (%d bytes)" % (p, len(body))
        if name == "run_command":
            cmd = str(args.get("command", ""))
            if not _confirm("$ %s" % cmd, auto_yes):
                return "refused by the user"
            r = subprocess.run(cmd, shell=True, cwd=root, capture_output=True, text=True, timeout=120)
            out = (r.stdout or "") + (r.stderr or "")
            if len(out) > MAX_OUT:
                out = out[:MAX_OUT] + "\n[truncated]"
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


def run(base, model, prompt=None, system=None, tools=False, yes=False, root=None):
    """Interactive chat.  tools=True enables the tool loop; yes=True skips confirmations."""
    root = root or os.getcwd()
    tools_on = bool(tools)
    auto_yes = bool(yes)
    messages = [{"role": "system", "content": system or (SYSTEM_WITH_TOOLS if tools_on else None)}] \
        if (system or tools_on) else []
    messages = [m for m in messages if m["content"]]

    def turn():
        """One user turn: stream, run any tool calls, repeat until the model answers."""
        for _ in range(MAX_ITER):
            reply, calls = stream(base, model, messages, tools=TOOLS if tools_on else None)
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
                print("%s    -> %s(%s)%s" % (DIM, c["name"], json.dumps(args)[:300], RESET), flush=True)
                out = call_tool(c["name"], args, root, auto_yes)
                shown = out if len(out) <= 600 else out[:600] + " ..."
                print("%s    <- %s%s" % (DIM, shown.replace("\n", "\n       "), RESET), flush=True)
                messages.append({"role": "tool", "tool_call_id": c["id"] or ("call_%d" % i),
                                 "content": out})
        print("%s(tool loop hit its %d-step limit)%s" % (DIM, MAX_ITER, RESET))
        messages.append({"role": "assistant", "content": "(stopped: too many tool calls)"})

    if prompt:
        messages.append({"role": "user", "content": prompt})
        turn()
        return
    print("%schatting with %s on the GTX 1080 Ti -- /exit, /clear, /system <text>, /tools, /yes, "
          "\"\"\" for multi-line%s" % (DIM, model, RESET))
    print("%stools %s -- %s; working root %s%s"
          % (DIM, "ON" if tools_on else "off (/tools on)", "no confirmations" if auto_yes
             else "write/run ask first (/yes)", root, RESET))
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
        if cmd.startswith("/tools"):
            arg = cmd[6:].strip().lower()
            if arg == "list":
                print("\n".join("  %-12s %s" % (t["function"]["name"], t["function"]["description"])
                                for t in TOOLS))
                continue
            tools_on = arg in ("on", "1", "yes") if arg else not tools_on
            print("(tools %s)" % ("ON" if tools_on else "off")); continue
        if cmd == "/yes":
            auto_yes = not auto_yes
            print("(confirmations %s)" % ("off: write/run execute immediately" if auto_yes else "on")); continue
        if not cmd:
            continue
        messages.append({"role": "user", "content": line})
        try:
            turn()
        except KeyboardInterrupt:
            print("\n(interrupted)"); messages.pop(); continue
