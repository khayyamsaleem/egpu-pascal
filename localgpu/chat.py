"""Streaming terminal chat over an OpenAI-compatible endpoint (used by `local-gpu chat`).

Commands: /exit, /clear, /system <text>, /tools, /load, /unload, /auto, /yes, /help.  Type `/` and
press TAB to list them all (TAB completes commands, and bundle/tool names after /load and /unload;
an ambiguous TAB prints the candidates), `/help <fuzzy>` ranks them by a query, and an unknown
command gets a did-you-mean.  Thinking (reasoning_content) is shown dimmed.  Standard library only.

Tools are a **registry, not a fixed list**: every tool belongs to a bundle (web, repo, compute, market,
edit, memory, time, demo) and only the active ones are sent in a request, so the prompt stays small.
Activation is conditional -- the bundle whose keywords match the user's message is loaded for that
turn -- and summonable -- the model can call `use_tools` to load more, and calling a tool that
isn't loaded yet loads it and retries.  `edit` (write_file, run_command, python) never auto-loads
and every call in it asks first unless `--yes`.  `--tools` turns the harness on (conditional
loading only), `--tools=web,repo` pins those bundles, `--tools=all` pins everything; `/load` and
`/unload` do the same inside the session.  File paths are confined to `--root`.
"""
import ast, calendar, datetime, html, json, os, re, subprocess, sys, time, urllib.error, urllib.parse, urllib.request
from typing import Any

DIM, RESET = ("\033[2m", "\033[0m") if sys.stdout.isatty() else ("", "")
MAX_ITER = 8                                   # tool round trips per turn
MAX_OUT = 4000                                 # bytes of tool output kept (context is 16k)
NOTES = os.path.expanduser("~/.local/state/local-gpu/chat-notes.md")

SUMMONER = "use_tools"
BUNDLES = {
    "web": "search the web and read pages",
    "repo": "files, grep and read-only git in the working root",
    "compute": "arithmetic (no side effects)",
    "sandbox": "run code in a locked-down container: no network, no host filesystem, capped",
    "market": "stock quotes and exchange rates",
    "edit": "write files and run commands ON THE HOST -- each asks first, nothing here auto-loads",
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
    "sandbox": ("python", "script", "pandas", "numpy", "matplotlib", "plot", "parse", "regex",
                "sandbox", "unit test", "jupyter", "csv file", "dataframe", "parse this"),
    "market": ("stock", "ticker", "share price", "shares of", "market cap", "trading at",
               "nasdaq", "nyse", "s&p", "exchange rate", "fx rate", "stock price", "quote for",
               "candlestick", "last 7 days", "historical price", "price history"),
    "time": ("what time", "the date", "today's date", "what day", "utc"),
}

# Short usage notes ("skills") for tools whose *interface* is only half the story.  They are not in
# the prompt by default: a note is attached to the system message only while a bundle that needs it
# is active (BUNDLE_SKILLS), so a session that never runs code pays nothing for them.  Each note is
# deliberately a handful of lines -- the tool schema says what the arguments are, these say how to
# not waste a round trip with them.
SKILLS = {
    "python": (
        "One self-contained script per call, no state between calls: print what you need back. "
        "Helper modules go in files={'x.py': ...}. Batch deps=[...] -- each install asks the user. "
        "Charts: matplotlib is installed, and a ready candlestick helper is in the repo: "
        "python3 /repo/sandbox/tools/candles.py --out /work/chart.png < data.csv (takes the "
        "stock_history CSV as-is). Use it rather than hand-rolling a candlestick."),
    "sandbox": (
        "Throwaway VM: no network, 512 MB, 30 s default timeout. Only /work persists; /repo is "
        "read-only; /mnt/<tag> are folders the user shared."),
}
BUNDLE_SKILLS = {"sandbox": ("python", "sandbox")}
SKILL_HEADER = "\n\nNotes for the tools you have loaded:\n"

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


def _skill_text(names):
    return "".join("- %s\n" % SKILLS[n] for n in sorted(names) if n in SKILLS)


def _apply_skills(messages, state):
    """Show exactly the attached notes at the end of the system message (idempotent: the base is
    whatever precedes the header, so re-applying after a note changes replaces them cleanly)."""
    if not messages or messages[0].get("role") != "system":
        return
    base = messages[0]["content"].split(SKILL_HEADER)[0]
    notes = _skill_text(state.get("skills", ()))
    messages[0]["content"] = base + ((SKILL_HEADER + notes) if notes else "")


def active_names(state, text=""):
    """The tool set for one request: pinned + the summoner, plus keyword-matched bundles."""
    names = {SUMMONER} | set(state.get("pinned", ()))
    if state.get("auto") and text:
        low = text.lower()
        hit = {b for b, kws in BUNDLE_KEYWORDS.items() if any(k in low for k in kws)}
        names |= names_in_bundles(hit)
    return names


# ---------------------------------------------------------------- slash commands
COMMANDS = {
    "/exit": "leave the session (also /quit, /bye)",
    "/clear": "forget the conversation, keep the system prompt",
    "/system": "/system <text> -- replace the system prompt",
    "/tools": "/tools [on|off] -- list the tool bundles, or switch the harness",
    "/load": "/load web,repo -- pin bundles or single tools (also: all)",
    "/unload": "/unload web -- drop them again",
    "/auto": "toggle loading a bundle just because the wording matches it",
    "/yes": "toggle the write_file/run_command/python confirmation",
    "/temp": "/temp [0..2] -- sampling temperature; lower is shorter thinking (default 0)",
    "/mount": "/mount <host>[:<tag>][:ro|rw] -- share a host folder into the sandbox (ro default)",
    "/skill": "/skill [name|none] -- attach/detach a tool's usage note (they auto-attach with a bundle)",
    "/help": "/help [fuzzy] -- this list, ranked by your query",
}
ALIASES = {"/quit": "/exit", "/bye": "/exit", "/?": "/help", "/h": "/help", "/cmds": "/help"}


def fuzzy_score(needle, haystack):
    """Subsequence score, fzf-ish: runs of consecutive characters and segment starts score higher.
    None when the needle is not a subsequence of the haystack."""
    n, h = str(needle).lower(), str(haystack).lower()
    if not n:
        return 0
    score, i, run = 0, 0, 0
    for ch in n:
        j = h.find(ch, i)
        if j < 0:
            return None
        if j == i:
            run += 1
            score += 2 * run                      # keeping the letters together
        else:
            run, score = 0, score + 1
        if j == 0 or h[j - 1] in "/-_ ":
            score += 3                            # a segment start
        i = j + 1
    return score - (len(h) - len(n)) // 4          # prefer the shorter candidate, mildly


def fuzzy_rank(query, candidates):
    """Candidates that match `query` as a subsequence, best first."""
    hits = [(-s, c) for c in candidates if (s := fuzzy_score(query, c)) is not None]
    return [c for _, c in sorted(hits)]


def help_text(query=""):
    names = list(COMMANDS)
    if query:
        names = fuzzy_rank(query, names)
        if not names:
            return "no command matches %r -- /help lists them all" % query
    return "\n".join("  %-9s %s" % (c, COMMANDS[c]) for c in names)


def completions(line):
    """TAB candidates for the current line: command names, or bundles/tools for /load and /unload."""
    if not line.startswith("/"):
        return []
    head, sep, tail = line.partition(" ")
    if not sep:
        names = sorted(set(COMMANDS) | set(ALIASES))
        return [c for c in names if c.startswith(head)] or fuzzy_rank(head[1:], names)
    if head in ("/load", "/unload"):
        pre, comma, part = tail.rpartition(",")
        opts = sorted(set(BUNDLES) | set(_registry) | {"all"})
        return [pre + comma + o for o in opts if o.startswith(part)]
    return []


def _columns(items, width=78):
    """Pack candidates into columns the way `ls` would."""
    pad = max(len(i) for i in items) + 2
    per = max(1, width // pad)
    return "\n".join("".join(i.ljust(pad) for i in items[i:i + per]).rstrip()
                     for i in range(0, len(items), per))


def setup_readline():
    """TAB completion (and up-arrow history) for the slash commands.  macOS Python links libedit,
    which needs the `bind ^I rl_complete` form as well as the GNU one.  libedit itself only beeps
    when a TAB is ambiguous, so an ambiguous TAB prints the candidates here instead."""
    try:
        import readline
    except ImportError:
        return False

    def completer(text, state):
        hits = completions(readline.get_line_buffer())
        if state == 0 and len(hits) > 1:
            print("\n%s%s%s" % (DIM, _columns(hits), RESET))
        return hits[state] if state < len(hits) else None

    readline.set_completer(completer)
    readline.set_completer_delims(" \t\n")
    for bind in ("tab: complete", "bind ^I rl_complete"):
        try:
            readline.parse_and_bind(bind)
        except Exception:
            pass
    try:
        readline.set_history_length(500)
    except Exception:
        pass
    return True


# The server's own default (1.0) makes the model ramble in its thinking channel: measured on the
# "what is the stock price for docn?" turn, temperature 1.0 burned ~270 generated tokens where 0.0
# burned ~139 for the same answer.  The harness used to send no temperature at all.
TEMP = float(os.environ.get("LOCAL_GPU_TEMP", "0.0"))
# Short, and deliberately not a numbered rule list -- an earlier prompt enumerated its rules and the
# model mirrored them back as a "constraint checklist" inside its thinking, at ~80 tokens a go.
SYSTEM_BASE = (
    "You are a careful engineering assistant running locally on a GTX 1080 Ti. Answer in one short "
    "line: no preamble, no plan, no checklist, no confidence score. Never restate a tool result -- "
    "the user can already see it.")
SYSTEM_TOOLS = (
    " Tools are loaded on demand: use_tools loads the rest (bundles: %s). Calling a tool that is "
    "not loaded loads it, and you can call it again." % "; ".join(BUNDLES))
# Shows the *answer format* only.  A few-shot that demonstrated a tool call worked slightly better on
# the tool question but pushed the model into calling a tool for arithmetic it could do itself, which
# costs a whole extra round trip -- and every round trip pays for another thinking block.
FORMAT_SHOT = [
    {"role": "user", "content": "how many seconds are in an hour?"},
    {"role": "assistant", "content": "3,600."},
    {"role": "user", "content": "and how big is 5 GiB in bytes?"},
    {"role": "assistant", "content": "5,368,709,120 bytes."},
]

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
# market
register("stock_quote", "market",
         "Quote(s) for stocks, ETFs, indices and crypto: price, change, day range, volume. "
         "Symbols like AAPL, MSFT, NVDA, 7203.T, ^GSPC, BTC-USD. Give `when` for a historical "
         "price instead of the live one: \"2026-11-10\" for that day's close, or \"2026-11-10 "
         "14:30\" for the price at that minute (exchange time).",
         {"symbols": S, "detail": {"type": "boolean"}, "when": S}, ["symbols"])
register("fx_rate", "market", "Exchange rates (ECB, via frankfurter.dev)",
         {"base": S, "to": S}, [])
register("stock_history", "market",
         "Daily open/high/low/close/volume for the last N trading days (default 7), returned as "
         "CSV rows (date,open,high,low,close,volume) -- use this rather than one stock_quote per "
         "day when you need a series, a trend or a chart. The sandbox has matplotlib.",
         {"symbols": S, "days": I}, ["symbols"])
# sandbox: code runs in a VM of its own (hvi microVM) or a locked-down container (localgpu/sandbox_exec.py)
register("python", "sandbox",
         "Run Python 3.12 in a sandboxed machine of its own (no network, no host filesystem except "
         "the mounts the user declared, capped memory/CPU) and return its output. /work is yours and "
         "persists between calls; the repo is read-only at /repo; user-declared host folders appear "
         "under /mnt/. Preinstalled: numpy pandas matplotlib scipy sympy tabulate requests bs4 lxml "
         "pyyaml dateutil. For anything else pass deps=[\"pkg\"]: pip installs it into /deps, which "
         "is on PYTHONPATH and shared by every later call (that step asks first and needs network).",
         {"code": S, "deps": {"type": "array", "items": S}, "files": {"type": "object"},
          "timeout": I, "net": {"type": "boolean"}}, ["code"])
register("shell", "sandbox", "Run a shell command in the same sandboxed machine (sh -c)",
         {"command": S, "timeout": I, "net": {"type": "boolean"}}, ["command"])
register("show", "sandbox",
         "Open a file the sandbox wrote (e.g. /work/chart.png) on the Mac in its default app, so the "
         "user can see it. Asks first; only files inside the sandbox work dir are allowed.",
         {"path": S}, ["path"], gated=True)
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
# Yahoo's edge 429s the full browser UA above on sight, but answers a plain one -- measured, not
# guessed: "Mozilla/5.0" -> 200 with a real quote, the long Chrome string -> 429, same request.
UA_MIN = "Mozilla/5.0"
# The self-hosted SearXNG behind `web_search`: either a directly reachable base URL, or
# "ssh-host|docker-ip:port" for the one-hop ssh route (cherryblossom's instance publishes no port).
SEARXNG_URL = os.environ.get("LOCAL_GPU_SEARXNG_URL", "")
SEARXNG_SSH = os.environ.get("LOCAL_GPU_SEARXNG_SSH", "root@cherryblossom|172.17.0.19:8080")


def _get(url, timeout=25, headers=None):
    h = {"User-Agent": UA, "Accept-Language": "en-US,en"}
    h.update(headers or {})
    req = urllib.request.Request(url, headers=h)
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


def _num(x, dp=2):
    """1234567.891 -> '1,234,567.89' (a big number with 0 dp for volumes)."""
    if x is None:
        return "?"
    try:
        if dp == 0 or abs(float(x)) >= 100000:
            return "{:,.0f}".format(float(x))
        return "{:,.{d}f}".format(float(x), d=dp)
    except (TypeError, ValueError):
        return str(x)


def _chg(price, prev):
    if price is None or not prev:
        return ""
    d, p = price - prev, (price - prev) / prev * 100
    return "%+.2f (%+.2f%%) vs prev close %s" % (d, p, _num(prev))


def _yahoo_raw(sym, p1=None, p2=None, interval="1d"):
    """One chart response.  Two things this endpoint needs: the plain UA (its edge 429s a full
    browser one) and a host/backoff retry, because it rate-limits per IP on a burst.  Give p1/p2
    for a historical window, otherwise it is the last day."""
    qs = ("period1=%d&period2=%d&interval=%s" % (int(p1), int(p2), interval)) if p1 and p2 \
        else "range=1d&interval=1d"
    url = "https://%s.finance.yahoo.com/v8/finance/chart/%s?" + qs
    last = None
    for host in ("query1", "query2"):
        for attempt in (0, 1):
            try:
                return json.loads(_get(url % (host, urllib.parse.quote(str(sym))),
                                       headers={"User-Agent": UA_MIN}))
            except urllib.error.HTTPError as e:
                last = e
                if e.code in (429, 500, 502, 503) and attempt == 0:
                    time.sleep(2.0)
                    continue
                break
            except Exception as e:
                last = e
                break
    raise RuntimeError("yahoo: %s" % last)


def _yahoo_quote(sym, detail=False):
    """Yahoo's keyless chart endpoint: works worldwide (non-US listings, indices, BTC-USD), gives
    OHLCV -- and rate-limits, which is why the Nasdaq route exists for plain US tickers."""
    j = _yahoo_raw(sym)
    chart = j.get("chart") or {}
    if chart.get("error") or not chart.get("result"):
        raise RuntimeError((chart.get("error") or {}).get("description") or "no data")
    r = chart["result"][0]
    m = r.get("meta") or {}
    q = ((r.get("indicators") or {}).get("quote") or [{}])[0]
    price = m.get("regularMarketPrice") or (q.get("close") or [None])[0]
    prev = m.get("previousClose") or m.get("chartPreviousClose")
    bits = ["%s %s %s" % (m.get("symbol") or sym, _num(price), m.get("currency") or ""),
            _chg(price, prev)]
    if m.get("marketState"):
        bits.append(m["marketState"].lower())
    if q.get("low") and q.get("high"):
        bits.append("day %s-%s" % (_num(q["low"][0]), _num(q["high"][0])))
    if q.get("volume") and q["volume"][0]:
        bits.append("vol %s" % _num(q["volume"][0], 0))
    if m.get("exchangeName"):
        bits.append(m["exchangeName"])
    if m.get("regularMarketTime"):
        bits.append(time.strftime("%Y-%m-%d %H:%M", time.localtime(m["regularMarketTime"])))
    return "[yahoo] " + ", ".join(b for b in bits if b)


def _nasdaq_quote(sym, detail=False):
    """api.nasdaq.com (declared User-Agent, no key): US listings only, but it has the company
    name, market cap and 52-week range that the Yahoo chart endpoint does not."""
    h = {"Accept": "application/json"}
    info = json.loads(_get("https://api.nasdaq.com/api/quote/%s/info?assetclass=stocks"
                           % urllib.parse.quote(str(sym)), headers=h)).get("data") or {}
    if not info:
        raise RuntimeError("no data (US listings only)")
    p = info.get("primaryData") or {}
    bits = ["%s %s" % (info.get("symbol") or sym, p.get("lastSalePrice") or "?"),
            "%s %s" % (p.get("netChange") or "", p.get("percentageChange") or "")]
    if info.get("companyName"):
        bits.append(info["companyName"])
    if p.get("volume") and p["volume"] != "N/A":
        bits.append("vol %s" % p["volume"])
    if info.get("marketStatus"):
        bits.append(info["marketStatus"].lower())
    if detail:
        try:
            s = (json.loads(_get("https://api.nasdaq.com/api/quote/%s/summary?assetclass=stocks"
                                 % urllib.parse.quote(str(sym)), headers=h)).get("data") or {}).get("summaryData") or {}
            for k in ("MarketCap", "FiftTwoWeekHighLow", "PreviousClose", "Sector"):
                if (s.get(k) or {}).get("value"):
                    bits.append("%s %s" % (s[k]["label"].lower(), s[k]["value"]))
        except Exception:
            pass
    return "[nasdaq] " + ", ".join(b.strip() for b in bits if b and b.strip())


COINGECKO = {"BTC": "bitcoin", "ETH": "ethereum", "SOL": "solana", "DOGE": "dogecoin",
             "XRP": "ripple", "ADA": "cardano", "LTC": "litecoin", "BNB": "binancecoin",
             "AVAX": "avalanche-2", "DOT": "polkadot", "LINK": "chainlink", "MATIC": "matic-network",
             "TON": "the-open-network", "SHIB": "shiba-inu", "TRX": "tron", "XMR": "monero"}
CRYPTO_PAIR = re.compile(r"[A-Z]{2,6}[-/](USD|USDT|EUR|USDC)")


def _coingecko_quote(sym, detail=False):
    """Crypto pairs (BTC-USD) through CoinGecko's keyless endpoint: the one source here that is not
    rate-limits-by-IP on a burst."""
    base = re.split(r"[-/]", str(sym))[0]
    cid = COINGECKO.get(base)
    if not cid:
        raise RuntimeError("no CoinGecko id for %s (known: %s)" % (base, ", ".join(sorted(COINGECKO))))
    d = (json.loads(_get("https://api.coingecko.com/api/v3/simple/price?ids=%s&vs_currencies=usd"
                         "&include_24hr_change=true" % cid)) or {}).get(cid) or {}
    if not d.get("usd"):
        raise RuntimeError("no price for %s" % cid)
    chg = d.get("usd_24h_change")
    return "[coingecko] %s $%s USD%s" % (sym, _num(d["usd"]),
                                         (" (%+.2f%% 24h)" % chg) if chg is not None else "")


def _parse_when(when):
    """'' -> None (live).  'YYYY-MM-DD' -> a daily bar.  'YYYY-MM-DD HH:MM' -> that minute.
    'today'/'yesterday' are taken on the host's clock (which is US Eastern, like the market)."""
    s = str(when or "").strip()
    if not s:
        return None
    low = s.lower()
    today = datetime.date.today()
    if low in ("today", "now"):
        return ("%s" % today.isoformat(), False, None)
    if low == "yesterday":
        return ((today - datetime.timedelta(days=1)).isoformat(), False, None)
    m = re.match(r"^(\d{4})-(\d{1,2})-(\d{1,2})(?:[ T](\d{1,2}):(\d{2}))?$", s)
    if not m:
        raise ValueError("use YYYY-MM-DD or 'YYYY-MM-DD HH:MM' (or today/yesterday)")
    y, mo, d, hh, mi = m.groups()
    date_s = "%s-%02d-%02d" % (y, int(mo), int(d))
    return (date_s, bool(hh), (int(hh), int(mi)) if hh else None)


def _at(ts_list, off, date_s, hhmm):
    """Pick the bar to report: the one at that minute, else the last bar of that local day."""
    want_day = date_s
    bars = [(t, datetime.datetime.utcfromtimestamp(t + off)) for t in ts_list]
    if hhmm:
        h, mi = hhmm
        epoch = calendar.timegm(datetime.datetime.strptime(date_s, "%Y-%m-%d").timetuple()) \
            + h * 3600 + mi * 60 - off
        nearest = min(ts_list, key=lambda t: abs(t - epoch)) if ts_list else None
        return nearest, datetime.datetime.utcfromtimestamp(nearest + off) if nearest else None, \
            (nearest == epoch)
    same = [t for t, local in bars if local.date().isoformat() == want_day]
    if same:
        last = same[-1]
        return last, datetime.datetime.utcfromtimestamp(last + off), False
    return None, None, False


def _when_label(when):
    date_s, has_time, hhmm = when
    return "%s %02d:%02d" % (date_s, hhmm[0], hhmm[1]) if has_time else date_s


def _yahoo_at(sym, when):
    """A historical bar from Yahoo's keyless chart endpoint (1m bars for a time, 1d for a date).

    Yahoo refuses an intraday request whose window is too wide (HTTP 422) and only keeps fine
    granularity for recent periods, so both the interval and the window are chosen by the age of
    the target: 1m/2 days inside a week, 5m/10 days inside two months, 1h/30 days inside two years,
    daily beyond that.
    """
    date_s, has_time, hhmm = when
    d0 = datetime.datetime.strptime(date_s, "%Y-%m-%d")
    age = (datetime.datetime.now() - d0).days
    if not has_time:
        iv, span = "1d", 5
    elif age <= 7:
        iv, span = "1m", 1
    elif age <= 55:
        iv, span = "5m", 5
    elif age <= 700:
        iv, span = "60m", 15
    else:
        iv, span = "1d", 5
    p1 = calendar.timegm((d0 - datetime.timedelta(days=span)).timetuple())
    p2 = calendar.timegm((d0 + datetime.timedelta(days=span)).timetuple())
    url = ("https://%s.finance.yahoo.com/v8/finance/chart/%s?period1=%d&period2=%d&interval=%s"
           "&includePrePost=false")
    last = None
    for host in ("query1", "query2"):
        try:
            j = json.loads(_get(url % (host, urllib.parse.quote(str(sym)), p1, p2, iv),
                                headers={"User-Agent": UA_MIN}))
            last = j
            break
        except Exception as e:
            last = e
    if not isinstance(last, dict):
        raise RuntimeError("yahoo: %s" % last)
    chart = last.get("chart") or {}
    res = chart.get("result") or []
    if chart.get("error") or not res:
        raise RuntimeError((chart.get("error") or {}).get("description") or "no data")
    r = res[0]
    m = r.get("meta") or {}
    off = m.get("gmtoffset")
    off = -18000 if off is None else int(off)
    ts = r.get("timestamp") or []
    q = ((r.get("indicators") or {}).get("quote") or [{}])[0]
    if not ts:
        raise RuntimeError("no bars in that window (market closed, or a holiday)")
    t, local, exact = _at(ts, off, date_s, hhmm)
    if t is None or local is None:
        raise RuntimeError("no trading day near %s" % date_s)
    i = ts.index(t)
    close = (q.get("close") or [None])[i]
    cur = m.get("currency") or ""
    bits = ["%s %s %s" % (m.get("symbol") or sym, _num(close), cur)]
    if has_time:
        bits.append("at %s %s" % (local.strftime("%Y-%m-%d %H:%M"),
                                  m.get("exchangeTimezoneName") or "exchange time"))
    else:
        bits.append("session %s" % local.date().isoformat())   # a daily bar has a date, not a time
    bits.append("%s bar" % iv)
    if not exact and has_time:
        bits.append("NEAREST bar, not an exact %s hit" % _when_label(when))
    if not has_time:
        for k, lab in (("open", "open"), ("high", "high"), ("low", "low")):
            v = (q.get(k) or [None])[i]
            if v is not None:
                bits.append("%s %s" % (lab, _num(v)))
    if q.get("volume") and q["volume"][i]:
        bits.append("vol %s" % _num(q["volume"][i], 0))
    return "[yahoo] " + ", ".join(b for b in bits if b)


def _nasdaq_at(sym, when):
    """Daily history from api.nasdaq.com (keyless, US listings, no intraday)."""
    date_s, has_time, _hhmm = when
    d0 = datetime.datetime.strptime(date_s, "%Y-%m-%d")
    frm = (d0 - datetime.timedelta(days=7)).strftime("%Y-%m-%d")
    to = (d0 + datetime.timedelta(days=3)).strftime("%Y-%m-%d")
    j = json.loads(_get("https://api.nasdaq.com/api/quote/%s/historical?assetclass=stocks"
                        "&fromdate=%s&todate=%s&limit=30" % (urllib.parse.quote(str(sym)), frm, to),
                        headers={"Accept": "application/json"}))
    rows = (((j.get("data") or {}).get("tradesTable") or {}).get("rows")) or []
    if not rows:
        raise RuntimeError("no rows (US listings only)")
    want = d0.strftime("%m/%d/%Y")
    row = next((x for x in rows if x.get("date") == want), None)
    extra = ""
    if row is None:
        row = min(rows, key=lambda x: abs((datetime.datetime.strptime(x["date"], "%m/%d/%Y") - d0).days))
        extra = ", NEAREST trading day"
    bits = ["%s %s" % (sym, row.get("close") or "?"), "on %s" % row.get("date"),
            "open %s" % row.get("open"), "high %s" % row.get("high"), "low %s" % row.get("low"),
            "vol %s" % row.get("volume")]
    if has_time:
        extra += ", daily close only (this source has no intraday)"
    return "[nasdaq] " + ", ".join(b for b in bits if b and b.strip()) + extra


def _coingecko_at(sym, when):
    """Crypto at a date (CoinGecko history) or a minute (its keyless market_chart/range)."""
    base = re.split(r"[-/]", str(sym))[0]
    cid = COINGECKO.get(base)
    if not cid:
        raise RuntimeError("no CoinGecko id for %s" % base)
    date_s, has_time, hhmm = when
    if not has_time:
        dd, mm, yy = date_s.split("-")[2], date_s.split("-")[1], date_s.split("-")[0]
        d = (json.loads(_get("https://api.coingecko.com/api/v3/coins/%s/history?date=%s-%s-%s"
                             "&localization=false" % (cid, dd, mm, yy))) or {})
        usd = ((d.get("market_data") or {}).get("current_price") or {}).get("usd")
        if not usd:
            raise RuntimeError("no price for that date")
        return "[coingecko] %s $%s USD on %s" % (sym, _num(usd), date_s)
    off = -18000                                   # report in US Eastern, like the equity path
    epoch = calendar.timegm(datetime.datetime.strptime(date_s, "%Y-%m-%d").timetuple()) \
        + hhmm[0] * 3600 + hhmm[1] * 60 - off
    d = json.loads(_get("https://api.coingecko.com/api/v3/coins/%s/market_chart/range?vs_currency=usd"
                        "&from=%d&to=%d" % (cid, epoch - 900, epoch + 900)))
    pts = d.get("prices") or []
    if not pts:
        raise RuntimeError("no intraday points (CoinGecko keeps about a day of minute data)")
    t, p = min(pts, key=lambda x: abs(x[0] / 1000.0 - epoch))
    local = datetime.datetime.utcfromtimestamp(t / 1000.0 + off)
    return "[coingecko] %s $%s USD at %s (US Eastern, UTC-05:00), nearest minute point" % (
        sym, _num(p), local.strftime("%Y-%m-%d %H:%M"))


def _yahoo_hist(sym, days=7):
    """Daily OHLCV for the last `days` trading days -- the series a chart needs (one request)."""
    p2 = int(time.time())
    p1 = p2 - max(10, int(days * 2.2)) * 86400
    j = _yahoo_raw(sym, p1, p2, "1d")
    chart = j.get("chart") or {}
    res = chart.get("result") or []
    if chart.get("error") or not res:
        raise RuntimeError((chart.get("error") or {}).get("description") or "no data")
    r = res[0]
    m = r.get("meta") or {}
    off = int(m.get("gmtoffset") or -18000)
    ts = r.get("timestamp") or []
    q = ((r.get("indicators") or {}).get("quote") or [{}])[0]
    rows = []
    for i, t in enumerate(ts):
        c = (q.get("close") or [None])[i]
        if c is None:
            continue
        rows.append((datetime.datetime.utcfromtimestamp(t + off).date().isoformat(),
                     (q.get("open") or [None])[i], (q.get("high") or [None])[i],
                     (q.get("low") or [None])[i], c, (q.get("volume") or [0])[i] or 0))
    if not rows:
        raise RuntimeError("no daily rows in that window")
    rows = rows[-days:]
    head = "[yahoo] %s, %d daily rows to %s -- CSV follows (a header row, then the data)" % (
        m.get("symbol") or sym, len(rows), rows[-1][0])
    body = "date,open,high,low,close,volume\n" + \
        "\n".join("%s,%.2f,%.2f,%.2f,%.2f,%d" % row for row in rows)
    return head + "\n" + body


def _nasdaq_hist(sym, days=7):
    """Same shape from api.nasdaq.com (keyless, US listings); newest first, reversed here."""
    d0 = datetime.date.today()
    frm = (d0 - datetime.timedelta(days=max(10, int(days * 2.2)))).strftime("%Y-%m-%d")
    j = json.loads(_get("https://api.nasdaq.com/api/quote/%s/historical?assetclass=stocks"
                        "&fromdate=%s&todate=%s&limit=%d"
                        % (urllib.parse.quote(str(sym)), frm, d0.strftime("%Y-%m-%d"),
                           max(10, days * 2)),
                        headers={"Accept": "application/json"}))
    rows = (((j.get("data") or {}).get("tradesTable") or {}).get("rows")) or []
    if not rows:
        raise RuntimeError("no rows (US listings only)")
    out = []
    for x in reversed(rows[:days]):
        try:
            d = datetime.datetime.strptime(x["date"], "%m/%d/%Y").date().isoformat()
        except Exception:
            continue
        f = lambda k: float(str(x.get(k, "0")).replace("$", "").replace(",", "") or 0)   # noqa: E731
        out.append("%s,%.2f,%.2f,%.2f,%.2f,%d" % (d, f("open"), f("high"), f("low"), f("close"),
                                                  int(f("volume"))))
    return ("[nasdaq] %s, %d daily rows -- CSV follows (a header row, then the data)\n"
            "date,open,high,low,close,volume\n%s" % (sym, len(out), "\n".join(out)))


def stock_history(symbols="", days=7):
    """The daily open/high/low/close/volume series for the last `days` trading days, as CSV rows --
    one request per symbol instead of one per day.  Feed it straight into a script or a chart."""
    syms = [s for s in re.split(r"[,\s]+", str(symbols).upper()) if s][:4]
    if not syms:
        return "no symbol given -- try DOCN, AAPL, NVDA"
    try:
        days = max(2, min(int(days or 7), 60))
    except (TypeError, ValueError):
        days = 7
    out = []
    for s in syms:
        why = []
        for fn in (_yahoo_hist, _nasdaq_hist):
            try:
                out.append(fn(s, days))
                break
            except Exception as e:
                why.append("%s: %s" % (fn.__name__.strip("_").split("_")[0], e))
        else:
            out.append("%s: no history (%s)" % (s, "; ".join(why)))
    return "\n".join(out)


def stock_quote(symbols="", detail=False, when=None):
    """Quotes for one or more symbols (AAPL, MSFT, 7203.T, BTC-USD ...).

    A plain US ticker goes to Nasdaq first (no rate limit, and it carries the company name, market
    cap and 52-week range); a crypto pair (BTC-USD) goes to CoinGecko; anything else -- a foreign
    listing like 7203.T, an index like ^GSPC -- goes to Yahoo first, because Nasdaq only serves US
    listings.  Either way the answer says which source it came from.
    """
    syms = [s for s in re.split(r"[,\s]+", str(symbols).upper()) if s][:8]
    if not syms:
        return "no symbol given -- try AAPL, MSFT, NVDA, 7203.T, ^GSPC, BTC-USD"
    try:
        w = _parse_when(when)
    except ValueError as e:
        return "bad date: %s" % e
    out = []
    for s in syms:
        if w is not None:                       # historical: Yahoo has intraday and the world;
            order = (_coingecko_at, _yahoo_at, _nasdaq_at) if CRYPTO_PAIR.fullmatch(s) \
                else (_yahoo_at, _nasdaq_at)    # CoinGecko is the crypto-native source
        elif CRYPTO_PAIR.fullmatch(s):
            order = (_coingecko_quote, _yahoo_quote)
        elif re.fullmatch(r"[A-Z]{1,5}", s):
            order = (_nasdaq_quote, _yahoo_quote)
        else:
            order = (_yahoo_quote, _nasdaq_quote)
        why = []
        for fn in order:
            try:
                arg: Any = detail if w is None else w
                out.append(fn(s, arg))
                break
            except Exception as e:
                why.append("%s: %s" % (fn.__name__.strip("_").split("_")[0], e))
        else:
            out.append("%s: no quote for %s (%s)" % (s, _when_label(w) if w else "now",
                                                     "; ".join(why)))
    return "\n".join(out)


def fx_rate(base="USD", to=""):
    """ECB reference rates through frankfurter.dev -- keyless, daily, no crypto."""
    base = (base or "USD").upper()
    syms = ",".join(x for x in re.split(r"[,\s]+", str(to).upper()) if x)
    j = json.loads(_get("https://api.frankfurter.dev/v1/latest?base=%s%s"
                        % (base, ("&symbols=" + syms) if syms else "")))
    rates = j.get("rates") or {}
    if not rates:
        raise RuntimeError("no rates for %s" % base)
    return "[frankfurter/ECB %s] %s -> %s" % (
        j.get("date"), base,
        ", ".join("%s %s" % (k, _num(v, 4 if abs(v) < 10 else 2)) for k, v in sorted(rates.items())))


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


def _load_sandbox():
    """Import the sandbox runner (localgpu/sandbox_exec.py)."""
    import sandbox_exec
    return sandbox_exec


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
        if name in ("python", "shell"):
            try:
                sandbox = _load_sandbox()
            except Exception as e:
                return "could not load localgpu/sandbox_exec.py: %s" % e
            deps = [str(d) for d in (args.get("deps") or [])]
            net = bool(args.get("net"))
            what = "%s with %s" % (name, "network on" if net else "a deps install (network for pip)")
            why = _refused(what, auto_yes) if (deps or net) else None
            if why:
                return why
            return sandbox.run(code=args.get("code") if name == "python" else None,
                               command=args.get("command") if name == "shell" else None,
                               files=args.get("files"), deps=deps,
                               timeout=int(args.get("timeout") or 30), net=net, root=root,
                               mounts=ctx.get("mounts", ()))
        if name == "show":
            sandbox = _load_sandbox()
            raw = str(args.get("path") or "")
            p = os.path.realpath(os.path.join(sandbox.WORK, os.path.basename(raw)))
            if not p.startswith(os.path.realpath(sandbox.WORK)):
                return "refused: only files inside %s can be opened" % sandbox.WORK
            if not os.path.isfile(p):
                return "no such file: %s (files the run made are listed with each python call)" % p
            why = _refused("open %s on the Mac" % os.path.basename(p), auto_yes)
            if why:
                return why
            subprocess.run(["open", p], check=False)
            return "opened %s on the Mac (%d bytes) -- path: %s" % (
                os.path.basename(p), os.path.getsize(p), p)
        if name == "remember":
            return remember(args.get("text", ""))
        if name == "recall":
            return recall(args.get("query", ""))
        if name == "git_cmd":
            return git_cmd(args.get("subcommand", ""), root)
        if name == "grep_repo":
            return grep_repo(args.get("pattern", ""), root, args.get("path") or ".",
                             args.get("glob"), int(args.get("max_hits") or 40))
        if name == "stock_quote":
            return stock_quote(args.get("symbols", ""), bool(args.get("detail")),
                               args.get("when"))
        if name == "stock_history":
            return stock_history(args.get("symbols", ""), args.get("days") or 7)
        if name == "fx_rate":
            return fx_rate(args.get("base", "USD"), args.get("to", ""))
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


def stream(base, model, messages, tools=None, temp=None):
    req_body = {"model": model, "messages": messages, "stream": True,
                "temperature": TEMP if temp is None else temp,
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
        pin="", auto=True, temp=None, examples=True, mounts=(), initial_skills=()):
    """Interactive chat.

    tools: turn the tool harness on.  pin: "all" or a comma list of bundles/tool names to load up
    front ("" = nothing, conditional loading only).  auto: let the user's wording load bundles.
    temp: sampling temperature (default 0.0 -- the server's 1.0 makes the model ramble).
    examples: include the terse-answer few-shot (FORMAT_SHOT) in the prompt prefix.
    """
    root = root or os.getcwd()
    harness = bool(tools)
    auto_yes = bool(yes)
    temperature = TEMP if temp is None else float(temp)
    pins = set()
    for w in str(pin or "").split(","):
        pins |= resolve(w)
    state = {"pinned": sorted(pins), "auto": bool(auto), "mounts": list(mounts),
             "skills": set(initial_skills)}
    sys_prompt = system or (SYSTEM_BASE + (SYSTEM_TOOLS if harness else ""))
    prefix = ([{"role": "system", "content": sys_prompt}] if sys_prompt else []) + \
             (list(FORMAT_SHOT) if examples else [])
    messages: list = list(prefix)

    def turn(text=""):
        """One user turn: stream, run any tool calls, repeat until the model answers."""
        ctx = {"root": root, "auto_yes": auto_yes, "state": state,
               "mounts": tuple(state.get("mounts", ()))}
        for _ in range(MAX_ITER):
            names = active_names(state, text) if harness else set()
            # attach the usage notes that belong to the bundles in play, and only those
            for b in {bundle_of(n) for n in names}:
                state["skills"] |= set(BUNDLE_SKILLS.get(b, ()))
            _apply_skills(messages, state)
            reply, calls = stream(base, model, messages, tools=tool_defs(names) if harness else None,
                                  temp=temperature)
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
        """What the harness knows and what it is sending (bare /tools, or /tools list)."""
        print("%stool harness %s | pinned: %s | loaded by wording: %s%s"
              % (DIM, "ON" if harness else "off", ", ".join(state["pinned"]) or "none",
                 "on" if state["auto"] else "off", RESET))
        rows = [("(always)", [SUMMONER], "the summoner -- use_tools loads anything below")]
        rows += [(b, sorted(n for n, t in _registry.items() if t["bundle"] == b), BUNDLES[b])
                 for b in BUNDLES]
        for b, ts, what in rows:
            print("  %-9s %-40s %s" % (b, ",".join(ts), what))

    if prompt:
        messages.append({"role": "user", "content": prompt})
        turn(prompt)
        return
    tab = setup_readline() if sys.stdin.isatty() else False
    print("%schatting with %s on the GTX 1080 Ti -- /help lists the commands%s; \"\"\" for "
          "multi-line%s" % (DIM, model, ", TAB completes them" if tab else "", RESET))
    print("%stool harness %s -- %s; root %s%s"
          % (DIM, ("on (loads per message; /tools lists, /load pins)"
                   + ((" -- pinned: " + ", ".join(state["pinned"])) if state["pinned"] else ""))
             if harness else "off (/tools on)",
             "no confirmations" if auto_yes else "write/run ask first (/yes)", root, RESET))
    print("%stemperature %.2f%s%s" % (DIM, temperature, "" if examples else ", no few-shot", RESET))
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
        first, _, rest = cmd.partition(" ")
        first = ALIASES.get(first, first)          # /quit, /?, /cmds ...
        cmd = ("%s %s" % (first, rest)).strip() if rest else first
        if cmd in ("/", "/help") or cmd.startswith("/help "):
            print(help_text(cmd[5:].strip() if cmd.startswith("/help ") else "")); continue
        if cmd in ("/exit", "/quit", "/bye"):
            return
        if cmd == "/clear":
            messages = list(prefix); print("(cleared)"); continue
        if cmd.startswith("/system "):
            tail = [m for m in messages[len(prefix):] if m.get("role") != "system"]
            prefix = [{"role": "system", "content": cmd[8:]}] + (list(FORMAT_SHOT) if examples else [])
            messages = list(prefix) + tail
            print("(system prompt set)"); continue
        if cmd == "/temp" or cmd.startswith("/temp "):
            arg = cmd[5:].strip()
            if arg:
                try:
                    temperature = max(0.0, min(2.0, float(arg)))
                except ValueError:
                    print("(temperature must be a number)"); continue
            print("(temperature %.2f -- lower means shorter thinking; 0 is the default)" % temperature)
            continue
        if cmd == "/tools" or cmd.startswith("/tools "):
            arg = cmd[6:].strip().lower()
            if arg in ("on", "off"):
                harness = arg == "on"
                print("(tool harness %s)" % ("ON" if harness else "off")); continue
            tool_report(); continue
        if cmd.startswith("/load ") or cmd.startswith("/unload "):
            load = cmd.startswith("/load ")
            words = [w for w in cmd.split()[1].split(",") if w]
            harness = harness or load          # /load implies the harness
            for w in words:
                got = resolve(w)
                if not got:
                    print("(unknown bundle or tool %r)" % w); continue
                state["pinned"] = sorted((set(state["pinned"]) | got) if load
                                         else (set(state["pinned"]) - got))
                print("(%s %s: %s)" % ("loaded" if load else "unloaded", w, ", ".join(sorted(got))))
            continue
        if cmd == "/mount" or cmd.startswith("/mount "):
            arg = cmd[6:].strip()
            if arg:
                try:
                    added = _load_sandbox().parse_mounts(arg)
                except Exception as e:
                    print("(could not parse that mount: %s)" % e); continue
                state["mounts"] = list(state.get("mounts", [])) + list(added)
                for h, t, ro in added:
                    print("(sandbox will see %s at /mnt/%s%s)" % (h, t, ":ro" if ro else ""))
            cur = state.get("mounts", [])
            print("\n".join("  %s -> /mnt/%s%s" % (h, t, ", read-only" if ro else ", read-write")
                            for h, t, ro in cur) or "  (no host folders shared)")
            continue
        if cmd == "/skill" or cmd.startswith("/skill "):
            arg = cmd[6:].strip().lower()
            if arg in ("none", "off", "clear"):
                state["skills"] = set(); print("(notes cleared)"); continue
            if arg:
                if arg not in SKILLS:
                    print("(no note called %s -- have: %s)" % (arg, ", ".join(sorted(SKILLS))))
                elif arg in state["skills"]:
                    state["skills"] = set(state["skills"]) - {arg}
                    print("(note %s detached)" % arg)
                else:
                    state["skills"] = set(state["skills"]) | {arg}
                    print("(note %s attached, %d words)" % (arg, len(SKILLS[arg].split())))
                continue
            print("attached: %s" % (", ".join(sorted(state["skills"])) or "none"))
            for n in sorted(SKILLS):
                print("  %-8s %s" % (n, SKILLS[n][:78] + ("..." if len(SKILLS[n]) > 78 else "")))
            print("  (a bundle attaches its own notes when it loads; /skill none clears them)")
            continue
        if cmd == "/auto":
            state["auto"] = not state["auto"]
            print("(keyword auto-load %s)" % ("on" if state["auto"] else "off")); continue
        if cmd == "/yes":
            auto_yes = not auto_yes
            print("(confirmations %s)" % ("off: write/run execute immediately" if auto_yes else "on")); continue
        if cmd.startswith("/"):
            near = fuzzy_rank(cmd.split()[0][1:], list(COMMANDS))
            print("(unknown command %s%s -- /help lists them all)"
                  % (cmd.split()[0], (" -- did you mean %s?" % ", ".join(near[:3])) if near else ""))
            continue
        if not cmd:
            continue
        messages.append({"role": "user", "content": line})
        try:
            turn(line)
        except KeyboardInterrupt:
            print("\n(interrupted)"); messages.pop(); continue
