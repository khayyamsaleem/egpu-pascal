"""Streaming terminal chat over an OpenAI-compatible endpoint (used by `local-gpu chat`).

Commands: /exit, /clear (forget the conversation), /system <text>, /tools [on|off|list], /yes,
\"\"\" starts and ends a multi-line message.  Thinking (reasoning_content) is shown dimmed.
Standard library only.

With tools on (`local-gpu chat --tools`) the model gets a small, explicit tool set it can call in
a loop: read-only tools run immediately, anything that writes or runs a command asks first unless
`--yes` was given.  File paths are confined to `--root` (default: the working directory).
"""
import json, os, subprocess, sys, time, urllib.request

DIM, RESET = ("\033[2m", "\033[0m") if sys.stdout.isatty() else ("", "")
MAX_ITER = 8                                   # tool round trips per turn
MAX_OUT = 4000                                 # bytes of tool output kept (context is 16k)

SYSTEM_WITH_TOOLS = (
    "You are a careful engineering assistant running locally on a GTX 1080 Ti. You have tools: "
    "call one when it answers the question better than guessing, then use its result verbatim "
    "instead of inventing one. Keep answers short and state assumptions.")

TOOLS = [
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


def call_tool(name, args, root, auto_yes):
    """Execute one tool call and return its result as a string (never raises)."""
    try:
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
