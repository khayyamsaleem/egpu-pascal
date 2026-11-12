"""Minimal streaming terminal chat over an OpenAI-compatible endpoint (used by `local-gpu chat`).

Commands: /exit, /clear (forget the conversation), /system <text>, \"\"\" starts and ends a
multi-line message. Thinking (reasoning_content) is shown dimmed. Standard library only.
"""
import json, sys, time, urllib.request

DIM, RESET = ("\033[2m", "\033[0m") if sys.stdout.isatty() else ("", "")


def stream(base, model, messages):
    body = json.dumps({"model": model, "messages": messages, "stream": True,
                       "stream_options": {"include_usage": True}}).encode()
    req = urllib.request.Request(base + "/chat/completions", data=body, headers={"Content-Type": "application/json"})
    text, think, timings, t0, first = "", False, None, time.time(), None
    with urllib.request.urlopen(req, timeout=3600) as r:
        for line in r:
            line = line.strip()
            if not line.startswith(b"data:") or line[5:].strip() == b"[DONE]":
                continue
            j = json.loads(line[5:])
            timings = j.get("timings") or timings
            for ch in j.get("choices") or []:
                d = ch.get("delta") or {}
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
    return text


def run(base, model, prompt=None, system=None):
    messages = [{"role": "system", "content": system}] if system else []
    if prompt:
        messages.append({"role": "user", "content": prompt})
        stream(base, model, messages)
        return
    print("%schatting with %s on the GTX 1080 Ti -- /exit, /clear, /system <text>, \"\"\" for multi-line%s"
          % (DIM, model, RESET))
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
        if not cmd:
            continue
        messages.append({"role": "user", "content": line})
        try:
            reply = stream(base, model, messages)
        except KeyboardInterrupt:
            print("\n(interrupted)"); messages.pop(); continue
        messages.append({"role": "assistant", "content": reply})
