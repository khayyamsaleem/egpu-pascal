"""Ollama-compatible front end for one OpenAI-compatible backend (llama-server or `max serve`).

Listens on 127.0.0.1:11434 (Ollama's port) so Ollama clients (Open WebUI, Continue, Zed, the
`ollama` CLI with OLLAMA_HOST, ...) and OpenAI clients (base URL http://127.0.0.1:11434/v1) both
work against the GTX 1080 Ti, whichever runtime serves it:

  /v1/*              passed through unchanged (streaming included)
  /api/chat          -> /v1/chat/completions  (streaming NDJSON, tools, thinking, format)
  /api/generate      -> /v1/chat/completions  (prompt as one user turn; raw -> /v1/completions)
  /api/tags, /api/show, /api/ps, /api/version, /api/pull (the served model only)

One model is served at a time; any requested model name maps to it.
Usage: python3 ollama_proxy.py --backend http://127.0.0.1:8080 --model gemma4-e4b [--port 11434]
Standard library only.
"""
import argparse, http.client, json, sys, threading, time, urllib.parse
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ARGS = None


def now():
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def backend_conn():
    u = urllib.parse.urlparse(ARGS.backend)
    return http.client.HTTPConnection(u.hostname, u.port or 80, timeout=3600)


def model_entry():
    return {"name": ARGS.model + ":latest", "model": ARGS.model + ":latest", "modified_at": now(),
            "size": ARGS.size, "digest": "0" * 64,
            "details": {"parent_model": "", "format": ARGS.format, "family": ARGS.family,
                        "families": [ARGS.family], "parameter_size": ARGS.params, "quantization_level": ARGS.quant}}


# ------------------------------------------------------------------ Ollama -> OpenAI
def oai_messages(msgs):
    out, pending = [], []                 # pending: tool_call ids of the last assistant turn
    for m in msgs:
        m = dict(m)
        role = m.get("role", "user")
        msg = {"role": role, "content": m.get("content", "")}
        if m.get("images"):
            parts = [{"type": "text", "text": msg["content"]}] if msg["content"] else []
            parts += [{"type": "image_url", "image_url": {"url": "data:image/png;base64," + im}} for im in m["images"]]
            msg["content"] = parts
        if role == "assistant" and m.get("tool_calls"):
            calls, pending = [], []
            for i, tc in enumerate(m["tool_calls"]):
                f = tc.get("function", {})
                args = f.get("arguments", {})
                cid = tc.get("id") or "call_%d_%d" % (len(out), i)
                pending.append(cid)
                calls.append({"id": cid, "type": "function",
                              "function": {"name": f.get("name", ""),
                                           "arguments": args if isinstance(args, str) else json.dumps(args)}})
            msg["tool_calls"] = calls
        if role == "tool":
            msg["tool_call_id"] = m.get("tool_call_id") or (pending.pop(0) if pending else "call_0")
            if m.get("tool_name"):
                msg["name"] = m["tool_name"]
        if m.get("thinking") and role == "assistant":
            msg["reasoning_content"] = m["thinking"]
        out.append(msg)
    return out


def oai_body(req, messages):
    o = req.get("options") or {}
    body = {"model": ARGS.model, "messages": messages, "stream": req.get("stream", True)}
    if body["stream"]:
        body["stream_options"] = {"include_usage": True}
    for ok, bk in (("temperature", "temperature"), ("top_p", "top_p"), ("top_k", "top_k"), ("min_p", "min_p"),
                   ("seed", "seed"), ("stop", "stop"), ("repeat_penalty", "repeat_penalty"),
                   ("presence_penalty", "presence_penalty"), ("frequency_penalty", "frequency_penalty")):
        if ok in o:
            body[bk] = o[ok]
    if "num_predict" in o and o["num_predict"] not in (None, -1, -2):
        body["max_tokens"] = o["num_predict"]
    if req.get("tools"):
        body["tools"] = req["tools"]
    fmt = req.get("format")
    if fmt == "json":
        body["response_format"] = {"type": "json_object"}
    elif isinstance(fmt, dict):
        body["response_format"] = {"type": "json_schema", "json_schema": {"name": "output", "schema": fmt}}
    if req.get("think") is False:
        body["chat_template_kwargs"] = {"enable_thinking": False}
    return body


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "local-gpu-ollama/1"

    def log_message(self, fmt, *a):
        if ARGS.verbose:
            sys.stderr.write("[ollama-proxy] " + fmt % a + "\n")

    def _json(self, code, obj):
        b = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def _body(self):
        n = int(self.headers.get("Content-Length") or 0)
        return self.rfile.read(n) if n else b""

    # -------------------------------------------------------------- routing
    def do_HEAD(self):
        self.send_response(200); self.send_header("Content-Length", "0"); self.end_headers()

    def do_GET(self):
        p = self.path.split("?")[0]
        if p in ("/", ""):
            b = b"Ollama is running"
            self.send_response(200); self.send_header("Content-Type", "text/plain")
            self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b)
        elif p == "/api/version":
            self._json(200, {"version": "0.12.6"})
        elif p == "/api/tags":
            self._json(200, {"models": [model_entry()]})
        elif p == "/api/ps":
            e = model_entry(); e.update(expires_at="2999-01-01T00:00:00Z", size_vram=ARGS.size)
            self._json(200, {"models": [e]})
        elif p.startswith("/v1/"):
            self._passthrough("GET", b"")
        else:
            self._json(404, {"error": "not found"})

    def do_POST(self):
        p = self.path.split("?")[0]
        body = self._body()
        if p.startswith("/v1/"):
            return self._passthrough("POST", body)
        try:
            req = json.loads(body or b"{}")
        except ValueError:
            return self._json(400, {"error": "invalid JSON"})
        if p == "/api/chat":
            self._chat(req, oai_messages(req.get("messages") or []), generate=False)
        elif p == "/api/generate":
            if not req.get("prompt") and not req.get("messages"):     # a load / keep_alive probe
                return self._json(200, {"model": ARGS.model, "created_at": now(), "response": "",
                                        "done": True, "done_reason": "load"})
            msgs = ([{"role": "system", "content": req["system"]}] if req.get("system") else [])
            msgs.append({"role": "user", "content": req.get("prompt", ""), "images": req.get("images")})
            self._chat(req, oai_messages(msgs), generate=True)
        elif p == "/api/show":
            self._json(200, {"modelfile": "", "parameters": "", "template": "", "license": "",
                             "details": model_entry()["details"], "modified_at": now(),
                             "model_info": {"general.architecture": ARGS.family,
                                            "%s.context_length" % ARGS.family: ARGS.ctx},
                             "capabilities": ["completion", "tools"] + (["thinking"] if ARGS.thinking else [])})
        elif p == "/api/pull":
            self._json(200, {"status": "success"})
        elif p in ("/api/embed", "/api/embeddings"):
            self._json(501, {"error": "embeddings are not served by local-gpu"})
        else:
            self._json(404, {"error": "not found"})

    # -------------------------------------------------------------- OpenAI passthrough
    def _passthrough(self, method, body):
        c = backend_conn()
        hdrs = {k: v for k, v in self.headers.items() if k.lower() in ("content-type", "authorization", "accept")}
        try:
            if body and ARGS.rewrite_model:
                try:
                    j = json.loads(body); j["model"] = ARGS.model; body = json.dumps(j).encode()
                except ValueError:
                    pass
            c.request(method, self.path, body=body or None, headers=hdrs)
            r = c.getresponse()
        except OSError as e:
            return self._json(502, {"error": "backend unavailable: %s" % e})
        self.send_response(r.status)
        for k, v in r.getheaders():
            if k.lower() not in ("transfer-encoding", "connection", "content-length"):
                self.send_header(k, v)
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()
        while True:
            chunk = r.read1(65536) if hasattr(r, "read1") else r.read(65536)
            if not chunk:
                break
            self.wfile.write(b"%x\r\n%s\r\n" % (len(chunk), chunk)); self.wfile.flush()
        self.wfile.write(b"0\r\n\r\n")
        c.close()

    # -------------------------------------------------------------- Ollama chat / generate
    def _chat(self, req, messages, generate):
        stream = req.get("stream", True)
        body = oai_body(req, messages)
        t0 = time.time()
        c = backend_conn()
        try:
            c.request("POST", "/v1/chat/completions", body=json.dumps(body).encode(),
                      headers={"Content-Type": "application/json"})
            r = c.getresponse()
        except OSError as e:
            return self._json(502, {"error": "backend unavailable: %s" % e})
        if r.status != 200:
            err = r.read().decode(errors="replace")
            return self._json(r.status, {"error": err})
        key = "response" if generate else "message"

        def piece(text="", thinking="", tool_calls=None):
            d = {"model": ARGS.model, "created_at": now()}
            if generate:
                d["response"] = text
                if thinking:
                    d["thinking"] = thinking
            else:
                d["message"] = {"role": "assistant", "content": text}
                if thinking:
                    d["message"]["thinking"] = thinking
                if tool_calls:
                    d["message"]["tool_calls"] = tool_calls
            d["done"] = False
            return d

        def final(usage, timings, reason, extra=None):
            d = extra or piece()
            d.update(done=True, done_reason=reason or "stop", total_duration=int((time.time() - t0) * 1e9),
                     load_duration=0,
                     prompt_eval_count=(usage or {}).get("prompt_tokens", (timings or {}).get("prompt_n", 0)),
                     prompt_eval_duration=int((timings or {}).get("prompt_ms", 0) * 1e6),
                     eval_count=(usage or {}).get("completion_tokens", (timings or {}).get("predicted_n", 0)),
                     eval_duration=int((timings or {}).get("predicted_ms", 0) * 1e6))
            if generate:
                d["context"] = []
            return d

        def ollama_tools(calls):
            out = []
            for tc in calls:
                f = tc.get("function", {})
                a = f.get("arguments") or "{}"
                try:
                    a = json.loads(a) if isinstance(a, str) else a
                except ValueError:
                    a = {"_raw": a}
                out.append({"id": tc.get("id"), "function": {"name": f.get("name", ""), "arguments": a}})
            return out

        if not stream:
            j = json.loads(r.read())
            ch = (j.get("choices") or [{}])[0]
            m = ch.get("message") or {}
            d = piece(m.get("content") or "", m.get("reasoning_content") or "",
                      ollama_tools(m.get("tool_calls") or []) or None)
            return self._json(200, final(j.get("usage"), j.get("timings"), ch.get("finish_reason"), d))

        self.send_response(200)
        self.send_header("Content-Type", "application/x-ndjson")
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()

        def emit(obj):
            b = (json.dumps(obj) + "\n").encode()
            self.wfile.write(b"%x\r\n%s\r\n" % (len(b), b)); self.wfile.flush()

        calls, usage, timings, reason, buf = {}, None, None, None, b""
        try:
            while True:
                chunk = r.read1(65536) if hasattr(r, "read1") else r.read(4096)
                if not chunk:
                    break
                buf += chunk
                while b"\n" in buf:
                    line, buf = buf.split(b"\n", 1)
                    line = line.strip()
                    if not line.startswith(b"data:"):
                        continue
                    data = line[5:].strip()
                    if data == b"[DONE]":
                        continue
                    j = json.loads(data)
                    usage = j.get("usage") or usage
                    timings = j.get("timings") or timings
                    for ch in j.get("choices") or []:
                        reason = ch.get("finish_reason") or reason
                        dl = ch.get("delta") or {}
                        for tc in dl.get("tool_calls") or []:
                            e = calls.setdefault(tc.get("index", 0), {"id": None, "function": {"name": "", "arguments": ""}})
                            e["id"] = tc.get("id") or e["id"]
                            f = tc.get("function") or {}
                            e["function"]["name"] += f.get("name") or ""
                            e["function"]["arguments"] += f.get("arguments") or ""
                        if dl.get("content") or dl.get("reasoning_content"):
                            emit(piece(dl.get("content") or "", dl.get("reasoning_content") or ""))
            extra = piece(tool_calls=ollama_tools([calls[i] for i in sorted(calls)])) if calls else None
            emit(final(usage, timings, "stop" if reason == "tool_calls" else reason, extra))
            self.wfile.write(b"0\r\n\r\n")
        except (BrokenPipeError, ConnectionResetError):
            pass
        finally:
            c.close()


def main():
    global ARGS
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="http://127.0.0.1:8080")
    ap.add_argument("--model", default="local-gpu")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=11434)
    ap.add_argument("--size", type=int, default=0)
    ap.add_argument("--ctx", type=int, default=8192)
    ap.add_argument("--family", default="llama")
    ap.add_argument("--params", default="")
    ap.add_argument("--quant", default="")
    ap.add_argument("--format", default="gguf")
    ap.add_argument("--thinking", action="store_true")
    ap.add_argument("--rewrite-model", action="store_true", help="set \"model\" in /v1 requests to the served name")
    ap.add_argument("--verbose", action="store_true")
    ARGS = ap.parse_args()
    srv = ThreadingHTTPServer((ARGS.host, ARGS.port), Handler)
    srv.daemon_threads = True
    print("ollama-proxy: http://%s:%d -> %s (model %s)" % (ARGS.host, ARGS.port, ARGS.backend, ARGS.model), flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
