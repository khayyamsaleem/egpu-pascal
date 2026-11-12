"""Is the token stream gated by the card? Freeze the driver server mid-generation.

Streams a completion straight from llama-server (127.0.0.1:8080), timestamps every
token, and SIGSTOPs pascal/shimd.py (the process holding the card) part-way through
for a fixed window, then SIGCONT. If the tokens are computed on the card, the stream
stops for exactly that window and resumes; if they were computed on the CPU, freezing
the card's driver server changes nothing.
"""
import json
import os
import signal
import threading
import time
import urllib.request

PID_FILE = os.path.expanduser("~/.local/state/local-gpu/shimd.pid")
URL = "http://127.0.0.1:8080/completion"
FREEZE_AT, FREEZE_FOR = 1.5, 3.0

pid = int(open(PID_FILE).read())
stamps, tokens, err = [], [], []

body = {"prompt": "Count from 1 to 60, one number per line, nothing else.",
        "n_predict": 220, "stream": True, "temperature": 0, "cache_prompt": False}
req = urllib.request.Request(URL, data=json.dumps(body).encode(),
                             headers={"Content-Type": "application/json"})


def reader():
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=600) as r:
            for line in r:
                line = line.strip()
                if not line.startswith(b"data:"):
                    continue
                j = json.loads(line[5:])
                c = j.get("content")
                if c:
                    stamps.append(time.perf_counter() - t0)
                    tokens.append(c)
    except Exception as e:                                  # noqa: BLE001
        err.append("%s: %s" % (type(e).__name__, e))


th = threading.Thread(target=reader, daemon=True)
t0 = time.perf_counter()
th.start()

time.sleep(FREEZE_AT)
n_before = len(stamps)
print("freezing shimd (pid %d) at t=%.2fs, %d tokens so far" % (pid, time.perf_counter() - t0, n_before), flush=True)
os.kill(pid, signal.SIGSTOP)
t_stop = time.perf_counter()
try:
    time.sleep(FREEZE_FOR)
finally:
    os.kill(pid, signal.SIGCONT)
    t_resume = time.perf_counter()
print("resumed at t=%.2fs (frozen %.2fs)" % (t_resume - t0, t_resume - t_stop), flush=True)

th.join(timeout=180)
n_after = len(stamps)
print("tokens: %d before the freeze, %d after; total %d" % (n_before, n_after - n_before, n_after))

gap_start = t_stop - t0
if n_before and n_after > n_before:
    first_after = stamps[n_before]
    last_before = stamps[n_before - 1]
    print("last token before freeze : t=%.2fs" % last_before)
    print("freeze window            : t=%.2fs .. %.2fs" % (gap_start, t_resume - t0))
    print("first token after resume : t=%.2fs  (gap %.2fs)" % (first_after, first_after - last_before))
    ok = last_before < gap_start + 0.5 and first_after > t_resume - t0 - 0.5
    print("RESULT: %s" % ("PASS - the stream stalled for the whole freeze window: the tokens are gated by the card's driver server"
                          if ok else "INCONCLUSIVE - the gap does not line up with the freeze window"))
else:
    print("RESULT: INCONCLUSIVE - %s" % ("; ".join(err) if err else "no tokens"))
print("context around the freeze:")
print("  ... " + " ".join(t.strip() for t in tokens[max(0, n_before - 6):n_before]) + " ||| " +
      " ".join(t.strip() for t in tokens[n_before:n_before + 6]) + " ...")
