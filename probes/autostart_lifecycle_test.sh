#!/bin/bash
# The autostart supervisor's lifecycle, driven through the real launchd job with the plug/unplug
# simulated by the project's own hook, so nothing has to be unplugged:
#
#   PASCAL_SIMULATE_DISCONNECT_FILE=<file>  in the job's EnvironmentVariables (the plist's comment
#                                           shows the key); a present file means the card is gone.
#
# Then: bash probes/autostart_lifecycle_test.sh   (~5 min, one bring-up per simulated replug)
#
#   T1  restart the job around a serving stack -> clean release (no wedged shimd, lock free), fresh serve
#   T2  ... and 0 "acquire lock file" errors after it
#   T3  touch the hook -> driver server released, lock free, not serving
#   T4  rm the hook    -> bring-up on its own + a real completion through /v1
#   T5  local-gpu stop -> still stopped 40 s later, user-stop present (the marker race)
#   T6  touch+rm       -> re-armed and serving again
#
# Probe notes: `/` and `/v1/models` are answered by the proxy itself and stay "up" behind a dead
# backend, so every "is it serving?" check here is a real completion. And "did it answer?" is judged on
# usage.completion_tokens, not on content: with a short budget this model spends all of it in its
# thinking channel (reasoning_content) and content comes back empty on a healthy card.
JOB=gui/501/com.khayyamsaleem.local-gpu.autostart
PLIST=$HOME/Library/LaunchAgents/com.khayyamsaleem.local-gpu.autostart.plist
HOOK=/tmp/egpu-gone-hook
LOCK=$TMPDIR/nv_usb4.lock
S=$HOME/.local/state/local-gpu
cd "$(dirname "$0")/.." || exit 1
say() { echo "[$(date +%H:%M:%S)] $*"; }
tokens() { curl -s -m 120 http://127.0.0.1:11434/v1/chat/completions -H 'Content-Type: application/json' \
  -d "{\"model\":\"gemma4-e2b\",\"temperature\":0,\"max_tokens\":96,\"messages\":[{\"role\":\"user\",\"content\":\"$1\"}]}" \
  2>/dev/null | python3 -c "
import json,sys
try: d = json.load(sys.stdin)
except Exception: print(''); raise SystemExit
print((d.get('usage') or {}).get('completion_tokens', 0))"; }
serves() { T=$(tokens "Reply with exactly: ok"); case "$T" in ''|*[!0-9]*) return 1 ;; esac; [ "$T" -gt 0 ]; }
ask() { curl -s -m 120 http://127.0.0.1:11434/v1/chat/completions -H 'Content-Type: application/json' \
  -d "{\"model\":\"gemma4-e2b\",\"temperature\":0,\"max_tokens\":96,\"messages\":[{\"role\":\"user\",\"content\":\"$1\"}]}" \
  2>/dev/null | python3 -c "
import json,sys
try: d = json.load(sys.stdin)
except Exception: print('<no response>'); raise SystemExit
c = d.get('choices') or [{}]
m = c[0].get('message') or {}
text = (m.get('content') or '').strip() or '(thinking) ' + (m.get('reasoning_content') or '').strip()
t = d.get('timings') or {}
print('\"%s\"  [%s tokens, %s tok/s]' % (text.splitlines()[0][:60], (d.get('usage') or {}).get('completion_tokens'),
      round(t.get('predicted_per_second', 0), 1)))"; }
wait_serving() { for i in $(seq 1 60); do serves && { say "serving after ~$((i*5))s"; return 0; }; sleep 5; done; say "TIMEOUT: not serving"; return 1; }
shimd_pid() { cat $S/shimd.pid 2>/dev/null; }
alive() { ps -p "$1" >/dev/null 2>&1; }
lock_owner() { lsof -t "$LOCK" 2>/dev/null | head -1; }
hook_live() { SUP=$(launchctl print $JOB | awk '/pid = /{print $3; exit}'); ps eww -p "$SUP" 2>/dev/null | tr ' ' '\n' | grep -q "PASCAL_SIMULATE_DISCONNECT_FILE=$HOOK"; }

rm -f "$HOOK"
say "== T0 a serving stack, then restart the job around it"
serves && say "   serving, shimd $(shimd_pid), lock owner $(lock_owner)" || say "   NOT serving (the test needs a live stack)"
MARK=$(wc -l < $S/auto.log)

say "== T1 bootout with a live stack: the SIGTERM path must release the card itself"
launchctl bootout $JOB 2>/dev/null; sleep 6
alive "$(shimd_pid)" && say "T1a FAIL  shimd still alive: $(shimd_pid)" || say "T1a PASS  driver server released"
[ -z "$(lock_owner)" ] && say "T1a PASS  TinyGPU lock free (no wedged shimd)" || say "T1a FAIL  lock held by $(lock_owner)"
serves && say "T1b FAIL  still serving" || say "T1b PASS  not serving"

say "== T2 bootstrap again: it must bring the stack back by itself, cleanly"
launchctl bootstrap gui/501 "$PLIST" 2>/dev/null || { sleep 4; launchctl bootstrap gui/501 "$PLIST"; }
wait_serving
sed -n "$((MARK+1)),\$p" $S/auto.log | sed 's/^/   | /'
echo "   lock-file errors in auto.log: $(grep -c 'acquire lock file' $S/auto.log)"
[ "$(lock_owner)" = "$(shimd_pid)" ] && say "T2 PASS  lock held by the live shimd $(shimd_pid)" || say "T2 FAIL  lock owner '$(lock_owner)' vs shimd '$(shimd_pid)'"
hook_live && say "T2 PASS  the job carries the simulation hook (T3+ are meaningful)" || say "T2 FAIL  no PASCAL_SIMULATE_DISCONNECT_FILE in the job -- T3+ would be meaningless"
say "   answer from the card: $(ask 'Reply with exactly: ok')"

say "== T3 simulate UNPLUG (touch $HOOK)"
touch "$HOOK"; sleep 12
alive "$(shimd_pid)" && say "T3 FAIL  shimd still $(shimd_pid)" || say "T3 PASS  driver server released"
[ -z "$(lock_owner)" ] && say "T3 PASS  lock free" || say "T3 FAIL  lock held by $(lock_owner)"
serves && say "T3 FAIL  still serving" || say "T3 PASS  not serving"
tail -2 $S/auto.log | sed 's/^/   | /'

say "== T4 simulate REPLUG (rm $HOOK): the stack comes back on its own"
MARK=$(wc -l < $S/auto.log)
rm -f "$HOOK"; wait_serving
sed -n "$((MARK+1)),\$p" $S/auto.log | sed 's/^/   | /'
say "   answer from the card: $(ask 'Reply with exactly: live on replug')"

say "== T5 'local-gpu stop' must not be undone by the supervisor (the marker race)"
python3 localgpu/local_gpu.py stop >/dev/null 2>&1
sleep 40
serves && say "T5 FAIL  the supervisor restarted it" || say "T5 PASS  stayed stopped"
[ -f $S/user-stop ] && say "   marker present" || say "   NO MARKER -- stop did not park the supervisor"

say "== T6 a re-plug re-arms it (the marker is cleared on the attach edge)"
touch "$HOOK"; sleep 8; rm -f "$HOOK"; wait_serving
say "   answer from the card: $(ask 'Reply with exactly: re-armed')"

say "== T7 job state"
launchctl print $JOB | grep -E "state = running|pid = " | sed 's/^/   /'
say "DONE"
