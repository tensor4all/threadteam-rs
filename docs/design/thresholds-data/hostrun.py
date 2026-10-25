#!/usr/bin/env python3
"""Run one measurement process under the host protocol: wait until the machine is at least 85 percent
idle with no other cargo/fbench/test process, record idle, load and the top 3 CPU consumers before and
after (and an idle sample every 25 s during), append a JSON line to the log, mark the run busy if any
sample is below 85 percent.

usage: hostrun.py LOG TAG OUTFILE -- command...   (the command's stdout goes to OUTFILE)
"""
import json, os, re, subprocess, sys, threading, time

IDLE_MIN = 85.0

def sh(c):
    return subprocess.run(c, shell=True, capture_output=True, text=True).stdout.strip()

def idle():
    m = re.search(r"([\d.]+)% idle", sh("top -l 2 -s 1 -n 0 | grep 'CPU usage' | tail -1"))
    return float(m.group(1)) if m else None

def top3():
    out = []
    for r in sh("ps -Ao pcpu,comm -r | head -4 | tail -3").splitlines():
        a = r.split(None, 1)
        out.append(f"{a[0]} {os.path.basename(a[1])}")
    return out

def others():
    bad = []
    for l in sh("ps -Ao pid,command").splitlines():
        pid, _, cmd = l.strip().partition(" ")
        if pid.isdigit() and int(pid) != os.getpid() and re.search(r"(^|/)(fbench|thr-[a-z]+|dispatch_cost|thresholds-[a-z]+)( |$)|cargo (build|test|bench|check|clippy)|rustc ", cmd) and "grep" not in cmd:
            bad.append(cmd[:80])
    return bad

log, tag, out = sys.argv[1:4]
cmd = sys.argv[5:]
waited, seen = 0, []
while True:
    b = others(); u = idle(); seen.append(u)
    if not b and u is not None and u >= IDLE_MIN: break
    if waited >= 300: break
    time.sleep(15); waited += 17
before = {"idle": seen[-1], "load": os.getloadavg(), "top3": top3(), "others": b}
samples, stop = [], threading.Event()
def poll():
    while not stop.wait(25):
        samples.append(idle())
th = threading.Thread(target=poll, daemon=True); th.start()
t0 = time.time()
r = subprocess.run(cmd, capture_output=True, text=True)
t1 = time.time(); stop.set()
after = {"idle": idle(), "load": os.getloadavg(), "top3": top3()}
busy = bool(b) or before["idle"] is None or before["idle"] < IDLE_MIN or any(s is not None and s < IDLE_MIN for s in samples) or (after["idle"] or 100) < IDLE_MIN
open(out, "w").write(r.stdout)
open(log, "a").write(json.dumps({"tag": tag, "cmd": cmd, "rc": r.returncode, "wall_s": round(t1 - t0, 1), "waited_s": waited, "idle_polls": seen,
                                  "before": before, "during_idle": samples, "after": after, "busy": busy, "stderr": r.stderr[-300:]}) + "\n")
print(f"{tag}: rc={r.returncode} {t1-t0:.0f}s idle_before={before['idle']} during={samples} after={after['idle']} busy={busy}")
