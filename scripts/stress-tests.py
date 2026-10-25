#!/usr/bin/env python3
"""Run test binaries many times, in parallel, with a hard per-run timeout.

usage: scripts/stress-tests.py RUNS PARALLEL TIMEOUT_SECS BINARY [BINARY...]

A run that exceeds the timeout is killed and counted as a hang. Exit status is
nonzero when any run failed or hung.
"""
import subprocess, sys, concurrent.futures as cf

runs, par, limit = int(sys.argv[1]), int(sys.argv[2]), float(sys.argv[3])
bins = sys.argv[4:]

def one(args):
    b, i = args
    try:
        r = subprocess.run([b], capture_output=True, timeout=limit)
        return (b, i, r.returncode, r.stdout[-2000:] + r.stderr[-2000:])
    except subprocess.TimeoutExpired as e:
        return (b, i, "HANG", (e.stdout or b"")[-2000:])

bad = 0
with cf.ThreadPoolExecutor(par) as ex:
    for b, i, rc, out in ex.map(one, [(b, i) for b in bins for i in range(runs)]):
        if rc != 0:
            bad += 1
            print(f"FAIL {b} run {i}: {rc}\n{out.decode(errors='replace')}")
print(f"{len(bins) * runs} runs, {bad} bad")
sys.exit(1 if bad else 0)
