#!/usr/bin/env python3
"""Median over rounds and processes of raw/confirm-p*.csv; prints ratio to serial per mode and the new rule's
prediction (the width the token's rule picks for the serial time seen in the same sweep, by the class table).

usage: summarize_confirm.py [raw dir]
"""
import csv, glob, statistics as st, sys, collections, json

raw = sys.argv[1] if len(sys.argv) > 1 else "raw"
cells = collections.defaultdict(list)
for f in sorted(glob.glob(f"{raw}/confirm-p*.csv")):
    for r in csv.DictReader(open(f)):
        cells[(r["family"], int(r["shape"]), r["mode"])].append(float(r["ns"]))
med = {k: st.median(v) for k, v in cells.items()}
fams = []
for (f, s, m) in sorted(med):
    if f not in fams:
        fams.append(f)
modes = ["plain_out", "plain_in", "team"]
unit = {"copy": "elements", "map": "elements", "exp": "elements", "sum": "elements", "tc_matmul": "n", "einsum_chain": "n"}
print("| family | size | serial | plain_out/seq | plain_in/seq | team/seq |")
print("| --- | --- | --- | --- | --- | --- |")
for f in fams:
    sizes = sorted({s for (ff, s, m) in med if ff == f})
    for s in sizes:
        seq = med.get((f, s, "seq"))
        if seq is None:
            continue
        us = seq / 1000
        cols = []
        for m in modes:
            v = med.get((f, s, m))
            cols.append("n/a" if v is None else f"{v / seq:.2f}")
        print(f"| {f} | {s} | {us:.1f} us | " + " | ".join(cols) + " |")
