#!/usr/bin/env python3
"""Summarise raw/*.csv into tables.md: median over processes and rounds per cell, speedup against
serial, and the break-even shape (first shape from which the mode stays faster than serial by at least
5 percent for every larger shape of the family)."""
import csv, glob, json, os, re, statistics, sys, collections

here = os.path.dirname(os.path.abspath(__file__))
raw = os.path.join(here, "raw")
busy = {}
try:
    for l in open(os.path.join(raw, "processes.log")):
        j = json.loads(l)
        busy[j["tag"]] = j
except FileNotFoundError:
    pass

def load(prefix):
    cells = collections.defaultdict(lambda: collections.defaultdict(list))
    order = []
    for p in sorted(glob.glob(os.path.join(raw, prefix + "*.csv"))):
        tag = os.path.basename(p)[:-4]
        for r in csv.DictReader(open(p)):
            k = (r["family"], r["shape"])
            if k not in order: order.append(k)
            cells[k][r["mode"]].append((float(r["ns"]), tag))
    return cells, order

def us(x): return f"{x/1000:.2f}" if x < 1e5 else f"{x/1000:.0f}"

def table(prefix, modes=("seq", "plain_out", "plain_in", "team"), thr=0.95):
    cells, order = load(prefix)
    out, fams = [], []
    for k in order:
        if k[0] not in fams: fams.append(k[0])
    be = {}
    for f in fams:
        out.append(f"\n### {prefix} / {f}\n")
        out.append("| shape | " + " | ".join(f"{m} us" for m in modes) + " | " + " | ".join(f"{m}/seq" for m in modes[1:]) + " | n |")
        out.append("|" + "---|" * (len(modes) * 2 + 1))
        rows = [k for k in order if k[0] == f]
        med = {}
        for k in rows:
            med[k] = {m: statistics.median(v for v, _ in cells[k][m]) for m in modes if cells[k][m]}
            n = len(cells[k][modes[0]])
            r = med[k]
            out.append(f"| {k[1]} | " + " | ".join(us(r[m]) for m in modes) + " | " + " | ".join(f"{r[m]/r['seq']:.2f}" for m in modes[1:]) + f" | {n} |")
        for m in modes[1:]:
            # first index i such that all rows j>=i have ratio < thr
            idx = None
            for i in range(len(rows) - 1, -1, -1):
                r = med[rows[i]]
                if r[m] / r["seq"] < thr: idx = i
                else: break
            be[(f, m)] = (rows[idx][1], med[rows[idx]]["seq"]) if idx is not None else None
        # any large-size loss
    out.append(f"\n### {prefix}: break-even (first shape after which the mode stays below {thr} x serial), with the serial time there\n")
    out.append("| family | " + " | ".join(modes[1:]) + " |")
    out.append("|" + "---|" * len(modes))
    for f in fams:
        out.append(f"| {f} | " + " | ".join((f"{be[(f,m)][0]} ({us(be[(f,m)][1])} us)" if be[(f, m)] else "none") for m in modes[1:]) + " |")
    return "\n".join(out), be

def dispatch():
    rows = collections.defaultdict(list)
    for p in sorted(glob.glob(os.path.join(raw, "dispatch-p*.csv"))):
        for r in csv.DictReader(open(p)):
            rows[(r["kind"], int(r["width"]))].append(float(r["ns"]))
    kinds = []
    for k, _ in rows:
        if k not in kinds: kinds.append(k)
    out = ["| kind | w=2 | w=4 | w=6 | processes |", "|---|---|---|---|---|"]
    for k in kinds:
        cell = lambda w: f"{statistics.median(rows[(k,w)])/1000:.2f}" if (k, w) in rows else "-"
        out.append(f"| {k} | {cell(2)} | {cell(4)} | {cell(6)} | {len(rows[(k,6)])} |")
    return "## Dispatch cost (us per section, median over processes; the `load` rows are call time minus the 10 us of work in every share)\n\n" + "\n".join(out)

def hosts():
    out = ["| process | idle before % | load1 before | idle after % | top 3 consumers before | marked busy |", "|---|---|---|---|---|---|"]
    for t, j in busy.items():
        b, a = j["before"], j["after"]
        bad = bool(b["others"]) or (b["idle"] or 0) < 85 or (a["idle"] or 0) < 85
        out.append(f"| {t} | {b['idle']} | {b['load'][0]:.1f} | {a['idle']} | {'; '.join(b['top3'])} | {'**yes**' if bad else 'no'} |")
    return "## Host state per process (busy = idle below 85 percent before or after; in-run samples include the benchmark's own 6 threads and are not used)\n\n" + "\n".join(out)

if __name__ == "__main__":
    parts = [dispatch()]
    for pre in ["faer-off", "faer-default", "stack", "lusweep"]:
        if glob.glob(os.path.join(raw, pre + "*.csv")):
            t, _ = table(pre, modes=("seq","plain_in","team") if pre=="lusweep" else ("seq","plain_out","plain_in","team"))
            parts.append(f"## {pre}\n" + t)
    parts.append(hosts())
    open(os.path.join(here, "tables.md"), "w").write("\n\n".join(parts) + "\n")
    print("\n\n".join(parts))
