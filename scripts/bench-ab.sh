#!/usr/bin/env bash
# Interleaved A/B of the wait benchmarks (docs/design/waiting.md): builds `benches/wait_scenarios.rs`
# at two commits (the base is the bench commit a3a9320, which runs the bench on the 0.1.3 waits) and runs them alternately, so slow drifts of a shared host (other users' jobs)
# fall on both sides. Rows go to $CSV (default docs/design/waiting-data/<host>-ab.csv), labelled
# `<name>#<round>`; the `waits per call` lines go to $LOG.
#
#   scripts/bench-ab.sh <base-rev> <head-rev> <rounds> [bench args...]
#
# Example, the held-pool arms at width 64 (threadteam-rs#14, broccoli):
#   taskset -c 0-63 scripts/bench-ab.sh a3a9320 feat/wait-openmp 20 \
#     --scenario acquire --widths 64 --calls 40 --impl threadteam
#
# HEAD_VARIANTS adds head runs with extra arguments in every round, `;`-separated, each named by
# its arguments, e.g. HEAD_VARIANTS='--spin-polls 300000;--own-hold-ms 100;--pressure-events 0'.
# OMP=1 also runs the libgomp driver (benches/omp/wait_omp.c) in every round, with the same
# --scenario (sections, gaps or acquire = libgomp's held team), --widths, --calls and --load.
set -euo pipefail
base=${1:?base rev}; head=${2:?head rev}; rounds=${3:?rounds}; shift 3
host=$(hostname -s)
csv=${CSV:-docs/design/waiting-data/$host-ab.csv}
log=${LOG:-docs/design/waiting-data/$host-ab.log}
mkdir -p "$(dirname "$csv")"
work=$(mktemp -d)
trap 'git worktree remove --force "$work/base" 2>/dev/null; git worktree remove --force "$work/head" 2>/dev/null; rm -rf "$work"' EXIT
build() { # $1 = rev, $2 = dir name; prints the bench binary
  git worktree add -q --detach "$work/$2" "$1"
  (cd "$work/$2" && cargo build -j "${JOBS:-4}" --release --bench wait_scenarios >/dev/null 2>&1)
  ls -t "$work/$2"/target/release/deps/wait_scenarios-* | grep -v '\.d$' | head -1
}
bin_base=$(build "$base" base)
bin_head=$(build "$head" head)
omp=""
if [[ ${OMP:-0} == 1 ]]; then
  gcc -O2 -fopenmp "$work/head/benches/omp/wait_omp.c" -o "$work/wait_omp"
  omp="$work/wait_omp"
fi
arg() { # value of bench option $1 in "$@" after it, or $2
  local name=$1 def=$2; shift 2
  while [[ $# -gt 0 ]]; do [[ $1 == "$name" ]] && { echo "$2"; return; }; shift; done
  echo "$def"
}
omp_scenario=$(arg --scenario all "$@"); omp_widths=$(arg --widths "$(nproc)" "$@")
omp_calls=$(arg --calls 200 "$@"); omp_load=$(arg --load "" "$@")
IFS=';' read -r -a variants <<<"${HEAD_VARIANTS:-}"
echo "base $(git rev-parse --short "$base")  head $(git rev-parse --short "$head")  rounds $rounds  args $*" | tee -a "$log"
for r in $(seq 1 "$rounds"); do
  order=(base head); (( r % 2 == 0 )) && order=(head base)
  for side in "${order[@]}"; do
    if [[ $side == base ]]; then
      "$bin_base" --label "base#$r" --csv "$csv" "$@" >>"$log"
    else
      "$bin_head" --label "head#$r" --csv "$csv" "$@" >>"$log"
      if [[ -n $omp ]]; then
        for w in ${omp_widths//,/ }; do
          WAIT_OMP_SCENARIO=$omp_scenario WAIT_OMP_LOAD=$omp_load \
            "$omp" "$w" "$omp_calls" "$csv" "libgomp#$r" >>"$log"
        done
      fi
      for v in "${variants[@]}"; do
        [[ -z $v ]] && continue
        # shellcheck disable=SC2086
        "$bin_head" --label "head[$v]#$r" --csv "$csv" "$@" $v >>"$log"
      done
    fi
  done
done
echo "rows in $csv, waits in $log"
python3 - "$csv" <<'PY'
import collections, statistics, sys
# `param` may hold commas (`work=5us,sections=200`), unquoted: the fields around it are fixed.
cells = collections.defaultdict(list)
for line in open(sys.argv[1]).read().splitlines()[1:]:
    f = line.split(",")
    host, label, cpus, load, scenario, impl, width = f[:7]
    metric, median = f[-5], f[-4]
    param = ",".join(f[7:-5])
    name = label.split("#")[0]
    if impl not in ("threadteam", "libgomp"):
        name = f"{name} ({impl})"
    cells[(scenario, width, param, metric, name)].append(float(median))
print(f"{'scenario':<9} {'width':>5} {'param':<24} {'metric':<6} {'side':<40} {'n':>3} {'median of medians us':>21}")
for k in sorted(cells):
    v = cells[k]
    print(f"{k[0]:<9} {k[1]:>5} {k[2]:<24} {k[3]:<6} {k[4]:<40} {len(v):>3} {statistics.median(v):>21.1f}")
PY
