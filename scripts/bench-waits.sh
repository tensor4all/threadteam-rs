#!/usr/bin/env bash
# Runs the wait benchmarks (docs/design/waiting.md, README "Wait benchmarks") for one checkout:
# the Rust bench (threadteam and Rayon) and, when `gcc -fopenmp` works, the libgomp driver, with
# and without foreign load. Rows are appended to $CSV.
#
#   scripts/bench-waits.sh <label> [widths] [load K:P]
#
# label   the version or commit under test, written to every row (e.g. `main-4c73c2b`)
# widths  comma separated (default: all CPUs of the affinity set and three quarters of them)
# load    foreign load for the second pass (default 8:10, eight processes at 10 % each);
#         `none` skips the loaded pass
# CSV     output file (default docs/design/waiting-data/<host>.csv)
set -euo pipefail
label=${1:?label}
widths=${2:-}
load=${3:-8:10}
host=$(hostname -s)
csv=${CSV:-docs/design/waiting-data/$host.csv}
mkdir -p "$(dirname "$csv")"
cargo build -j "${JOBS:-4}" --release --bench wait_scenarios >/dev/null
bench=$(ls -t target/release/deps/wait_scenarios-* | grep -v '\.d$' | head -1)
wargs=()
[[ -n $widths ]] && wargs=(--widths "$widths")

omp=""
if echo 'int main(void){return 0;}' | gcc -fopenmp -x c - -o /dev/null 2>/dev/null; then
  gcc -O2 -fopenmp benches/omp/wait_omp.c -o target/wait_omp
  omp=target/wait_omp
else
  echo "gcc -fopenmp not available: libgomp reference skipped" >&2
fi
cpus=$(nproc)
ws=${widths:-$cpus,$((cpus * 3 / 4))}

run_pass() { # $1 = load spec or none
  local largs=()
  [[ $1 != none ]] && largs=(--load "$1")
  "$bench" --label "$label" --csv "$csv" "${wargs[@]}" "${largs[@]}"
  if [[ -n $omp ]]; then
    local pids=()
    if [[ $1 != none ]]; then
      for _ in $(seq "${1%%:*}"); do "$bench" --load-child "${1##*:}" & pids+=($!); done
      sleep 0.2
    fi
    for w in ${ws//,/ }; do WAIT_BENCH_LOAD=$([[ $1 == none ]] && echo none || echo "${1%%:*}x${1##*:}%") \
      "$omp" "$w" 200 "$csv" "$label"; done
    [[ ${#pids[@]} -gt 0 ]] && kill "${pids[@]}" 2>/dev/null; wait 2>/dev/null || true
  fi
}
run_pass none
[[ $load != none ]] && run_pass "$load"
echo "rows appended to $csv"
