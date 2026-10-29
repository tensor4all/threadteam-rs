# Wait benchmark results

One CSV per host, written by `scripts/bench-waits.sh` (columns: host, label, cpus, load, scenario,
impl, width, param, metric, median_us, p90_us, max_us, overhead_us). `label` names the commit or
version measured; `load` is the foreign load of that pass (`none` or `KxP%`). Times are per call in
microseconds; `overhead_us` is the median minus the ideal `sections * work`.
