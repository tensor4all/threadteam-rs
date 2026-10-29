# Serial and parallel thresholds derived from the token's dispatch cost

Status: phase 2 implemented for `threadteam` and every gate of the faer fork (`exp/par-team-lifetime`, draft PR
tensor4all/faer#61); strided-rs, tcontract-rs, cpueinsum-rs and tlinalg-rs still use their own constants
(section 3.4). All numbers are from one Apple M5 Max (6 threads, QoS `UserInitiated`) on an interactive
laptop, so they are rough; final acceptance needs a quiet Linux host, including x86_64.

## Shape

A token states its own dispatch cost. Each kernel family states its own serial throughput. A kernel
runs a region in parallel when

```
estimated_serial_time(work)  >=  M * D(token, width)
```

with one fixed multiple `M` for the whole stack, `D` owned and measured in `threadteam`, and the
estimate `work * ns_per_unit` using one throughput constant per kernel family, owned by the kernel's
crate. The four constants listed in the issue (faer GEMM floor, LU `par_threshold`, strided
`MINTHREADLENGTH`, tcontract `WidthPolicy`) and the other faer thresholds in the inventory below
become derived values instead of separate tunables.

Data: [`thresholds-data/tables.md`](thresholds-data/tables.md) (every table), raw CSV and the host log in
[`thresholds-data/raw/`](thresholds-data/raw/), bench sources in
[`thresholds-data/src/`](thresholds-data/src/), the dispatch bench in
[`benches/dispatch_cost.rs`](../../benches/dispatch_cost.rs).

## 1. Inventory of thresholds

"Work" is what the comparison counts. Kinds of token: **T** team token, **P** plain token (outside or
inside its pool). A threshold written once applies to every kind of token today, which is the problem.

### faer (t4a-faer `exp/par-team-lifetime` at `7b60550`; paths under `faer/src/`)

Every gate is `pays(estimate, sections, width)` with `PAYS = 3` (section 3.1). The estimate is work times the
serial rate of the family (section 2.3), `sections` is counted in the code of the gate. A shape condition that
states a concern other than cost stays and is named. Token: **T** team, **P** plain; the rule reads the token.

| Where | Work compared | Estimate (rate) | Sections |
| --- | --- | --- | --- |
| `linalg/matmul/mod.rs` (x86_64 private route, aarch64 and generic `gemm` route), `utils/team.rs` (`gemm_tiled`) | `m*n*k` multiply-adds | GEMM rate of the dtype; the `gemm` crate keeps its own lower floor under it | 1 |
| `linalg/lu/partial_pivoting/factor.rs` trailing update | `cols * (b*b/2 + rows*b)` multiply-adds | GEMM rate | 2 (solve and product, the second on cold caches) |
| same, swap stage | `n * cols` row swaps | 0.55 ns per swap (memory bound) | 4 (provisional) |
| `linalg/lu/full_pivoting/factor.rs` per step | `(m-k-1)*(n-k-1)` elements | LU full pivoting rate | 1 |
| `linalg/cholesky/bunch_kaufman/factor.rs` (full pivoting) per step | `rem*rem` | Bunch-Kaufman rate | 1 |
| `linalg/qr/col_pivoting/factor.rs` per step | `(m-k-1)*(n-k-1)` | QR column pivoting rate | 1 |
| `linalg/qr/no_pivoting/factor.rs` | no gate of its own | delegates to the Householder and product gates | |
| `linalg/svd/bidiag.rs` per step | `(m-k-1)*(n-k-1)` | bidiagonalization rate | 2 (fused update or `y2`, and `z2`) |
| `linalg/evd/hessenberg.rs` (unblocked) per step | `n*n` of the trailing block | Hessenberg rate | 2 (two products) |
| `linalg/evd/tridiag.rs` per step | `rem*rem/2` | tridiagonalization rate | 1 |
| `linalg/householder.rs` `upgrade_householder_factor` (two sites) | `deep_rows * n * n` multiply-adds | GEMM rate | 1 |
| same, shape conditions | `deep_rows >= 2n` (reduction deep against its `n x n` output), `>= 4 * threads` (rows per task) | not a cost, kept | |
| `linalg/householder.rs` rank one apply (`N == 1`) | `M*K` target elements | rank one apply rate | 1 |
| same, row split | `M*N*K` multiply-adds; shape: `M >= 2K`, `M >= 4 * threads`, chunk covers `N` | GEMM rate | 1 |
| same, column split | `(1 or 2) * M*N*K` multiply-adds at the GEMM rate plus `M*K` target elements at the rank one apply rate (`cost::block_apply_estimate`), at width `n_tasks` | GEMM and rank one rates | 1 |
| `linalg/matmul/mod.rs` matvec (both layouts) | `nrows*ncols` | matvec rate | 1 |
| `linalg/matmul/internal/mod.rs` `spicy_matmul` | `nrows*ncols*depth` | GEMM rate | 1 |
| `linalg/matmul/triangular.rs` product with a triangular right side (a join) | `m*n*n/2` | GEMM rate | 1 per level, width 2 |
| same, product into a triangle (joins) | `n*n*k/2` | GEMM rate | 2 per level (outer join, join in its second half), width 2 |
| `linalg/triangular_solve.rs` (two sites) | `n*n*k/2`; the split for `k > 64 && n <= 128` is a shape rule, taken for every parallelism | GEMM rate | 1 (the join), width 2 |
| LLT, LDLT, Bunch-Kaufman with partial pivoting | no gate of their own | the product and solve gates | |
| `linalg/svd/bidiag_svd.rs` recursion join | `2 (n/2)^2` (two whole solves of half the size) | bidiagonal D&C rate per `m^2` | 1, width 2 |
| same, merge products (`update_v`, `update_u`, and their join when both run) | `(k^2 + rem^2) * n` multiply-adds (`update_v`), `((k+1)^2 + (rem+1)^2) * (n+1)` (`update_u`), the sum for the join of both | GEMM rate | 1 each, width 2 |
| `linalg/evd/tridiag_evd.rs` recursion join | `2 (n/2)^2` | tridiagonal D&C rate per `m^2` | 1, width 2 |
| same, merge products | `(n1^2 + (n-n1)^2) * n` multiply-adds | GEMM rate | 1, width 2 |
| `linalg/triangular_inverse.rs` join (lower and unit lower) | `2 (n/2)^3` | triangular inverse rate per `m^3` | 1, width 2 |
| `sparse/linalg/matmul.rs` numeric product | the symbolic flop count (`SparseMatMulInfo`, already computed) | sparse product rate per multiply-add | 1, width = degree (one flat section over the column ranges) |
| norm reductions (`norm_l2`, `norm_l2_sqr`, `norm_max`, `norm_l1`), `zip` and the elementwise kernels | no parallel region exists (no `par` argument, no Rayon call, serial pairwise SIMD reductions and loops) | nothing to gate | |

`ParThreshold { Derived, Fixed(n), Never }` is the one type of every `par_threshold` field of these families
(`Derived` is the default and is the table above; `Fixed(n)` is an explicit floor in the unit of the field).
The new join gates have no params field (nothing was configurable there before).

A join (`join_raw`, a split of the budget into two halves) is a section of **width 2**: its cost is the
dispatch cost at width 2 (team 3.0 us, plain outside 14 us on the M5 table), not at the full degree (1.4 and 19 us
at width 6). Evidence: a 64 x 64 triangular product on a team of 6 is 7.0 us serial and 8.1 us with the join at an
estimate of 4.8 us (`data/probes/20261026-qr128-regions/raw/trimm.csv`).

### strided-rs (main, `d24474d`)

| Where | Work compared | Value today | Token |
| --- | --- | --- | --- |
| `strided-perm/src/lib.rs:43` (`MINTHREADLENGTH`) | total elements of the output | `1 << 15 = 32768` | T, P |
| `strided-basic/src/threading.rs:106` (`parallel_threads_for_len`), `:115` (`for_len`), `:122` (`is_parallel`) | elements (`reduce`, `sum`: `dest_total*reduce_total`) | `len <= MINTHREADLENGTH` serial | T, P |
| `strided-basic/src/threading.rs:442`, `:517` (`ThreadedMapReduce`) | total elements, then `dims[i] > min(block, 1024)` for each split (`:327` to `:339`) | `32768`, then 2 way splits down to 1024 elements | T, P |
| `strided-basic/src/map_view.rs:861` and the callers of `is_parallel`/`for_len` in `ops_view.rs`, `map_view.rs`, `reduce_view.rs`, `update_view.rs`, `copy_plan.rs`, `gather_plan.rs`, `dense_update.rs`, `static_indexing_plan.rs`, `erased.rs:1167`, `:1305`, `:1381`, `erased/line.rs:232`, `kernel.rs:703` | same elements | same constant | T, P |
| `strided-perm/src/copy.rs:130` | elements of a copy | `<= MINTHREADLENGTH` runs on one task | T, P |
| `strided-fused/src/fused.rs:1529` | elements | `parallel_threads_for_len` | T, P |
| `strided-nn/src/softmax.rs:26` (`MIN_BYTES_PER_TASK`), `:23` (`GROUP_BYTES`) | bytes per task, `len*size_of::<T>() / 32 KiB` tasks | `32 KiB` | T, P |
| `strided-perm/src/hptt/execute.rs:115`, `:478` | budget only: tasks `min(budget, tiles)` | no work floor | T, P |

### tcontract-rs (main, `96d775a`)

| Where | Work compared | Value today | Token |
| --- | --- | --- | --- |
| `tcontract-exec/src/width.rs:27` to `29`, rule at `:40` to `:52` (`WidthPolicy`, `width_for`) | estimated serial ns; serial below `serial_below_ns`, else the `k` minimising `entry_base + entry_per_thread*k + ns/k` | `50 us`, `5 us`, `5 us` (35 us at k = 6) | T, P |
| `tcontract-exec/src/batch.rs:93` to `96`, `:182` to `189` (`DEFAULT_BATCH_POLICY`, `WorkModel::default`) | same `WidthPolicy`; `muladds_per_ns`, `min_work_per_lane_ns`, `min_items`, `min_items_per_lane` | `10`, `0`, `2`, `1` | T, P |
| `tcontract-exec/src/batch.rs:582` | tiny item rule `item_ns < serial_below_ns` keeps lanes | 50 us | T, P |
| `tcontract/src/plan/mod.rs:74` (`NS_PER_FLOP`), `:776` (`Plan::width`) | flops to ns | `0.05 ns/flop` (20 GFLOP/s) | T, P |

### cpueinsum-rs (main, `da71bb2`) and tlinalg-rs (main, `495e1f2`)

| Where | Work compared | Value today | Token |
| --- | --- | --- | --- |
| `cpueinsum/src/grouped.rs:705` to `708` | `2 * work * NS_PER_FLOP` through `Exec::width_for` | 50 us rule above | T, P |
| `cpueinsum-blas/src/lib.rs:80` (`DEFAULT_MIN_MACS`) | MACs of a GEMM step, BLAS against tcontract (not a parallel floor) | `1 << 15` | none |
| `tlinalg/src/batch.rs:832` to `851` | `item_muladds / muladds_per_ns * batch` through `Exec::width_for` | 50 us rule, 10 MAC/ns | T, P |
| `tlinalg/src/lib.rs:102` (`faer_par`) | width only: `Par::rayon(width)` | no floor of its own, faer decides | P |
| `tlinalg/src/householder.rs:46` | `rows*cols` blocking, not a parallel floor | `48*48` | none |

## 2. Measurements

Host protocol: before every process the runner
([`hostrun.py`](thresholds-data/hostrun.py)) polled until idle was at least 85 percent with no other
cargo, rustc or bench process, recorded idle, load and the top 3 CPU consumers before and after, and
marked the process busy when idle was below 85 percent before or after. The host table is at the end of
[`tables.md`](thresholds-data/tables.md). Idle before each process was 85.2 to 87.8 percent (the
desktop session and a browser renderer were the top consumers, not a benchmark). Idle after was below
85 percent for `dispatch-p3`, `faer-off-p3`, `faer-default-p1`, `stack-p2`, `stack-p3` and `lusweep-p1`;
those processes are marked busy and are included in the medians (the ratio of one cell between
processes varied by 15 percent at the median, up to 1.6 for tiny cells, so a break-even is located to
within one grid step). Run times: 55 s for a dispatch process, 47 s for a faer process, 51 s for a strided
and tcontract process, 33 s for the LU sweep; 12 processes, about 10 minutes of measurement in all.

Setup: `threadteam` `feat/team` at `6e87862` (path), t4a-faer `exp/par-team-lifetime` at `8f45c43`
(`std`, `rayon`, `threadteam` features, `gemm` 0.19), strided-rs `d24474d` and tcontract-rs `96d775a`
built against that `threadteam`, `rustc` 1.96.0. A pool of 6 threads and a team of width 6 (leader plus 5
members). Medians of 11 batches of at least 4 ms, 2 rounds per process, 3 processes per sweep.

Modes in every sweep: `seq`, `plain_out` (plain token, the call made from a thread outside the pool, one
install per call), `plain_in` (plain token, the loop runs inside `Exec::external`, so calls are made by a
pool worker), `team` (`Team::exec(6)`, faer through `Par::Team`).

To find where the parallel route itself starts to pay, the sweeps run with the floors removed: the
`gemm` threading threshold and the LU `par_threshold` set to 0 in faer (`thr-faer off`), `MINTHREADLENGTH`
lowered to 16 in a scratch checkout of strided-rs
([patch](thresholds-data/src/strided-minthreadlength.patch)), and a zero `WidthPolicy` in tcontract
(`tc_off`, forcing the full budget; `tc_default` keeps the policy). One further faer process keeps faer's
own floors (`faer-default`).

### 2.1 Dispatch cost (microseconds per section, median of 3 processes)

The `load` rows give every share 10 us of work and subtract it: the cost of getting work onto all
participants and waiting for them. An empty body can finish on the caller before another worker has
started, so the empty rows are a lower bound.

| token and state | w=2 | w=4 | w=6 |
| --- | --- | --- | --- |
| team, back to back, empty shares | 0.76 | 0.77 | 0.56 |
| team, back to back, **10 us shares** | **3.03** | **2.22** | **1.40** |
| team, after a 1 ms gap | 5.3 | 6.8 | 6.4 |
| team, after a 10 ms gap | 6.1 | 5.3 | 7.2 |
| team, after a 300 ms gap (members parked) | 27.7 | 27.4 | 26.6 |
| plain from outside, back to back, empty shares | 10.8 | 12.1 | 12.0 |
| plain from outside, back to back, **10 us shares** | **14.0** | **18.8** | **18.6** |
| plain from outside, after a 1 ms gap | 19.3 | 16.2 | 33.2 |
| plain from outside, after a 10 ms gap | 31.2 | 35.9 | 45.4 |
| `external` from outside, empty body, back to back | 11.5 | 10.7 | 12.2 |
| `external` from outside, empty body, after a 1 ms gap | 18.3 | 20.0 | 21.3 |
| plain in the pool, empty shares (the caller runs them itself) | 0.03 | 0.11 | 0.20 |
| plain in the pool, **10 us shares** | **6.1** | **10.8** | **5.6** |

The 10 us share rows are the dispatch cost used below. The 10 us share rows agree within 10 percent between
processes (the empty rows within 40 percent). The plain in pool value of 10.8 at width 4 is reproducible
in all three processes and unexplained. The team cost is not monotone in width (3.0 at 2, 1.4 at 6); the cause is not established
(candidates: which cores the first members land on, and the shares of a narrower token being served by
members that are the first to be woken).

### 2.2 Break-even sizes at 6 threads, floors removed

The first shape from which the mode stays below 0.95 times serial for every larger shape of the family,
with the serial time at that shape in brackets. `<` means the smallest shape swept already wins. Full
tables with every shape: [`tables.md`](thresholds-data/tables.md).

| family | team | plain, in pool | plain, outside |
| --- | --- | --- | --- |
| faer GEMM f64 square `n^3` | `40^3` (2.5 us); `32^3` is 0.95 | `64^3` (10 us) | `96^3` (33 us) |
| faer GEMM f64 tall, `m x 64 x 64` | `< 64` rows (`< 10 us`) | `< 64` rows (`< 10 us`) | 256 rows (38 us) |
| faer GEMM f64 shallow, `L x L x 16` | `L = 48` (1.5 us) | `L = 128` (9.5 us) | `L = 256` (36 us) |
| faer GEMM c64 square | `24^3` (2.3 us) | `48^3` (19 us) | `64^3` (45 us) |
| faer LU f64, all gates off | `n` 192 to 256 (150 to 320 us) | `n = 512` (2300 us) | `n = 512` (2300 us) |
| faer LU f64, faer's own gates | `n = 384` (910 us) | `n = 384` (910 us) | `n = 384` (910 us) |
| strided copy f64 | 32768 el. (2.9 us) | 65536 el. (5.8 us) | 524288 el. (53 us) |
| strided map `2x+1` | 32768 el. (3.1 us) | 65536 el. (6.4 us) | 262144 el. (34 us) |
| strided map `exp` | 4096 to 8192 el. (6 to 11 us) | 8192 el. (11 us) | 32768 el. (46 us) |
| strided `sum` | 32768 el. (2.7 us) | 65536 el. (5.4 us) | 262144 el. (21 us) |
| tcontract matmul, width policy off | `n = 96` to `128` (33 to 75 us) | `n = 96` (33 us) | `n = 96` (33 us) |
| tcontract matmul, default policy | `n = 96` (33 us) | `n = 96` (33 us) | `n = 96` (33 us) |

In serial time the break-even is 1.5 to 3 us for the team token on GEMM, copy, map and `sum` (1 to 2
times the 6 thread dispatch cost of 1.4 us; 6 to 11 us for `exp`), 6 to 19 us for a plain token in the
pool (1 to 3 times 5.6 us) and 33 to 53 us for a plain token from outside (2 to 3 times 18.6 us). The constants of
today do not follow this: faer's GEMM floor is about 22 us of serial work on every kind of token, and
`MINTHREADLENGTH` is 3 us of copy and 45 us of `exp`.

Findings that change the proposal:

1. **`MINTHREADLENGTH` is wrong in both directions.** At 32768 elements a plain token from outside is
   5.6 times slower than serial for a copy (16.3 us against 2.9 us) and 5.1 times for a map; the team
   token is at break-even. For `exp` the same 32768 elements are 46 us of work, 11 times past the team
   break-even. A count of elements cannot be right across kernels whose cost per element differs by 15
   times. The premise that today's constants were tuned for entering a pool does not hold for strided:
   the constant is Julia's, and it is only right for the team token on copy and map.
2. **The team route of faer ignores the dtype.** `utils/team.rs:280` compares `m*n*k` MACs with the real
   floor for complex products as well, where the `gemm` crate divides it by 16 for c64 on the plain
   route. With faer's own floors, c64 `48^3` and `64^3` (18 and 44 us serial) stay serial on a team (1.04
   and 1.02 times serial) while the plain token in the pool is 0.43 and 0.29 times serial.
3. **LU is a composite.** With all gates off a team LU opens 36 sections at `n = 192` (25 GEMM tiles and
   11 triangular solve joins, [`raw/sections.csv`](thresholds-data/raw/sections.csv)); the total
   overhead against one sixth of the serial time is about 110 us, 3 us per section, twice the 1.4 us
   of a section with 10 us shares. It loses at 128 (1.17 times serial), reaches break-even at 192 to 256 and wins from 384 (0.60).
   With faer's own gates at `n = 192` and `256` no GEMM section opens, but the swap stage
   (`factor.rs:183`, 16384 elements) opens one split, and the team LU is 1.08 and 1.02 times serial there
   (plain in pool 1.06 and 1.09): a parallel swap of 18 thousand elements costs more than it saves.
   Raising `par_threshold` alone does not fix it (it also gates the trailing update): at 65536 the team
   LU is 1.05 times serial at 384 and 512 and wins only at 768, at 262144 it never wins up to 768
   ([`tables.md`](thresholds-data/tables.md), `lusweep`).
4. **tcontract on a team token does not reach the team's dispatch cost.** `tc_off` at `n = 8` costs 7.9 us
   on a team token against 9.9 us on a plain token from outside (0.08 us serial): tcontract hands faer a
   width through `Exec::external`, which lends the members and costs about 8 us. From `n = 96` on the
   team token is no faster than the plain token (0.99 and 0.91 times serial at 96). The threshold for a
   team token can only drop where the consumer's parallel path actually uses sections (faer through
   `Par::Team`, strided through `for_ranges` and `fork`, both of which showed the gain above).
5. **tcontract's rate constant is 2.5 times pessimistic.** `NS_PER_FLOP = 0.05` against 0.0185 measured
   (0.037 ns per MAC, GEMM and tcontract agree: 10.2 us against 10.1 us at `64^3`). The existing
   `serial_below_ns = 50 us` is therefore about 20 us of real work, close to faer's floor and below the
   33 us break-even for a plain token from outside (the default policy measures 0.94 times serial at 96
   on a plain token from outside and 0.66 on a team).

### 2.3 Serial throughput constants (nanoseconds per unit, one thread)

Median of the one thread (`seq`) measurements. The first block is from the L2 resident shapes of the sweeps
([`tables.md`](thresholds-data/tables.md)). The faer blocks are from the dedicated program of faer-benchmark
`data/probes/20261026-threshold-recalibration/constants/` (the in-place routine of each family timed with
`Par::Seq`, shapes 96 to 384, 768 for matvec, primed 250 ms, median of at least 9 repetitions, two processes per
group, QoS `UserInitiated`; the unit is the one the gate counts and the time is the whole serial operation over the
sum of its per-step units, so a step includes its serial parts). Host protocol of that program: idle at least
85 percent and no other cargo, rustc, fbench or test process before each process, idle, load and the top 3
consumers recorded before and after, a process below 85 percent idle before or after is marked busy. Idle before
was 85.2 to 92.6 percent; 4 of 22 processes ended below 85 percent idle (74 to 78 percent for a few seconds,
cause not identified), among them both Hessenberg processes, whose values (range 0.42 to 0.53 for f64) are used
as measured and marked below.

The last faer rows (block apply streaming term, divide and conquer solves, triangular inverse, sparse product)
are from `faer-benchmark` `data/probes/20261026-qr128-regions/` (`consts/`, `raw/applyrate.csv`, `raw/apply.csv`):
the whole serial operation at one thread over shapes `m` 64 to 512 (block apply: `M` 24 to 512, `N` 2 to 32),
primed 250 ms, median of at least 15 repetitions, QoS `UserInitiated`, the same host protocol; 2 of the 9
constant processes (the bidiagonal SVD on Gaussian data and the triangular inverse with c64) are marked busy in
its `processes.log`; the first agrees with its clean rerun within 2 percent and the f64 inverse with the clean
run within 1 percent, the c64 inverse has no clean run.

| family | unit | f64 | c64 | range over shapes (f64) |
| --- | --- | --- | --- | --- |
| faer GEMM (and tcontract matmul) | MAC | 0.037 | 0.165 | 0.034 to 0.040 (`16^3` is 0.07) |
| faer LU f64 (partial pivoting, whole) | flop (`2/3 n^3`) | 0.030 | | 0.059 at 64, 0.038 at 128, 0.029 at 256, 0.021 at 768 |
| faer LU full pivoting step | element of the trailing block | 0.216 | 0.600 | 0.204 to 0.248 |
| faer Bunch-Kaufman (full pivoting) step | element `rem^2` | 0.238 | 0.380 | 0.205 to 0.263 |
| faer QR with column pivoting step | element of the trailing block | 0.202 | 0.713 | 0.187 to 0.232 |
| faer bidiagonalization step | element of the trailing block | 0.480 | 1.36 | 0.429 to 0.600 |
| faer Hessenberg step (unblocked), busy processes | element `n^2` | 0.451 | 1.57 | 0.419 to 0.535 |
| faer tridiagonalization step | element of the triangle `rem^2/2` | 0.339 | 1.16 | 0.299 to 0.422 |
| faer matvec, column major / row major | matrix element | 0.112 / 0.087 (used: 0.100) | 0.281 / 0.425 (used: 0.340) | 0.087 to 0.138 |
| faer rank one Householder apply | target element `M*K` | 0.212 | 0.725 | 0.204 to 0.223 |
| faer block apply, triangular product, triangular solve | MAC | 0.037, 0.039, 0.051 (use the GEMM rate) | 0.165, 0.176, 0.187 | |
| faer block apply, streaming term (with the GEMM term, `block_apply_estimate`) | target element `M*K` | 0.236 (the rank one rate 0.212 is used) | 0.538 (0.725 is used) | fit over `M` 24 to 512, `N` 2 to 32: median error 7 percent f64, 4 percent c64 |
| faer bidiagonal SVD divide and conquer, whole solve (both vectors) | `m^2` (time `45 m^2 + 0.023 m^3`) | 53 | not used (real types) | 47.6 to 61.7 over `m` 64 to 512 |
| faer tridiagonal EVD divide and conquer, whole solve (eigenvectors) | `m^2` | 31 | not used (real types) | 26.6 to 36.2 over `m` 64 to 512 |
| faer triangular inverse, whole serial inverse | `m^3` | 0.0096 | 0.0344 | 0.0081 to 0.0125 (`m` 128 to 512; 0.022 at 64) |
| faer sparse by sparse numeric product | symbolic multiply-add | 1.5 | 1.8 | 1.15 to 2.07 (six shapes, 4 to 40 entries per column, `n` 2000 to 20000) |
| strided copy f64 | element | 0.090 | | 0.065 to 0.105 |
| strided map `2x+1` f64 | element | 0.100 | | 0.073 to 0.129 |
| strided map `exp` f64 | element | 1.38 | | 1.36 to 1.39 |
| strided `sum` f64 | element | 0.081 | | 0.078 to 0.083 |

`f32` and `c32` are half of `f64` and `c64` (twice the lanes per register, half the bytes); only the GEMM was
checked (0.019 and 0.084). A scalar that is not native takes the `f64` or `c64` rate, a lower bound of its time.

## 3. Design

### 3.1 API in `threadteam` (implemented, `src/cost.rs`)

```rust
pub const PAYS: f64 = 3.0;
pub enum DispatchKind { Team, InPool, Outside }
pub struct WidthCosts;     // measured anchors: const new([Duration; 3]) (widths 2, 4, 6), const flat(Duration),
                           //   const from_anchors(&[(usize, Duration)]) -> Option<Self>, const MAX_ANCHORS = 12,
                           //   at(width), min()
pub struct DispatchCosts;  // new(team, in_pool, outside), host_default(), for_class(HostClass), with(kind, costs), cost(kind, width), min_cost(kind)
impl PoolBuilder { pub fn dispatch_costs(self, c: DispatchCosts) -> Self; }
impl Pool        { pub fn dispatch_costs(&self) -> DispatchCosts; }
impl Exec<'_> {
    pub fn dispatch_cost(&self, width: NonZeroUsize) -> Duration;  // MAX: serial or width < 2
    pub fn pays(&self, estimate: Duration, sections: u32, width: NonZeroUsize) -> bool;
    pub fn width_for(&self, estimate: Duration, sections: u32) -> NonZeroUsize;
}
impl TeamExec<'_> { dispatch_cost, pays, width_for, dispatch_costs() }
pub fn pays_at(estimate: Duration, sections: u32, dispatch_cost: Duration) -> bool;
pub fn width_for_by(estimate, sections, budget, cost: impl Fn(NonZeroUsize) -> Duration) -> NonZeroUsize;
```

The rule is `pays_at`: `estimate >= PAYS * sections * D`. `width_for` minimises `sections * D(k) +
estimate / k` over `1..=budget`, counting only widths at which the region pays, and returns 1 when none does
(ties go to the smaller width). The kind is read from the token and the calling thread when asked: team
authority (including `fork` children of a team token), else a worker of the pool (in pool), else outside. No
thread-local state, no allocation (`tests/dispatch_cost.rs`).

Lookups are cheap: a `WidthCosts` stores its measured anchors (integers, widths and nanoseconds) and `at` is a
forward scan over at most `MAX_ANCHORS` = 12 of them, so the lookup allocates nothing and `pays_at` /
`width_for_by` compare integer nanoseconds (the objective of `width_for_by` stays `f64`). `cost_lookup` bench on
cores 4-11 (previous phase, flat table whose `at` was an index): `at` 7.9 to 1.2 ns, `pays_at` after `at` 10.1 to
2.2 ns, `width_for` at budget 8 83 to 32 ns. The anchor scan costs an integer division per interpolated width where
the table held the value: on the 4-core run of issue #6 (cores 56-59, best of 9 x 2M calls, same bench file on both
trees, with the 64-bit fast path that falls back to 128 bits only when a product overflows) `at` 0.89 -> 4.12 ns,
`pays_at` after `at` 1.97 -> 7.84 ns, `width_for` budget 8 25.9 -> 61.8 ns and budget 64 222 -> 522 ns, `min_cost`
6.7 -> 4.8 ns, a by-value `DispatchCosts` copy 4.5 -> 11.0 ns (144 -> 384 bytes), `Exec::dispatch_cost` on a team
3.3 -> 6.3 ns. That stays far below one section, which the rows above charge 0.6 to 427 us, but a consumer that
calls `width_for` on every tiny region should use `min_cost` first. `min_cost(kind)` /
`WidthCosts::min()` is the cheapest cost over all widths: the minimum over the anchors (an interpolated value lies
between its two anchors, and the extrapolation never falls below the last anchor), so a region below `PAYS * min_cost` pays at no width.
A test compares the rows with the former interpolation for widths 1 to 6 and pins the extrapolation above them.

Width model: the value at the first anchor below it, linear interpolation between two anchors, and above the
**largest** anchor `w_m` linear extrapolation, `cost(w_m) * w / w_m` (issue #6). The largest anchor is 6 for the
rows built with `new` (widths 2, 4 and 6); the `x86_64-avx512` team row also carries the measured width-8 anchor,
and `from_anchors` accepts measured anchors up to `u16::MAX`, so a row measured on a many-core host replaces the
extrapolation where it has data. Above the last anchor the cost is never flat: a wide section is charged at least
in proportion to its width, so `width_for` no longer answers the full width just because the largest anchor pays (a
flat 1 us row gives 1, 2, 4 and 8 us at widths 6, 12, 24 and 48; a 24 us region on such a row selects width 12 of a
budget 64). Section 3.6 keeps the historical reason for the earlier flat model; section 3.7 is the `x86_64-avx2` team
row with anchors to width 64. A host that is not measured gets the
flat generic row (team 3, in pool 15, outside 40 microseconds at the anchors 2, 4 and 6, and the width proportion
above), which errs towards staying serial.

faer: `ParShape::Team(n, DispatchCosts)` carries the costs of its token (copied by `Par::shape`);
`ParShape::dispatch_cost(width)` and `pays(estimate, sections, width)` delegate to threadteam for team shapes.
`Rayon(n)` is entered from outside the pool by faer, so it uses the `Outside` cost of the host default table
(the same numbers are kept in `faer::utils::cost` for a build without the `threadteam` feature). `Seq` is
`Duration::MAX`.

Decisions of the lead:

* **`sections` is the real number of dispatches**, not tuned to the data. A kernel that enters the token twice
  declares 2. The strided kernels, which the data shows to carry a fixed overhead of about two sections on a
  team, will declare their real count when they are converted (later task); `exp` on a team token then follows
  from the count and not from a larger multiple.
* **Costs are steady state.** The wake penalty after the members parked (about 27 microseconds on a team; a
  plain token from outside after a gap of 10 ms costs 31 to 45) is documented here and is not part of the
  table or of `PAYS`. A caller with isolated sections raises its own estimate or calls `Pool::release_idle`.
* `PAYS = 3`: at the threshold a region that scales three times on six threads is 1.5 times faster and one
  that scales twice is not slower.

Steady-state defaults (Apple M5 Max, 6 threads, `UserInitiated`), microseconds, from the 10 us share rows:

| kind | w=2 | w=4 | w=6 | generic fallback |
| --- | --- | --- | --- | --- |
| team | 3.0 | 2.3 | 1.4 | 3 |
| plain in pool | 6.0 | 11 | 5.5 | 15 |
| plain outside | 14 | 18 | 19 | 40 |

### 3.2 Throughput constants (one per kernel family, in the crate that owns the kernel)

The values of section 2.3 are named constants with a reference to the bench that produces them. faer:
`utils::cost::ns_per_mac` (GEMM) and `utils::cost::rate` (`Rate { real, cplx }`: `LU_FULL_PIVOT`,
`BUNCH_KAUFMAN`, `QR_COL_PIVOT`, `BIDIAG`, `HESSENBERG`, `TRIDIAG`, `MATVEC`, `HOUSEHOLDER_RANK_ONE`,
`GEMM_MAC`, `BIDIAG_SVD_DC`, `TRIDIAG_EVD_DC`, `TRIANGULAR_INVERSE`, `SPARSE_MATMUL`), `NS_PER_SWAP = 0.55`. Planned for the other crates: `NS_PER_ELEM_COPY = 0.090`,
`NS_PER_ELEM_MAP = 0.100`, `NS_PER_ELEM_EXP = 1.38`, `NS_PER_ELEM_SUM = 0.081`; tcontract's `NS_PER_FLOP`
becomes 0.0185 and `muladds_per_ns` 27 (from 10). The throughput does not depend on the token, so one set serves
all kinds.

### 3.3 Resulting thresholds at 6 threads (`PAYS = 3`, `sections = 1`)

Minimum serial time: team 4.2 us, plain in pool 16.5 us, plain outside 57 us.

| work | today (every token) | team | plain in pool | plain outside | measured break-even (2.2) team / in pool / outside |
| --- | --- | --- | --- | --- | --- |
| GEMM f64, MACs | 589824 (`84^3`) | 113 thousand (`48^3`) | 446 thousand (`76^3`) | 1.54 million (`115^3`) | `40^3` / `64^3` / `96^3` |
| GEMM c64, complex MACs | team: 589824; rayon route: 36864 (`33^3`) | 25 thousand (`29^3`) | 100 thousand (`46^3`) | 345 thousand (`70^3`) | `24^3` / `48^3` / `64^3` |
| strided copy, elements | 32768 | 47 thousand | 183 thousand | 633 thousand | 33 thousand / 66 thousand / 524 thousand |
| strided map, elements | 32768 | 42 thousand | 165 thousand | 570 thousand | 33 thousand / 66 thousand / 262 thousand |
| strided `exp`, elements | 32768 | 3.0 thousand | 12 thousand | 41 thousand | 4 to 8 thousand / 8 thousand / 33 thousand |
| strided `sum`, elements | 32768 | 52 thousand | 204 thousand | 704 thousand | 33 thousand / 66 thousand / 262 thousand |
| tcontract matmul, MACs | est. 50 us at 0.05 ns/flop = 500 thousand (`79^3`) | 113 thousand, **only once team tokens reach `Par::Team` or sections** (finding 4) | 446 thousand | 1.54 million | `96` to `128` / `96` / `96` |

The derived values sit at or slightly above the measured break-evens (the margin is the multiple), except
that the team and in pool plain thresholds for GEMM are about two times above them, which costs a
little at the very bottom and is the price of not being slower anywhere. On a plain token from outside
the new GEMM threshold is 2.6 times today's floor, and the strided thresholds are 17 to 21 times
today's constant for copy, map and `sum` (a copy at 32768 elements is 5.6 times slower than serial today)
and 1.3 times for `exp`. On the team token the strided thresholds are 1.3 to 1.6 times today's constant
for copy, map and `sum` and 11 times lower for `exp`, and the GEMM threshold is 5.2 times lower.

Composite kernels (LU, QR, Cholesky, SVD and EVD steps): the rule applies per region, with the region's own work
at the family's own rate and the number of dispatches the region performs, in place of the per kernel
`par_threshold` constant; the field is the one type `ParThreshold { Derived, Fixed, Never }`, so an explicit
override is `Fixed`. The LU swap stage (memory bound, a split costs about 13 us on a team at 18 thousand elements)
declares `sections = 4` and the trailing update of an LU level `sections = 2` (solve and product, the second on
cold caches, from the 3 us per section measured in the LU at `n = 192`); both numbers are provisional, every
other count is read from the code. Derived floors at width 6 (`PAYS * sections * D / rate`, f64, team 1.4 us and
outside the pool 19 us of dispatch): full pivoting LU 19 thousand elements of the trailing block on a team and 264
thousand outside; column pivoting QR 21 thousand and 282 thousand; bidiagonalization (two sections) 17.5 thousand
and 237 thousand; tridiagonalization 12 thousand and 168 thousand (of the triangle); matvec 42 thousand and 570
thousand matrix elements. The previous constants were `192 * 256 = 49152` elements for the last four families on
every token.

Joins are sections of width 2 (3.0 us on a team, 14 us outside the pool, so 9 us and 42 us of estimated work):
the triangular inverse joins from `n` about 154 on a team and 260 outside the pool (`2 (n/2)^3` at 0.0096 ns), the
divide and conquer joins from `n` about 20 and 40 (`2 (n/2)^2` at 53 ns, f64), the triangular product joins at
`m*n*n/2` of 247 thousand multiply-adds on a team (`n` about 79) and 1.15 million outside. The numeric sparse
product is one flat section over the column ranges at the full degree: 2.8 thousand multiply-adds on a team,
38 thousand outside the pool (1.5 ns each). The block apply column split on a team of 6 starts at `M = 80` for a
basis of 8 columns (estimate 4.6 us, measured serial 4.7 us).

### 3.4 What changed and what remains

Done in `threadteam`: the API above, the defaults table, tests, `placement.md` and the README table.

Done in faer (`exp/par-team-lifetime`, `7b60550`): every gate of the inventory table of section 1.

* GEMM floors (`matmul/mod.rs`, `utils/team.rs`): `pays(estimate, 1, width)` with the GEMM rate of the dtype
  (`utils::cost::ns_per_mac`; the team route used to compare `m*n*k` with the real floor for complex products).
* LU (`partial_pivoting/factor.rs`): trailing update 2 sections, swap stage 4 sections (provisional).
* The other composite regions and the Householder, matvec, `spicy_matmul`, triangular product and triangular solve
  gates of the table: `ParThreshold::region_pays` or `par_pays` with the rate and the section count of the table.
  `ParThreshold { Derived, Fixed(n), Never }` is the type of every `par_threshold` field (`Fixed(n)` keeps the old
  floor in the unit the field documents; the C API maps 0 to `Derived` and `usize::MAX` to `Never`); the unused
  `QrParams::par_threshold` and its `faer-ffi` mirror are removed.
* Shape conditions that are not costs (`2n`, `4 * threads`, `M >= 2K`, the chunk covering the unit block, the
  `k > 64 && n <= 128` split) stay and are commented at the gate.
* Tests: the derived floors per family against the rule on the M5 table (`utils::cost::tests`), forced parallel
  steps of the six reductions equal the serial ones, the Householder, matvec, triangular product and solve
  parallel routes equal the serial ones above the floors (`householder::gate_tests`); `cargo test --release
  --lib` with `std rayon threadteam` and with default features passes except the three known `strict_triangle`
  cases.

* Joins (`utils::cost::join_pays`, one section of width 2 per join, `sections` counted per site): the recursion and
  merge joins of `bidiag_svd.rs` and `tridiag_evd.rs`, the join of `triangular_inverse.rs`, and (changed from the
  full degree) the joins of the triangular products and of the split triangular solve. The sparse numeric product
  (`sparse/linalg/matmul.rs`) is gated on its symbolic flop count and runs its column ranges as one range when
  the rule fails. The block apply column split is estimated as `2 M N K` multiply-adds at the GEMM rate plus `M K`
  target elements at the rank one apply rate (`cost::block_apply_estimate`): median error 7 percent over the
  measured bases, against a single GEMM rate that was 0.6 to 0.8 of the measured time at `N = 8`.
* The norm reductions, `zip` and the elementwise kernels have no parallel region (no `par` argument, no Rayon call),
  so there is nothing to gate; a parallel norm would be a new feature.
* `utils::cost::par_pays` and `ParThreshold::region_pays` return false for a sequential token before computing the
  estimate (a `128 x 128` QR asks 500 times).
* Probe feature `par-trace`: `utils::cost::trace` records every gate decision (site, dimensions, units, sections,
  estimate, dispatch cost, decision).

Remaining work in faer: the f32 and c32 rates of the families other than the GEMM are halved, not measured. The
Hessenberg constant is from processes marked busy. The rate of a family falls with the size (the triangular inverse
is 0.022 ns per `m^3` at 64 and 0.0095 at 256), so a model with one constant under estimates the halves of a small
join (5.0 us against 11.6 us at `n = 128`); a size dependent rate would be a second constant per family.
`ParShape::pays` has no kind of section for a split of a team (the LU swap stage splits cost about 13 us), so the
width 2 cost of a join is the best available number, not a measured cost of `Team::split`.

Remaining work in the other repositories (later task):

| repository | change |
| --- | --- |
| strided-rs | `MINTHREADLENGTH` replaced by `exec.pays(est, sections, width)` per kernel with the constants of 3.2 and the real section count of each kernel (a kernel that enters twice declares 2); softmax's `MIN_BYTES_PER_TASK` in time too |
| tcontract-rs | `WidthPolicy` removed in favour of `Exec::width_for`; `NS_PER_FLOP` 0.0185; hand faer `Par::Team` on a team token instead of `external` (`external` on a team token costs 8 to 9 us, it lends the members) |
| cpueinsum-rs, tlinalg-rs | follow tcontract; `faer_par` builds `Par::Team` when the token carries a team |

For those crates: use `Exec::pays` and `Exec::width_for` (they read the token kind at call time, so a function
that may be called from inside or outside the pool needs no branch of its own); take the throughput constant
from section 3.2; `TeamExec` has the same three methods; `Par::shape()` of a team in faer carries the pool's
costs, so a `PoolBuilder::dispatch_costs` override reaches faer's team decisions but not its Rayon ones.

### 3.5 Rough confirmation of the faer floors (M5, 6 threads)

`faer-benchmark` `bench/par-team-lifetime`, `data/probes/20261026-dispatch-cost-floors-m5-6t/` (two processes
per cell, no busy process, new floors against the previous revision; times are the ratio to one thread serial):

| cell | before | now |
| --- | --- | --- |
| GEMM f64 `64^3`, team | 1.00 | 0.23 |
| GEMM c64 `32^3`, `48^3`, `64^3`, team | 1.07, 1.07, 1.06 | 0.31, 0.21, 0.21 |
| GEMM c64 `48^3`, plain from outside | 1.73 | 1.07 (serial, run to run noise) |
| GEMM f64 `96^3`, plain from outside | 1.27 | 1.07 (serial, noise) |
| LU f64 128, 192, 256, 384, 512, team | 1.07, 1.02, 1.01, 0.59, 0.50 | 0.98, 0.79, 0.67, 0.51, 0.45 |
| LU f64 128, 192, 256, 384, 512, plain | 1.02, 1.14, 1.03, 0.86, 0.68 | 1.00, 0.98, 0.95, 0.80, 0.63 |

Serial rates on one thread: f32 0.019 and c32 0.084 ns per MAC, within 10 percent of the halved f64 and c64
constants used for them.

Remaining regions (`data/probes/20261026-threshold-recalibration/`, f64, 6 threads, four processes per build at 6
threads and three serial, none busy; ratio to one thread serial, before is t4a-faer `9cd64f0`, after `6788b20`;
times in microseconds are in the probe README):

| cell | plain before | plain after | team before | team after |
| --- | --- | --- | --- | --- |
| QR factorization 128, 256, 512 | 2.49, 1.99, 1.22 | 1.01, 0.80, 0.48 | 0.72, 0.61, 0.41 | 0.85, 0.58, 0.41 |
| Cholesky 128, 256, 512 | 1.00, 0.96, 0.57 | 1.00, 1.00, 0.60 | 1.00, 0.68, 0.41 | 1.00, 0.68, 0.41 |
| SVD (values) 128, 256, 512 | 0.82, 1.05, 1.09 | 0.82, 0.87, 0.88 | 0.82, 1.05, 1.09 | 0.82, 0.87, 0.88 |
| EVD (eigh) 128, 256, 512 | 1.24, 0.81, 0.69 | 1.01, 0.81, 0.66 | 1.23, 0.79, 0.69 | 1.01, 0.81, 0.66 |

The plain token is no longer slower than serial at 128 and 256 for QR and at 128 for the eigensolver. QR at 128 on
a team was worse than before (86 us to 102 us, serial 120 us); the cause is established below.

QR 128 on a team and the join gates (`data/probes/20261026-qr128-regions/`, same host and protocol, 2 to 8 passes
per cell, many of them marked busy by the desktop session; ratio to one thread serial, before is `6788b20`, after is
`7b60550`; microseconds and every raw file are in the probe README):

| cell | plain before | plain after | team before | team after |
| --- | --- | --- | --- | --- |
| QR 128, 256, 512 | 1.02, 1.14, 0.68 | 1.02, 1.16, 0.71 | 0.82, 0.61, 0.41 | 0.80, 0.62, 0.47 (check: 0.41 to 0.47, noise) |
| SVD (U, V) 128, 256, 512 | 0.88, 0.88, 0.75 | 0.83, 0.87, 0.76 | 0.76, 0.61, 0.44 | 0.76, 0.63, 0.42 |
| EVD (eigenvectors) 128, 256, 512 | 0.98, 0.86, 0.67 | 0.99, 0.85, 0.66 | 0.97, 0.56, 0.39 | 0.98, 0.58, 0.39 |
| triangular inverse 128, 256, 512 | 6.62, 1.67, 0.96 | 1.00, 1.00, 0.90 | 0.81, 0.48, 0.29 | 1.04, 0.52, 0.31 |
| sparse product (4 per column) 500, 2000, 8000 | 15.0, 3.09, 0.83 | 1.02, 1.01, 0.84 | 0.44, 0.27, 0.25 | 0.47, 0.28, 0.25 |

Cause of QR 128 (evidence: a decision trace of the 512 gate decisions of one factorization, and forced parallel
against serial timing of every block apply it contains): of those gates only the column split of the block apply
(`N = 8`, `M = 56` to 128) changed its decision between `9cd64f0` (10 of 59 parallel) and `6788b20` (5). Its estimate
(`2 M N K` multiply-adds at the GEMM rate) was 0.58 to 0.77 of the measured serial time because the apply also
streams its target; with the GEMM term plus a streaming term at the rank one apply rate it is 0.85 to 1.05, and the
split starts at `M = 80` (7 of 59 parallel), 100 us against 105 us in one process. The three regions left serial
(`M` 56, 64, 72) gain 1.1, 0.9 and 2.0 us when forced parallel and have estimates of 2.2, 2.9 and 3.7 us against the
rule's 4.2 us: that is the margin of `PAYS = 3` (break-even is about 1.2 times the dispatch at width 6), the rule is
kept, and 4 us of the remaining distance to the 87 to 94 us of `9cd64f0` is theirs. `sections = 1` was right.

Norm and `zip` cases are not in the table: they have no parallel region.

### 3.6 Linux row (x86_64, quiet host, tensor4all-meta#51)

`faer-benchmark` `bench/awake-team-linux`, `data/probes/20261026-awake-team-linux/` (README and `tables.md` there):
AMD Ryzen AI 9 HX 470 (12 cores, two L3 domains), Linux, **boost off** (every core at 2.0 GHz), governor and EPP
`performance`, measured on cores 4-11 with the observer on the other L3 domain, idle guard on the whole measured
domain, 101 processes all accepted at the first attempt. Same benches and the same 10 us share rows as section 2.1,
pool of 8, median of 3 processes, microseconds per section:

| kind | w=2 | w=4 | w=6 | w=8 |
| --- | --- | --- | --- | --- |
| team | 2.8 | 2.2 | 1.7 | 1.1 |
| plain in pool | 3.1 | 3.6 | 3.5 | 3.0 |
| plain outside | 9.5 | 10.1 | 8.8 | 6.7 |

After the per-member parker fix (see "Width dependence" below; same bench, pool of 8, 10 us share rows, median of 3
processes, all accepted; `tables-dispatch-exec2.md`), the team row is 0.6, 0.7, 0.9 us at widths 2, 4, 6 and 0.98 at
w=8; in pool 3.0, 3.6, 3.5 and outside 9.7, 10.1, 8.7 were re-measured and are essentially unchanged. The table
above predates the parker fix; `for_class(HostClass::X86_64Avx512)` holds the post-fix row. The team row carries all
four measured widths as anchors, so width 8 keeps the measured 0.98 us; in pool and outside keep the widths of the
post-fix run only (their width-8 columns are from the pre-fix run and are not used), so the model extrapolates
there: width 8 is 4.67 us in the pool and 11.6 us from outside.

| kind | w=2 | w=4 | w=6 | w=8 |
| --- | --- | --- | --- | --- |
| team (post-fix) | 0.6 | 0.7 | 0.9 | 0.98 |
| plain in pool | 3.0 | 3.6 | 3.5 | |
| plain outside | 9.7 | 10.1 | 8.7 | |

The row is `DispatchCosts::for_class(HostClass::X86_64Avx512)` in `cost.rs`, and `host_default()` returns it on a host whose runtime detection finds `avx512f` (`x86_64-avx2` has its own team row since section 3.7; `x86_64-scalar` takes the generic row until measured). The row and the rates of the consumer crates for the same class come from this machine at 2.0 GHz (boost off); boost on moves both by about 1.66x with a stable ratio (`raw-exec3-boost`, `tables-*-exec3-boost.md`):

```rust
HostClass::X86_64Avx512 => DispatchCosts {
    team: WidthCosts::from_anchors(&[
        (2, us_f(0.6)), (4, us_f(0.7)), (6, us_f(0.9)), (8, us_f(0.98)),
    ]).unwrap(),
    in_pool: WidthCosts::new([us_f(3.0), us_f(3.6), us_f(3.5)]),
    outside: WidthCosts::new([us_f(9.7), us_f(10.1), us_f(8.7)]),
},
```

What the Linux data says about the design:

* The team cost is the M5's (1.1 to 2.8 us); a plain token costs about half the M5's. Parked members are dearer:
  a team after a 300 ms gap costs 82 to 84 us (M5 27), the first share starts 63 to 83 us after a park.
* The width model above the last anchor was 2 times pessimistic when it grew in proportion to the width; the flat
  model that replaced it (section 3.1) had the opposite error and was 8 % optimistic at width 8 on the post-fix team
  row (0.9 against 0.98 us). Since issue #6 the model grows in proportion above the largest anchor again, with the
  measured width-8 team anchor covering width 8 exactly. The historical numbers: width 8 measures 1.11 us on a team,
  3.0 in the pool and 6.7 outside (pre-fix run) against 2.21, 4.69 and 11.7 from `cost(6) * 8 / 6`. With the generic
  row the extrapolation refuses a team GEMM of `64^3` at width 8 (estimate 9.7 us against `3 * 4.0`) that pays at
  width 4 (0.32 of serial).
* Every gate measured follows `estimate >= PAYS * sections * D` exactly; the deviations are in the inputs: the
  generic table on Linux, the M5 serial rates (the 2.0 GHz rates here are 1.4 to 3.7 times larger: GEMM 0.064 ns
  per MAC f64, 0.257 c64, 0.032 f32, 0.128 c32), and strided's class assignment (a contiguous `copy_into` and every
  opaque `map_into` closure, `exp` included, are gated as `Map`).
* The triangular inverse rate is size dependent on Linux too (0.047 ns per `m^3` at the smallest shape, median
  0.019).
* QR 128 on a team is 1.06 times the held row at 8 threads and 0.99 at 4: no case for changing `PAYS = 3`.

#### Width dependence of a team section (root cause and fix)

The table above falls with width (2.8 to 1.1 us), the opposite of a plain token. Cause: all reserved members of a
pool slept on one pool-wide `Parker`, and every publish ended with `notify_all` on it. A member that parked
(after the 200 ms park delay, or in the pool's idle gap) and was *not* addressed by a narrower section was woken
by each publish, found its generation unchanged and went back to sleep: a lock, a `futex` wake of every sleeper
and the sleepers' re-lock, per section, growing with the number of unused members (6 for `w=2` of 8; none for
`w=8`). A narrower token is not a split (`Team::exec(w)` takes the first `w - 1` reserved members and
`spread_raw` publishes to those only), so there was no split bookkeeping, no extra barrier and no shared counter
involved; `strace -f -c` showed ~440k `futex` calls in 2 s. Fix (`src/awake.rs`): one `Parker` per member slot;
`publish` bumps the targeted generations and wakes the targeted slots' parkers only; `release` / `lend` wake the
slots they changed. Loom: `a_wake_of_one_slot_leaves_the_other_asleep_until_its_own_change`.

Pool of 8, `dispatch_cost` bench (`DISPATCH_TEAM_ONLY=8`), cores 4-11, microseconds per section, before ->
after (two runs each, the two agree within 5 %):

| row | w=2 | w=4 | w=6 | w=8 |
| --- | --- | --- | --- | --- |
| hot, empty shares | 0.67-1.35 -> 0.47 | 1.39 -> 0.59 | 1.31 -> 0.81 | 1.14 -> 0.97 |
| after 1 ms | 5.2 -> 0.84 | 3.8 -> 1.0 | 2.4 -> 1.07 | 1.33 -> 1.23 |
| 10 us shares (minus 10) | 2.8 -> 0.54 | 2.24 -> 0.61 | 1.7 -> 0.77 | 1.06 -> 0.92 |

The cost now rises with width like the plain tokens, and the full team is not slower (the wake checks run after
all generations are bumped, so a spinning member starts first). Evaluated and rejected: running a narrower
section on the whole team with empty shares for the members outside the width costs the `w=8` row at every width
(0.97 / 1.23 / 0.92) against 0.47 to 0.81 for the targeted publish, wakes members the section does not need, and
needs the ledger and `Participant::size` to exclude the empty shares to keep "a width-k token never runs more
than k tasks of real work". the `x86_64-avx512` row now holds the post-fix row (0.6, 0.7, 0.9), re-measured by the Linux
campaign.

### 3.7 Zen2 team row (x86_64-avx2, threadteam-rs#6)

`for_class(HostClass::X86_64Avx2)` carries a measured team row with ten anchors. Host `broccoli`: AMD EPYC 7702P
(Zen2, 64 physical cores, SMT off, NUMA 1, 16 MiB L3 per 4-core CCX), `cpufreq/boost = 0`, governor `performance`,
nothing else running. This crate's `benches/dispatch_cost.rs` (the Linux port, unchanged) built against threadteam
0.1.0, `TT_POOL=64`, `TT_WIDTHS=2,4,6,8,12,16,24,32,48,64`, process pinned to CPUs 0-63, 2026-10-29. The row is
`team/load10us-minus-10us`, the same quantity as the other rows (back to back sections whose shares carry 10 us of
work, minus the 10 us), rounded to the nearest nanosecond. Raw output: `thresholds-data/raw/broccoli-dispatch-20261029.csv`.

| width | 2 | 4 | 6 | 8 | 12 | 16 | 24 | 32 | 48 | 64 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| team, us | 1.42 | 1.99 | 2.33 | 2.74 | 3.56 | 4.08 | 4.29 | 4.84 | 6.40 | 9.33 |

The cost is sublinear in the width (about `2.33 * (w / 6)^0.45` between 6 and 64), so the linear extrapolation above
the largest anchor stays conservative against it; with anchors to 64 a 64-core host reads measured values at every
width it can use.

What the row does not claim:

* **Mixed provenance.** For `x86_64-avx512` and `aarch64-apple` the dispatch row and the consumer crates' rates of
  the class come from one machine in one clock state. The consumer crates (faer, tcontract-rs, strided-rs) still
  hold their generic rates for `x86_64-avx2`, so on this class the gate compares a measured dispatch cost with a
  conservative rate. It is a heuristic exception, recorded here, until those rates are measured on the same host.
* **One class, many hosts.** `x86_64-avx2` covers every AVX2 + FMA host, small desktops included. The width 2 team
  floor moves from the generic 9 us (`PAYS * 3 us`) to 4.3 us. The row is a representative default for the class, not
  a guarantee for a given topology; a pool owner with its own measurement overrides it with
  `PoolBuilder::dispatch_costs`.
* **Plain kinds unchanged.** In pool and outside deliberately keep the generic row (15 and 40 us at the anchors). The
  run has plain-token numbers too (in the CSV), but they are a control the measurement was not designed for and are
  not promoted to the row.
* **Fresh team entry.** tensor4all/faer#91 saw, on the same host, an intermittent section of about 0.62 ms when a
  caller enters a fresh `Team::exec(w).team(...)` per call; the anchor instrument reuses one token and never shows it.
  Its cause is not attributed. The row is the steady-state cost of a reused token and does not include it.

## 4. Limits of this data

* One host, rough, 6 threads only (widths 2 and 4 only for the dispatch cost), one process per mode
  and shape and per round, the desktop session running (see the host table). Linux x86_64 is measured in
  section 3.6 (boost off); the table is keyed by host class so that a Linux table can be added without an API
  change.
* The dispatch cost is measured with 10 us shares; a 1 us share would show a different join time.
* The break-even is located on a grid (for GEMM 8 to 16 points per decade); thresholds in section 3.3
  are derived, not fitted, and are checked against the grid only.
* The `seq` mode and the parallel modes run in separate phases of a process; at shapes where both run
  the same sequential kernel (for instance `gemm_sq 32^3`) they differ by up to 5 percent, so ratios
  between 0.95 and 1.05 are not significant.
* The LU `sections` multiples, the team token inside tcontract and the c64 team floor are findings or
  estimates to be verified when implemented, not results. The rates of the faer reductions include the serial
  parts of a step and are measured on L2 resident shapes; the QR 128 team cell shows that small shapes can sit
  above them.
