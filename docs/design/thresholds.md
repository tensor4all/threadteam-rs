# Serial and parallel thresholds derived from the token's dispatch cost

Status: phase 2 implemented for `threadteam` and the faer fork (`exp/par-team-lifetime`, draft PR
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

### faer (t4a-faer `exp/par-team-lifetime`, measured at `8f45c43`; paths under `faer/src/`)

| Where | Work compared | Value today | Token |
| --- | --- | --- | --- |
| `linalg/matmul/mod.rs:1442` to `1465` (x86_64 private GEMM route) | `2 or 8 * m*n*k` flops against `2 * gemm::get_threading_threshold()` | `m*n*k >= 589824` (real), `>= 147456` (complex) | P |
| `gemm-common-0.19.0/src/gemm.rs:109`, `:398` to `:407`, `:516` (aarch64 and generic route, dependency) | `m*n_chunk*k_chunk` MACs, threshold divided by 4 for c32 and 16 for c64 | `48*48*256 = 589824` (real), `36864` (c64) | P |
| `utils/team.rs:280` (`gemm_tiled`, team route on every arch) | `m*n*k` MACs, no scaling by dtype | `>= 589824` for real and complex | T |
| `linalg/matmul/mod.rs:774`, `:894` (matvec variants) | `nrows*ncols` of the matrix | `< 256*256` stays serial | T, P |
| `linalg/matmul/internal/mod.rs:72` to `77` | `m*n*k` MACs | `> 32*32*32 = 32768` | T, P |
| `linalg/matmul/triangular.rs:125` | `n*n*m` | `< 128*128*64` serial | T, P |
| `linalg/matmul/triangular.rs:694` | `n*n*k` | `< 128*128*128` serial | T, P |
| `linalg/lu/partial_pivoting/factor.rs:137` (default at `:274`) | trailing update area, `2 or 8 * rows*cols` against `2 * par_threshold` | `par_threshold = 128*128 = 16384` | T, P |
| `linalg/lu/partial_pivoting/factor.rs:183` | swap stage `m*(ncols-n)` elements | `> 16384`, not scaled by dtype | T, P |
| `linalg/lu/full_pivoting/factor.rs:372` (default at `:446`) | `(m-k-1)*(n-k-1)` per step | `par_threshold = 256*512` | T, P |
| `linalg/cholesky/bunch_kaufman/factor.rs:328` (default at `:911`) | `rem*rem` per step | `par_threshold = 128*128` | T, P |
| `linalg/qr/col_pivoting/factor.rs:244` (default at `:106`) | `(m-k-1)*(n-k-1)` per step | `par_threshold = 192*256` | T, P |
| `linalg/qr/no_pivoting/factor.rs:126` | declared, not read | `192*256` | none |
| `linalg/svd/bidiag.rs:108` (default at `:31`) | `(m-k-1)*(n-k-1)` per step | `par_threshold = 192*256` | T, P |
| `linalg/evd/hessenberg.rs:291` (default at `:20`) | `n*n` per step | `par_threshold = 192*256` | T, P |
| `linalg/evd/tridiag.rs:322` (default at `:17`) | `rem*rem/2` per step | `par_threshold = 192*256` | T, P |
| `linalg/householder.rs:160` | rows below the basis, `n`, threads, `deep_rows*n*n` | `deep_rows >= 2n`, `>= 4*threads`, `deep_rows*n*n >= 8192` | T, P |
| `linalg/householder.rs:224`, `:478`, `:609` | same family: `rows*n*n`, `nrows*ncols`, `M*K` | `>= 8192` (with the `2n` and `4*threads` shape rules) | T, P |
| `linalg/triangular_solve.rs:439`, `:520` | right hand sides `k` and order `n` (a split, not a work floor) | `k > 64 && n <= 128` | T, P |
| `sparse/linalg/matmul.rs:115`, `:171` | none | no floor, width taken as given | T, P |

The `par_threshold` fields in `bidiag`, `col_pivoting`, `hessenberg` and `tridiag` count elements of a
trailing submatrix, so they are time floors of different sizes per kernel (a rank-one update, a
Householder pass, a matrix-vector product). The data below shows that one number cannot be right for
both kinds of token.

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

Median of the `seq` mode, from the L2 resident shapes of the sweeps ([`tables.md`](thresholds-data/tables.md)).

| family | unit | ns per unit | range over shapes |
| --- | --- | --- | --- |
| faer GEMM f64 (and tcontract matmul) | MAC | 0.037 | 0.034 to 0.040 (`16^3` is 0.07) |
| faer GEMM c64 | complex MAC | 0.165 | 0.162 to 0.172 |
| faer LU f64 | flop (`2/3 n^3`) | 0.030 | 0.059 at 64, 0.038 at 128, 0.029 at 256, 0.021 at 768 |
| strided copy f64 | element | 0.090 | 0.065 to 0.105 |
| strided map `2x+1` f64 | element | 0.100 | 0.073 to 0.129 |
| strided map `exp` f64 | element | 1.38 | 1.36 to 1.39 |
| strided `sum` f64 | element | 0.081 | 0.078 to 0.083 |

## 3. Design

### 3.1 API in `threadteam` (implemented, `src/cost.rs`)

```rust
pub const PAYS: f64 = 3.0;
pub enum DispatchKind { Team, InPool, Outside }
pub struct WidthCosts;     // values at widths 2, 4, 6: new([Duration; 3]), flat(Duration), at(width)
pub struct DispatchCosts;  // new(team, in_pool, outside), host_default(), apple_m5(), generic(), with(kind, costs), cost(kind, width)
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

Width model: linear interpolation between the widths 2, 4 and 6, the width 2 value below, and above width 6
the width 6 value grown in proportion to the width. A host that is not measured gets the flat generic table
(team 3, in pool 15, outside 40 microseconds), which errs towards staying serial.

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

The values of section 2.3 become named constants with a reference to the bench that produces them
(`NS_PER_MAC_F64 = 0.037`, `NS_PER_MAC_C64 = 0.165`, `NS_PER_FLOP_LU = 0.030`, `NS_PER_ELEM_COPY = 0.090`,
`NS_PER_ELEM_MAP = 0.100`, `NS_PER_ELEM_EXP = 1.38`, `NS_PER_ELEM_SUM = 0.081`). tcontract's
`NS_PER_FLOP` becomes 0.0185 and `muladds_per_ns` 27 (from 10). The throughput does not depend on the
token, so one set serves all kinds.

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

Composite kernels (LU, QR, Cholesky, SVD and EVD steps): the rule applies per region, with the region's
own flops or bytes and one dispatch cost, in place of the per kernel `par_threshold`; the field stays as
an explicit override. Measured against that, two rules need their own multiple: the swap stage of LU
(memory bound, a split costs about 13 us on a team at 18 thousand elements) is treated as a region with
`sections = 4` so that it needs about 17 us of serial work, and regions inside a factorisation that
run on cold caches (every region after the first) use `sections = 2`, from the 3 us per section
measured in the LU at `n = 192`. These two numbers are provisional; the rough M5 check below does not
contradict them, and acceptance is on a quiet Linux host.

### 3.4 What changed and what remains

Done in `threadteam`: the API above, the defaults table, tests, `placement.md` and the README table.

Done in faer (`exp/par-team-lifetime`):

* The GEMM floors of `matmul/mod.rs` (x86_64 private kernel and the aarch64 and generic `gemm` crate route)
  and `utils/team.rs` (`gemm_tiled`) are `pays(estimate, 1, width)` with the serial throughput of the dtype
  (`utils::cost::ns_per_mac`: f64 0.037, c64 0.165 per complex MAC; f32 and c32 are those halved, documented
  scaling, see the worklog for the one-thread check). This fixes the team route, which compared `m*n*k`
  against the real floor for complex products too. Derived floors at width 6 (`PAYS * D`: team 4.2 us,
  outside 57 us): f64 113 thousand MACs on a team (`48^3`) and 1.54 million outside (`115^3`); c64 25 thousand
  (`29^3`) and 345 thousand (`70^3`). The `gemm` crate keeps its own floor under faer's, which is lower than
  faer's on every kind of token that reaches it.
* LU (`partial_pivoting/factor.rs`): `par_threshold` is a per-region rule. `PartialPivLuParams::par_threshold
  == 0` (the new default) derives each region from its serial time. The trailing update (triangular solve and
  product, `cols * (b*b/2 + rows*b)` MACs at the dtype's GEMM rate) declares `sections = 2`; the swap stage
  (`n * cols` row swaps at 0.55 ns each, measured one thread, random pivots, cache resident) declares
  `sections = 4`. Both counts are provisional and stay separate rules. Any other value of `par_threshold`
  restores the old fixed floors (`work > 2 * par_threshold` for the update, `m * cols > par_threshold` for the
  swap) and `usize::MAX` disables parallelism; `0` therefore no longer means "always parallel" (use 1).
* The unused `par_threshold` of `qr/no_pivoting` (`QrParams`) is removed.

Remaining work in faer (thresholds unchanged, each a time floor of a different size per kernel):
`lu/full_pivoting/factor.rs:372` (`256*512`), `cholesky/bunch_kaufman/factor.rs:328` (`128*128`),
`qr/col_pivoting/factor.rs:244`, `svd/bidiag.rs:108`, `evd/hessenberg.rs:291` and `evd/tridiag.rs:322`
(`192*256`), `householder.rs:160`, `:224`, `:478`, `:609`, `:626` (`gemm::get_threading_threshold`),
`matmul/mod.rs:774`, `:894` (matvec), `matmul/internal/mod.rs:72`, `matmul/triangular.rs:125`, `:694`,
`triangular_solve.rs:439`, `:520`. The `faer-ffi` mirror of `QrParams` (outside the workspace) still names
the removed field.

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

## 4. Limits of this data

* One host, rough, 6 threads only (widths 2 and 4 only for the dispatch cost), one process per mode
  and shape and per round, the desktop session running (see the host table). x86_64 and Linux are not
  measured; the table is keyed by host class so that a Linux table can be added without an API change.
* The dispatch cost is measured with 10 us shares; a 1 us share would show a different join time.
* The break-even is located on a grid (for GEMM 8 to 16 points per decade); thresholds in section 3.3
  are derived, not fitted, and are checked against the grid only.
* The `seq` mode and the parallel modes run in separate phases of a process; at shapes where both run
  the same sequential kernel (for instance `gemm_sq 32^3`) they differ by up to 5 percent, so ratios
  between 0.95 and 1.05 are not significant.
* The LU `sections` multiples, the team token inside tcontract and the c64 team floor are findings or
  estimates to be verified when implemented, not results.
