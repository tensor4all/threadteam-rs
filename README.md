# threadteam

Splittable CPU placement token for the tensor4all CPU stack. `threadteam` is the one place
that starts tasks: kernels take an `Exec` token and split it, and no scheduler
type appears in any public signature.

Used by strided-rs, tcontract-rs, cpueinsum-rs, tlinalg-rs and tenferro-rs.
Design record: [`docs/design/placement.md`](docs/design/placement.md).

## Usage

```rust
use std::sync::atomic::{AtomicUsize, Ordering};
use threadteam::{Exec, Pool};

// The owner builds a pool once and keeps it.
let pool = Pool::builder().threads(4).build()?;

// Per call, the owner mints a token for `budget` slots of the pool.
let mut exec = Exec::new(&pool, 4)?;

// Several phases in sequence: reborrow, because entries consume the token.
let total = AtomicUsize::new(0);
exec.reborrow().for_ranges(1000, 64.try_into().unwrap(), |r| {
    total.fetch_add(r.len(), Ordering::Relaxed);
});

// Independent parts: each job gets a child token with its share of the slots.
exec.fork(2, |_job, child| {
    let _ = child.budget();
});
# Ok::<(), threadteam::Error>(())
```

`Exec::serial()` needs no pool and runs everything inline.

## Intended consumer pattern

The owner of the CPU resources (an engine, a contraction resource bundle) holds
a `threadteam::Pool` and mints `Exec::new(&pool, budget)` for each call, passing
the token to kernels by value. Kernels never store a token and never mint one.

## API

| Item | Role |
| --- | --- |
| `Pool`, `PoolBuilder` | fixed set of workers; `threads`, `thread_name`, `stack_size` (default 16 MiB, sized for recursive LU), `on_start`, `build` |
| `Exec::new(&pool, budget)` | mint a token; the only clamp (to the pool width); `budget == 0` is an error |
| `Exec::serial()` | the calling thread only; `const` |
| `Exec::budget`, `is_serial` | inspect the token (`is_serial` means budget 1) |
| `Exec::reborrow(&mut self)` | same pool and budget while the parent is exclusively borrowed |
| `Exec::limit(max)` | narrow to `min(budget, max)` |
| `Exec::fork(jobs, f)` | `min(jobs, budget)` tasks; shares differ by at most one and sum to the budget; job `i` gets a child token |
| `Exec::for_ranges(len, grain, f)` | at most `budget` tasks, ranges of at least `grain` except possibly the last |
| `Exec::broadcast(f)` | `f(i)` for `i in 0..budget`, all concurrent, or `Err(Refused)` |
| `Exec::can_broadcast()` | non-consuming: true exactly when `broadcast` would not refuse right now; ask before building an expensive setup |
| `Exec::carries_team()` | true when the token has team authority (minted by a `Team` or a child of one); `team()` then returns the carried team without acquisition, so a consumer that wants a `TeamExec` asks this and keeps `external` for plain tokens |
| `Exec::team(f)` / `team_with(over, f)` | reserve `budget - 1` awake pool members for a scope and run barrier-bearing sections through `TeamExec` (`run`, `run_with`, `split`, `fork`, `reborrow`), or `Err(Refused::Unavailable)`; see the design record |
| `Pool::into_team(width)` -> `Team` (owns the pool; `IntoTeamError` gives it back), `Team::exec(budget)` / `try_exec`, `Team::pool`, `Team::into_pool` | persistent reservation for an engine that owns its pool (`Pool` is not `Clone`); minted tokens are ordinary `Exec`s whose `fork`, `for_ranges`, `broadcast`, `team` dispatch on the reserved members; one leader at a time (`exec` blocks, `try_exec` -> `Refused::Busy`); idle members park while reserved |
| `PoolBuilder::qos(Qos)`, `raise_current_thread_qos(Qos)` | worker QoS on macOS (default `UserInitiated`); raise the leading thread without ever lowering it |
| `PoolBuilder::idle_spin`, `idle_runs_rayon`, `acquire_timeout`; `Pool::release_idle` | waiting policy of idle team members (an idle member returns to Rayon after `idle_spin` or the spin budget, whichever comes first) |
| `PoolBuilder::spin_polls`, `Pool::spin_polls` | spin budget of every team wait, in polls (the shape of libgomp's `GOMP_SPINCOUNT`): a reserved member waiting for work, the leader in a join and a participant in a barrier spin that many polls, yield a few times, then sleep until woken; throttled to 100 polls in an oversubscribed scope or after a waiter was descheduled ([`docs/design/waiting.md`](docs/design/waiting.md)) |
| `PoolBuilder::pin_participants(bool)`, `Pool::pin_participants()` | opt-in placement of the **members** of a root team scope (Linux; off by default, which is consistent with the rule it protects): member `k` (participant `k >= 1`) pins itself onto the `k`-th CPU of the calling thread's affinity set, `S[0]` is left to the leader, and a member restores its own mask before it returns to Rayon or serves foreign work. **The leader is never pinned**, because a vendor library creates its thread team from the calling thread and a one-CPU caller would confine the whole provider team; a member that calls such a library directly inside a share does confine it, so use `Exec::external` (which restores the member's mask first) or leave pinning off. For tensor4all/faer#107; give each team its own CPU set (`taskset -c`), there is no CPU allocator |
| `DispatchCosts`, `WidthCosts`, `PoolBuilder::dispatch_costs`, `Exec::dispatch_cost(width)`, `TeamExec::dispatch_cost(width)` | cost of one parallel section per kind of token (team, in pool, outside), steady state; one row per `HostClass` (`HostClass::detect()`, keyed by kernel dispatch: `x86_64-avx512`, `x86_64-avx2`, `x86_64-scalar`, `aarch64-apple`, `aarch64`, `generic`), `DispatchCosts::for_class(class)`; measured rows for `x86_64-avx512` (team anchors at widths 2, 4, 6 and 8) and `aarch64-apple`, a measured Zen2 team row for `x86_64-avx2` (anchors at ten widths from 2 to 64; its plain kinds stay generic), every other class takes the conservative `generic` row (3 / 15 / 40 us at the anchors 2, 4, 6); a row holds measured anchors (`WidthCosts::new`, `WidthCosts::flat`, or `WidthCosts::from_anchors` for measured anchors up to 64 and beyond), interpolates between them and grows linearly above the largest one, so a width above the anchors is never charged the cost of the largest anchor; `DispatchCosts::host_default()` is the detected class's row; `Duration::MAX` when serial; override per pool |
| `pays_at(est, sections, cost)`, `Exec::pays(est, sections, width)`, `Exec::width_for(est, sections)`, `width_for_by`, `PAYS` | the one rule: parallel iff `est >= PAYS * sections * D(width)`; `width_for` minimises `sections * D(k) + est / k` |
| `Exec::external(f)` | enter the pool once and hand the budget as a plain width to an outside library (faer, a vendor BLAS); how it divides the width is that library's concern |

## Contracts

* **No duplication.** `Exec` is neither `Clone` nor `Copy` (compile-fail
  doctests), and every parallel entry consumes it.
* **Slots are conserved.** Under one root token the number of tasks running at
  once is at most its budget at every nesting depth. A parent waiting for its
  children keeps no slot.
* **Lazy and inline.** A serial token, budget one, or one task runs inline on the
  calling thread and makes no scheduler call, so the process-global Rayon pool is
  never started implicitly.
* **Placement by caller.** A worker of the target pool spawns into the current
  scope without installing; a plain thread or a worker of another pool enters the
  target pool once and waits. No thread-local state exists in the crate.
* **`broadcast`** refuses with `Refused::ForkChild` for a token derived from
  `fork` (through `reborrow` or `limit` too), and with `Refused::CallerInPool`
  when the caller is a worker of the same pool. Roots, and their `reborrow`s and
  `limit`s, are eligible. Callers must handle `Refused` by falling back to a
  barrier-free path. Concurrent broadcasts on one pool queue in the same order on every
  worker (no interleaving); participants are the pool's workers `0..budget`, so
  blocking work of another token on those workers delays the broadcast.
  `can_broadcast` applies the same rule through one private helper, so the two
  cannot disagree.
* **Deterministic shutdown.** Dropping a `Pool` joins every worker thread: when
  `drop` returns, all workers have exited and their thread-local destructors
  (including state created in an `on_start` hook) have run. A pool dropped from
  one of its own workers cannot join itself, so that drop only signals shutdown
  and returns without joining. Without `parallel` there is nothing to join.
* **No scheduler type is public.** There is no constructor from, and no accessor
  returning, a scheduler pool. `scripts/check-no-public-rayon.sh` runs
  `cargo public-api -p threadteam` (when installed) and fails if `rayon::`
  appears in the output.

## Features

| Feature | Effect |
| --- | --- |
| `parallel` (default) | run on a private pool. Without it the same types exist, `PoolBuilder::build` returns a width-1 pool and every entry runs inline |
| `ledger` | test support: `ledger::max_live(root)`, `live`, `installs`, and `Exec::ledger_root`. Records live tasks per root token and pool installs; compiled out when off |

Consumers assert `ledger::max_live(root) <= budget` in their own tests, and keep
a test binary that runs serial and pooled work and then checks that
`rayon::ThreadPoolBuilder::new().build_global()` still succeeds.

## Wait benchmarks

`benches/wait_scenarios.rs` measures how team threads wait, the way callers use them; the design
is in [`docs/design/waiting.md`](docs/design/waiting.md). A change to the waits is judged by running
it on the base and on the head of the change, on a quiet host and on a many-core shared host.

| scenario | a call is | reports |
| --- | --- | --- |
| `sections` | 200 sections of 5, 20 or 50 µs, team held | per call, and the overhead over `sections * work` |
| `gaps` | 10 sections of 5 µs, the caller sleeps 0 / 100 µs / 1 ms / 10 ms between calls | the first section (wake latency) and the call |
| `acquire` | 50 sections of 20 µs: the team held (a token per call), the pool held (members acquired per call), a new pool per call | per call; for a new pool also the total from build to drop |
| `nested` | 100 splits of the team into pairs, each pair one 20 µs section | per call |

Each scenario runs at width = all CPUs of the affinity set and at three quarters of them, on
threadteam and on a Rayon `scope`; `benches/omp/wait_omp.c` runs `sections` and `gaps` on libgomp
(one `omp parallel` region per section, runtime defaults). `--load K:P` adds `K` child processes,
each busy `P` % of every 10 ms, for the whole run: the stand-in for another user's processes.

```sh
# one checkout, both passes (quiet, then 8 processes at 10 %), rows appended to
# docs/design/waiting-data/<host>.csv
scripts/bench-waits.sh <label>                 # e.g. main-4c73c2b, then the PR head
scripts/bench-waits.sh <label> 64,48 8:10      # explicit widths and load
cargo bench --bench wait_scenarios -- --scenario gaps --widths 8   # one scenario; options in the file header
```

* **evo** (quiet): `scripts/bench-waits.sh <label> "" none` on the base and on the head. The
  acceptance check is that `sections` and `gaps` do not regress.
* **broccoli** (64 CPUs, shared): `taskset -c 0-63 scripts/bench-waits.sh <label> 64,48 8:10` on the
  base and on the head, plus the faer#107 arms of `acquire_probe_107` (tensor4all/faer branch
  `probe/107-acquire-per-call`). The acceptance check is that width 64 under load is no longer
  slow and width 48 does not regress.

Results are committed as CSV under [`docs/design/waiting-data/`](docs/design/waiting-data/), one
file per host, with the label naming the commit; the summary tables live in `waiting.md`.

## License

Licensed under either of MIT or Apache-2.0 at your option.
