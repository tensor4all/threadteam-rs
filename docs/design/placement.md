# CPU placement: one `Exec` across the tensor4all CPU stack

`threadteam` is the one crate in the tensor4all stack that starts tasks. This
document records the design. It is used by strided-rs, tcontract-rs,
cpueinsum-rs, tlinalg-rs and tenferro-rs; the cross-repository rules are in
tensor4all/tensor4all-meta#41 (`docs/design/core-boundary-and-execution-ownership.md`,
"Where placement is decided").

## Problem

The CPU stack places work with **four** mechanisms at once:

| Mechanism | Where it lives today | Failure mode |
| --- | --- | --- |
| Raw Rayon (ambient) | fixed in tensor4all/tenferro-rs#2273 | serial engine starts the process-global pool |
| Thread-local scope | a per-thread pool or budget scope | a job running on the blocked thread inherits another operation's pool/budget |
| Explicit handle | tcontract-rs `Exec`, tenferro `FanOut`, `with_faer_width` | 3 separate implementations of clamp / inline / suspend; the cross-pool wait inherits per-thread scope state |
| Width only (faer `Par`) | faer, BLAS adapters | silently ambient unless the caller installed first |

Failure modes observed when placement is spread over several mechanisms: `install` costs 3.6 to 13 µs against 41 ns of work for a
1 KiB operation, so placement must stay lazy; budget clamping exists in four
places (`Exec::width`, `clamp_fanout_width`, `width.min(native_thread_count)`,
`FanOut::task_len`); entry counters exist in three systems that do not compose;
and because the produced numbers are correct, every one of these defects passes
the existing tests.

## Goals

1. One placement implementation in the whole stack.
2. Placement misuse is unrepresentable or a compile error.
3. No ambient fallback: the global Rayon pool is never reached implicitly.
4. Lazy: small operations never enter a pool.
5. Correct when the caller is a plain thread, a worker of the target pool, or a
   worker of a **different** pool.
6. The number of tasks running at once under a token is at most its budget at
   every nesting depth, by construction and checked by tooling, not by
   convention.
7. No Rayon type in any public signature, so the scheduler can be replaced
   inside `threadteam` alone.

## Non-goals

- Batch and work-model policy (`WorkModel`, `BatchPolicy`, `BatchQuery`). It is
  calibrated against contraction/GEMM benchmarks and stays in `tcontract-exec`.
- Pool ownership, arena and workspace lifetime. `threadteam` defines the
  `Pool` type but never holds one; instances stay with the owner
  (`tcontract-exec::Pool`, engine resources).
- faer / BLAS scheduling. They receive a plain width through `Exec::external`;
  how a library divides it (for faer: spindle's `with_lock` region and
  `join_raw`, owned by t4a-faer) is that library's concern. `threadteam`
  depends on neither faer nor spindle, and its `cargo deny` bans both.
- Runtime provider registry, autotuner, per-handle provider switching.

## Design

### The type (`threadteam`)

```rust
pub struct Pool { /* rayon::ThreadPool, private */ }

impl Pool {
    pub fn builder() -> PoolBuilder;
    pub fn width(&self) -> NonZeroUsize;
}

pub struct PoolBuilder { /* threads, name, stack size, start hook */ }

impl PoolBuilder {
    pub fn threads(self, n: usize) -> Self;
    pub fn thread_name(self, f: impl Fn(usize) -> String + 'static) -> Self;
    pub fn stack_size(self, bytes: usize) -> Self;   // default sized for LU recursion
    pub fn on_start(self, f: impl Fn(usize) + Send + Sync + 'static) -> Self; // e.g. pinning
    pub fn build(self) -> Result<Pool>;
}

pub struct Exec<'p> {
    pool: Option<&'p Pool>,              // None = the calling thread; never exposed
    budget: NonZeroUsize,                // clamped to the pool's width in `new`
}
// Not Clone, not Copy.

impl<'p> Exec<'p> {
    pub const fn serial() -> Exec<'static>;
    pub fn new(pool: &'p Pool, budget: usize) -> Result<Self>;
    pub fn budget(&self) -> NonZeroUsize;
    pub fn is_serial(&self) -> bool;                    // budget == 1

    /// Same pool and budget while `self` is exclusively borrowed: one token
    /// drives several parallel phases in sequence.
    pub fn reborrow(&mut self) -> Exec<'_>;
    /// Budget min(budget, max); the dropped slots are simply unused.
    pub fn limit(self, max: NonZeroUsize) -> Exec<'p>;

    /// Run `jobs` jobs on min(jobs, budget) tasks; job i gets a child token.
    pub fn fork(self, jobs: usize, f: impl Fn(usize, Exec<'_>) + Sync);
    /// Data-parallel loop over 0..len, ranges of at least `grain`, at most `budget` tasks.
    pub fn for_ranges(self, len: usize, grain: NonZeroUsize, f: impl Fn(Range<usize>) + Sync);
    /// The only place a bare width exists: enter the pool, hand the budget to
    /// an outside library (faer, a vendor BLAS) as a plain width.
    pub fn external<R: Send>(self, f: impl FnOnce(NonZeroUsize) -> R + Send) -> R;
    /// Co-scheduling for barrier-bearing kernels: all `budget` participants or refuse.
    pub fn broadcast(self, f: impl Fn(usize) + Sync) -> Result<(), Refused>;
    /// Whether `broadcast` would run right now (non-consuming); same rule.
    pub fn can_broadcast(&self) -> bool;
}
```

`Pool` is the only way to make a pool. It wraps a Rayon pool privately and
exposes no Rayon type: there is no constructor from a `rayon::ThreadPool` and no
accessor returning one.

An `Exec` with budget `b` is `b` slots of its pool. The design makes the
number of concurrently running tasks under one token at most its budget by
construction:

- **Creation needs the pool.** `new` takes `&Pool`; only the pool's owner
  (the engine, tcontract-rs's pool) has one, and a token never gives its pool
  back out. A kernel therefore cannot mint a fresh full-budget token.
- **No duplication.** `Exec` is not `Clone` or `Copy`; every parallel entry
  consumes it.
- **Splits conserve slots.** `fork` divides the budget among
  `t = min(jobs, budget)` tasks in shares that differ by at most one and add up
  to the budget. When `jobs > budget` the `t` tasks take job indices from a
  shared counter, so the bound holds and the load balances. A job's child token
  can be passed to any kernel, which may `fork` again inside its share.
- **No other spawner.** Only `threadteam` depends on Rayon among tensor4all
  crates (`cargo deny` with `wrappers`), so nothing else can start a task on
  the pool.

`reborrow` and `limit` conserve slots as well: the parent of a reborrowed
token is unusable while the child lives, and `limit` only drops slots.

`broadcast` needs its participants to run at the same time, so it is accepted
only for a token that owns its slots outright: one from `new`, or derived from
it by `reborrow` or `limit`. A token produced by `fork` (or derived from one) is
refused with `Refused::ForkChild`, because its slots may share workers with its
siblings; a call from a worker of the same pool is refused with
`Refused::CallerInPool`, because the caller would occupy a participant. A
caller that receives `Refused` takes its barrier-free path. Participants are
the pool's workers `0..budget`, entered through one Rayon broadcast; a long
task of another token on one of those workers delays the broadcast but cannot
deadlock it. `can_broadcast` reports, without consuming the token, whether
`broadcast` would be accepted on the calling thread: both go through one private
refusal rule, so they cannot disagree. A kernel asks it before building a setup
that only a broadcast needs.

Dropping a `Pool` is deterministic. The pool starts its workers through Rayon's
`spawn_handler` with `std::thread::Builder` (same name and stack size) and keeps
the join handles. `Drop` first drops the Rayon pool, which signals shutdown, and
then joins every handle, so when it returns all workers have exited and their
thread-local destructors have run. A pool dropped from one of its own workers
cannot join itself: that drop skips joining. Steady-state entries are unaffected.

A parent waiting for its children keeps no slot; Rayon may let the waiting
thread run one of its children, which is one of the tasks already counted.

### Mapping of today's patterns

| Today | With the token |
| --- | --- |
| tlinalg batch: `lanes = width`, each lane `Sequential` | `exec.for_ranges(batch, grain, ...)` or `exec.fork(lanes, \|i, _\| ...)` |
| tlinalg batch: one lane, faer gets the whole width | `exec.external(\|w\| faer_call(Par::rayon(w.get())))` |
| per-chunk fan-out under a thread-local pool scope | `exec.for_ranges(len, grain, ...)` |
| tcontract batched GEMM, batch x inner split | `exec.fork(b, \|i, child\| gemm(child, ...))`, `b` from `BatchPolicy` |
| tenferro two independent graph nodes | `exec.fork(2, \|i, child\| run_node(i, child))` |

### Ownership across repositories

| Crate | Owns | Never owns |
| --- | --- | --- |
| `threadteam` (the only Rayon dependent) | the `Pool` and `PoolBuilder` types, `Exec`, `fork`, `for_ranges`, `external`, `broadcast`, the test ledger | a pool instance, arena, policy |
| `strided-basic` / `-fused` / `-perm` | kernels; take `Exec` by value; re-export it | its own placement (no thread-local scope exists) |
| `tcontract-exec` | `Pool` (borrowed/owned/shared, holding a `threadteam::Pool`), `ArenaProvider`, workspace, `WorkModel`, `BatchPolicy`; mints tokens from its pool | a task spawner, a Rayon type |
| `cpueinsum` | nothing new | threading |
| `tlinalg` | batch driver expressed with `for_ranges` / `external` | threading |
| tenferro-rs | engine pool ownership (a `threadteam::Pool`, re-exported), dispatch, dtype, pooling; mints and passes `Exec` | fan-out, a Rayon type |

### Behaviour decisions

- **Serial / budget one / one task**: runs inline. No Rayon call, so the global
  registry is never touched.
- **Already in the target pool**: spawn into the current scope, no install
  (cheap re-entry).
- **Cross-pool or plain thread**: enter the pool once with `in_place_scope` and
  wait. No thread-local state exists, so nothing is inherited by work that runs
  on the waiting thread.
- **External width**: `external` hands over the budget unchanged. A library
  that rounds or splits it does so itself (t4a-faer's `join_raw` splits a width
  into halves that add up to it).
- **Plans** never store `Exec`; replay takes it per call.
- **Without the `parallel` feature** the same types and signatures exist;
  every pool has width 1 and every entry runs inline.

## Invariants and how they are checked mechanically

| Invariant | Check |
| --- | --- |
| Only `threadteam` starts tasks | `cargo deny` bans `rayon` with `wrappers = [threadteam, faer, gemm, ...]` in every consumer; clippy `disallowed-methods` for `std::thread::spawn` / `scope` in kernel crates |
| Tokens cannot be duplicated | compile-fail doctest: `Exec: Clone` and `Exec: Copy` do not hold |
| Splits conserve slots | unit test (property): shares of `fork` / `for_ranges` add up to the budget, each share at least one |
| Concurrency per root token at most its budget | test-feature ledger in `threadteam` (live tasks, max) asserted in every consumer test; faer measured at the kernel boundary |
| No implicit global pool | `build_global().is_err()` test in each consumer, own process |
| Serial runs on the calling thread | `std::thread::current().id()` test |
| Same-pool re-entry does not install | entry counter (test-only) |
| Cross-pool runs on the target pool | thread identity test from a foreign pool |
| Budget clamped once | constructor test; no other clamp sites exist (follows from rule 1) |
| No Rayon type is public | `cargo public-api` for `threadteam`, `tcontract-exec`, `tenferro-cpu` in CI; output must not contain `rayon::` |

## Consumers

- Kernel crates (strided-basic, strided-perm, strided-fused) take `Exec` by
  value, re-export it, and carry no direct `rayon` dependency.
- tcontract-rs holds a `threadteam::Pool` in its resource bundle and mints
  `threadteam::Exec` tokens from it; arena and policy stay there.
- tlinalg-rs expresses its batch driver with `for_ranges` / `external`.
- tenferro-rs owns the engine pool (a `threadteam::Pool`) and passes `Exec`
  down; it contains no fan-out of its own.
- cpueinsum-rs needs no threading of its own.

Each consumer keeps `cargo test`, `cargo clippy --all-targets`,
`cargo fmt --check`, `cargo deny check bans`, the ledger assertions and the
global-pool test green.

## Acceptance criteria

- `cargo deny` rejects a direct `rayon` dependency outside `threadteam` and
  the vendor crates in every repository of the stack.
- `cargo public-api` of `threadteam`, `tcontract-exec` and `tenferro-cpu`
  contains no `rayon::` path.
- The ledger shows, in every consumer's test suite, maximum live tasks per root
  token within its budget, including nested `fork`.
- tenferro, cpueinsum and tlinalg each carry a "global pool not started" test
  that passes for serial and for pooled engines.
- No consumer holds a thread-local or a raw pointer to a pool.

## Risks

- `Exec<'a>` becomes viral if any type stores it. Mitigation: per-call only;
  a non-`Clone` token is also useless to store.
- Static shares: a `fork` task that finishes early does not hand its slots to
  a sibling. `for_ranges` and `fork` with more jobs than budget balance through
  the shared counter; uneven `fork` with few jobs does not.
- External BLAS keeps its own worker team; that asymmetry stays documented and
  is out of scope for the budget.
- The cross-pool hand-off remains a microsecond-scale cost on a rare nesting.

## Planned: one pool per cache domain

A `Pool` built with a domain layout holds one Rayon pool per shared-cache
domain (an L3 on multi-CCD or multi-socket x86, an L2 cluster on Apple
Silicon; an M5 Max has three). A `fork` on a token that spans domains splits
at domain boundaries first, and each child receives the slots of one domain.
Each domain is a separate Rayon pool, and a thread is a worker of at most one
pool at a time (rayon-core 1.13.0, `src/registry.rs:710`), so no task of that
child runs outside its domain at any depth; inside a domain the budget is
divided as above. Inside a single Rayon pool, by contrast, the children of a
task spread over every worker: a task placed on workers 0 to 3 of an 8-worker
pool with `broadcast` had its `scope` children run on all 8.

What this needs:

- `PoolBuilder::domains(...)` and a topology helper
  (`/sys/devices/system/cpu/cpu*/cache/index3/shared_cpu_list` on Linux,
  `hw.perflevelN.cpusperl2` on macOS).
- Weighted shares in `fork` for domains with cores of different speed.
- Cross-domain cuts of large contractions decided by `WorkModel` /
  `BatchPolicy`; a GEMM larger than one domain is cut into blocks in
  `tcontract`, not inside faer.

Entering a domain's pool from outside costs about 15 µs on an M5 Max, paid only
by operations large enough to span domains. Because `Pool` exposes no Rayon
type, this is a change inside `threadteam`; `Exec`, `fork` and kernel
signatures do not change.

## Open questions

1. Whether weighted shares (`fork_with(&[w0, w1], ...)`) are needed; omitted
   until a caller has uneven parts.
