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
    pub fn pin_participants(self, on: bool) -> Self;  // Linux; on by default
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
    /// One child per share, child i gets exactly shares[i] slots; the shares
    /// sum to at most the budget (never clamped). `fork` is its even case.
    pub fn fork_sized(self, shares: &[usize], f: impl Fn(usize, Exec<'_>) + Sync) -> Result<(), Error>;
    /// `fork_sized` where child i also gets exclusive `&mut slots[i]`;
    /// `slots.len() != shares.len()` is an error.
    pub fn fork_sized_slots<S: Send>(self, shares: &[usize], slots: &mut [S], f: impl Fn(usize, &mut S, Exec<'_>) + Sync) -> Result<(), Error>;
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
  `fork_sized` takes explicit shares instead; their sum must not exceed the
  budget (an error names both, nothing is clamped), each child's budget is
  exactly its share, and slots the shares leave over run nothing during the
  call and are the parent's again after the join. Both entries go through one
  splitter over the same `Plan` that `TeamExec::split` uses, so join, panic
  propagation and job-to-task placement are one code path.
  `fork_sized_slots` is `fork_sized` plus one caller-owned state slot per
  child: it checks `slots.len() == shares.len()` (`Error::SlotsMismatch`
  otherwise, so no slot is silently left out), then calls `fork_sized`, whose
  child `i` runs exactly once and derives `&mut slots[i]` from the base
  pointer of the exclusively borrowed slice. Distinct children get distinct
  elements, the reference lives for one call of `f`, and the borrow of
  `slots` ends after the join (also when a child's panic is re-raised), so
  per-child state needs no lock and the call allocates nothing.

  *Safety argument.* The slice is not split before dispatch. One private
  `SlotBase<S>` wraps `slots.as_mut_ptr()` (`unsafe impl<S: Send> Sync`), and
  each child computes `&mut *base.at(i)` from its own index right before its
  callback. Exclusivity rests on one internal invariant: `fork_sized` calls
  `fork_plan` with as many jobs as tasks, so there is no shared counter and
  each `i in 0..shares.len()` is dispatched exactly once on every path (the
  single inline child, the disjoint `lo..hi` halves of the `rayon::join`
  tree, the team section's leader and distinct published participants).
  With `shares.len() == slots.len()` every `i` is in bounds. The caller
  cannot touch `slots` while the call holds it, the callback's slot lifetime
  is higher-ranked so the reference cannot escape (a compile-fail doctest),
  and `fork_sized` returns or unwinds only after every child has finished
  (`rayon::join` joins both halves before propagating; the team section's
  `JoinGuard` waits on unwind), so no derived reference outlives the
  borrow. `S: Send` suffices because a child receives exclusive `&mut S`,
  never shared access. A safe alternative was rejected: `split_at_mut`
  recursion fits the join tree but not the team section's shared
  `Fn + Sync` callback, and per-slot `Mutex` extraction allocates (a native
  lock is boxed lazily on macOS).
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

The pool starts its workers through Rayon's `spawn_handler` with
`std::thread::Builder` (same name and stack size) and keeps the join handles.
Dropping a pool with dedicated workers (a `thread_name`, an `on_start` hook, a
`stack_size` or a `qos` of its own) is deterministic: `Drop` first drops the
Rayon pool, which signals shutdown, and then joins every handle, so when it
returns all workers have exited and their thread-local destructors have run.
A pool with the default thread settings waits until its team members left
their loops and then keeps the Rayon pool and its handles in a process-wide
cache keyed by width; the next default pool of that width takes it and each
worker applies the building thread's CPU set to itself (a Rayon `broadcast`).
Only idle entries are handed out, so two live pools never share workers. The
cache holds at most the larger of the available parallelism and the widest
pool built so far, in threads, and drops and joins its oldest entries beyond
that. A pool dropped from one of its own workers can neither join nor wait for
itself: that drop skips joining and caching. Steady-state entries are
unaffected.

The cache is the Rayon-based answer to tensor4all/faer#107's per-call pools
(threadteam-rs#18, #19): the slow mode came from the fresh threads of every
pool, not from the waits. Own `std::thread` workers (#16) solved the same
problem with a much larger change; maintainer decision, 2026-10-30: 「しばらくは
Rayon版を出版し続けて，Rayonで何ができないか熟考してから判断する」. That
version is kept on the `develop` branch and would ship as 0.3.0.

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
- **Cross-pool or plain thread**: enter the pool once with `ThreadPool::install`
  and wait. Scheduler and budget state are not inherited through thread-local
  storage by work that runs on the waiting thread. The crate's only thread-local
  is the placement record (`pin::PLACED`), which says *where this thread was
  placed*; it is read when a nested team places its leader, and only there: a
  nested placement is allowed while the current mask still belongs to that
  record, so the leader can be narrowed onto the nested team's home and is never
  widened out of the set its caller gave it (`pin::home_pin`,
  threadteam-rs#54).
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
  the shared counter; uneven `fork` with few jobs does not. `fork_sized` lets
  the caller size the shares to the work instead.
- External BLAS keeps its own worker team; that asymmetry stays documented and
  is out of scope for the budget.
- The cross-pool hand-off remains a microsecond-scale cost on a rare nesting.

## Reserved awake members: `TeamExec` (prototype)

`Exec::team` reserves pool workers as members of a team for the duration of a
callback and hands the callback exclusive authority, `TeamExec<'team>`, over
the team: the calling thread (participant 0, the leader) plus `budget - 1`
members. It exists for barrier-bearing kernels that need all participants on
distinct threads, and for nested teams, without a per-section Rayon dispatch.
A token without team authority behaves exactly as described in the rest of this
document. Two ways to hold a team exist and share one implementation of sections
and dispatch: a scoped acquisition per call (`Exec::team`, for pools that are not
owned by one engine) and a persistent reservation owned by the engine (`Team`,
below), whose tokens carry the authority into the ordinary `Exec` methods.

```rust
impl Exec<'_> {
    pub fn team<R>(&mut self, f: impl for<'team> FnOnce(TeamExec<'team>) -> R) -> Result<R, Refused>;
    pub fn team_with<R>(&mut self, over: WaitOverride, f: impl for<'team> FnOnce(TeamExec<'team>) -> R) -> Result<R, Refused>;
}
impl<'team> TeamExec<'team> {            // not Clone, not Copy
    pub fn size(&self) -> NonZeroUsize;
    pub fn policy(&self) -> WaitPolicy;
    pub fn with_wait(self, over: WaitOverride) -> Self;
    pub fn reborrow(&mut self) -> TeamExec<'_>;
    pub fn run<F: Fn(usize, usize) + Sync>(self, f: F);                 // f(index, size)
    pub fn run_with<F: Fn(&Participant<'_>) + Sync>(self, f: F);        // adds barrier()
    pub fn split<F>(self, sizes: &[usize], f: F) -> Result<(), Error>
        where F: for<'c> Fn(usize, TeamExec<'c>) + Sync;
    pub fn fork<F>(self, children: usize, f: F) where F: for<'c> Fn(usize, TeamExec<'c>) + Sync;
}
impl Participant<'_> { pub fn index(&self) -> usize; pub fn size(&self) -> usize;
    pub fn barrier(&self) -> Result<(), Cancelled>; pub fn is_cancelled(&self) -> bool; }
```

### Authority and lifetimes

The `'team` lifetime is the reservation. The scoped callback keeps the
authority from escaping; entries consume it; `reborrow` locks the parent while
the lent authority lives; `split` and `fork` consume the parent and lend
children through `for<'c>` callbacks, so a child can neither escape nor be used
by the parent, and a sibling cannot be reached. Compile-fail doctests cover
`Clone`, `Copy`, reuse after an entry, escape, parent use while reborrowed,
parent use inside a `split` callback, and child escape. `team` takes `&mut
Exec`: the token is borrowed while the team lives (slots stay conserved) and is
available afterwards, in particular after a refusal.

### Members and acquisition

A member is a pool worker running a loop as one Rayon job; while it runs the
worker serves no other Rayon job. Each worker index owns one cache-line padded
slot with a state: `ABSENT` (no loop), `IDLE` (loop alive, unreserved),
`YIELD` (idle member inside `rayon::yield_now`, not claimable), `RESERVED`.

1. Idle resident members are claimed with a compare-and-swap, in slot order.
   Members inside `yield_now` are retried for at most the acquisition timeout.
2. Missing members are requested with claim jobs spawned into the pool. A job
   that lands on a free worker registers itself in the acquisition and starts
   the member loop; a job that lands on a resident member serving Rayon hands
   itself back to the pool. A worker executing any other Rayon job takes no
   claim job, so it is simply unavailable.
3. The leader waits at most `acquire_timeout` (default 2 ms) for free workers to
   wake. This bounds the wait for *idle* workers; it is never a wait for a
   running job. At the deadline the claim is closed (a late job does nothing),
   every tentative member is released and `Refused::Unavailable` is returned
   before any callback runs. The caller keeps a barrier-free fallback.
4. With all members present, every participant (leader included) passes a
   rendezvous section. All are inside it at once, so they are on distinct
   threads: this is the acknowledgement that precedes any user section.

Two acquisitions that each hold part of the free workers can both time out and
refuse; the fallback applies. The caller may itself be a pool worker (for
example inside a `fork` job): it leads on its own thread and the members come
from the other workers. Members hold no thread-local state; identity is the slot
index.

### Dispatch

A section is a stack object (`Section`): a type-erased pointer to the
closure, the participant count, a padded completion counter, a sense-reversing
barrier, a cancellation flag and an atomic cell for the first panic payload.
The leader stores the section address and the participant index into each
member slot and bumps its `gen` counter with release ordering; members poll
`gen` and `state`, run their share and decrement the completion counter. The
leader runs share 0, then waits for the counter. Every one of these waits
(a member waiting for work, the leader's join, a barrier) is a staged episode,
spin, a few yields, then sleep, described in [`waiting.md`](waiting.md). Nothing is allocated, and a dispatch wakes only members that are asleep
(`tests/team_alloc_free.rs`; acquisition allocates a small `Vec` and, for
members not already awake, the claim jobs).

`split`/`fork` partition the members by team index: child `c` covers a
contiguous index range; child 0 is led by the calling thread and the other
children by the first member of their range, which receives `f(c, child)` as
its share of one dispatch. The remaining members of a child wait on their own
slots for the child leader's sections; no slot is shared between children.
When the dispatch returns, all children finished and the members are the
parent's again. Members the sizes leave over stay parked.

### Persistent reservation: `Team` and team tokens

```rust
impl Pool {                               // Pool stays a unique owner: not Clone
    pub fn into_team(self, width: usize) -> Result<Team, IntoTeamError>;
    pub fn into_team_with(self, width: usize, over: WaitOverride) -> Result<Team, IntoTeamError>;
}
impl IntoTeamError { pub fn reason(&self) -> Refused; pub fn into_pool(self) -> Pool; }
impl Team {                               // owns the Pool; no lifetime; Drop drops the pool
    pub fn pool(&self) -> &Pool;          // borrow only
    pub fn into_pool(self) -> Pool;       // releases the members, gives the pool back
    pub fn size(&self) -> NonZeroUsize;
    pub fn exec(&self, budget: usize) -> Exec<'_>;                     // blocks; panics on re-entry
    pub fn try_exec(&self, budget: usize) -> Result<Exec<'_>, Refused>; // Err(Refused::Busy)
}
```

An owner that has its pool to itself (tenferro's engine) moves the pool into a
`Team` with `into_team`, which reserves `width - 1` members once, with the same
acquisition and acknowledgement rendezvous as `Exec::team`, and stores the `Team`
(or a plain `Pool` when it wants no team). `Pool` is a unique owner: it is not
`Clone`, no method returns a second handle, and sharing a pool between callers
stays outside this crate (for example an `Arc<Pool>` held by the caller, where
per-call acquisition with `Exec::team` applies). The `Team` holds the `Pool` by
value and has no lifetime and no `Arc` around the pool. When the members cannot
all be reserved, `IntoTeamError` returns the pool together with the `Refused`
reason and the caller falls back to plain tokens. Dropping the `Team` releases the
members (zero idle spin, woken) and then drops the pool, which ends its team
members: a pool with the default thread settings leaves its idle workers in a
process-wide cache keyed by width, which the next default pool of that width
takes instead of starting threads, and only a pool with dedicated workers
(`thread_name`, `on_start`, `stack_size`, `qos`) joins them; `into_pool` releases
the members and returns the pool instead. The
only internal sharing is an `Arc` of the awake-member state that claim jobs
hold; it is not a pool handle. `exec(budget)` mints a
root token, `budget` clamped to `1..=size`, that carries `TeamAuth`: the token's
members, a slice of the team's with `ids.len() + 1 == budget`. The slice stays
consistent through `limit` (narrowed), `reborrow` (copied) and `fork` (the sub-slice
of the child's share).

On a token with team authority the `Exec` methods dispatch through the same
`section` function as `TeamExec`:

| Method | Behaviour |
| --- | --- |
| `fork(jobs, f)` | `tasks = min(jobs, budget)`; task `k` runs on the first member of its index range of the same even plan as `split`, with the share of `split::share`; the child token carries that range's sub-team; more jobs than tasks pull from the shared counter as before |
| `fork_sized(shares, f)` | the plan of `split(shares)`: child `k` leads its index range of `shares[k]` participants and carries that sub-team; members past the sum stay parked |
| `fork_sized_slots(shares, slots, f)` | as `fork_sized`; child `k` also gets `&mut slots[k]` |
| `for_ranges` | the leader and the first `tasks - 1` members run the chunks; no Rayon entry |
| `broadcast` | every participant of the budget at once on the reserved members; never refused (a `fork` child owns its members too), `can_broadcast` is true |
| `team(f)` / `team_with` | hands out the carried (sub)team at once, no acquisition |
| `external(f)` | enters the pool as before and hands the width (the budget). While `f` runs, the token's members are **lent**: each sets aside its spin or park and runs `rayon::yield_now` (the pool's jobs) until the call returns, then resumes team service. A lent job still running on return delays the next section until it finishes. `tests/team_owner.rs` runs four mutually waiting tasks on a team of four from inside `external`: they can only complete on four threads |
| `limit`, `reborrow`, `budget`, `is_serial` | as on any token |

Budget conservation and the ledger hold as on plain tokens: the same bodies
count each participant as a live task for the time it runs a share, the leader
is released while it waits, and `tests/team_ledger.rs` runs nested forks,
for_ranges and broadcasts on team tokens against the budget. A token with a
budget of one runs inline.

**One leader at a time.** The calling thread leads every section, so minting
serializes callers: `Team::exec` blocks on a mutex and condition variable until
the previous root token and everything lent from it (`reborrow`) was dropped, then
takes leadership; `try_exec` returns `Refused::Busy` instead. A thread that still
holds a token and mints again would wait for itself, so `exec` panics. The guard
lives in the root token and releases on drop; consuming entries (`fork(self)`)
release it when they return, so a caller that needs the team across several
entries keeps the token and lends it with `reborrow`. No thread-local state is
involved (the owner's thread id is stored in the team).

**Parking.** A reserved member that finds no new section within its staged
wait (spin budget, a few yields; [`waiting.md`](waiting.md)) parks and stays
reserved. Each member slot has its own `Parker`, and a leader wakes only
the slots it changed (a publish, a release, a lend); the joins and barriers
sleep on one pool-wide `Parker`. The protocol of every `Parker`:

* A sleeper raises `parked` (acquire-release increment), takes the lock,
  re-checks the state it waits on and waits only while it is unchanged; it
  lowers `parked` when it leaves.
* A waker changes the state (SeqCst), then reads `parked` with an
  acquire-release `fetch_add(0)`. If it is nonzero, it takes and drops the lock
  and calls `notify_all`. Either the sleeper sees the changed state after
  raising `parked`, or the waker sees `parked` (both are read-modify-writes of
  one atomic, so the argument needs only its modification order and release
  sequences, not a total order of SeqCst accesses, which `loom` does not
  model); and because the waker takes the lock before notifying, the
  notification cannot fall between a sleeper's check and its wait.
* With nobody asleep a wake costs one uncontended atomic read-modify-write.
  Spurious wake-ups loop. Which parker each wait uses, and when it sleeps, is
  in [`waiting.md`](waiting.md).

The handshake is the small `Parker` type in `src/park.rs`, built on a `sync`
shim (`std`, or `loom` under `--cfg loom`), and checked three ways:

* **Model check.** `tests/loom_park.rs` runs the real `Parker` under `loom` with
  one leader and two members: a published section, a released reservation, a
  member that parks twice, and per-slot parkers; and the exact join and barrier
  code (`Completion`, `Barrier`): a leader asleep in a join, consecutive
  barrier rounds, and a cancellation waking a sleeper. The correctness argument therefore uses only what loom
  models (modification order and release sequences of `parked`). CI job `loom`:
  `RUSTFLAGS="--cfg loom" cargo test --release --test loom_park`.
* **Forced interleavings.** With the feature `test-hooks`, `test_hooks` lets a
  test pause the member between raising `parked` and taking the lock, between its
  re-check and its wait, and the leader between the state change and reading
  `parked` (`tests/team_hooks.rs`). Deliberate breakages (`Mutation`: the member
  skips its re-check, the leader skips the lock handoff) must stall in exactly
  these interleavings, which shows the scenarios can see a lost wake-up; the real
  protocol completes them in well under a second. The hooks cost nothing when the
  feature is off.
* Every team test has a watchdog that aborts the process when the test body
  stalls.

No allocation happens on the dispatch path. Dropping the team stores `IDLE`
with a zero idle spin and broadcasts, so the members return to Rayon at once.

### How a reserved member and a joining leader wait

Replaced by [`waiting.md`](waiting.md): every team wait spins for the pool's
spin budget in polls, yields a few times, then sleeps; the leader's join and
the barriers sleep on a pool-wide parker; an oversubscribed root scope throttles
its waits to 100 polls, and no wait reads a clock or detects descheduling. The
earlier design (PAUSE-only spin with a wall-time park delay for members and no
sleep at all in joins and barriers) is in the history of this file.

### Participant placement (on by default)

`PoolBuilder::pin_participants(bool)` (default **on**; `Pool::pin_participants()`
reads it back, `Pool::pin_skips()` counts the scopes it could not place) makes a
root team scope place its participants. It exists because of the measurement in
tensor4all/faer#107: `qr_factor` f64 512 at 16 threads on
16 allowed CPUs, 20 processes on broccoli (EPYC 7702P, boost 0, governor
`performance`), 20 repetitions each.

| arm | slow processes (>= 6 ms) | samples with two runnable team threads on one CPU | publish-to-share |
| --- | --- | --- | --- |
| A, width 16, unpinned | 11 / 20 | 92 % in the slow half | 225 us fast / 835 us slow |
| B, width 15 on the same 16 CPUs (one spare CPU) | 7 / 20 | 91 % in the slow half | 196 / 757 us |
| C, width 16, every participant pinned to CPUs 48-63 | **0 / 20** | **0 %** | 218 us |

Arm C pinned the leader too, as `threadteam` now does (the rule below). An
earlier version of the crate pinned only the members and kept the leader's mask
whole; that is the history told after the evo2 table.

12.4 ms is the one-thread time of the call, so the slow mode is the members'
work not overlapping: they are alive, unparked and receiving the same shares,
but queued on one CPU. A spare CPU does not stop the scheduler from stacking the
pair; deterministic placement does. That is why placement is the load-bearing
part of the change and parking is not.

The second measurement is evo2 (Ryzen AI MAX+ 395, two 8-core L3 domains, SMT
on, boost on; tensor4all `lu-chol-opt` campaign, LU f64 column major, faer fork
at 8, 11 and 14 threads, one CPU per physical core, n = 128-1024, four processes
per cell):

| arm | stalled cells | leader on a member's CPU (8 threads) |
| --- | --- | --- |
| A, unpinned | stalls at 11 and 14 threads (3 ms scheduler slices, one per wake of a parked member on a busy CPU) | n/a |
| B, every participant pinned, the leader to `S[0]` for the scope (**the rule now implemented**) | 0 of 12 | n/a |
| C, members pinned to `S[k]`, leader left where it was (0.3.0's rule) | catastrophic at 8 threads: LU 512 in 231 ms against 0.9 ms | 74-94 % of samples |
| C3, as C plus the leader nudged onto `S[0]` at scope entry, its mask left whole (n <= 1024 only; the nudge, removed in this change) | **0 of 12, equal to B** | **1.5 %** |

0.3.0 pinned only the members and left the leader free: the scheduler then put
the leader on a pinned member's CPU most of the time, and the two shared it for
whole slices. C3 fixed that for short scopes only (and B and C3 were equal
there). At n = 4096 (LU f64, T = 4-8, one root scope over many calls with about
50 ms of serial leader work between them) a leader that was placed once and kept
its whole mask drifted back onto a member's CPU after the first serial gap: 90 %
of the samples showed leader and member runnable on one CPU, only 1 % had the
leader on `S[0]`, and the calls ran 5-38x slower than unpinned.

**The rule: the leader is pinned to `S[0]` for the life of the team scope**, and
for one section when a token of a `Team` dispatches one without opening a scope
(see "A `Team`'s leader" below). One
`sched_setaffinity` at scope entry (or at section entry), one at its exit (a
guard, so also on unwind), and nothing in between for a `TeamExec` dispatch: no
per-section check, no move and restore around it. A *token* section dispatched
inside the scope is the exception the next subsection describes: with a nonzero
`home` it calls `home_pin`, which reads the mask and declines, so it pays the read
and no write.
Members keep `S[k]`, `k >= 1`. Arm B above is this rule for a scope.

The crate used to do something cleverer: it placed the leader on `S[0]`
immediately before **every section** and handed its whole mask back at once, so
that the leader could call vendor libraries between sections and a section
always opened with the leader home (the nudge, `Awake::place_leader`, with a
`sched_getcpu` check per dispatch). Measured at n = 4096 that was 106.3 ms per
call against 107.1 unpinned and 107.1 with the leader pinned for the whole
scope. The two ways of keeping the leader home were therefore not told apart by
that measurement (about 0.8 ms in 107, and it was not repeated to learn the run
to run spread), and for short scopes B and C3 were equal. The nudge is removed
because the pin is simpler, not because it was shown faster or slower; no
measurement here favours it. It was not re-measured on evo2 for this change (a
campaign is running there). The vendor-call half of the reason the nudge existed
is answered by the widening call below; the drift half by the pin itself, held
by the scope or, for a `Team`, by the section (see "A `Team`'s leader" below).

#### Widening the leader for a vendor call: `TeamExec::external`

A leader pinned to one CPU confines any vendor thread team it creates
(OpenMP, OpenBLAS, a LAPACK provider all build their team from the calling
thread; `tenferro-rs` `PERFORMANCE_TIPS.md`: 3-20x slower `dgemm` and LU at 4
threads). So the leader's vendor call goes through `TeamExec::external(f)`,
named after `Exec::external` because the two are halves of one rule: the
**member** path (`Exec::external`) lends the members and restores the member's
own mask; the **leader** path (`TeamExec::external`) sets the leader's mask to
`S` for the duration of `f` and puts `{S[0]}` back when `f` returns.

* **Cost, paid per call.** Two `sched_setaffinity` calls every call, inside
  whatever the caller times. Nothing is amortised and nothing caches "already
  wide". Measured on broccoli (EPYC 7702P, release build, empty closure, 200 000
  calls, once on CPUs 8-15 and once on 32-39): mean **2.30 us**, median 2.26 us,
  p99 2.85 us, max 20-28 us, against 0.05 us for the timer alone. This is the
  cost of the syscalls with an idle machine; it is not a measurement of a vendor
  call and does not include what the kernel does when the vendor threads start
  on the widened mask. One call around the whole vendor region is cheaper than
  one per vendor call.
* **The restore is not skipped**, including when `f` panics (a drop guard).
* **A refused syscall leaves the thread as it was.** A refused widening leaves
  the leader on `{S[0]}`, `f` still runs (narrowed) and no restore is made,
  because nothing changed. A refused restore (a cpuset that shrank) leaves the
  leader wide until the scope ends, when the exit guard sets `S` again.
* **Where nothing was placed** (pinning off, a team of one, `|S|` smaller than
  the team, a non-Linux target, a thread that does not lead a placed scope) it
  is a plain call to `f`.
* **The default path takes no vendor call** and so pays nothing: a scope that
  only runs sections has the leader on `S[0]` from entry to exit.
* **The members are not lent.** They stay reserved on `S[1..]`, so the vendor
  team may share those CPUs with them; to lend the members to Rayon use
  `Exec::external` on a token, not the leader path.

The check for a section dispatched while the leader is wide **is not
implemented, because it cannot happen**: `external` takes `&mut self`, so the
closure cannot reach the team, and a `compile_fail` doctest asserts it. Nothing
runs before a section, which is the costless version of the check. (A different
token's section dispatched from inside the closure is another scope's, and it
reads `S` back, so it is placed on its own team's home rather than on this one's
— threadteam-rs#54.) If `TeamExec` ever offers a way to reach the
team from the closure, the leader must place itself back on `S[0]` before the
shares start, and that dispatch needs a check.

#### Why it is on by default

Placement used to be opt-in, for three reasons. Two do not hold for this rule
and one holds with a bound:

* *It fights the caller's `taskset`.* It does not: every participant is placed
  inside the calling thread's own mask `S`, never outside it, and the leader's
  mask is `S` again when the scope has closed (it is `{S[0]}` while it is open).
* *It confines a vendor thread team a member creates.* That is the
  `Exec::external` path, which restores a member's own mask before the vendor
  call; and the leader, the thread engines call vendor libraries from, is widened
  to `S` for that call by `TeamExec::external` (above), at two syscalls per call.
* *It takes CPUs away from other users of the process.* It can: a placed member
  owns its CPU of `S` while the scope runs. It never takes a CPU outside `S`,
  and a set too small for the team places nobody (below). A pool that shares
  its CPUs with other work on purpose passes `pin_participants(false)`.

An unpinned team is not a configuration the tensor4all stack ships, so a default
of off only gave the stalling configuration to every caller who did not know
about the switch. All four compared runtimes (libgomp, libomp, OpenBLAS, faer)
leave threads unpinned unless asked; this crate differs on purpose.

How it works:

* At the start of a **root team scope** — `Exec::team_with` on the acquisition
  path, and on the carried path when the token is the root token minted by a
  persistent `Team` (`Team::exec`/`try_exec`; not a child, a reborrow or a
  sub-team) — the leader reads **its own** mask `S` with
  `sched_getaffinity(0, ...)` and runs one pin section through the ordinary
  section machinery (it also serves as the rendezvous, replacing
  `acknowledge`). The leader pins itself to `S[0]` with `sched_setaffinity(0, {S[0]})`
  before the pin section and keeps it for the scope; a guard gives it `S` back
  when the scope ends or unwinds. Nested scopes on one thread stack. Member `k` (participant `k >= 1`)
  pins itself with `sched_setaffinity(0, ...)` onto the `k`-th set CPU of `S`,
  so the members take `S` minus `S[0]`. A team of one is left alone.
* **Nothing wraps.** When `|S|` is smaller than the number of participants the
  scope places nobody: it runs one *unpin* section instead (the same
  rendezvous), in which every member restores the mask it saved before an
  earlier pin section of the reservation (a no-op for a member that is not
  pinned), and `Pool::pin_skips()` counts the scope. Wrapping would put two
  members on one CPU, the arrangement the placement exists to prevent (evo2: 8
  participants wrapped onto one CPU ran LU f64 512 in 3.1 ms against 0.89 ms
  unpinned and 1.7 ms on one thread, n = 1024 in 14.3 ms against 3.4); such a
  team runs with the scheduler's placement, and the counter makes that visible
  instead of silent. Such a scope is exactly an *oversubscribed* one, so its
  waits are already throttled.
* **The leader is pinned to `S[0]` for the scope, and widened for vendor
  calls.** Consequently a **pinned member must not build a foreign thread team
  directly inside a share** (it would confine that library to its one CPU), and
  neither must the leader outside `TeamExec::external`. A member routes the call
  through `Exec::external`, which restores the member's own mask first; the
  leader through `TeamExec::external` (above); or pinning is turned off for that
  pool.
* **Every affinity change is made by the thread itself.** A member
  saves its previous mask in its own slot (an `UnsafeCell<cpu_set_t>` plus a
  flag that only the member thread of that slot touches) the first time it is
  pinned, and restores it itself, in the member loop, before it leaves
  `RESERVED` (so before it returns its thread to Rayon) and before it serves
  Rayon work while lent by `Exec::external` — and it does the same when it is
  the thread that runs work outside its team itself, for example the member
  that leads a `fork` child and calls `external` (there the call runs inline on
  that worker, so it restores its own mask on its own thread before entering);
  it re-pins only at the next pin section. A persistent team therefore keeps the
  mask of its first scope and re-pins at each scope relative to that scope's
  leader.
* **Best effort.** A refused syscall (a kernel mask larger than `cpu_set_t`, a
  cpuset that changed under us) leaves that thread as it was; an empty mask pins
  nobody; a failed pin or restore never fails the scope. If the kernel
  refuses the leader's pin the leader keeps `S` and the members are still
  placed; a refused restore at scope exit leaves it on `S[0]`, the mask it
  already had (`TeamExec::external` has the same rule for its two calls).
* **Linux only.** Other targets accept the option and do nothing. No libc type
  appears in the public API; the masks are stack `cpu_set_t` values and the
  `k`-th set CPU is found by scanning one, so the path allocates nothing
  (`tests/team_alloc_free.rs`).
* **Assumption: the caller's CPU set belongs to this team.** Placement maps the
  participants onto the caller's mask, so two teams whose leaders see the same
  mask stack on the same CPUs. Give each team (in practice: each process) its
  own CPU set with `taskset -c`; there is no CPU allocator in this crate and none
  is planned for this fix. Overlapping teams and pools of one process stay
  caller-managed.
* The members' placement is the team's; the leader's pin is the scope's, or the
  section's. A token of a `Team` that never opens a team scope
  (`exec.for_ranges(...)`, `fork_sized`, directly) is not re-pinned: its members
  keep the placement of the last whole-team root scope, or of
  `Pool::into_team`, and **its leader is placed for each section it dispatches**
  and for each scope it opens without being the root token (see "A `Team`'s
  leader" for the cost of that).

#### Where the placement lives

The members' pins are owned by the authority over them and set and cleared by
the sections that pin and unpin them; the leader's pin is not part of that
placement, it belongs to one root scope, or to one section of a `Team` token
that opens none, and is held by a guard in that scope's or section's frame.

* **Scoped acquisition** (`Exec::team` on a plain token): the members are
  reserved, and pinned, for the scope only; they restore their masks when the
  reservation ends, and the leader gets `S` back at the same moment.
* **Persistent `Team`** (`Pool::into_team`): the members stay reserved, and
  pinned, from the pin section of a root scope (the first one is the
  acknowledgement scope of `into_team`) until an unpin section or the end of
  the `Team`. The leader is pinned to `S[0]` during each such root scope and
  has `S` again when it closes; a section that a token of the team dispatches
  without a scope pins it for that section and gives `S` back when the section
  returns (see "A `Team`'s leader"). Between scopes and between sections the
  leader, and the thread that called `into_team`, keep their own mask.
* **Only a root scope over the whole team changes the placement.** A root
  token from `Team::exec(budget)` with a smaller budget (or narrowed by
  `limit`) carries only the first `budget - 1` members. Letting it pin or unpin
  that prefix would leave the suffix as it was, and a pin from another mask
  could put the leader on a CPU an untouched member holds. So a narrower root
  scope runs no pin or unpin section, is not counted in `pin_skips`, and
  dispatches under the team's current placement like any plain team token; its
  oversubscription and the members' spin budget are still decided from its
  caller's set. Its leader is placed all the same, on `home` of that placement
  through `pin::home_pin` like a reborrowed scope's (threadteam-rs#53: it used
  to stay wide and run on the CPUs its members are pinned to). A team that
  wants a new placement opens a root scope with its whole budget.
* **A placed leader is paced by `S`, not by its one-CPU mask.** Inside a scope
  or a section the leader's mask is `{S[0]}` while "CPUs the caller's team is
  spread over" is still `|S|`, and oversubscription is the second question.
  `pin::available` answers `|S|` on a thread this crate placed (it reads the
  thread's `PLACED` record), and `Team::token` and `Exec::team_with` measure
  oversubscription between placements, where the surprise cannot happen: the
  leader of a `Team` is no longer pinned between sections, so the caller's own
  mask is the team's set. 0.5.0 answered 1 on the team-pinned thread, so every
  token a `Team` minted there was marked oversubscribed and its members ran with
  `THROTTLED_POLLS` (threadteam-rs#33). A team whose `S` is genuinely too small
  places no leader, so it is still paced by its mask.
* **Lending is the exception.** `Exec::external` lends the token's members to
  the pool for the call; each restores its own mask before it serves Rayon
  work and is re-pinned only at the next pin section, that is at the next
  whole-team root scope.

**A `Team`'s leader (tensor4all/threadteam-rs#33, #45).** An engine that owns a
`Team` (tenferro's `CpuEngine`) mints a token per operation with
`Team::exec`/`try_exec` and dispatches on it with `fork_sized`, `for_ranges` or
`broadcast`; it never opens a team scope. `into_team` opens one scope, to
acknowledge, which places the members on `S[1..]` for the life of the
reservation; with the leader's pin held by the scope, the leader got `S` back
when that scope ended, before the first real dispatch, and nothing placed it
again. That is the path reported in #33 (`fork_sized` at width 96 on a 96-CPU
two-socket host: 7 of 16 processes at about 11 ms per call instead of about 5.8
ms, leaders off `S[0]`).

**The rule: the leader's pin belongs to the section** (tensor4all/threadteam-rs#52).
`into_team` pins nothing: the caller gets its own mask back the moment the
acknowledgement scope ends, and the team's tokens place their leader only for
the section (or reborrowed scope) they dispatch. The team keeps one record of
where its members are, `pin::Home` — `S[0]` of the set the last whole-team root
scope placed them in, `0` for none — written by that scope and read by every
section. A team token the calling thread leads opens every section it dispatches
outside a scope (`fork`, `fork_sized`, `for_ranges`, `broadcast`) with one
`sched_setaffinity` onto `home` and one back to its own mask at the end of the
section, also on unwind (`pin::home_pin`, a guard); a section inside a team
scope is already covered by the scope's pin and is not pinned again, because the
mask this thread reads there is the one CPU it was placed on and a mask of one CPU
is never narrowed. So the leader is on `S[0]` while a section runs, which is what
#33 is about, at one `sched_getaffinity` and two `sched_setaffinity` calls per
outermost section (2.1 us on an empty 8-way section). A token section
(`Exec::fork`, `for_ranges`, `broadcast`, a reborrowed team scope) with a nonzero
`home` inside a placed scope reads the mask and is refused by it, so it pays that
one `sched_getaffinity` and no `sched_setaffinity`. The read is what buys the
refusal, not a memory of the enclosing scope — and it comes after the `home`
test, so a team whose placement is `0` (pinning off, or a scope that placed
nobody) pays neither. A `TeamExec` section
(`run`, `run_with`) dispatches without calling `home_pin` at all, so it pays no
affinity call either way. A token section inside `TeamExec::external` reads `S`
back and is pinned in full, on the nested team's home (threadteam-rs#54). The scope itself pays one pin (`LeaderPin` for a
whole-team scope, `home_pin` for one that is narrower). The
leader is placed for a scope that is not the root token's too (a reborrow, which
carries no leader guard of its own: faer's `with_team` shape), and for a child
that was handed out without a dispatch of its own (`fork_sized` with one share
runs inline). Three things are never placed by `home_pin`, which is what a token
section or a reborrowed scope uses: a thread whose own mask has one CPU or does
not hold `home` (a reserved member pinned to `S[k]`, a caller that narrowed
itself away from `S[0]`; a leader is never moved outside the mask its caller gave
it), a child that a member leads (its share was not given the leader's CPU), and a
thread whose mask has left the placement it already has — a caller that widened
itself out of `S` — which is refused rather than moved outside that placement.
Being placed is not itself a reason to decline: a team scope entered inside
`TeamExec::external` places this thread on the nested team's home and puts the
enclosing placement back when it ends (threadteam-rs#54). A whole-team root scope
is not described by this list either: it places through `pin_scope`, which reads
the caller's mask as the scope's own set and enters `LeaderPin` with it, and then
attempts `home_pin` (`src/exec.rs:1045`, `:1055`) — a no-op there, since the
mask is already the one CPU the scope pinned, and it costs the read.

Why not the `Team`-held pin of #45: it keeps the calling thread on `{S[0]}` for
the life of the team, and the caller's thread is not ours. `tenferro`'s
`CpuEngine::new` runs `pool.into_team(width)` on the thread that builds the
engine, keeps the `Team` for the engine's life, and constructs its vendor
backend from that same thread, so a lazily created MKL/OpenBLAS thread pool, any
plain `std::thread::spawn`, and a second `Pool` built there all inherited a
one-CPU mask (measured on evo2, 32 CPUs, width 8, see #52). Why not the same pin
held by the token instead: a token is not the unit that needs the pin either,
and a guard that is entered on one thread and dropped on another has to restore
nothing on the second one, which the section guard cannot get wrong.

The record belongs to the team. A `Team` is `Send` and `Sync`: a token used on
another thread places that thread for its sections too, from the same record,
and only inside the mask that thread has; its mask is what it was afterwards. A
whole-team root scope from another thread re-places the team, so the record
follows the latest placement and a section never sends its leader to a CPU a
member of an older placement holds. A scope that places nobody clears the
record. A team of one, a mask with fewer CPUs than participants and pinning
turned off place nothing, as before.

The earlier fix of #33, a home for the leader kept on the `Team` and a nudge onto
it before every section (#43, `e4fe0f4`), **landed, not measured**: `01bb587d`,
which the reporter re-ran, is the commit before that change, and the first nodes
of the same model did not reproduce #33 on 0.4.0 either (null controls), so no
run has measured #43 either way. It was removed in #45 for the scope-held pin.
The per-section pin added for #52 is the same unit as that nudge but *holds* the
pin for the section instead of handing `S` back immediately, because the drift
measured below at n = 4096 shows that a leader which keeps its whole mask does
not stay where it was moved.

Measured on broccoli (64 CPUs, otherwise idle), `into_team(64)` and then a fresh
`team.exec(64).fork_sized(&[1; 64], ..)` per call, about 2.8 ms of arithmetic per
share, 40 timed calls per run. With the scope-held pin (`e59dbbc`): with a 20 ms
sleep before each call, the leader was on `S[0]` in 17, 2 and 22 of 40 calls in
three runs, medians 13.8, 15.7 and 10.5 ms against a minimum of 2.9 ms; three
later runs had it on `S[0]` in 40 of 40 and medians of 2.9-3.0 ms, so that mode
depends on where the scheduler wakes the leader. With the caller moved onto `S[1]`
for one share's worth of work before each call, the leader was on `S[0]` in 0 of
40 in all six runs, median 15.7 ms. With the team-held pin, alternating
`fork_sized` and `for_ranges` on fresh tokens, three runs of each mode (no gap,
sleep, move): the leader on `S[0]` (`sched_getcpu`) with mask `{S[0]}` in 40 of 40
calls of every run, every member alone on its own pinned CPU in 40 of 40, medians
3.1-3.2 ms (minimum 3.0 ms), and `TeamExec::external` widened to `S` and restored
`{S[0]}`. The probe is not in the repository; the tests
`a_team_places_its_leader_for_the_sections_it_dispatches_and_no_longer`,
`into_team_leaves_the_callers_cpu_set_as_it_was`,
`a_thread_spawned_while_a_team_lives_is_not_confined_to_one_cpu` and
`the_reported_wide_fork_sized_path_keeps_every_participant_alone_on_its_cpu`
assert the placement on this path.

Measured on evo2 (32 CPUs allowed, `taskset -c 0-7`, release, `into_team(8)` and
a fresh `team.exec(8).fork_sized(&[1; 8], ..)` per call, 40 calls, one run per
cell; the shares do no work, so the cell times the dispatch and the placement,
not arithmetic). The mask rows are `/proc/thread-self/status`
`Cpus_allowed_list` of the calling thread:

| phase | 0.5.1 (team-held pin) | this change |
| --- | --- | --- |
| before the `Team` | `0-7` | `0-7` |
| after `into_team(8)` | `0` | `0-7` |
| thread started while the team lives | `0` | `0-7` |
| 4 workers of a second pool built there | all `0` | all `0-7` |
| after the `Team` | `0-7` | `0-7` |
| leader in its share, caller left where it is | `S[0]` 40/40, 1.7 us per call | `S[0]` 40/40, 3.8 us per call |
| leader in its share, caller moved off `S[0]` first | `S[0]` 0/40, **5.9 ms per call** | `S[0]` 40/40, 10 us per call |

The last row is #33 itself: on 0.5.1 the leader's placement is the team's, and a
caller that is not on `S[0]` dispatches with the leader on a member's CPU, which
costs a scheduler slice per section (5.9 ms against 10 us). With the section
pin the leader is home for every section, at **one `sched_getaffinity` and two
`sched_setaffinity` calls per outermost section**: 1.6 us for the set/restore
pair on this host, 2.1 us for the whole hook on an empty 8-way section. A token
section with a nonzero `home` inside a team scope pays the `sched_getaffinity` and
none of the writes, because the mask it reads has one CPU; a `TeamExec` section
pays nothing, because it does not call `home_pin`; a section inside
`TeamExec::external` pays the full
pair again, because `external` hands `S` back (threadteam-rs#54).

`Pool::pin_skips` counts every root scope that could not be placed, as before.
The doc-hidden `Pool::leader_moves` counter belonged to the nudge and is gone.

### Waiting policy

`PoolBuilder::idle_spin` (default 200 ms, may be zero) and `idle_runs_rayon`
govern an *unreserved* idle member, the state a member is in after a scoped
team (`Exec::team`) ended: it stays claimable without a Rayon wake-up until
`idle_spin` passes or it has polled for the pool's spin budget, whichever
comes first, then returns to Rayon. `WaitOverride` overrides them per scope.
Reserved members do not use `idle_spin` any more; their waits, and the joins
and barriers, follow [`waiting.md`](waiting.md).

### Unwind

Every section joins all dispatched shares before `run`, `run_with`, `split` or
`fork` returns, also on unwind (a drop guard waits on the completion counter).
A panic in any participant is caught on its thread, stored (first one wins),
and the section's barriers are cancelled: `Participant::barrier` returns
`Err(Cancelled)` so peers leave instead of waiting for the panicker. After the
join the payload is re-raised on the leader. A panic in a child leader
propagates through the parent's section the same way. The reservation is
released when the `team` scope ends, on unwind too.

### Ledger and fork children

Each participant counts as one live task of its root for the time it runs its
share, and the leader stops counting while it waits, so live tasks never exceed
the budget (`tests/team_ledger.rs`). A `fork` child may call `Exec::team`; it
leads on its own thread and reserves free workers, or is refused. Merely running
on a pool worker grants no reservation.

### Not provided

Reproducible reductions are left to numerical kernels (fixed logical chunks,
chunk-owned partials, fixed merge order): the crate gives no stealing within a
team, so a section's shares are exactly the participants. Cache-domain layout is
not handled, and placement is per team (on by default, see above): there is no CPU
allocator, so two teams whose leaders see the same CPU set would place onto the
same CPUs.

## Dispatch cost and the parallel rule

A token states what one parallel section costs it; a kernel states how long its
region takes on one thread (`docs/design/thresholds.md` has the data). A region
runs in parallel when

```
serial_estimate >= PAYS * sections * D(width)        (PAYS = 3)
```

* `Exec::dispatch_cost(width)` and `TeamExec::dispatch_cost(width)` return
  `D(width)`, the steady-state cost of one section (back to back, shares that
  carry work). A serial token, `width < 2`, or a team of one gives
  `Duration::MAX`; `width` is clamped to the budget.
* The kind of token is read at call time: team authority (`Team` tokens and their
  `fork` children) is `DispatchKind::Team`; a plain token whose caller is a worker
  of its pool is `InPool`; a plain token called from outside is `Outside`. No
  thread-local state is involved and nothing allocates.
* `DispatchCosts` holds one `WidthCosts` per kind. A `WidthCosts` stores the
  measured anchors of its row (widths 2, 4 and 6 from `new` / `flat`, or any
  strictly increasing widths from 2 to `u16::MAX` from `from_anchors`, at most
  `WidthCosts::MAX_ANCHORS` = 12 of them): the value at the first anchor below
  it, linear interpolation between two anchors, and **linear extrapolation above
  the largest anchor**, `cost(w_m) * w / w_m`. A wide section therefore never pays
  the cost of the largest measured anchor; a row measured at width 8 keeps its
  measured value there instead of the extrapolation. The model is evaluated on
  every lookup by a scan over at most 12 anchors (no table, no allocation).
  `DispatchCosts::for_class(HostClass)` is the row of a host class, and
  `DispatchCosts::host_default()` the row of `HostClass::detect()`. Classes are keyed by the kernel dispatch the
  machine takes, not by the CPU model: `x86_64-avx512` (runtime `avx512f`), `x86_64-avx2` (`avx2` + `fma`),
  `x86_64-scalar`, `aarch64-apple` (aarch64 on macOS), `aarch64`, `generic`; detected once per process.
  `x86_64-avx512` holds the measured Ryzen row (team 0.6/0.7/0.9 at widths 2/4/6 and the measured 0.98 at width 8,
  in pool 3.0/3.6/3.5, outside 9.7/10.1/8.7 us), `aarch64-apple` the measured Apple M5 row at widths 2, 4 and 6,
  `x86_64-avx2` a measured Zen2 team row at ten widths from 2 to 64 (1.42 to 9.33 us; in pool and outside generic;
  thresholds section 3.7), and every class without a measurement the conservative generic row (3, 15, 40 microseconds at the anchors; the
  extrapolation charges 32, 160 and 427 microseconds at width 64). A consumer that measured its own host at more
  widths overrides the row with `WidthCosts::from_anchors` (up to `WidthCosts::MAX_ANCHORS` = 12 anchors). The rates
  of the consumer crates (strided-rs, tcontract-rs, faer) are
  tables of their own, keyed by the same class names, and a measured class's rates and dispatch costs come from the
  same machine in the same clock state (the exception is the `x86_64-avx2` team row, whose consumer rates are still
  generic; thresholds section 3.7): boost on/off moves both by about 1.66x and leaves the ratio stable within a
  few percent (`faer-benchmark` `raw-exec3-boost`, `tables-*-exec3-boost.md`). Each crate carries an identical
  `HostClass` and `detect()`, kept in sync with `threadteam`'s by hand.
  `PoolBuilder::dispatch_costs` overrides it.
* `pays_at(estimate, sections, cost)` is the one rule; `Exec::pays` and
  `TeamExec::pays` apply it to the token's cost. `Exec::width_for(estimate,
  sections)` (and `width_for_by` for a consumer's own cost function) returns the
  width in `1..=budget` that minimises `sections * D(k) + estimate / k` among the
  widths that pay, or 1.
* `sections` is the number of dispatches the region really performs. It is a
  count, not a tuning knob: a region that enters the token twice declares 2.
* The costs are steady-state values. The wake penalty after members parked (about
  27 microseconds on the measured host) is documented and is not folded into
  `PAYS` or the table.

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

None at present. Explicit uneven shares are `Exec::fork_sized`.
