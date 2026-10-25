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
member slot and bumps its `gen` counter with release ordering; members spin on
`gen` and `state` (pure spin, then `yield_now` between polls) and run their
share and decrement the completion counter. After the reservation's park delay
(the effective `idle_spin`) without work, a member parks (see below). The leader runs
share 0, then waits for the counter. Nothing is allocated and nobody is woken
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
members (zero idle spin, woken) and then drops the pool, which joins the
workers; `into_pool` releases the members and returns the pool instead. The
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

**Parking.** A reserved member that finds no new section for the park delay
(the effective `idle_spin` of the reservation: the pool default or the
`reserve_with` / `team_with` override) parks and stays reserved. All parked
members of a pool wait on one condition variable (`park_lock`, `park_cv`), with
a counter `parked` of members inside or entering the wait. The protocol:

* A member raises `parked` (acquire-release increment), takes the lock,
  re-checks `gen`, `state` and `lent` and waits only while none changed; it
  lowers `parked` when it leaves.
* A leader changes the state a member waits on (bumps `gen`, stores `IDLE`,
  sets `lent`), then reads `parked` with an acquire-release `fetch_add(0)`. If it
  is nonzero, the leader takes and drops the lock and calls `notify_all`. Either the member sees the changed
  state after raising `parked`, or the leader sees `parked` (both are read-modify-writes of one atomic, so the
  argument needs only its modification order and release sequences, not a total
  order of SeqCst accesses, which `loom` does not model); and because the
  leader takes the lock before notifying, the notification cannot fall between a
  member's check and its wait (the usual condition-variable handshake).
* With nobody parked a dispatch pays one uncontended atomic read-modify-write. A broadcast also wakes
  parked members of other teams of the same pool, which re-check and sleep again;
  one broadcast replaces a chain of per-thread unparks, so the cost on the
  leader is one syscall instead of one per member.
* Spurious wake-ups loop. There is no per-member wake state and no tree: an
  earlier per-member design (unparks fanned out through the members) lost a
  wake-up when a woken member unparked a child before the leader had published
  the child's generation; the broadcast design has no such ordering to get wrong.
  `tests/team_owner.rs` keeps the stress that reproduced it (zero spin, cold
  acquisitions, CPU burners, a watchdog).

The handshake is the small `Parker` type in `src/park.rs`, built on a `sync`
shim (`std`, or `loom` under `--cfg loom`), and checked three ways:

* **Model check.** `tests/loom_park.rs` runs the real `Parker` under `loom` with
  one leader and two members: a published section, a released reservation, and a
  member that parks twice. An earlier version of the handshake (a SeqCst store/load pair,
  Dekker style) was reported as a deadlock by loom, which led to the
  read-modify-write form above. CI job `loom`:
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

**Quality of service.** `PoolBuilder::qos(Qos)` sets the class of each worker
when it starts (`pthread_set_qos_class_self_np` on Apple targets, nothing
elsewhere; no platform type is public). The default is `UserInitiated` on
macOS. `threadteam` never changes the caller's class, but the calling thread
leads every section, so `raise_current_thread_qos(Qos)` raises the calling
thread to at least the given class and never lowers it (the current class is
read first); an engine calls it once for its leading thread. In the wake probe
the class (builder default, explicit `Default`, `UserInteractive`) made no
measurable difference to the wake latency on this host. Dropping the
team stores `IDLE` with a zero idle spin and wakes them, so the members return to
Rayon at once.

### Waiting policy

`PoolBuilder::idle_spin` (default 200 ms, may be zero) and
`PoolBuilder::idle_runs_rayon` (default `false`) set `WaitPolicy` defaults;
`acquire_timeout` is separate. `WaitOverride` (partial policy) applies to one
operation: `Exec::team_with` for a root reservation, `TeamExec::with_wait` for a
scope. Children inherit the effective policy and may override it for their own
scope. The root reservation's effective policy is stored in its members' slots
when it ends and governs the idle interval that follows; the interval starts
when the reservation ends, not when the member last computed. A reserved member
never leaves on a timeout and never runs an unrelated Rayon job (it parks, still
reserved, after the park delay), whatever the policy, so an override cannot weaken a reservation, and a child's override
expires when the child returns and cannot release its parent's members.
With `idle_runs_rayon`, an idle member calls `rayon::yield_now`, whose job runs
to completion before the member is claimable again. Without it, an idle member
leaves its worker out of Rayon service for up to `idle_spin`; a following
`broadcast` or a Rayon job that needs every worker then waits for them, so a
caller that alternates with such work sets a short spin or calls
`Pool::release_idle`.

`Pool::release_idle` bumps an epoch: idle members leave at once, reserved
members leave when their reservation ends. It never revokes a live token and
does not wait, so it may be called inside a reservation. Dropping the pool
signals shutdown, which idle members honour. `Pool::resident_members` reports
the number of awake members (diagnostic).

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
team, so a section's shares are exactly the participants. Cache-domain layout and
pinning are not handled.

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
