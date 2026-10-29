# Own worker threads (threadteam-rs#15)

threadteam stops running on a private Rayon `ThreadPool`. Its workers are `std::thread`s kept in a process-wide **reserve**; a `Pool` leases a fixed number of them and returns them when it is dropped. The waits stay libgomp's (`waiting.md`). Rayon becomes an opt-in interop layer.

Design inputs:
- pi-council, plan mode (GPT-6-astra chair, DeepSeek critic, GLM minimalist). The brief and the synthesis are in the shinaoka-advisor run directory `work/runs/threadteam-own/`.
- The runtime sources read for it, without copying code: libgomp, LLVM libomp, oneTBB.
- The measurements of threadteam-rs#14, which stays a draft as their record.

## Why: the one-call pool

faer's `with_team` builds a `Pool`, and with it 64 fresh threads, per call. On a 64-CPU set under foreign load those threads start stacked: the scheduler places a new thread before it knows its load. The team's spinning waiters then never leave a CPU idle that could pull one over, and every 20 µs section of the call waits for a descheduled member (faer#107).

The structural difference is the spawn, not the waiting. In one broccoli run (faer#107, 13:10Z, 0.1.3, 64 threads, `gemm128`), the same acquisition path at the same width gave:

| arm | what happens per call | median call |
| --- | --- | --- |
| `into_team_per_call` | the pool is held; members are acquired and released per call | 0.03 ms |
| `fresh_pool` | a new pool, so 64 new threads, per call | 15.99 ms |

On #14, no waiting policy fixed the fresh pool and kept held teams at libgomp's speed at the same time:
- the pool-wide throttle fixed (a) but cost (c);
- libgomp's waits met (b) and (c) but left the fresh pool at 11 ms;
- a wall-clock window could not tell the two apart.

libgomp creates its threads once per process (per master thread) and reuses them for every region, docked on a futex between regions. TBB's RML server owns one process-wide set of workers that every arena reuses, each sleeping on its own monitor when idle.

The answer to question (i) of #15 is therefore (ii): if no `Pool` spawns threads, there is no fresh-thread stacking to detect.

**Caveat.** The first pool of a process still spawns its workers. The acceptance criterion (a) judges a process by its median call over 50, so one cold first call does not make it slow, and the benchmarks report the cold first call separately.

## Decisions

### Reserve and lease
- **One process-wide reserve: a list of idle workers.** It is created by the first explicit `Pool::build` that wants shared workers. Serial, budget-one and one-task entries never reach it, and no API submits work to it without a pool, so "no implicit global pool" holds.
- **Compatibility.** A worker of the reserve was spawned with the default stack (16 MiB), the platform's default QoS and no hooks. A pool with any `thread_name`, `on_start`, a different stack size or a QoS gets **dedicated** workers: spawned for it and joined when it is dropped, exactly as before. tcontract's C API names its workers and sets a thread-local in `on_start`; it keeps today's semantics.
- **`Pool::build`.** It takes `width` idle workers from the reserve and spawns the missing ones; a spawn failure is a build error. A worker learns its pool and its index when leased.
- **`Pool` drop (shared workers).** Every worker is told to detach. Each one finishes, restores its own affinity mask (a pinned member) and acknowledges. Only then is the worker back in the reserve. The drop waits for all acknowledgements (quiescence) and does not join.
  - The reserve keeps at most `C` idle workers, `C` being the available parallelism when it was created; a worker beyond the cap exits and is joined by the drop.
  - A pool dropped from one of its own workers (a worker running a task that owns the pool) cannot wait for itself: it detaches the others, and that worker detaches when its task returns.
- **Contract change.** "Dropping a `Pool` joins every worker thread" now holds for dedicated workers only; a shared worker outlives the pool, parked in the reserve.

### The worker loop
Each leased worker runs one native loop over its member slot, the slot of the existing team machinery (`awake.rs`). Slot states:

| state | meaning |
| --- | --- |
| `IDLE` | serves the pool's queue and its own mailbox; claimable by a team (CAS) |
| `RUNNING` | runs a task of the pool's queue or a broadcast share; not claimable |
| `RESERVED` | a team member, as before; with `lent`, it also serves the pool's queue |
| `DETACH` | the pool is being dropped; the worker acknowledges and leaves |

Work, in the order the loop looks for it:
1. A **broadcast share** in its mailbox.
2. A **task** from the pool's queue.
3. Otherwise the staged wait: spin, yield, then park on its slot's `Parker`, until the queue generation, the mailbox, or the state changes.

The `ABSENT` state, the claim jobs (`claim_cold`) and the `YIELD` state of the Rayon design disappear: every leased worker is always in its loop.

### Plain tokens: `fork`, `for_ranges`, `external`
- **A call publishes a stack-allocated batch.** The batch holds the task function, the next unclaimed index, the task count and a completion counter (`Completion`). It is linked into the pool's queue: an intrusive list under a `std::sync::Mutex`, so nothing is allocated.
- **Claiming and freeing.**
  - A worker claims an index **under the queue lock**, and the claim that takes the last index unlinks the batch.
  - The batch's owner frees it only after the completion counter reaches zero.
  - So no worker can reach a batch after it was unlinked, and an unlinked batch is gone only after all of its tasks finished.
- **The caller.**
  - A caller that is a worker of the pool claims its own batch's remaining indices before it waits.
  - A caller from outside the pool waits (staged; it sleeps on the pool's `sync` parker, woken by the last completion), as "a plain thread enters the pool once and waits" says today.
- **No deadlock.** Every unclaimed task is claimable by its owner, and an owner that waits only waits for tasks that are already running. A running task blocks only on its own children, which it can run itself. So no cycle of waits is possible; the worst case is a nested fork that runs its children on its own thread.
- **Budget conservation.** The structure is unchanged: a batch has at most its token's budget of tasks, and the shares of `fork` sum to the budget.
- **`external`.**
  - A plain token: the callback runs on a pool worker, as a one-task batch, or inline when the caller is a worker of the pool.
  - A team token: its members are lent to the pool's queue for the call, and serve tasks the callback's code publishes to the same pool.

### Broadcast
- The pool holds a broadcast mutex for the whole broadcast. The caller is never a worker of the pool (`CallerInPool` refusal), so holding it cannot deadlock a worker, and concurrent broadcasts run one after the other, in the same order for every worker.
- The share for worker `i` goes into slot `i`'s mailbox. The caller waits for all `budget` shares, as today; a reserved member serves no broadcast, again as today.

### Identity without thread-locals
Each slot records its worker's `ThreadId` when leased. "Is the caller a worker of this pool, and which one" is a scan of the pool's slots (`width` comparisons), done at entry points only, never on the team dispatch path.

### Rayon interop: opt-in
The Rayon lending of `external` has no production caller (downstream use: tests and benchmarks). It becomes a non-default feature, `rayon`: `Exec::external_rayon(f)` runs `f` inside `install` on a private Rayon pool of the token's width, built lazily per pool. Those are separate threads, which the feature documents. Without the feature, Rayon is not a dependency at all.

### Unchanged
- **The team machinery:** sections, `Completion`, `Barrier`, `Parker`, pinning, root-scope oversubscription, `Team`/`TeamExec` and the spare worker. A pool of `w` workers gives a team of `w` participants with an outside leader, so one worker always stays free for the queue and for `external`. threadteam-rs#12 stays open.
- **The waits** (`waiting.md`): 300000 polls, 16 yields, futex; 100 polls in a statically oversubscribed scope.

### Public semantics that change (for the maintainer)

| item | before (Rayon) | now |
| --- | --- | --- |
| `Pool` drop | joins every worker | dedicated workers: joins. Shared workers: waits for every worker to detach, then returns them to the reserve |
| `idle_runs_rayon` | an idle member runs queued Rayon jobs | no effect: an idle worker always serves the pool's queue (kept for compatibility) |
| `idle_spin` | how long an unreserved member spins before returning to Rayon | how long an unreserved worker may spin before it sleeps; the spin budget bounds it as well |
| `release_idle` | idle members return to Rayon | idle workers stop spinning and sleep now |
| `resident_members` | members in a member loop (reserved or idle) | workers reserved by a team, or idle and awake |
| `external` on a team token | lends members to Rayon | lends members to the pool's queue; Rayon lending is `external_rayon` under the `rayon` feature |
| feature `parallel` | pulls Rayon | pulls nothing; `rayon` is a separate opt-in |

## Pre-review resolutions

The gpt-6.1-sol pre-review (`work/runs/threadteam-own/pre.out`) found one deadlock and seven gaps in the first draft. The design above is amended as follows.

1. **Cross-pool liveness (A → B → A).**
   - **The problem.** A worker of pool A that waits for a batch it published on pool B must not just sleep. If every worker of A does so, and B's tasks publish back on A, nobody runs A's queue.
   - **The rule.** A waiting worker of any pool keeps serving **its own pool's queue** (and its broadcast sequence) until its batch completes.
   - **The wake.** The batch records who to wake: the waiting worker's own slot parker, or the target pool's `sync` parker for a plain outside thread. That waiter sleeps on a predicate that covers both its batch and its pool's work.
   - **Same pool.** A worker caller of the same pool still claims its own batch's indices first.
2. **Broadcast.**
   - **Publication only.** The pool's broadcast lock covers publication, not completion.
   - **Ordering.** Broadcasts are numbered in publication order and linked into the pool's broadcast list. Each worker runs the broadcasts that target it in sequence order, skipping the others. A broadcast stays linked, and alive on its caller's stack, until all of its targets have finished; its caller unlinks it.
   - **No lock held while waiting**, so a cooperative waiter re-entering a broadcast cannot block on a lock held lower on its own stack.
3. **One state word per slot.** Bits `RESERVED`, `LENT`, `BUSY`, `DETACH`.
   - A team claims a slot only from `0` (compare-and-swap).
   - A free worker takes work only by winning `0 → BUSY`, and a lent member only by winning `RESERVED|LENT → RESERVED|LENT|BUSY`. Both happen before it removes a task from the queue; a failed compare-and-swap leaves the task unclaimed.
   - Releasing a reservation clears `RESERVED|LENT` only. A worker still running a lent task stays `BUSY`, so it is not claimable until it finishes.
   - Detach sets `DETACH`. Finishing a task clears `BUSY` only, so it cannot erase `DETACH`.
   - Nested execution (a waiting worker helping its queue) keeps `BUSY`.
   - These transitions are a small type (`SlotState`) that `loom` checks directly.
4. **Lease state and self-drop.**
   - **Ownership.** Every leased worker holds an `Arc` of the pool's shared state through its epilogue. Its detach acknowledgement is its last access to that state.
   - **Drop.** `Pool` drop marks every slot `DETACH`, wakes the workers and waits for the acknowledgements of all workers but itself.
   - **Self-drop from a worker.** It does not wait for the dropping worker, which acknowledges after its task returns. The other workers cannot be running work of this pool, because any such work borrows the pool, so their acknowledgements come promptly.
   - **Locks.** No reserve lock is held while waiting.
   - **Workers above the cap.** They exit, and their handles are detached; only dedicated workers are joined.
5. **Affinity on lease.** `Pool::build` reads the calling thread's affinity mask, and every worker applies it to itself when it is leased (Linux), as a newly spawned thread would inherit it. A worker whose pin restore failed is not returned to the reserve; it exits.
6. **Reserve cap.** The cap is the largest of the available parallelism and every width leased so far. So repeated wide pools (faer's per-call `with_team`) keep reusing their workers, whatever the first caller's affinity was.
7. **Panics.**
   - Every claimed task and broadcast share runs under `catch_unwind` and always reaches its completion. The first payload is kept on the batch.
   - The owner waits for completion (all claimed tasks, including tasks still unclaimed, which it runs itself) before it resumes the first panic.
   - A broadcast share is unlinked by its caller only after completion; a worker never clears a share after signalling it.
   - `Completion::finish`'s rule holds: nothing touches a batch after its final decrement.
8. **Verification.**
   - The loom list grows to the transitions above: state word, queue claim against reserve, last completion against reclamation, broadcast sequence, detach against task end, and lease attach against first publication.
   - The allocation test of the plain paths drops its Rayon-era minimum-window exemption and counts strictly after warm-up.

Kept against the review's delete list: the opt-in Rayon layer, which the maintainer asked for ("Rayon との相互運用が要る機能は opt-in の層として残す"), and `idle_runs_rayon`, as a documented no-op, so the public API does not break.

## Verification
- **loom, new paths, small models:**
  - a batch published with one owner and two claimers, the owner asleep until completion;
  - an idle worker's wait on the queue generation, its mailbox and its state, against a publish and a detach;
  - the detach handshake of a pool drop.
- **The existing tests:** alloc-free (the plain paths and the team paths, warm), ledger, compile-fail, no-public-rayon, and the feature matrix.
- **New tests:**
  - reuse: thread creation counted; a second pool of the same width spawns none;
  - quiescence at drop; dedicated pools keep join-on-drop;
  - nested forks; a worker caller running its own batch;
  - concurrent broadcasts; a pinned member restoring its mask on detach.
- **Benchmarks:** `scripts/bench-ab.sh` against the base (`a3a9320`) and libgomp (`OMP=1`), and faer's `acquire_probe_107` (all arms), on broccoli and evo. The `fresh_pool` rows (`build`, `acquire`, `call`, `release`, `drop`) show the reuse, and the cold first call is reported separately.

## Acceptance
As on #14:
- **(a)** faer#107 slow mode zero in every arm;
- **(b)** quiet hosts within 10 % of 0.1.3 (`held_team`, `held_pool`, gaps 0-1 ms);
- **(c)** under foreign load no worse than libgomp;
- **(d)** the 10 ms gap governed by the spin budget.
