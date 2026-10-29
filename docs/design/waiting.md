# How team threads wait

This record covers how the threads of a team wait:
- a reserved member, between sections and between calls;
- the leader, in `Section::join`;
- every participant, in `Section::barrier`.

It replaces the waits of 0.1.3 with the shape of the OpenMP runtimes: a bounded spin, then a short yield phase, then a sleep on a futex. Maintainer decision, 2026-10-29: 「threadteam の待ち方を OpenMP に近づけるのがいいと思う」. The trigger is tensor4all/faer#107.

## What 0.1.3 does, and why it stacks

| wait | 0.1.3 |
| --- | --- |
| reserved member polling `gen` (between sections and between calls) | one PAUSE per poll; parks after `idle_spin` of **wall time** (200 ms), or after 64 polls when the scope is oversubscribed |
| leader in `Section::join` | one PAUSE per poll, **never sleeps**; when oversubscribed, ~100 PAUSEs, then `yield_now` per poll |
| participant in `Section::barrier` | same as `join` |
| unreserved `IDLE` member | `backoff` (2^14 PAUSEs, then `yield_now`) for `idle_spin` of wall time, then returns to Rayon |

"Oversubscribed" means more participants than CPUs in the caller's affinity set. Foreign load is not counted.

faer#107, broccoli, 64 threads on 64 CPUs with another user's processes at 5-13 %:
- a fresh pool per call is slow in 20/20 processes (gemm 128: a fixed 16 ms against 0.02 ms);
- chol 2048 is slow in 10/10 processes even with the pool held;
- at width 48 on 64 CPUs the slow mode disappears (0.02 ms, 0 % stacked).

The mechanism:
- At width = CPUs, every CPU holds a participant that is either working or spinning, so no CPU is ever idle.
- Linux moves a runnable thread from a busy CPU to another CPU mainly when that CPU goes idle (newidle balancing). Periodic balancing has nothing to gain from a 2-to-1 move.
- So once a member shares a CPU with another runnable thread (a peer, the spare worker, or a foreign process), the pair stays put. Every section then waits for the descheduled member's next time slice.
- `yield_now` does not help: it hands the CPU to a thread that is already on it, and it leaves no CPU idle.

The only wait that frees a CPU is a sleep.

## What the OpenMP runtimes do (read from their sources; no code is taken)

**libgomp** (gcc `libgomp/config/linux/wait.h`, `bar.c`, `team.c`, `env.c`):
- **Wait.** `do_wait` polls the futex word up to `gomp_spin_count_var` times, with one `cpu_relax` (PAUSE) per poll, then calls `futex_wait`.
- **Spin count.** The count is in polls, not in time. The default without `OMP_WAIT_POLICY` is 300000, "a rough estimation of 100000 spins per msec", so about 3 ms, "e.g. 5 times longer or 5 times shorter" depending on the CPU. It is 0 under `PASSIVE`, 3·10^10 under `ACTIVE`, and 1 on x86 hybrid CPUs (`config/linux/x86/spincount.h`).
- **Throttling.** When libgomp's own threads outnumber the available CPUs (`gomp_managed_threads > gomp_available_cpus`), the count drops to `gomp_throttled_spin_count_var`: 100 by default, 1000 under `ACTIVE`.
- **Foreign load.** It is not considered.
- **Barriers.** A barrier is a generation word. The last arriving thread bumps it and calls `futex_wake(INT_MAX)`; the others `do_wait` on it.
- **Region start.** Pool threads wait between regions on the `threads_dock` simple barrier, and the master's arrival releases them. That barrier covers the whole team, because a region always uses the whole team.

**LLVM libomp** (`openmp/runtime/src/kmp_wait_release.h`, `kmp.h`, `kmp_global.cpp`, `kmp_runtime.cpp`):
- **Wait.** `__kmp_wait_template` spins with `KMP_CPU_PAUSE` (TPAUSE where available). It yields only when the runtime's own threads exceed the available processors (`KMP_OVERSUBSCRIBED`, `__kmp_use_yield`). It suspends once `KMP_BLOCKTIME` is exhausted.
- **Blocktime.** The blocktime is **wall time**, checked against `rdtsc` (`KMP_BLOCKING`). The default is 200 ms (`KMP_DEFAULT_BLOCKTIME`), and 0 on hybrid CPUs. With `KMP_ADJUST_BLOCKTIME`, it drops to 0 when there are more threads than processors.
- **Library mode.** The default `KMP_LIBRARY=throughput` keeps the blocktime. `turnaround` yields only when oversubscribed.
- **Hot teams.** A team's threads are kept across parallel regions and wait in the fork barrier under the same blocktime rule.
- **Foreign load.** It is not considered either.

**Implications for threadteam:**
1. Both runtimes eventually sleep in **every** wait, and neither leaves a waiter spinning forever. That is what lets an idle CPU appear and pull a stacked thread.
2. libgomp counts **polls**, and libomp counts **time**. A count is consumed only while the waiter runs, so a waiter that is descheduled behind a peer on the same CPU does not use up its budget by waiting in the run queue. A time budget expires there too, and makes the waiter sleep for the wrong reason. threadteam 0.1.3 uses time, like libomp, but 200 ms is far longer than any gap that matters.
3. Neither runtime reads foreign load. threadteam does, below, because the faer#107 configuration (width = CPUs, foreign processes) is exactly where the static participants-against-CPUs test says "not oversubscribed".

## Design

### Waits

Every wait is an episode of three phases.

1. **Spin.** Poll the condition with one PAUSE (YIELD on aarch64) per poll, for at most `spin_polls` polls. The budget is a count, so it is consumed only while the waiter runs.
2. **Yield.** Then `yield_now` between polls, for at most `YIELD_POLLS = 16` polls. A peer queued on the same CPU then gets the CPU without paying a futex round trip.
3. **Sleep.** Then sleep on a `Parker`, until the condition changes or the episode is ended for that waiter.

| wait | condition | sleeps on |
| --- | --- | --- |
| reserved member | `gen` changed, the reservation ended, or the member was lent | its slot's `Parker` (unchanged: the leader wakes only the slots it changed) |
| leader in `join` | `remaining == 0` | the pool's `sync` `Parker`, woken by the member that brings `remaining` to 0 |
| participant in `barrier` | `epoch` changed, or the section was cancelled | the pool's `sync` `Parker`, woken by the last arrival and by `fail` |
| unreserved `IDLE` member (after a scoped `Exec::team` ended) | claimed, released, `idle_spin` elapsed, **or the spin budget of polls spent** | returns to Rayon (whose idle worker sleeps after 33 rounds); a team released between calls no longer keeps its members polling for the whole `idle_spin` |
| acquisition (`acquire`, `claim_cold`) | unchanged (`backoff`) | — |

The `sync` parker is **one per pool** (in `Awake`), not one per section:
- Sections are stack objects on the hot path. A `Parker` is touched once when it is built (some platforms allocate lazily), so building one per section would cost on every section.
- A member may touch the parker after its final decrement of `remaining`. By then the leader may have returned and freed the section, but the pool outlives every member loop.

Every leader on a pool shares the parker: two teams on one pool, and also the child leaders of one team's `split` and `fork`, which join at the same time. A wake wakes every sleeper, and each re-checks its own condition. That is correct, and it is not rare: a split of 64 participants into 32 pairs can have 32 child leaders asleep at once. It is kept simple until a measurement shows the broad wake costs something; the `nested` bench scenario (100 splits into pairs per call) is that measurement.

Both new wake points follow the existing `Parker` protocol, which `loom` already models:
1. change the state with SeqCst: the final `remaining` decrement, the epoch bump, the `cancelled` store;
2. then call `Parker::wake`, one `fetch_add(0)` that is enough when nobody is parked.

The last member of a section therefore pays one extra uncontended read-modify-write. It reads the pool's parker before its final decrement and never touches the section after it (the leader may free the section as soon as it sees zero); the pool outlives every member loop. The join and the barrier are small types in `src/park.rs` (`Completion`, `Barrier`) that the sections use and that `tests/loom_park.rs` checks as they are: the last of two members waking a sleeping leader, two consecutive barrier rounds, and a cancellation waking a sleeper. Each model fails (deadlock) when its wake is removed.

### Budget

- **A count, as in libgomp.** The budget is `PoolBuilder::spin_polls`, a number of polls (one PAUSE, or YIELD on aarch64, per poll). A waiter that is descheduled does not spend it while it waits for its CPU.
- **Default: 131072 polls on x86-64** (2^21 on aarch64, not measured yet). One poll measured about 24 ns on Zen 3 (EPYC 7713P), so this is about 3 ms there, the order of libgomp's default (300000 polls, "about 3 ms"); Intel cores with a slower PAUSE take about twice as long.
  - The first default, 12288 polls (about 300 µs), made the first section after a 1 ms or 10 ms gap 2.3x and 10.8x slower than 0.1.3 on a quiet host (evo, 6 → 16 → 75 µs): the members had gone to sleep, and the next call paid futex wakes. With about 3 ms, a team called again within a few milliseconds is still spinning, as under libgomp; after a longer gap it sleeps, as under libgomp.
  - libomp keeps 200 ms (`KMP_BLOCKTIME`); 0.1.3 kept 200 ms of wall time for members and never slept in joins.
- **No calibration.** Converting a time to polls at pool build was considered and dropped (pre-review): a calibration taken while the CPU is contended yields too few polls, and a bare PAUSE loop is not the real poll anyway. The count is approximate in time, as libgomp's is.
- **Throttled budget.** 100 polls (libgomp's value) when the root scope is statically oversubscribed, or while the pool is under sustained pressure (below).

### Scheduling pressure

- **Detection.** A spinning waiter reads the clock (`Instant::now`, vDSO) every 64 polls. If the gap since its previous read exceeds 200 µs, it was descheduled while it spun. Sixty-four polls take well under 10 µs, and a scheduler slice is 0.75 ms or more.
- **What it means.** This is a pressure heuristic, not proof of foreign load. A foreign process, another thread of the team, interrupts, page faults or VM steal all look the same, and a short preemption can escape the threshold. It is acceptable because the response only shortens active waiting; it never changes affinity or correctness.
- **Response.** The waiter ends its spin at once (yields, then sleeps). The pool counts the observations: once 4 come within 100 ms of each other, every wait of the pool adopts the throttled budget at its first clock read, for 100 ms (moved forward with a `fetch_max`, so an older observation never shortens a newer one).
  - **Not on one observation.** Throttling the whole pool on a single preemption made every section of a width-64 team on a mildly shared host sleep and pay futex wakes (broccoli, `held_team` 1.48 → 6.12 ms, `c0c024b`'s predecessor). An isolated preemption now ends only its own waiter's spin.
  - **Not per thread.** `2773c36` tried giving the thread that saw a preemption a 100 ms deadline of its own instead of throttling the pool. On a quiet broccoli that made held teams 1.79x / 1.21x slower than 0.1.3 (against 1.03 / 1.07 with the pool threshold): one member preempted once then slept at every section for the hold, and every join waited for its wake-up. The per-thread hold remains a benchmark knob (`--own-hold-ms`, off by default).
  - **The pool-wide throttle at all:** on a saturated host (primerose), throttling only the waiter that saw the preemption left the pool as slow as 0.1.3.
  - **Knobs and counters.** The threshold, the hold, the throttled budget and the per-thread hold can be varied for the benchmarks (`PoolBuilder::wait_tuning`, hidden; `--pressure-events`, `--pressure-hold-ms`, `--throttled-polls`, `--own-hold-ms`). `Pool::wait_counters` (hidden) counts join and barrier sleeps, member parks and observed preemptions, printed under every bench row.
- **Cost.** A wait that ends within 64 polls never reads the clock; otherwise one clock read per 64 polls. There is no `getrusage` or other syscall in the poll loop.
- **Limit.** A pool built for a single call (`with_team`) cannot carry the deadline to the next call. What helps it is the short budget and the sleeps in `join` and `barrier`, which give the scheduler idle CPUs within the call. With `pin_participants`, a sleeping waiter frees its CPU, but a pinned member that is descheduled behind a foreign process cannot move.

### Start of a section: no arrival barrier

A section starts when its members observe their `gen`. There is no barrier in which every participant first announces itself. libgomp's region start is a whole-team dock barrier, because a region always uses the whole team. threadteam sections can be narrower than the team (faer narrows regions to the width that pays), and waking or waiting for members a section does not use would make narrow sections pay for the whole team.

A member that slept is woken by the leader's publish on its slot. That wake only makes it runnable; no arrival acknowledgement is needed for a section's correctness, because the reservation's acknowledgement at acquisition already put every member on its own thread, and the section's publication and completion protect its storage. The join at the end of the section is the only barrier. `Section::barrier` inside a share is unchanged in meaning and gains the sleep phase.

### The spare worker (threadteam-rs#12): not in this change

A pool of `w` workers gives a team of at most `w` participants: the leader counts as one, and `w - 1` workers are reserved, so one worker is a spare that only Rayon uses. Letting `into_team(width)` reach `w + 1` from an outside thread would reserve every worker. The pre-review found that this breaks `Exec::external` on a team token:
- `external` lends the token's `budget - 1` members and installs the callback on a pool worker.
- Today the spare is the worker that hosts it. Without a spare, the callback occupies one of the lent members, so a callback that waits for `width` Rayon jobs at once deadlocks, and a full-width token advertises `w + 1` threads to a pool of `w`.

Removing the spare therefore needs a new contract for `external`'s width and lending, covering full-width and narrowed tokens. broccoli also showed that the spare is not what separates the slow arms (faer#107, 13:10Z): `team_per_call` runs the same 65 threads and is fast. The spare stays, and threadteam-rs#12 carries the `external` constraint.

## Evaluation

`benches/wait_scenarios.rs` (README, "Wait benchmarks"), run on the base and on the head of the change:
- `sections`: many short sections per call (5-50 µs of work, 200 sections);
- `gaps`: the wake latency after a gap of 0 / 100 µs / 1 ms / 10 ms between calls;
- `acquire`: a pool held with a team per call, against a new pool per call;
- `nested`: 100 splits into pairs per call (the shared parker under many child leaders);
- width = CPUs against width < CPUs;
- with and without a foreign-load generator: child processes that each burn a few % of a CPU;
- the same work under libgomp (`benches/omp/wait_omp.c`, when `gcc -fopenmp` is available) and under a Rayon `scope`.

The change is accepted when (criteria set by the monitor, 2026-10-30):
- **(a)** the faer#107 slow mode is zero in every arm of `acquire_probe_107` (tensor4all/faer branch `probe/107-acquire-per-call-wait`);
- **(b)** on quiet hosts (evo, an unloaded broccoli), `held_team`, `held_pool` and `gaps` (0-1 ms) are within 10 % of the base;
- **(c)** under foreign load (`--load 8:10`), the same scenarios are no worse than libgomp's. The base is not the reference there: it keeps its spinners on CPUs that other processes need;
- **(d)** the 10 ms gap is tuned with the spin budget; if 300000 polls closes it, that is accepted as the OpenMP default.

The comparison is `scripts/bench-ab.sh` with `OMP=1`, which runs libgomp's counterpart (`acquire`/`held_team`: calls of 50 sections of 20 µs on the same hot team) in every round, with and without load.

**First reading, primerose (not an acceptance run).** EPYC 7713P, 64 CPUs, saturated by other users (load average 61-63). Width 6 on `taskset -c 0-5`, 40 calls, threadteam only. Base = the bench commit on 0.1.3; head = this change. Median µs per call:

| scenario | param | base | head | ideal |
| --- | --- | --- | --- | --- |
| sections | 5 µs × 200 | 775 725 | 14 497 | 1 000 |
| sections | 20 µs × 200 | 978 060 | 21 991 | 4 000 |
| sections | 50 µs × 200 | 1 462 996 | 39 042 | 10 000 |
| gaps (first section) | 0 | 10 972 | 19 | 5 |
| gaps (first section) | 1 ms | 10 895 | 47 | 5 |
| gaps (first section) | 10 ms | 12 003 | 1 033 | 5 |
| gaps (call) | 10 ms | 124 883 | 9 832 | 50 |
| nested | pairs, 100 × 20 µs | 741 969 | 37 894 | 2 000 |
| acquire | team held, token per call | 16 055 | 9 551 | 1 000 |
| acquire | pool held, members per call | 245 584 | 2 501 | 1 000 |
| acquire | new pool per call (call) | 393 990 | 61 622 | 1 000 |
| acquire | new pool per call (build to drop) | 408 038 | 67 548 | 1 000 |

Empty-section dispatch (`benches/team_dispatch.rs`, same host, median ns): width 2, 519 (base) against 530 (head); width 6, 767 against 862; the load average moved from 64 to 74 between the two runs, so this bounds a regression only coarsely.

On a host whose CPUs are all busy, 0.1.3's spinning waiters stack and stay stacked, and calls take hundreds of times their work. The staged waits bring them within about 2-10x of the ideal, the cost of sharing CPUs with other users' processes. A quiet host and broccoli decide the acceptance; these numbers only show the mechanism.
