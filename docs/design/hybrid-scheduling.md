# Hybrid scheduling: OpenMP-style teams with stealable tasks (threadteam-rs#22)

Status: design study. Nothing here is implemented. It applies to the own-thread version on `develop` (`own-threads.md`), whose workers, reserve, slot state word, batch queue and broadcast list it builds on.

The question: threadteam's main axis is OpenMP-style. That means:
- concurrent teams whose members are co-scheduled;
- barriers;
- stable member numbers for static partitioning;
- places and pinning;
- a wait policy chosen by the caller;
- sub-teams.

Can Rayon-style flexibility live on the same workers? That means work stealing for irregular and nested parallelism, `join` and `scope`, and composition without multiplying threads. The note answers that question and evaluates an alternative next to it: an engine that owns two pools, a threadteam pool for teams and a Rayon pool for stealing.

Design inputs:
- **pi-council, plan mode** (GPT-6-astra chair, DeepSeek critic, GLM minimalist). It ran in two meetings, one on questions 1 to 4 and one on questions 5 to 7 plus the two-pool items. The brief and both syntheses are in the shinaoka-advisor run directory `work/runs/threadteam-hybrid/`.
- **A source digest.** Every citation below was opened and checked against the source text. Claims that could not be checked are marked **not verified**.
- **The decision record** [threading-substrate.md](https://github.com/tensor4all/tensor4all-meta/blob/main/docs/design/threading-substrate.md) in tensor4all-meta: the correspondence table of OpenMP 6.0 and Rayon, interoperation, and the conditions for 0.3.0.

## 1. Summary

**Recommended: one runtime, team first, with scoped stealing.** A worker runs two kinds of work:

- **Team sections.** These are today's reserved, co-scheduled members, with barriers and stable numbers. Nothing changes for them, and a plain barrier never runs other work.
- **Tasks.** These are units created by `join`, `scope` and dependent-task graphs. They are stolen only inside a **domain**: the set of workers the creating token holds authority over. Free workers of the pool may run a task only when the task opts in, as with OpenMP 6.0's `threadset(omp_pool)`.

The rules that make the two coexist are taken from runtimes that already combine teams with tasks:
- A waiting member runs only **descendants** of what it is waiting for (the OpenMP task scheduling constraint, and oneTBB's `isolate`).
- A task carries **no team authority**, so it can never wait at a team barrier.
- Stealing **never creates authority**: a domain's members are the only threads its tasks can occupy, so budget conservation holds.

Child stealing (Rayon's `join` shape) is the stealing discipline, because continuation stealing needs compiler support or a cactus stack.

**The two-pool option is the engine-level baseline, not the destination.** It is available today on the Rayon version, with no runtime change. Two costs follow from its structure:
- An item of the Rayon pool cannot hand its unused share of CPUs to the team side.
- The two pools have to be kept from running at once on one CPU set.

The note keeps it as the reference every native stage has to beat, and gives the measurements that decide (section 13).

## 2. Model and terms

- **Worker.** A thread of the reserve, leased to one pool (`own-threads.md`).
- **Token (`Exec`).** Authority over `budget` slots of a pool. It is not `Clone`; its splits conserve slots (`placement.md`).
- **Team section.** Work run by every member of a reservation at once, with barriers between phases and a fixed member number. The leader is the calling thread.
- **Task.** A closure that runs once, on any worker of its domain, with no guarantee of running at the same time as any other task.
- **Domain.** The set of workers a token holds authority over at the moment a task is created. For a token minted from a `Team`, it is the team's members plus the leader. For a plain token of budget `b`, it is the `b` slots the token was given.
- **Descendant.** Task `t` is a descendant of task or section `s` when `t` was created, directly or transitively, by code running inside `s`.

## 3. Question 1: tasks and teams on the same workers

### Recommendation

- **Two queues per worker slot, one domain per task.** Each slot gets a task deque: the owner pushes and pops at one end, and thieves take from the other. This is next to the existing section mailbox (the slot's generation counter) and the pool's batch queue.
- **What a worker looks for, in order.** This extends the order of the worker loop in `own-threads.md`:
  1. a broadcast share;
  2. a team section, if reserved;
  3. its own deque;
  4. a steal from a worker of the same domain;
  5. the pool's batch queue;
  6. otherwise the staged wait.
- **A reserved member between sections** is an idle worker of its team's domain. It runs tasks of that domain when it has nothing else to do, and the next section's generation bump is checked before each task, so a section never waits behind a task the member had not yet started.
- **Free workers of the pool** run a task outside its domain only when the task was created with pool scope. The default is domain scope, as OpenMP 6.0's `threadset` clause defaults to `omp_team` (OpenMP 6.0, section 14.8, cited in section 4).

### Rejected

- **Teams as tasks on a stealing pool (Rayon's model with a team layer on top).** Rayon gives no guarantee that `n` jobs run at the same time, so `n` jobs that wait for each other at a barrier can deadlock (threading-substrate.md, section 4). threadteam 0.1 and 0.2 build teams this way only by occupying Rayon workers for the whole reservation, which is the two-pool option inside one pool.
- **One shared task queue per pool, as libgomp does** (gcc `libgomp/task.c` keeps one team-wide priority queue under `team->task_lock`). Every push and pop then takes one lock. That adds to the work term of every task, which the work-first principle says to keep minimal (section 4). It remains the fallback for a first implementation if per-slot deques do not pay.

## 4. Question 2: what the literature and runtimes establish

| source | what it establishes | consequence here |
| --- | --- | --- |
| Frigo, Leiserson, Randall, "The implementation of the Cilk-5 multithreaded language", PLDI 1998 ([PDF](https://people.cs.umass.edu/~emery/classes/cmpsci691w-spring2006/papers/cilk5.pdf)) | The work-first principle: minimize the work overhead `c_1 = T_1/T_S` even at the cost of a larger critical-path overhead `c_∞`, assuming parallel slackness. Steal costs are borne on the critical path. | Anything the team machinery adds to `join` or spawn must be paid on steals, not on every task. |
| Blumofe and Leiserson, "Scheduling multithreaded computations by work stealing", JACM 46(5), 1999 ([PDF](https://cs.utexas.edu/~venkatar/sys_perf_analysis/ws_theory.pdf)) | Theorem 13: expected time `T_1/P + O(T_∞)`. Theorem 5: space at most `S_1·P`. Both are proved for **fully strict** computations (every join edge goes to the parent), and the scheduler pushes the parent on the deque and runs the child. | A team barrier joins siblings, not a parent, so team sections are outside the theorem. A child-stealing runtime does not inherit the space bound. |
| Robison, "A Primer on Scheduling Fork-Join Parallelism with Work Stealing", WG21 N3872, 2014 ([PDF](https://open-std.org/JTC1/sc22/wg21/docs/papers/2014/n3872.pdf)) | Definitions of child stealing and continuation stealing. TBB and PPL use child stealing, Cilk uses continuation stealing, OpenMP uses child stealing by default. Continuation stealing bounds space blow-up by a factor of `P`; child stealing does not (a loop that spawns `n` tasks spawns all of them first). Child stealing can be a library without compiler support. Greedy joins look for other work, and finding work that cannot deadlock "is not always possible". | threadteam is a library: child stealing. Loops split recursively (`join`), never as flat spawn loops. |
| Rayon `join_context` ([`rayon-core/src/join/mod.rs`](https://raw.githubusercontent.com/rayon-rs/rayon/main/rayon-core/src/join/mod.rs)) | Pushes job B, runs A inline, then pops B or waits on its latch. | Rayon is child-stealing-like, and its `join` is the shape to keep. |
| Lee et al., Cilk-M, PACT 2010 ([PDF](https://vm-web.pdos.csail.mit.edu/~sbw/papers/pact183a-lee.pdf)); Yang and Mellor-Crummey, Fibril, SPAA 2016 (DOI [10.1145/2935764.2935787](https://doi.org/10.1145/2935764.2935787)) | Cactus-stack platforms trade serial interoperability, speedup or bounded stack memory. Cilk-M changed the kernel to get all three. Fibril is a later approach to the same problem; **Fibril's mechanism is not verified** (only its abstract was read). | Continuation stealing in Rust needs either compiler support or stackful coroutines with their own stacks. Neither fits a library that runs closures on `std::thread`s. |
| Rayon `wait_until` ([`rayon-core/src/registry.rs`](https://raw.githubusercontent.com/rayon-rs/rayon/main/rayon-core/src/registry.rs)) and issues [#592](https://github.com/rayon-rs/rayon/issues/592), [#1105](https://github.com/rayon-rs/rayon/issues/1105), [#835](https://github.com/rayon-rs/rayon/issues/835) | A waiting worker pops and steals any job. That re-enters a held lock (#592), re-enters a loop body meant to bound memory (#1105), and deadlocks a one-thread pool blocked inside `scope` (#835). | A waiter that runs unrelated work breaks lock, memory and thread-local invariants. Section 5 restricts what a waiter may run. |
| oneTBB [`task_arena`](https://oneapi-spec.uxlfoundation.org/specifications/oneapi/latest/elements/onetbb/source/task_scheduler/task_arena/task_arena_cls), [`this_task_arena::isolate`](https://oneapi-spec.uxlfoundation.org/specifications/oneapi/latest/elements/onetbb/source/task_scheduler/task_arena/this_task_arena_ns), [`task_group`](https://oneapi-spec.uxlfoundation.org/specifications/oneapi/latest/elements/onetbb/source/task_scheduler/task_group/task_group_cls), [work isolation guide](https://uxlfoundation.github.io/oneTBB/main/tbb_userguide/work_isolation.html) | An arena limits concurrency (`max_concurrency`). A thread waiting in `task_group::wait()` may run unrelated tasks. `isolate` restricts the calling thread to tasks created in its scope, because otherwise a nested wait can run an outer task and break thread-local state or deadlock. | An arena whose concurrency equals the team, plus isolation while waiting, is the industrial form of "team domain plus descendants only". |
| LLVM libomp ([`kmp_tasking.cpp`](https://raw.githubusercontent.com/llvm/llvm-project/main/openmp/runtime/src/kmp_tasking.cpp), [`kmp_barrier.cpp`](https://raw.githubusercontent.com/llvm/llvm-project/main/openmp/runtime/src/kmp_barrier.cpp), [`kmp_wait_release.h`](https://raw.githubusercontent.com/llvm/llvm-project/main/openmp/runtime/src/kmp_wait_release.h)); OpenMP 5.2, section 12.9 ([HTML](https://www.openmp.org/spec-html/5.2/openmpse77.html)) | Per-thread deques. A thread steals from the last successful victim, then a random one, **within its team** only. A thread waiting at a barrier executes tasks (`__kmp_wait_template` calls `execute_tasks`). Implicit and explicit barriers are task scheduling points. The task scheduling constraint lets a thread start a new tied task only if it descends from every tied task suspended on that thread (outside a barrier). | Stealing scoped to a team, plus a descendant rule at waits, is the standard shape. Section 5 adopts it. |
| gcc libgomp ([`config/linux/bar.c`](https://gcc.gnu.org/git/?p=gcc.git;a=blob_plain;f=libgomp/config/linux/bar.c;hb=HEAD), [`task.c`](https://gcc.gnu.org/git/?p=gcc.git;a=blob_plain;f=libgomp/task.c;hb=HEAD)) | Barrier waiters run pending tasks (`gomp_barrier_handle_tasks`) until the barrier generation advances. A comment in `task.c` guards against running tasks of the next scheduling region (PR122314). | A waiter must not take work that belongs to the next phase. The section generation check in section 3 is that guard. |
| OpenMP 6.0 specification ([PDF](https://www.openmp.org/wp-content/uploads/OpenMP-API-Specification-6-0.pdf)): glossary; section 14.8 `threadset`; section 4.3.10 `OMP_THREADS_RESERVE`; section 14.3 `taskgraph` | A free-agent thread is an unassigned thread that runs an explicit task. With `threadset(omp_team)`, the default, tasks run only on the current team. With `omp_pool`, they may also run on unassigned pool threads. `OMP_THREADS_RESERVE` sets how many threads are reserved for structured parallelism and for free agents. `taskgraph` records the replayable constructs of a region and replays the record. | Domain scope by default and pool scope by opt-in (section 3), and prepared graphs with replay (section 7), follow the standard. |
| Iwasaki, Amer, Taura, Seo, Balaji, "BOLT", PACT 2019 ([PDF](https://conferences.computer.org/pact/2019/pdfs/PACT2019-2tEkxmW1ZYI1p1wcV5R7bj/6uxH49GIooPa3EKdNP5oP9/6Bf5AiwrWKoGYBVwGN7Nyd.pdf)) | OpenMP over user-level threads (Argobots): one OS thread per core, random work stealing of user-level threads, tied tasks kept correct. Nested parallel regions become cheap, where OS-thread runtimes disable nesting by default. A hybrid wait is active when flat, passive when oversubscribed. | Nested teams on OS threads either oversubscribe or must be narrowed to what is free. Without user-level threads, section 6 narrows. |
| Kokkos [TeamPolicy](https://kokkos.org/kokkos-core-wiki/API/core/policies/TeamPolicy.html), [team handle](https://kokkos.org/kokkos-core-wiki/API/core/policies/TeamHandleConcept.html), [tasking](https://kokkos.org/kokkos-core-wiki/API/core/Task-Parallelism.html) | Teams with `team_barrier` and team scratch are first-class. Task functors must run to completion without blocking. The task DAG was deprecated in 4.5 and removed in 5.0. | Teams and non-blocking tasks are separate kinds of work, and tasks never block. |
| HPX [runtime and resources](https://docs.hpx.dev/latest/html/manual/hpx_runtime_and_resources.html) | One queue per OS thread. With NUMA-sensitive scheduling, a thread steals in its NUMA domain first. A static policy has no stealing. **The default stealing order outside that mode, and HPX futures, are not verified.** | Victim order can follow topology; section 10, stage 5. |
| Huang et al., Taskflow, TPDS 2021 ([arXiv](https://arxiv.org/pdf/2004.10908), DOI 10.1109/TPDS.2021.3104255) | While any worker is active, at least one worker keeps trying to steal unless all are active; the last thief to become active wakes one more. The others sleep. | This is an alternative idle policy for free workers that bounds wasted steals. It is a candidate next to libgomp's spin count. |
| van Dijk and van de Pol, Lace, Euro-Par 2014 workshops (DOI [10.1007/978-3-319-14313-2_18](https://doi.org/10.1007/978-3-319-14313-2_18); [abstract](https://research.utwente.nl/en/publications/lace-non-blocking-split-deque-for-work-stealing/)) | A split deque with a movable boundary between a private and a shared part; fences only when the shared part shrinks. The paper itself was not opened; the abstract is verified. | A deque whose private part costs no fence keeps the work term low. |
| Olivier et al., IJHPCA 2012 (abstract via [OpenAlex](https://api.openalex.org/works/doi:10.1177/1094342011434065)); Deters, Wu, Xu, Lee, NUMA-WS, IISWC 2018 ([arXiv](https://arxiv.org/pdf/1806.11128)) | One thread steals for all cores that share a cache, which limits remote steals. NUMA-WS biases victims toward the local socket and keeps `T_1/P + O(T_∞)` with a slightly larger constant. **The ROSS 2011 paper behind the first is not verified.** | Bias victims toward the domain's own L3 or socket. Keep a nonzero chance of a remote steal so the bound survives. |
| Frigo, Halpern, Leiserson, Lewin-Berlin, reducer hyperobjects, SPAA 2009 ([PDF](https://www.fftw.org/~athena/papers/hyper.pdf)); OpenMP 6.0, section 7.6.7; oneTBB [`parallel_deterministic_reduce`](https://oneapi-spec.uxlfoundation.org/specifications/oneapi/latest/elements/onetbb/source/algorithms/functions/parallel_deterministic_reduce_func); libomp [`kmp_settings.cpp`](https://raw.githubusercontent.com/llvm/llvm-project/main/openmp/runtime/src/kmp_settings.cpp) | A reducer with an associative operator and a true identity gives the serial result whatever the schedule (floating-point excepted). OpenMP leaves the combining order unspecified and promises no bitwise reproducibility, even at one thread count. No `OMP_*` variable changes that; libomp's `KMP_DETERMINISTIC_REDUCTION` forces a tree reduction. TBB's deterministic reduce runs the same splits and joins at any thread count, but may differ from the serial result. | Two different contracts: the serial result, or reproducibility across runs. Section 8 offers the second. |
| Malakhov, SciPy 2016 ([PDF](https://proceedings.scipy.org/articles/Majora-629e541a-002.pdf)); Malakhov, Liu, Gorshkov, Wilmarth, SciPy 2018 ([PDF](https://proceedings.scipy.org/articles/Majora-4af1f417-003.pdf)); oneTBB [`rml_tbb.cpp`, `python/rml/ipc_server.cpp`](https://github.com/uxlfoundation/oneTBB); Intel oneMKL [threading layer](https://www.intel.com/content/www/us/en/docs/onemkl/developer-guide-linux/2025-1/dynamic-select-the-interface-and-threading-layer.html) | Section 11, item (6). | Section 11. |

## 5. Question 3: when a waiting member may run tasks

There are three kinds of wait:
1. a member waiting at a team barrier;
2. a leader waiting in a section's join;
3. a thread waiting for its own tasks (`join`, the end of a `scope`).

### Recommendation

- **A team barrier runs nothing by default.** A plain `Barrier` stays as `waiting.md` specifies: spin `spin_polls`, yield 16 times, park. Its latency is set by the last member to arrive, never by work a waiter picked up.
- **An opt-in draining barrier** (`barrier_draining`, a separate method) lets a waiting member run tasks, under three restrictions:
  1. **Descendants only.** It runs tasks that descend from the current section of the same team. This is libomp's task scheduling constraint, and oneTBB `isolate`.
  2. **Leaf tasks only.** A leaf task is one whose type gives it no team authority and no way to wait on a team (section 9). It may still `join` its own children; those are descendants too.
  3. **A check before each task.** The member checks the barrier generation before taking each task, so it never starts a task after the barrier opened.
- **A thread waiting for its own tasks** (`join`, end of `scope`) runs its own deque first, then steals descendants of the work it waits for from its domain. This is Rayon's greedy join, restricted by the descendant rule. It never runs a task of an outer scope, which closes the class of rayon#592 and #1105.
- **A leader waiting in a section's join** follows the barrier rule: it runs nothing by default, and the draining form is opt-in.

### Deadlock freedom (proof sketch)

Order every task and section by "is a descendant of". This is a strict partial order on the finite set of live work, because creation happens before execution.

A waiter blocked on work `w` only ever runs descendants of `w`. A descendant cannot wait for an ancestor:
- A task has no team authority, so it cannot wait at its ancestor's barrier.
- Its only waits are `join` and `scope` on its own descendants.

So every wait-for edge points from a node to one of its descendants, and the wait-for graph is acyclic. With finite work and terminating task bodies, some leaf can always run. This is the argument behind the task scheduling constraint. It does not cover user code that blocks on a lock or a channel inside a task, which stays the caller's responsibility, as in Kokkos ("run to completion without blocking").

### Latency bound of a draining barrier

Let `L` be the longest leaf task in the domain, and let `t_last` be the time the last member arrives.
- A member checks the generation before each task, so after `t_last` it starts no new task. The only task it can still be running is one it started before `t_last`.
- That task can `join` its own children and run them inline, but a join is bounded by the span of that task's subtree.
- So the release of each member is delayed by at most `L_span`, the longest span of a leaf task's subtree, plus the wake time.

This holds when no OS preemption occurs. Preemption adds to every bound in this note and is not modeled.

### Rejected

- **Unrestricted helping at barriers or joins, as Rayon's `wait_until`.** It runs outer work under held locks and thread-local state (rayon#592, #1105, #835). It also makes barrier latency depend on unrelated work.
- **A draining barrier as the default.** Most dense kernels use barriers between phases of 5 to 50 µs (threading-substrate.md, section 4). A barrier whose release can wait for a task is only worth it where measured idle time at the barrier exceeds the task lengths. Section 10, stage 3, gates it.
- **Inferring task safety from closure signatures alone.** A closure can capture a lock or a channel. The type restriction removes team authority, which is the cycle the runtime can see, and leaves other blocking to the caller.
- **Leapfrogging (a waiter steals only from the thief that took its work).** It needs the stolen-by link on every task. Its benefit is the same descendant restriction, which the domain plus descendant rule already gives without that bookkeeping.

## 6. Question 4: budgets, stealing and nested teams

### Recommendation

- **Authority is never created by a steal.** A task created under a token of budget `b` is stealable only by the `b` workers of that token's domain. So at most `b` threads run tasks of that root at any time, which is the existing budget conservation (`placement.md`).
- **Dynamic redistribution inside a domain.** When `fork(m)` gives each child a share and one child finishes early, its workers stay in the parent's domain and may steal tasks of the siblings. This is the case the two-pool option cannot serve (section 11, item 2). It conserves budget because the domain belongs to the parent, whose authority already counts those workers.
- **A nested team inside a task** may be opened only from workers of the task's domain that are idle at that moment:
  - it claims them with the slot state word, from `0`;
  - it refuses (`Refused::Unavailable`) instead of waiting;
  - the task then runs its work serially or with the members it got;
  - a team never grows through the process reserve, because the reserve is the pool's capacity, not the token's authority.
- **A team section never creates tasks with pool scope implicitly.** A share that wants tasks creates them in its member's sub-domain. That sub-domain is the member alone, or the members its `split` gives it.

### Invariant and proof sketch

**Invariant:** for every root token `r`, the threads running work of `r` at once are at most `budget(r)`.

**Proof by induction on splits and task creation:**
- A split's shares sum to at most the parent's (existing).
- A task runs only on a worker of its domain, and the domain is a subset of its creator's slots.
- A nested team claims only idle slots of that domain, through a compare-and-swap from `0`, so no slot is counted twice.

### Rejected

- **Credits that a parent keeps while waiting for its children.** They double-count the parent's slot. Today a parent waiting for its children keeps no slot.
- **Nested teams from the process reserve.** This is the N² oversubscription that the budget rule exists to prevent.
- **Waiting for a nested team's members.** The waiting time has no bound, because a member may be running a long task. Refusal is immediate, and the caller already handles `Refused::Unavailable`.

## 7. Question 5: dependent tasks

### Recommendation

- **A prepared graph** (`GraphPlan`) holds an immutable topology:
  - nodes;
  - predecessor counts;
  - successor lists;
  - an optional rank, the remaining critical path of each node.

  Building it checks acyclicity in `O(V + E)` and refuses a cycle before running anything.
- **Running a plan.** `plan.run(exec, scratch)` runs the plan on a token. It uses caller-owned scratch (the counters and a bounded ready queue), so a replay allocates nothing. This is OpenMP 6.0's `taskgraph`: record once, replay per iteration.
- **Nodes.** A node is a serial, terminating, non-blocking task, with the same restrictions as a leaf task (section 5). A node that needs a team gets one only through the nested-team rule of section 6, and refuses rather than waits.
- **Readiness.** A node becomes ready when its predecessor count reaches zero; the decrement that reaches zero pushes it. Ready nodes are taken highest rank first, so the panel of a look-ahead LU or the diagonal tile of a tiled Cholesky runs before trailing updates. Priority only orders ready nodes; it never leaves a worker idle while any node is ready.

### Proof sketch

- **Exactly once.** Each node is pushed exactly once, by the unique decrement to zero, and is taken exactly once from the queue.
- **Ordering.** A successor runs after its predecessors' writes, because the decrement publishes them (release on decrement, acquire on the zero test).
- **Progress.** A finite acyclic graph with terminating, non-blocking nodes always has a ready node until all have run.
- **Time.** The greedy bound for `b` workers, `T_b ≤ T_1/b + T_∞`, holds for any schedule that never idles a worker while a node is ready. This is the classical greedy-scheduling bound (Graham 1969, Brent 1974); **neither original was opened**. Reservations, steal costs and preemption come on top of it.

### Rejected

- **Inferring dependencies from addresses (`depend(in/out)` on memory).** The tiled kernels in scope have known graphs, and the alias analysis is runtime cost on every task.
- **Level-synchronous execution (a barrier per graph level).** One slow node delays every ready successor of the next level.
- **Priority as a preemptive order.** Leaving a worker idle for a higher-ranked node that is not yet ready loses the greedy bound.

## 8. Question 6: determinism

### Recommendation

There are two result classes, named in the API:

- **Reproducible.** The result is bitwise identical across runs, at a fixed width, compiler, target and floating-point environment. It is produced by a fixed partition into chunks whose boundaries depend on the data size and a declared chunk size, never on the thread count or on which worker ran a chunk. Chunk results are combined through a fixed tree, keyed by chunk index. Static team partitioning and stolen tasks both qualify, provided they write into the chunk's own slot.
  - When the chunk size is independent of the width, the result is also identical across widths. This is the contract of oneTBB's `parallel_deterministic_reduce`, which may differ from the serial result.
- **Schedule dependent.** This is any reduction whose combining order follows completion order or stealing, such as an adaptive split or an atomic accumulator. It is the default for `join` and `scope`, as OpenMP leaves combining order unspecified (OpenMP 6.0, section 7.6.7).

**API shape:**
- A reproducible reduction is a separate entry that takes the chunk size explicitly.
- A request for reproducibility that the entry cannot honor is a refusal, not a silent fallback.

**Proof sketch:** identical chunk results combined through an identical tree give identical bits, by induction on the tree. Which worker computed a chunk does not appear in the tree.

### Rejected

- **"Static scheduling is reproducible."** It is not when chunk boundaries follow the width, or when partial results are combined in arrival order.
- **"Stealing prevents reproducibility."** It does not when stolen chunks write into fixed slots.
- **Promising the serial result (Cilk reducer semantics) for floating point.** The reducer paper excludes non-associative operators, and floating-point addition is one.

## 9. Question 7: Rust API and Rayon interop

### Recommendation

- **`Exec` stays the only authority.** The new entries are its methods:
  - `exec.join(a, b)`;
  - `exec.scope(|s| ...)`, which spawns with `s.spawn`;
  - `exec.reduce_reproducible(...)`;
  - `plan.run(exec, ...)`.

  Closures that run on another thread are `FnOnce + Send`. Shared closures (`for_ranges`, sections) are `Fn + Sync`. A `scope` returns only after every task spawned in it finished, which lets tasks borrow the caller's stack, as `std::thread::scope` and Rayon's `scope` do.
- **A task's handle is a `TaskExec`, not an `Exec`.** It can `join`, `scope` and run a prepared plan within its domain. It has no `team`, no `broadcast` and no barrier. This is the type restriction behind sections 5 and 6, enforced at compile time.
- **Rayon interop on the own-thread version stays an opt-in feature (`rayon`)**, with two forms:
  1. **`external_rayon`**, as on `develop`: a private Rayon pool on separate threads, with the width of the token. The `develop` code sizes it by the pool's width while passing the token's width. That is to be aligned to the token width before 0.3.0.
  2. **Hosting a Rayon pool on leased threads** through `spawn_handler` (threading-substrate.md, section 5.1). The leased thread runs Rayon's worker loop and nothing else until the Rayon pool is dropped. This is the two-pool option within one owner. It does not integrate Rayon jobs with threadteam tasks: they are two schedulers on disjoint threads.
- **No Rayon type in the default public API**, as today (`check-no-public-rayon.sh`).

### Rejected

- **Implementing Rayon's `ThreadPool` API on threadteam's scheduler.** Rayon offers no hook to replace its scheduler, only its thread spawning. A compatible facade would have to follow Rayon's semantics, including the unrestricted waiting that section 5 rejects.
- **Capability wrappers that claim to stop a callback from using Rayon.** A callback can always call `rayon::spawn` or a parallel iterator on Rayon's global pool. Interop is a convention plus a test that the global pool was not started (`ledger::global_pool_unstarted`), not an enforcement.

## 10. Question 8: invariants, model checking and staging

### Invariants

| invariant | where it is argued |
| --- | --- |
| I1. Budget: threads running work of a root at once are at most its budget | section 6 |
| I2. A reservation is exclusive, and member numbers are stable for its lifetime | `own-threads.md`, slot state word |
| I3. No waiter runs work that is not a descendant of what it waits for | section 5 |
| I4. No task holds team authority | section 9, by type |
| I5. Deadlock freedom: the wait-for graph is acyclic | section 5, from I3 and I4 |
| I6. A plain barrier's release does not depend on tasks | section 5 |
| I7. Each graph node runs exactly once, after its predecessors | section 7 |
| I8. Reproducible entries give identical bits under any schedule | section 8 |
| I9. No storage is reclaimed before every admitted use finished, also on unwind and cancellation | `own-threads.md`, batch and broadcast rules, extended to deques and plans |
| I10. The steady state allocates nothing: deques are fixed-capacity per slot, and plans use caller scratch | as today |

### TLA+ properties

A model of one pool, two domains, and workers that may be leased, reserved, idle or stealing, with these actions:
- push, pop, steal;
- open and close a section;
- arrive at and leave a barrier;
- claim and release a nested team;
- detach.

Properties to check:
- **Safety.** Every invariant from I1 to I4, plus I7 and I9.
- **Deadlock freedom (I5).** Some action is always enabled while work remains.
- **Liveness under weak fairness of workers.** Every pushed task runs, every barrier opens, and every `scope` returns.
- **Bounded release.** After the last arrival, no member of a draining barrier starts a new task (the guard of section 5).

Loom covers the new wait and wake paths at small sizes (deque push and steal against the owner's pop, the generation check before a task), as for every existing path.

### Staging

Each stage ships only if it beats the previous stage, and the two-pool baseline, on its named kernels on a quiet host and a loaded host. Every stage keeps the acceptance criteria of `own-threads.md` (held teams, held pools and gaps within 10 % of the established baseline).

| stage | what it adds | kernels that justify it | measurement |
| --- | --- | --- | --- |
| 0 | the two-pool baseline (section 11), no runtime change | the kernels below, on the Rayon version | baseline numbers for every later stage |
| 1 | `join` and `scope` on `TaskExec`, child stealing inside a domain, no change to barriers | recursive irregular work: the sparse tree interpolation in tensor4all-rs, unequal batches of small factorizations | time against `fork` with static shares and against the two-pool baseline; work overhead `c_1` with one worker; allocations |
| 2 | dynamic redistribution inside a domain (section 6) | few outer items with inner parallelism: `m < P` independent factorizations of unequal size | the time lost to idle shares in stage 1, and how much stage 2 recovers |
| 3 | draining barrier, opt-in | a team kernel with skewed phases and independent side work: GEMM with packing imbalance, a QR panel with a trailing update | idle time at the barrier against the task lengths; release delay with and without draining |
| 4 | prepared graphs with rank priority | tiled Cholesky and look-ahead LU | time against level-synchronous execution and against the two-pool baseline with Rayon `scope` |
| 5 | topology-biased victims (L3 domain or socket first) | stage 1 and 4 kernels at full width on a multi-L3 part | remote steals per task and time |

## 11. The two-pool option

In this option, an engine (tenferro's `CpuEngine`, for example) owns a threadteam pool for team kernels and a Rayon pool for irregular and nested work. Both are built explicitly by the engine, and neither crate starts a global pool.

### (1) Two pools on one CPU set

- **Concurrent use oversubscribes.** A team of `w` members and a Rayon pool of `w` workers, both active on `w` CPUs, run `2w` threads. A team section then waits for its slowest member, and a member that is descheduled behind a Rayon worker stalls the whole section. This is the slow mode of tensor4all/faer#107 in another form.
- **The rule is phase exclusion.** At any moment only one pool runs work on a given CPU set. The engine enforces it by ownership: one `&mut` engine, one phase at a time.
- **CPU spent by the side that waits.**
  - **threadteam:** an idle member spins `spin_polls` polls (300000 on x86-64, about 3 to 7 ms; `waiting.md`), then yields 16 times, then parks.
  - **Rayon:** an idle worker searches every worker's deque and the injector, then calls `yield_now`, for 32 rounds. It announces itself sleepy, searches once more, and sleeps ([`rayon-core/src/sleep/mod.rs`](https://raw.githubusercontent.com/rayon-rs/rayon/main/rayon-core/src/sleep/mod.rs), `ROUNDS_UNTIL_SLEEPY = 32`; [README](https://raw.githubusercontent.com/rayon-rs/rayon/main/rayon-core/src/sleep/README.md)). A round is a full search, so its cost grows with the pool's width. It is bounded by 33 rounds, not by a time.
  - **So:** after a Rayon phase, the Rayon workers stop within 33 search rounds. After a team phase, the team members spin up to `spin_polls` polls, unless the engine shortens it (`spin_polls`, `idle_spin`, `release_idle`).
- **Phase switch latency.** This is the time from the end of one pool's work until the other pool runs at full width. It adds a wake of the other pool if it slept, and the CPU the first pool keeps while it is still spinning. It is measured, not assumed (section 13).

### (2) Nested budget

- **The serial-inner rule.** A kernel called from inside a Rayon task gets a serial `Exec` (budget 1). This prevents N² threads.
- **Its loss.** With `m` outer items on `P` CPUs:
  - when `m < P`, at most `m` CPUs are used;
  - for equal items whose inner work scales perfectly, the loss is a factor of up to `P/m`;
  - for unequal items, the CPUs freed by early finishers stay idle until the last item ends.

  The native option recovers this with dynamic redistribution inside a domain (section 6).
- **The reverse: Rayon from inside a team share.** A `par_iter` in a share either:
  - runs on the engine's Rayon pool, oversubscribing the share's CPUs (forbidden by phase exclusion); or
  - runs serially; or
  - runs through `Exec::external` with the share's sub-budget. On the Rayon version that is the pool's own Rayon pool; on the own-thread version it is `external_rayon`.

  The rule is: no ambient Rayon in a share, and `external` when a library needs Rayon.

### (3) Placement

- **Same CPU set.** Both pools take the engine's CPU set:
  - threadteam's `Pool::build` reads the calling thread's mask, and reused workers re-apply it;
  - the Rayon pool's `start_handler` sets the same mask on each worker.

  The coordinating thread keeps its whole mask (`placement.md`).
- **No global Rayon pool.** Rayon's process-global pool is never started. That is tested with `ledger::global_pool_unstarted`, and on the Rayon version with a test binary that calls `build_global` last.
- **What affinity does not do.** It does not remove the oversubscription of (1). Only phase exclusion does.

### (4) Ownership rule

Today's rule is: "threadteam is the one owner of CPU threads in the stack, and the pools are owned by the caller" (threading-substrate.md, section 1).

The two-pool option reads it as: "the engine owns the threads; threadteam and Rayon are its tools." Written as rules:
1. Only an engine builds pools.
2. An engine builds at most one pool of each kind per CPU set.
3. An engine runs one phase at a time on a CPU set.
4. A library below the engine receives either an `Exec` or a `&rayon::ThreadPool` from the engine, never both for one call, and starts no pool.

This changes the one-owner rule in threading-substrate.md and the core-boundary document it refers to. It is a maintainer decision, not a design choice of this note.

### (5) `Parallelism::Pool` in tensor4all-rs

`Parallelism::Pool { pool: &rayon::ThreadPool, budget }` in [`crates/tensor4all-treetn/src/interpolation.rs`](https://github.com/tensor4all/tensor4all-rs/blob/main/crates/tensor4all-treetn/src/interpolation.rs) is served directly by the engine's Rayon pool in the two-pool option, under phase exclusion.

**`budget` is advisory.** Nothing stops the callee from using every worker of the pool. Enforcing it takes a Rayon pool of exactly `budget` workers.

In the single-runtime option, that type becomes a threadteam `Exec` (threading-substrate.md, section 6, item 7).

### (6) The OpenMP and TBB precedent

- **The problem** (Malakhov, SciPy 2016):
  - Parallel tasks that each enter an OpenMP region cause quadratic oversubscription with default settings.
  - OpenMP workers busy-waiting keep other parallel work from starting.
  - The 2016 paper proposes a global OpenMP lock as future work only.
- **The remedies Intel shipped** (Malakhov, Liu, Gorshkov, Wilmarth, SciPy 2018):
  - `KMP_COMPOSABILITY=mode=exclusive` in the Intel OpenMP runtime: one parallel region at a time, a lock per region.
  - `KMP_COMPOSABILITY=mode=counting`: a semaphore that caps the total thread count. It shipped as an experimental preview with Intel Distribution for Python 2018. It is absent from LLVM libomp.
  - **For TBB:** `libirml`, an IPC thread-pool provider that takes a semaphore sized to the hardware threads before creating a worker (`python -m tbb --ipc`; oneTBB [`src/tbb/rml_tbb.cpp`](https://github.com/uxlfoundation/oneTBB/blob/master/src/tbb/rml_tbb.cpp) loads `libirml.so.1`, and [`python/rml/ipc_server.cpp`](https://github.com/uxlfoundation/oneTBB/blob/master/python/rml/ipc_server.cpp) implements it). Worst-case active threads stay within twice the hardware threads.
  - **For oneMKL:** `MKL_THREADING_LAYER` selects INTEL, SEQUENTIAL, GNU or TBB, so a caller can make MKL serial or route it onto TBB ([oneMKL guide](https://www.intel.com/content/www/us/en/docs/onemkl/developer-guide-linux/2025-1/dynamic-select-the-interface-and-threading-layer.html)).
  - oneTBB has no `src/irml` directory; the RML interface is `src/tbb/rml_base.h`.
- **The key observation** (2018 paper): even in counting mode, OpenMP waits for all requested threads before a region starts, while TBB lets threads join computations in progress. That is exactly the asymmetry between a threadteam team, which needs all its members at once, and a Rayon pool, which takes whoever is free.
- **The lesson for two pools:** Intel's remedies are admission control. That means one region at a time (exclusive), a global thread cap (counting, IPC semaphore), or making the inner library serial (threading layer). The two-pool option needs the same, and phase exclusion is the exclusive mode.

## 12. Comparison

| property | single runtime (sections 3 to 9) | two pools (section 11) |
| --- | --- | --- |
| teams, barriers, stable numbers, pinning, wait policy | native, unchanged | native, unchanged (threadteam pool) |
| irregular and nested work | stealing inside a domain | Rayon |
| CPUs of finished siblings flow to the rest | yes, inside the domain (section 6) | no: serial inner kernels, or a Rayon pool per budget |
| concurrent team and stealing work on one CPU set | yes, one set of threads | no, phase exclusion required |
| idle CPU of the inactive side | none: one set of workers, one wait policy | Rayon: 33 search rounds; threadteam: `spin_polls` |
| phase switch | none | measured: wake plus the other side's spin tail |
| dependent tasks with priority | prepared graphs (section 7) | hand-built on Rayon `scope`, no priority |
| reproducible reductions | entry with fixed chunks (section 8) | the same entry on the threadteam side; Rayon's own `reduce` is schedule dependent |
| `Parallelism::Pool` in tensor4all-rs | replaced by `Exec` | served as is |
| ownership rule | unchanged | rewritten (section 11, item 4) |
| implementation risk | a deque, a domain, a descendant rule, new waits; loom and TLA+ | small: engine conventions and tests |
| available | after stages 1 to 4 on the own-thread version | now, on the Rayon version |

## 13. Measurements that decide

All are run with the observers outside the measured CPUs, on a quiet host and on a host with foreign load, with the cold first call reported separately.

1. **Phase switch.** An engine alternates a team kernel (GEMM 512) and a Rayon `par_iter` over 64 items, with gaps of 0, 100 µs and 1 ms. Report p50 and p99 of the first section after a Rayon phase, and of the first Rayon item after a team phase, against each pool alone. This decides item (1).
2. **Idle CPU of the inactive side.** The same alternation. Report CPU time per wall time of the pool that is not running (from `/proc/<tid>/schedstat`), with the default `spin_polls`, and with `spin_polls` reduced to Rayon's search time.
3. **Stranded shares.** `m` independent factorizations of unequal sizes (`m` = 2, 4, 8 on 16 and 64 CPUs). Compare:
   - two pools with serial inner kernels;
   - `fork(m)` with static shares;
   - stage 2 redistribution.

   This decides item (2) and stage 2.
4. **Irregular recursion.** A tree evaluation with uneven branching (the tensor4all-rs interpolation). Compare Rayon on the engine's pool with stage 1 `join`. This decides whether stage 1 is worth its risk, or whether Rayon keeps this work.
5. **Graph kernels.** Tiled Cholesky and look-ahead LU, at tile counts where the critical path matters. Compare Rayon `scope` with hand-built counters against stage 4 plans with rank priority. This decides stage 4.
6. **Draining barrier.** A team kernel with measured idle time at its barriers and independent side work. Report the release delay against task length. This decides stage 3.

The single runtime is justified only where measurements 3 to 6 show a gain the two-pool baseline cannot reach. Measurements 1 and 2 decide whether the two-pool baseline is acceptable at all for engines that alternate phases quickly.

## 14. Open questions

1. **The ownership rule.** May an engine own a Rayon pool next to its threadteam pool (section 11, item 4)? This is a maintainer decision.
2. **Which kernels need stages 1 to 4 now.** Each stage needs a named kernel and its measurement (threading-substrate.md, section 7, item 1).
3. **Whether `TaskExec` needs a `team` entry later.** That would let a graph node or a task open a nested team with a guaranteed width. As specified, it can only request one and accept a refusal (section 6).
4. **`external_rayon`'s width** on `develop` follows the pool, not the token (section 9). It is to be fixed before 0.3.0.

## Not verified

- Fibril's mechanism (only its abstract was read).
- The ROSS 2011 paper by Olivier et al. (only the journal abstract was read).
- HPX's default stealing order and HPX futures.
- Graham 1969 and Brent 1974, the sources of the greedy bound in section 7.
- The paper behind Lace (only the abstract was read).
- OpenCilk and Cilk Plus documentation (not opened; continuation stealing is cited through N3872 and the Cilk-5 paper).
