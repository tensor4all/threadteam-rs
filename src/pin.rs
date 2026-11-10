//! Participant placement on the calling thread's CPU set (on by default).
//!
//! With [`PoolBuilder::pin_participants`](crate::PoolBuilder::pin_participants)
//! on (the default), the leader of a root team scope reads its own affinity mask
//! `S` and places the team inside it, never outside it:
//!
//! * **The leader is pinned to `S[0]` for one section at a time**, never for
//!   the life of a `Team`. One `sched_setaffinity` at scope entry
//!   ([`LeaderPin::enter`]) and one at scope exit (its drop, which also runs on
//!   unwind) restore `S`; a team token that dispatches a section without
//!   opening a scope pins its leader for that section and no longer
//!   ([`home_pin`], whose home is `S[0]` of the set the team recorded at
//!   `into_team`). Between sections the thread keeps `S`, so a vendor library
//!   that creates its thread team from the thread that calls it, or a plain
//!   thread started there, does not inherit a one-CPU mask (tenferro-rs
//!   `PERFORMANCE_TIPS.md`). Inside a section, or a scope, the leader's half of
//!   that rule is [`widened`], behind `TeamExec::external`: it sets the mask to
//!   `S` for the duration of one call and puts `{S[0]}` back when the call
//!   returns or unwinds. A team scope entered inside that call places this
//!   thread again, on the nested team's own home, and the placement before it
//!   comes back when *its* scope ends ([`home_pin`]: a placed thread refuses a
//!   mask that has left the placement, not every second placement), so two
//!   independent pools nest (threadteam-rs#54).
//! * **Member `k >= 1` pins itself onto `P[k]`**, the `k`-th CPU of the
//!   placement order [`order`] gives `S` under the pool's [`PinStrategy`], so
//!   the members take `n - 1` CPUs of `S` other than `S[0]`, one CPU each.
//!   `P[0]` is always `S[0]`, the leader's home. Under
//!   [`PinStrategy::Spread`] (the default) `P` deals the CPUs of `S`
//!   round-robin over the NUMA nodes `S` spans, so a team narrower than `S` is
//!   spread over every node it can reach (threadteam-rs#57); under
//!   [`PinStrategy::Pack`] `P` is `S` in ascending order, the placement before
//!   0.6 (`P[k] = S[k]`). A member saves its previous mask in its own slot and
//!   restores it itself before it stops being reserved.
//! * **No wrapping.** When `S` has fewer CPUs than the team has participants
//!   nothing is pinned (today's unpinned placement), the members of a persistent
//!   team that an earlier scope pinned restore their own masks, and the skip is
//!   counted ([`Pool::pin_skips`](crate::Pool::pin_skips)). Wrapping would stack
//!   two members on one CPU, which is worse than leaving them to the scheduler.
//!
//! Every affinity change is made by the thread that is being moved, with
//! `sched_setaffinity(0, ...)`, so no thread ever changes another thread's
//! placement and there is no thread-id bookkeeping. Linux only: on other
//! targets the option is accepted and does nothing. Nothing on a scope's path
//! allocates: the masks are `cpu_set_t` values on the stack and the placement
//! order is a stack array the leader computes once per root scope. The node
//! topology is read from sysfs once per process, when the first pool that
//! spreads is built ([`host_topology`]).
//!
//! **A mask is not a provider's thread count.** Widening the leader to `S`
//! ([`widened`], `TeamExec::external`) only decides *where* a vendor library's
//! threads may run; *how many* threads OpenMP, OpenBLAS or MKL starts is their
//! own setting (`OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS`,
//! or the library's API), not anything this module sets. A provider left at its
//! default may start one thread per CPU of the machine whatever the mask is;
//! one started from a one-CPU thread is confined to that CPU whatever its count
//! is. Both knobs have to be right; see tenferro-rs `PERFORMANCE_TIPS.md`.
//!
//! **Which CPUs, and when they are decided.** The [`PinStrategy`] is fixed for
//! a pool's life ([`PoolBuilder::pin_strategy`](crate::PoolBuilder::pin_strategy))
//! and the host's node topology for the process's ([`host_topology`]). The
//! order `P` ([`order`]) is computed by the leader once per placing root scope
//! ([`Placement::of`]); the sections inside a scope, and the sections a `Team`
//! token dispatches without a scope, never recompute it: the members keep the
//! CPUs that scope gave them. Under [`PinStrategy::Spread`] the NUMA effect is
//! **not verified on this project's hosts**, which all have one NUMA node; on a
//! one-node mask it places exactly as [`PinStrategy::Pack`].
//!
//! The same CPU set decides whether a root scope is *oversubscribed* (see
//! [`available`]): a team of more participants than the caller's set has CPUs
//! parks its members early instead of spinning (see `awake.rs`), and
//! [`oversubscribed`] is what the root-scope entries use. On a leader this
//! crate placed, the caller's set is the `S` it was placed in, not the one-CPU
//! mask it now runs on. An oversubscribed root scope is one whose pin is
//! skipped; the converse fails only for a root scope of another pool entered on
//! a placed leader, which reads the one-CPU mask, places nobody, and is paced
//! by `S`.

use core::sync::atomic::{AtomicBool, Ordering};

#[cfg(target_os = "linux")]
pub(crate) use sys::count;
#[cfg(not(target_os = "linux"))]
#[allow(unused_imports)]
pub(crate) use sys::count;
pub(crate) use sys::{contains, cpus, read_here, subset, CpuMask};

/// Number of CPUs the calling thread's team is spread over: the CPUs a root
/// team scope entered on this thread may place its participants on.
///
/// This is *not* "CPUs in the calling thread's current affinity mask" once
/// this crate has placed the thread. A placed leader (a [`LeaderPin`], held by
/// a team scope or by one section of a [`Team`](crate::Team) token) runs on the
/// one CPU `{S[0]}`, but it still leads a team spread over `S`, so the answer
/// is `|S|`, read from the placement this thread records. Answering 1 there
/// marked every token a `Team` mints on its own leader thread as oversubscribed
/// and throttled its members' waits (threadteam-rs#33).
///
/// Between sections, and on the thread that called `Pool::into_team`, nothing
/// is placed and the mask itself is the answer, which for that thread is the
/// same `S`.
///
/// On any other Linux thread it is the number of CPUs in its affinity mask,
/// i.e. the same set the pin section places participants on. Elsewhere
/// [`std::thread::available_parallelism`]. Zero is never returned.
///
/// The read is memoised per thread ([`AVAILABLE`]). It is a kernel call
/// (`sched_getaffinity` on Linux, `sysconf` then `__sysctl` on macOS, ~216 ns
/// there), and on the macOS host the mint cost this read almost entirely: 216 ns
/// of `Team::try_exec` before, 47 ns after (threadteam-rs#62). A caller that
/// mints once per small operation used to pay a syscall every time.
///
/// The memo is refreshed whenever this crate touches the mask of the calling
/// thread: it is rewritten by [`note_count`] from the pin section of a root
/// scope, which reads the mask in any case, and forgotten by [`set_here`] and
/// [`set_here_to`], that is, by every leader pin, home pin (including the one a
/// narrower root scope takes when the team has a placement), member restore and
/// unpin. A count another library changed therefore survives only until the next
/// of those on that thread: at most a whole-team scope, and often an earlier
/// narrower one. A pool with participant pinning off is the exception: no scope of
/// this crate touches the mask then, so there the count stands until the thread
/// ends. Off Linux neither setter moves a thread either, so the count stands there
/// too; nothing changes it in practice, `available_parallelism` being a property of
/// the process.
///
/// A mint reads the mask only on a miss, that is, the first mint on a thread and
/// the first after this crate moved its mask. Every other mint takes the count the
/// thread had when it was minted, so a scope can be paced by a stale count if the
/// mask changed while nothing was placed. That is the cost of not reading on every
/// mint, and the count paces member waits and decides nothing else.
pub(crate) fn available() -> usize {
    #[cfg(target_os = "linux")]
    if let Some(placed) = PLACED.with(core::cell::Cell::get) {
        let n = count(&placed.wide);
        if n > 0 {
            return n;
        }
    }
    AVAILABLE.with(|memo| {
        let cached = memo.get();
        if cached != 0 {
            return cached;
        }
        let n = unplaced_count();
        memo.set(n);
        n
    })
}

/// The CPU count of the calling thread with nothing placed on it: the CPUs of
/// its affinity mask on Linux, [`std::thread::available_parallelism`]
/// elsewhere. Zero is never returned.
fn unplaced_count() -> usize {
    #[cfg(target_os = "linux")]
    if let Some(set) = read_here() {
        let n = count(&set);
        if n > 0 {
            return n;
        }
    }
    std::thread::available_parallelism().map_or(1, |n| n.get())
}

thread_local! {
    /// The value [`unplaced_count`] last returned on this thread; zero until
    /// the first read. A `Cell` of a plain `usize`, so a mint neither allocates
    /// nor registers a destructor (threadteam-rs#62).
    static AVAILABLE: core::cell::Cell<usize> = const { core::cell::Cell::new(0) };
}

/// Applies `mask` to the calling thread, then forgets [`AVAILABLE`]: the CPU
/// set of the thread has just changed.
pub(crate) fn set_here(mask: &CpuMask) -> bool {
    let moved = sys::set_here(mask);
    if moved {
        AVAILABLE.with(|memo| memo.set(0));
    }
    moved
}

/// Records a CPU count that the calling thread just read for itself, so that
/// the next mint does not read the mask again.
///
/// Called by the pin section of a root scope, which reads the calling thread
/// mask anyway: that read is also the answer [`available`] would give on the
/// same thread, so it replaces the memo (threadteam-rs#62). A mask another
/// library changed while nothing was placed is therefore seen by the scope
/// that opens next, and by every mint after it.
pub(crate) fn note_count(n: usize) {
    if n > 0 {
        AVAILABLE.with(|memo| memo.set(n));
    }
}

/// Pins the calling thread onto `cpu`, then forgets [`AVAILABLE`].
pub(crate) fn set_here_to(cpu: usize) -> bool {
    let moved = sys::set_here_to(cpu);
    if moved {
        AVAILABLE.with(|memo| memo.set(0));
    }
    moved
}

/// Whether a root scope with `participants` participants is oversubscribed: it
/// has more participants than the calling thread's team has CPUs
/// ([`available`]: `|S|` on a placed leader, its mask elsewhere).
///
/// Evaluated on the thread that enters the root scope, whether or not pinning
/// is on. For a [`Team`](crate::Team) that is the thread that mints the root
/// token *and* again the thread that opens each team scope through it, so a
/// token that travels to a thread with a different CPU set is paced by that
/// thread's set.
pub(crate) fn oversubscribed(participants: usize) -> bool {
    participants > available()
}

/// How the participants of a root team scope are laid out on the calling
/// thread's CPU set `S` when [`PoolBuilder::pin_participants`] is on
/// ([`PoolBuilder::pin_strategy`]).
///
/// Both strategies place participant 0, the leader, on `S[0]`, give every
/// participant its own CPU of `S`, never use a CPU outside `S`, and never wrap.
/// They differ in *which* CPUs a team narrower than `S` takes, which matters
/// when `S` spans several NUMA nodes. A team as wide as `S` takes every CPU of
/// `S` under both, but **not in the same order**: on nodes `{0,1}` and `{2,3}`
/// with four participants `Spread` gives `[0, 2, 1, 3]` and `Pack` gives
/// `[0, 1, 2, 3]`, so participant 1's shard sits on another node. The aggregate
/// set is the same; the mapping is not. On a host (or a mask) with one node, or
/// where the kernel does not report nodes, they place identically.
///
/// [`PoolBuilder::pin_participants`]: crate::PoolBuilder::pin_participants
/// [`PoolBuilder::pin_strategy`]: crate::PoolBuilder::pin_strategy
///
/// # Examples
///
/// ```
/// use threadteam::{PinStrategy, Pool};
/// let pool = Pool::builder().threads(2).build()?;
/// assert_eq!(pool.pin_strategy(), PinStrategy::Spread); // the default
/// let pool = Pool::builder().threads(2).pin_strategy(PinStrategy::Pack).build()?;
/// assert_eq!(pool.pin_strategy(), PinStrategy::Pack);
/// # Ok::<(), threadteam::Error>(())
/// ```
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum PinStrategy {
    /// **The default.** Participant `k` goes round-robin over the NUMA nodes
    /// `S` spans, then in ascending CPU order within a node: the first
    /// participant on each node (nodes in the order of their lowest CPU in
    /// `S`), then the second on each node that has one left, and so on.
    ///
    /// **Invariant:** a team never sees a CPU outside `S`, and it is never
    /// concentrated on fewer nodes than `S` spans unless it has fewer
    /// participants than `S` spans nodes: a team of `n` participants on a mask
    /// spanning `m` nodes covers `min(n, m)` of them.
    ///
    /// Chosen as the default because a freshly allocated destination is the
    /// common case in the tensor4all stack, and first-touch places its pages on
    /// the nodes of the threads that write it: a packed team uses only part of
    /// the machine's memory controllers. **What it may cost:** the reporter of
    /// threadteam-rs#57 saw a warm destination, already in cache, run faster
    /// with the team packed, which they attribute to shared caches. Their
    /// report (Xeon Platinum
    /// 9242, 2 × 48 cores, 4 NUMA nodes, a strided permutation with 48
    /// participants, the reporter's measurement) priced both directions: with
    /// the team packed onto the lowest nodes, 190 µs on a fresh f64 destination
    /// against 121 µs unpinned, but 39 µs warm against 52 µs. Choose
    /// [`Pack`](Self::Pack) when you have measured that your destinations are
    /// warm.
    #[default]
    Spread,
    /// Participant `k` goes on `S[k]`, the `k`-th CPU of `S` in ascending
    /// order: a team narrower than `S` sits on its lowest CPUs, and on a
    /// multi-socket host on its first nodes. The placement of threadteam 0.4
    /// to 0.5. In the threadteam-rs#57 report (the reporter's measurement,
    /// packed against unpinned) it was ahead on a warm destination (39 against
    /// 52 µs) and behind on a fresh one (190 against 121 µs); this crate has not
    /// measured it on a multi-node host.
    Pack,
}

/// The most CPUs a placement can name: the bits of a Linux `cpu_set_t`.
pub(crate) const MAX_CPUS: usize = 1024;

/// Group index of a CPU whose node is not known: one group of its own, after
/// every node id a placement can see.
const UNKNOWN: usize = MAX_CPUS;

/// The NUMA node of each CPU: the input of [`order`].
///
/// Built from "node `i` holds these CPUs" lists ([`Topology::from_nodes`]),
/// once per process for the host ([`host_topology`]), or by a test for a
/// synthetic machine. A CPU no list names is in no node; all such CPUs are
/// placed as one group of their own.
#[derive(Debug, Default)]
pub(crate) struct Topology {
    /// `node_of[cpu]`, `u16::MAX` for a CPU no node names.
    node_of: Box<[u16]>,
}

impl Topology {
    /// The topology in which `nodes[i]` lists the CPUs of node `i`. A CPU named
    /// twice keeps its last node; CPUs and nodes past [`MAX_CPUS`] are ignored.
    // Only `read_host_topology` (Linux) and the tests call this; the doc link
    // above keeps it named on other targets, where nothing calls it.
    #[cfg_attr(not(any(target_os = "linux", test)), allow(dead_code))]
    pub(crate) fn from_nodes(nodes: &[Vec<usize>]) -> Self {
        let width = nodes
            .iter()
            .flatten()
            .copied()
            .filter(|&cpu| cpu < MAX_CPUS)
            .max()
            .map_or(0, |cpu| cpu + 1);
        let mut node_of = vec![u16::MAX; width].into_boxed_slice();
        for (node, cpus) in nodes.iter().enumerate().take(MAX_CPUS) {
            for &cpu in cpus {
                if let Some(slot) = node_of.get_mut(cpu) {
                    *slot = node as u16;
                }
            }
        }
        Topology { node_of }
    }

    /// The group `cpu` is placed in: its node, or [`UNKNOWN`].
    fn group(&self, cpu: u16) -> usize {
        match self.node_of.get(usize::from(cpu)) {
            Some(&node) if node != u16::MAX => usize::from(node),
            _ => UNKNOWN,
        }
    }
}

/// The host's NUMA topology, read once per process from
/// `/sys/devices/system/node/node*/cpulist`. Empty (every CPU in one group,
/// so [`PinStrategy::Spread`] places as [`PinStrategy::Pack`]) off Linux and
/// when sysfs does not list the nodes.
///
/// The first call allocates and reads files; `PoolBuilder::build` makes it for
/// a spreading pool, so a root scope never does.
///
/// Tests may pin it ([`set_test_topology`]) to place on a synthetic machine; the
/// placement rule then runs unchanged, only the node map is injected.
pub(crate) fn host_topology() -> &'static Topology {
    #[cfg(test)]
    if let Some(topology) = TEST_TOPOLOGY.with(std::cell::Cell::get) {
        return topology;
    }
    static HOST: std::sync::OnceLock<Topology> = std::sync::OnceLock::new();
    HOST.get_or_init(read_host_topology)
}

#[cfg(test)]
std::thread_local! {
    /// The topology [`host_topology`] returns on this thread, once a test sets it.
    /// Thread local so a placement test cannot place another test's team.
    static TEST_TOPOLOGY: std::cell::Cell<Option<&'static Topology>> =
        const { std::cell::Cell::new(None) };
}

/// Makes [`host_topology`] read `topology` on the calling thread instead of the
/// host's node map, and [`PinStrategy::Spread`] place on it. `None` restores the
/// host. Test-only: production always reads the machine it runs on.
///
/// The calling thread is the one that places a root scope
/// (`exec::pin_scope`), so a test that sets this and then enters a team scope
/// drives the real path from the pool's strategy to the members' CPUs.
#[cfg(test)]
#[cfg_attr(
    not(all(feature = "parallel", target_os = "linux")),
    allow(dead_code) // only the placement test (Linux, `parallel`) sets it
)]
pub(crate) fn set_test_topology(topology: Option<&'static Topology>) {
    TEST_TOPOLOGY.with(|slot| slot.set(topology));
}

#[cfg(target_os = "linux")]
fn read_host_topology() -> Topology {
    let Ok(dir) = std::fs::read_dir("/sys/devices/system/node") else {
        return Topology::default();
    };
    let mut nodes: Vec<Vec<usize>> = Vec::new();
    for entry in dir.flatten() {
        let name = entry.file_name();
        let Some(id) = name
            .to_str()
            .and_then(|n| n.strip_prefix("node"))
            .and_then(|n| n.parse::<usize>().ok())
        else {
            continue;
        };
        if id >= MAX_CPUS {
            continue;
        }
        let Ok(list) = std::fs::read_to_string(entry.path().join("cpulist")) else {
            continue;
        };
        if nodes.len() <= id {
            nodes.resize_with(id + 1, Vec::new);
        }
        nodes[id] = parse_cpulist(&list);
    }
    Topology::from_nodes(&nodes)
}

#[cfg(not(target_os = "linux"))]
fn read_host_topology() -> Topology {
    Topology::default()
}

/// Parses a sysfs CPU list (`"0-3,8,10-11"`). Malformed pieces are skipped.
#[cfg_attr(not(target_os = "linux"), allow(dead_code))]
fn parse_cpulist(list: &str) -> Vec<usize> {
    let mut cpus = Vec::new();
    for piece in list.trim().split(',').filter(|p| !p.is_empty()) {
        let mut ends = piece.splitn(2, '-').map(|n| n.trim().parse::<usize>());
        match (ends.next(), ends.next()) {
            (Some(Ok(lo)), None) => cpus.push(lo),
            (Some(Ok(lo)), Some(Ok(hi))) if lo <= hi => {
                cpus.extend(lo..=hi.min(MAX_CPUS - 1));
            }
            _ => {}
        }
    }
    cpus
}

/// The placement order of a mask: writes into `out` the CPU of participant
/// `k = 0, 1, ...` and returns how many it wrote, `min(out.len(), mask.len())`.
///
/// A pure function of `(mask, topology, strategy)`, so the placement rule is
/// tested without the hardware. `mask` lists the CPUs of `S` in ascending order
/// (at most [`MAX_CPUS`] are read). The result is a prefix of a permutation of
/// `mask`, so no CPU outside `S` is named and none twice, and `out[0]` is
/// `mask[0]`:
///
/// * [`PinStrategy::Pack`]: `mask` itself.
/// * [`PinStrategy::Spread`]: CPU `c` of node `g` gets the key `(r, o)`, where
///   `r` is the number of CPUs of `mask` in node `g` below `c` and `o` is the
///   rank of `g` among the nodes in the order of their lowest CPU in `mask`;
///   the CPUs are listed by ascending key. So the first `m` participants (`m`
///   nodes spanned) take one CPU on each node, the next take the second CPU of
///   each node that has one, and so on. Two nodes `{0, 1}` and `{2, 3}` give
///   `0, 2, 1, 3`.
///
/// No allocation: about 12 KiB of stack.
pub(crate) fn order(
    mask: &[u16],
    topology: &Topology,
    strategy: PinStrategy,
    out: &mut [u16],
) -> usize {
    let m = mask.len().min(MAX_CPUS);
    let n = out.len().min(m);
    if strategy == PinStrategy::Pack {
        out[..n].copy_from_slice(&mask[..n]);
        return n;
    }
    // Per group: CPUs of `mask` seen so far, and the group's rank by first
    // appearance (`u16::MAX` until it appears).
    let mut seen = [0u16; UNKNOWN + 1];
    let mut rank = [u16::MAX; UNKNOWN + 1];
    let mut groups = 0u16;
    let mut keyed = [(0u32, 0u16); MAX_CPUS];
    for (slot, &cpu) in keyed.iter_mut().zip(&mask[..m]) {
        let g = topology.group(cpu);
        if rank[g] == u16::MAX {
            rank[g] = groups;
            groups += 1;
        }
        *slot = ((u32::from(seen[g]) << 16) | u32::from(rank[g]), cpu);
        seen[g] += 1;
    }
    // Keys are distinct (one CPU per (group, position) pair), so an unstable
    // sort is deterministic.
    keyed[..m].sort_unstable_by_key(|&(key, _)| key);
    for (o, &(_, cpu)) in out[..n].iter_mut().zip(&keyed[..m]) {
        *o = cpu;
    }
    n
}

/// Participant `k`'s CPU on `mask` (ascending CPU list) under `topology` and
/// `strategy`: entry `k` of [`order`], `None` when the mask has `k` or fewer
/// CPUs (never wraps). The placement rule as a pure function of
/// `(mask, topology, k)`; [`Placement::of`] computes all of them at once for a
/// root scope. Under [`PinStrategy::Pack`] it is `mask[k]`, today's
/// `nth(S, k)`.
#[cfg_attr(not(test), allow(dead_code))]
pub(crate) fn cpu_for(
    mask: &[u16],
    topology: &Topology,
    strategy: PinStrategy,
    k: usize,
) -> Option<u16> {
    if k >= mask.len().min(MAX_CPUS) {
        return None;
    }
    let mut out = [0u16; MAX_CPUS];
    let n = order(mask, topology, strategy, &mut out[..=k]);
    (n > k).then(|| out[k])
}

/// The placement of one root scope: participant `k`'s CPU, for every
/// participant. Computed by the leader ([`Placement::of`]) and read by each
/// member in the pin section.
pub(crate) struct Placement {
    cpus: [u16; MAX_CPUS],
    len: usize,
}

impl Placement {
    /// The placement of `participants` participants on `set` under `strategy`,
    /// with the host's topology. Shorter than `participants` only when `set`
    /// has fewer CPUs (the root scope does not place then).
    pub(crate) fn of(set: &CpuMask, strategy: PinStrategy, participants: usize) -> Placement {
        // Pack never reads the topology; an empty one does not allocate, and
        // does not make the scope read sysfs.
        let empty = Topology::default();
        let topology = match strategy {
            PinStrategy::Pack => &empty,
            PinStrategy::Spread => host_topology(),
        };
        Placement::with_topology(set, strategy, participants, topology)
    }

    /// [`Placement::of`] with the topology supplied, so a test can place on a
    /// synthetic machine instead of this one. The whole production path from
    /// `strategy` to participant CPUs runs here: only the node map is injected.
    fn with_topology(
        set: &CpuMask,
        strategy: PinStrategy,
        participants: usize,
        topology: &Topology,
    ) -> Placement {
        let mut mask = [0u16; MAX_CPUS];
        let m = cpus(set, &mut mask);
        let mut placement = Placement {
            cpus: [0; MAX_CPUS],
            len: 0,
        };
        let want = participants.min(MAX_CPUS);
        placement.len = order(&mask[..m], topology, strategy, &mut placement.cpus[..want]);
        placement
    }

    /// Participant `k`'s CPU, `None` past the placement (never wraps).
    pub(crate) fn get(&self, k: usize) -> Option<usize> {
        (k < self.len).then(|| usize::from(self.cpus[k]))
    }
}

/// What a placed root scope remembers about its leader thread: the whole set
/// `S` it started with and the one CPU, `S[0]`, it is pinned to.
#[derive(Clone, Copy)]
struct Placed {
    wide: CpuMask,
    home: usize,
}

thread_local! {
    /// The placement of the root scope the calling thread leads, if it is
    /// placed. `const` initialised and `Copy`, so neither access allocates nor
    /// registers a destructor.
    static PLACED: core::cell::Cell<Option<Placed>> = const { core::cell::Cell::new(None) };
}

/// The calling thread, the leader of a placed root scope, pinned to `S[0]`.
///
/// Dropping it gives the thread `S` back, on the normal path and on unwind.
/// Nested scopes on one thread stack: the previous placement is put back.
pub(crate) struct LeaderPin {
    wide: CpuMask,
    prev: Option<Placed>,
}

impl LeaderPin {
    /// Pins the calling thread onto `home = S[0]`, remembering `wide = S`.
    /// `None` when the kernel refuses the pin: the thread keeps `S`, the scope
    /// places its members anyway and nothing is restored at exit.
    pub(crate) fn enter(wide: &CpuMask, home: usize) -> Option<LeaderPin> {
        if !set_here_to(home) {
            return None;
        }
        let prev = PLACED.with(|p| p.replace(Some(Placed { wide: *wide, home })));
        Some(LeaderPin { wide: *wide, prev })
    }
}

impl Drop for LeaderPin {
    fn drop(&mut self) {
        // Best effort: a refused restore (a cpuset that shrank) leaves the
        // thread on `S[0]`, the mask it already had.
        let _ = set_here(&self.wide);
        PLACED.with(|p| p.set(self.prev));
    }
}

/// The placement a [`Team`](crate::Team) holds: `0` when its members are not
/// placed, otherwise the CPU no member takes plus one.
///
/// The `Team` stores one of these, and every root scope over its whole budget
/// writes it: `S[0] + 1` of the set the scope placed the members in ([`home_of`]), or
/// `0` when the scope placed nobody. A token reads it before each section it
/// dispatches without a scope, and each scope it opens as a reborrow, so the
/// leader goes home to the CPU the members of *that* placement left free.
pub(crate) type Home = usize;

/// The [`Home`] of a placement: participant 0's CPU (`S[0]`) plus one.
pub(crate) fn home_of(placement: &Placement) -> Home {
    placement.get(0).map_or(0, |cpu| cpu + 1)
}

/// Pins the calling thread onto `home` for one section or scope and gives it its
/// own mask `S` back when the returned guard drops (also on unwind).
///
/// The leader of a [`Team`](crate::Team) token calls this before every section
/// it dispatches without opening a team scope: it then opens the section on the
/// one CPU no member takes, and is not confined to it afterwards. `None`, and
/// nothing changed, when the team placed nothing (`home` is `0`), when its own
/// mask is one CPU (a pinned member whose CPU it must keep, and also a leader an
/// enclosing placement already confined to one CPU) or does not hold `home` (a
/// caller that narrowed itself away from it: a leader is never moved outside the
/// mask its caller gave it), when the current mask has left the placement this
/// thread already has (see below), or when the kernel refuses.
///
/// A thread this crate placed is *not* a reason to decline by itself. A placed
/// leader runs on `{S[0]}`; [`widened`] puts `S` back for the duration of one
/// `TeamExec::external` call, and a team scope entered inside that call must be
/// able to place this thread again, on its own home (threadteam-rs#54). The
/// placement that decides this is the one recorded when the thread was placed:
/// narrowing into the nested team's home is allowed while the current mask still
/// belongs to it (`here ⊆ placed.wide`). A mask that reaches outside that
/// placement is a widening of the caller's own doing, and placing the thread
/// there would move it out of the set its caller gave it, so it is refused.
pub(crate) fn home_pin(home: Home) -> Option<LeaderPin> {
    let home = home.checked_sub(1)?;
    let here = read_here()?;
    if count(&here) < 2 || !contains(&here, home) {
        return None;
    }
    if let Some(placed) = PLACED.with(|p| p.get()) {
        if !subset(&here, &placed.wide) {
            return None;
        }
    }
    LeaderPin::enter(&here, home)
}

/// Runs `f` with the calling thread's mask widened to `S` when it leads a
/// placed root scope, and puts `{S[0]}` back when `f` returns or unwinds. On
/// any other thread, or in a scope that placed nobody, it just calls `f`.
///
/// Two `sched_setaffinity` calls per call, paid every time: nothing caches
/// "already wide" and nothing skips the restore. A refused widening leaves the
/// thread as it was (on `{S[0]}`) and `f` still runs, then no restore is
/// needed; a refused restore leaves the thread wide until the scope exits and
/// [`LeaderPin`] gives `S` back.
pub(crate) fn widened<R>(f: impl FnOnce() -> R) -> R {
    struct Restore(usize);
    impl Drop for Restore {
        fn drop(&mut self) {
            let _ = set_here_to(self.0);
        }
    }
    let placed = PLACED.with(|p| p.get());
    // `then`, not `then_some`: a refused widening must not even construct the
    // guard, whose `Drop` would issue the restore this path promises not to make.
    let _restore = placed.and_then(|p| set_here(&p.wide).then(|| Restore(p.home)));
    f()
}

/// The mask a member saved before it pinned itself, and whether it is pinned.
///
/// Written and read only by the thread that owns the slot: a slot id `i` is
/// only ever entered by the worker with index `i` (`claim_job` uses
/// `rayon::current_thread_index()`), and only that thread touches this value —
/// from its `member_loop`, and from `Awake::unpin_member` when the same thread
/// is about to run work that is not its team's (`Exec::external`). The leader
/// never reads it, which is why one saved mask per slot and no
/// cross-reservation cache is enough.
#[cfg_attr(not(feature = "parallel"), allow(dead_code))]
pub(crate) struct SavedMask {
    mask: core::cell::UnsafeCell<core::mem::MaybeUninit<CpuMask>>,
    pinned: AtomicBool,
}

// SAFETY: the `UnsafeCell` is only reached through `save_if_unpinned` and
// `restore`, and both are called exclusively from threads that own the slot: a
// pool worker with index `i` is one fixed thread, it runs the member loop of
// slot `i` only, and `Awake::unpin_member(i)` (`sched::unpin_worker`) is called
// on that same worker. The state those calls leave behind (a saved mask,
// `pinned`) is restored before the loop returns the thread to Rayon, so no
// other thread observes it. `pinned` itself is an atomic, so even a spurious
// read from elsewhere would be sound.
unsafe impl Sync for SavedMask {}

#[cfg_attr(not(feature = "parallel"), allow(dead_code))]
impl SavedMask {
    pub(crate) const fn new() -> Self {
        SavedMask {
            mask: core::cell::UnsafeCell::new(core::mem::MaybeUninit::uninit()),
            pinned: AtomicBool::new(false),
        }
    }

    /// Saves the calling thread's mask unless this member is already pinned.
    ///
    /// A persistent team re-pins its members at every root scope and keeps the
    /// first saved mask, so a member that was never returned to Rayon restores
    /// what it had before the first scope.
    pub(crate) fn save_if_unpinned(&self) {
        if self.pinned.load(Ordering::Relaxed) {
            return;
        }
        // `read_here` is `None` off Linux and when the kernel refuses a mask
        // larger than `cpu_set_t`: then nothing was changed and nothing has to
        // be restored.
        if let Some(mask) = read_here() {
            // SAFETY: this thread owns the cell (see the `Sync` note).
            unsafe { (*self.mask.get()).write(mask) };
            self.pinned.store(true, Ordering::Relaxed);
        }
    }

    /// Restores the saved mask on the calling thread, if this member is pinned.
    ///
    /// Best effort: when the kernel refuses the saved mask (a cpuset changed
    /// under us) the saved value is kept for a later attempt.
    pub(crate) fn restore(&self) {
        if !self.pinned.load(Ordering::Relaxed) {
            return;
        }
        // SAFETY: written by `save_if_unpinned` on this thread (see `Sync`).
        let mask = unsafe { (*self.mask.get()).assume_init_ref() };
        if set_here(mask) {
            self.pinned.store(false, Ordering::Relaxed);
        }
    }
}

#[cfg(target_os = "linux")]
mod sys {
    use core::mem;

    /// The most CPUs a `cpu_set_t` can hold: its representable bits (1024 on
    /// Linux, 128 bytes).
    const SETSIZE: usize = 8 * mem::size_of::<libc::cpu_set_t>();

    /// A thread's CPU set, on the stack (`cpu_set_t`, 128 bytes).
    #[derive(Clone, Copy)]
    pub(crate) struct CpuMask {
        set: libc::cpu_set_t,
    }

    /// A mask holding exactly `cpus`. Tests only: production masks come from
    /// the kernel ([`read_here`]).
    #[cfg(test)]
    pub(crate) fn mask_from(cpus: &[u16]) -> CpuMask {
        // SAFETY: a zeroed `cpu_set_t`; `CPU_ZERO`/`CPU_SET` only clear and set
        // bits, and every caller keeps the indices inside the set.
        unsafe {
            let mut set = mem::zeroed::<libc::cpu_set_t>();
            libc::CPU_ZERO(&mut set);
            for &cpu in cpus {
                libc::CPU_SET(usize::from(cpu), &mut set);
            }
            CpuMask { set }
        }
    }

    /// The calling thread's affinity mask, or `None` when the syscall refuses
    /// it (a kernel mask larger than `cpu_set_t`, no current thread).
    pub(crate) fn read_here() -> Option<CpuMask> {
        // SAFETY: a `cpu_set_t` is plain data; the call writes at most its size.
        let mut set = unsafe { mem::zeroed::<libc::cpu_set_t>() };
        // SAFETY: `set` is a valid, correctly sized `cpu_set_t`; pid 0 is the
        // calling thread, so this thread is the one that is read.
        let r = unsafe { libc::sched_getaffinity(0, mem::size_of::<libc::cpu_set_t>(), &mut set) };
        (r == 0).then_some(CpuMask { set })
    }

    /// Sets the calling thread's mask. `false` when the kernel refuses it; the
    /// thread keeps the mask it had.
    pub(crate) fn set_here(mask: &CpuMask) -> bool {
        // SAFETY: reads a valid `cpu_set_t` of the right size for pid 0 (the
        // calling thread): the calling thread is the one being moved.
        let r = unsafe { libc::sched_setaffinity(0, mem::size_of::<libc::cpu_set_t>(), &mask.set) };
        r == 0
    }

    /// Pins the calling thread onto exactly `cpu`.
    pub(crate) fn set_here_to(cpu: usize) -> bool {
        if cpu >= SETSIZE {
            return false;
        }
        // SAFETY: a zeroed `cpu_set_t`; `CPU_ZERO`/`CPU_SET` only clear and set
        // bits, and `cpu < SETSIZE` keeps the index inside the set.
        unsafe {
            let mut set = mem::zeroed::<libc::cpu_set_t>();
            libc::CPU_ZERO(&mut set);
            libc::CPU_SET(cpu, &mut set);
            set_here(&CpuMask { set })
        }
    }

    /// Whether `cpu` is in the set.
    pub(crate) fn contains(mask: &CpuMask, cpu: usize) -> bool {
        if cpu >= SETSIZE {
            return false;
        }
        // SAFETY: reads one bit of a valid set.
        unsafe { libc::CPU_ISSET(cpu, &mask.set) }
    }

    /// Whether every CPU of `a` is also in `b`.
    pub(crate) fn subset(a: &CpuMask, b: &CpuMask) -> bool {
        (0..SETSIZE).all(|cpu| {
            // SAFETY: reads one bit of each valid set.
            !unsafe { libc::CPU_ISSET(cpu, &a.set) } || unsafe { libc::CPU_ISSET(cpu, &b.set) }
        })
    }

    /// Number of CPUs in the set.
    pub(crate) fn count(mask: &CpuMask) -> usize {
        // SAFETY: only counts the bits of a valid set.
        let n = unsafe { libc::CPU_COUNT(&mask.set) };
        n.max(0) as usize
    }

    /// Writes the CPUs of the set into `out` in ascending order; returns how
    /// many (at most `out.len()`).
    pub(crate) fn cpus(mask: &CpuMask, out: &mut [u16]) -> usize {
        let mut n = 0;
        for cpu in 0..SETSIZE.min(usize::from(u16::MAX)) {
            if n == out.len() {
                break;
            }
            // SAFETY: reads one bit of a valid set.
            if unsafe { libc::CPU_ISSET(cpu, &mask.set) } {
                out[n] = cpu as u16;
                n += 1;
            }
        }
        n
    }
}

#[cfg(not(target_os = "linux"))]
#[allow(dead_code)] // the option is accepted and does nothing off Linux
mod sys {
    /// No CPU set off Linux: every operation is a no-op.
    #[derive(Clone, Copy)]
    pub(crate) struct CpuMask(());

    pub(crate) fn read_here() -> Option<CpuMask> {
        None
    }

    pub(crate) fn set_here(_mask: &CpuMask) -> bool {
        false
    }

    pub(crate) fn set_here_to(_cpu: usize) -> bool {
        false
    }

    pub(crate) fn contains(_mask: &CpuMask, _cpu: usize) -> bool {
        false
    }

    pub(crate) fn subset(_a: &CpuMask, _b: &CpuMask) -> bool {
        false
    }

    pub(crate) fn count(_mask: &CpuMask) -> usize {
        0
    }

    pub(crate) fn cpus(_mask: &CpuMask, _out: &mut [u16]) -> usize {
        0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `order` of `mask` on `nodes`, for `n` participants.
    fn place(nodes: &[Vec<usize>], mask: &[u16], strategy: PinStrategy, n: usize) -> Vec<u16> {
        let topology = Topology::from_nodes(nodes);
        let mut out = vec![0u16; n];
        let len = order(mask, &topology, strategy, &mut out);
        out.truncate(len);
        out
    }

    fn two_nodes() -> Vec<Vec<usize>> {
        vec![vec![0, 1], vec![2, 3]]
    }

    /// The placement object on a synthetic two-node machine, with both
    /// strategies and at two widths, so a mapping change is visible without
    /// multi-node hardware. Which strategy `exec::pin_scope` hands in is covered
    /// by `tests::placement::a_root_scope_places_under_the_pools_pin_strategy`,
    /// which drives a real team through the seam (`pin::set_test_topology`).
    #[cfg(target_os = "linux")]
    #[test]
    fn placement_takes_the_strategy_on_a_two_node_machine() {
        use super::sys::mask_from;
        let set = mask_from(&[0, 1, 2, 3]);
        let topology = Topology::from_nodes(&two_nodes());
        let placed = |strategy| {
            let placement = Placement::with_topology(&set, strategy, 2, &topology);
            (0..2).filter_map(|k| placement.get(k)).collect::<Vec<_>>()
        };
        assert_eq!(placed(PinStrategy::Spread), vec![0, 2]);
        assert_eq!(placed(PinStrategy::Pack), vec![0, 1]);
        // Full width: the set is the same, the mapping is not.
        let wide = |strategy| {
            let placement = Placement::with_topology(&set, strategy, 4, &topology);
            (0..4).filter_map(|k| placement.get(k)).collect::<Vec<_>>()
        };
        assert_eq!(wide(PinStrategy::Spread), vec![0, 2, 1, 3]);
        assert_eq!(wide(PinStrategy::Pack), vec![0, 1, 2, 3]);
    }

    fn four_nodes() -> Vec<Vec<usize>> {
        vec![vec![0, 1], vec![2, 3], vec![4, 5], vec![6, 7]]
    }

    #[test]
    fn spread_alternates_the_nodes_of_a_two_node_mask() {
        let got = place(&two_nodes(), &[0, 1, 2, 3], PinStrategy::Spread, 4);
        assert_eq!(got, [0, 2, 1, 3]);
    }

    #[test]
    fn pack_is_the_mask_in_ascending_order() {
        let got = place(&two_nodes(), &[0, 1, 2, 3], PinStrategy::Pack, 4);
        assert_eq!(got, [0, 1, 2, 3]);
        let got = place(&two_nodes(), &[0, 1, 2, 3], PinStrategy::Pack, 2);
        assert_eq!(got, [0, 1], "a narrow packed team sits on node 0");
    }

    #[test]
    fn a_narrow_spread_team_covers_both_nodes() {
        let got = place(&two_nodes(), &[0, 1, 2, 3], PinStrategy::Spread, 2);
        assert_eq!(got, [0, 2]);
    }

    #[test]
    fn a_team_of_one_on_a_four_node_mask_gets_one_cpu() {
        let mask: Vec<u16> = (0..8).collect();
        for strategy in [PinStrategy::Spread, PinStrategy::Pack] {
            assert_eq!(
                place(&four_nodes(), &mask, strategy, 1),
                [0],
                "{strategy:?}"
            );
        }
    }

    #[test]
    fn spread_never_leaves_the_mask() {
        // The mask covers parts of three of the four nodes.
        let got = place(&four_nodes(), &[1, 2, 3, 5], PinStrategy::Spread, 4);
        assert_eq!(got, [1, 2, 5, 3]);
    }

    #[test]
    fn uneven_nodes_drop_out_of_the_rotation_when_empty() {
        let nodes = vec![vec![0, 1, 2], vec![3]];
        let got = place(&nodes, &[0, 1, 2, 3], PinStrategy::Spread, 4);
        assert_eq!(got, [0, 3, 1, 2]);
    }

    #[test]
    fn nodes_are_visited_in_the_order_of_their_lowest_cpu() {
        // Node 1 holds the lowest CPU, so it leads and `S[0]` stays first.
        let nodes = vec![vec![4, 5], vec![0, 1]];
        let got = place(&nodes, &[0, 1, 4, 5], PinStrategy::Spread, 4);
        assert_eq!(got, [0, 4, 1, 5]);
        // Interleaved node numbering (node 0 = even CPUs, node 1 = odd).
        let nodes = vec![vec![0, 2], vec![1, 3]];
        let got = place(&nodes, &[0, 1, 2, 3], PinStrategy::Spread, 4);
        assert_eq!(got, [0, 1, 2, 3]);
    }

    #[test]
    fn without_a_topology_spread_places_as_pack() {
        let mask: Vec<u16> = (0..6).collect();
        let got = place(&[], &mask, PinStrategy::Spread, 6);
        assert_eq!(got, mask, "every CPU in one unknown group");
        // CPUs a partial topology does not name form one group of their own.
        let got = place(&[vec![0, 1]], &[0, 1, 8, 9], PinStrategy::Spread, 4);
        assert_eq!(got, [0, 8, 1, 9]);
    }

    #[test]
    fn a_placement_is_never_longer_than_the_mask() {
        let got = place(&two_nodes(), &[0, 2], PinStrategy::Spread, 3);
        assert_eq!(got, [0, 2], "no wrapping");
    }

    /// The invariant on `PinStrategy::Spread`, over every mask of an 8-CPU
    /// machine with two and with four nodes and every team size: the placement
    /// is a prefix of a permutation of the mask that starts at `S[0]`, and a team
    /// of `n` covers `min(n, m)` of the `m` nodes the mask spans.
    #[test]
    fn spread_covers_as_many_nodes_as_the_team_allows() {
        let two = vec![vec![0, 1, 2, 3], vec![4, 5, 6, 7]];
        for nodes in [two, four_nodes()] {
            let node_of = |cpu: u16| nodes.iter().position(|n| n.contains(&usize::from(cpu)));
            for bits in 1u32..256 {
                let mask: Vec<u16> = (0..8).filter(|c| bits & (1 << c) != 0).collect();
                let mut spanned: Vec<_> = mask.iter().map(|&c| node_of(c)).collect();
                spanned.sort();
                spanned.dedup();
                for n in 1..=mask.len() {
                    let got = place(&nodes, &mask, PinStrategy::Spread, n);
                    assert_eq!(got.len(), n);
                    assert_eq!(got[0], mask[0], "the leader is on S[0]");
                    let mut sorted = got.clone();
                    sorted.sort();
                    sorted.dedup();
                    assert_eq!(sorted.len(), n, "no CPU twice: {got:?}");
                    assert!(got.iter().all(|c| mask.contains(c)), "inside S");
                    let mut covered: Vec<_> = got.iter().map(|&c| node_of(c)).collect();
                    covered.sort();
                    covered.dedup();
                    assert_eq!(
                        covered.len(),
                        n.min(spanned.len()),
                        "mask {mask:?}, n {n}: {got:?}"
                    );
                }
            }
        }
    }

    #[test]
    fn on_a_one_node_mask_spread_and_pack_place_identically() {
        // A mask inside one node of a multi-node machine, a one-node machine,
        // and a machine whose nodes the kernel does not report.
        let cases: [(Vec<Vec<usize>>, Vec<u16>); 3] = [
            (four_nodes(), vec![2, 3]),
            (vec![(0..8).collect()], (0..8).collect()),
            (vec![], vec![1, 3, 4, 6]),
        ];
        for (nodes, mask) in cases {
            let topology = Topology::from_nodes(&nodes);
            for k in 0..=mask.len() {
                let spread = cpu_for(&mask, &topology, PinStrategy::Spread, k);
                let pack = cpu_for(&mask, &topology, PinStrategy::Pack, k);
                assert_eq!(spread, pack, "mask {mask:?}, k {k}");
                assert_eq!(pack, mask.get(k).copied(), "Pack is nth(S, k)");
            }
        }
    }

    #[test]
    fn cpu_for_is_entry_k_of_the_order() {
        let topology = Topology::from_nodes(&two_nodes());
        let mask = [0, 1, 2, 3];
        let got: Vec<_> = (0..5)
            .map(|k| cpu_for(&mask, &topology, PinStrategy::Spread, k))
            .collect();
        assert_eq!(got, [Some(0), Some(2), Some(1), Some(3), None]);
    }

    #[test]
    fn a_narrow_spread_team_covers_all_four_nodes() {
        let mask: Vec<u16> = (0..8).collect();
        let got = place(&four_nodes(), &mask, PinStrategy::Spread, 4);
        assert_eq!(got, [0, 2, 4, 6], "one CPU on each of the four nodes");
    }

    #[test]
    fn sysfs_cpu_lists_parse() {
        assert_eq!(parse_cpulist("0-3,8,10-11\n"), [0, 1, 2, 3, 8, 10, 11]);
        assert_eq!(parse_cpulist(""), Vec::<usize>::new());
        assert_eq!(parse_cpulist("5"), [5]);
        assert_eq!(
            parse_cpulist("x,2,3-1"),
            [2],
            "malformed pieces are skipped"
        );
    }

    /// The refusal assertion in `tests/pinning.rs` sees the mask, and a placement
    /// onto the CPU the thread is already on leaves the mask unchanged. What
    /// separates "refused" from "placed again on the same CPU" is the recorded
    /// placement `S`, which `available()` reports: a second placement would
    /// replace `S` with the one-CPU mask the thread runs on and halve the answer.
    #[test]
    fn a_refused_second_placement_keeps_the_recorded_set() {
        let Some(wide) = read_here() else {
            return; // not Linux, or no affinity support: nothing to place
        };
        let width = count(&wide);
        if width < 2 {
            return; // one allowed CPU: the placement is skipped by design
        }
        let home = Placement::of(&wide, PinStrategy::Pack, 1)
            .get(0)
            .expect("a CPU in a non-empty mask");
        // A refused pin is supported behaviour (`LeaderPin::enter` returns `None`
        // and the scope places nobody), so this is a skip, not a failure.
        let Some(first) = LeaderPin::enter(&wide, home) else {
            eprintln!("skipped: the kernel refused to pin this thread");
            return;
        };
        assert_eq!(
            available(),
            width,
            "the recorded set is the one the scope started in"
        );

        assert!(
            home_pin(home + 1).is_none(),
            "a second placement in the same scope is refused"
        );
        assert_eq!(
            available(),
            width,
            "the refused placement must not overwrite the recorded set"
        );

        drop(first);
        assert_eq!(
            available(),
            width,
            "the set comes back when the placement ends"
        );
    }

    /// threadteam-rs#62: the CPU count is read once per thread, and a mask
    /// change this crate makes forgets it. A mask changed behind the memo is
    /// deliberately not observed until this crate moves the thread again.
    #[cfg(target_os = "linux")]
    #[test]
    fn the_cpu_count_is_memoised_until_this_crate_moves_the_thread() {
        let whole = match read_here() {
            Some(mask) if count(&mask) >= 2 => mask,
            _ => {
                eprintln!("skipped: need at least 2 CPUs in the calling thread mask");
                return;
            }
        };
        let wide = count(&whole);
        assert_eq!(available(), wide, "the first read answers the mask");
        let mut first = [0u16; 1];
        assert_eq!(cpus(&whole, &mut first), 1, "the mask holds a CPU");
        let one = sys::mask_from(&first[..]);
        assert!(sys::set_here(&one), "restricting the calling thread");
        assert_eq!(
            available(),
            wide,
            "no kernel read on the next mint: the memo answers"
        );
        assert!(set_here(&one), "a mask change this crate makes");
        assert_eq!(available(), 1, "which forgets the memo");
        assert!(set_here(&whole), "restoring the calling thread mask");
        assert_eq!(available(), wide);
    }
}
