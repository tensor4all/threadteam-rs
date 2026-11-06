//! Participant placement on the calling thread's CPU set (on by default).
//!
//! With [`PoolBuilder::pin_participants`](crate::PoolBuilder::pin_participants)
//! on (the default), the leader of a root team scope reads its own affinity mask
//! `S` and places the team inside it, never outside it:
//!
//! * **The leader is pinned to `S[0]` for the life of the scope**, and the
//!   thread that created a `Team` for the life of the team ([`TeamPin`]),
//!   which is what places a team token that never opens a scope. One
//!   `sched_setaffinity` at scope entry ([`LeaderPin::enter`]) and one at scope
//!   exit (its drop, which also runs on unwind) restore `S`; there is no check
//!   and no move before a section. The price is that a vendor library creates
//!   its thread team from the thread that calls it, and a team created from a
//!   one-CPU thread silently confines the whole library team to that CPU
//!   (tenferro-rs `PERFORMANCE_TIPS.md`). The leader's half of that rule is
//!   [`widened`], behind `TeamExec::external`: it sets the mask to `S` for the
//!   duration of one call and puts `{S[0]}` back when the call returns or
//!   unwinds.
//! * **Member `k >= 1` pins itself onto `S[k]`**, so the members take `S` minus
//!   `S[0]`, one CPU each. A member saves its previous mask in its own slot and
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
//! targets the option is accepted and does nothing. Nothing here allocates: the
//! masks are `cpu_set_t` values on the stack and the `k`-th set CPU is found by
//! scanning one.
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
pub(crate) use sys::{nth, read_here, set_here, set_here_to, CpuMask};

/// Number of CPUs the calling thread's team is spread over: the CPUs a root
/// team scope entered on this thread may place its participants on.
///
/// This is *not* "CPUs in the calling thread's current affinity mask" once
/// this crate has placed the thread. A placed leader (a [`LeaderPin`], which a
/// [`TeamPin`] holds for the life of a `Team`) runs on the one CPU `{S[0]}`,
/// but it still leads a team spread over `S`, so the answer is `|S|`, read
/// from the placement this thread records. Answering 1 there marked every
/// token a `Team` mints on its own leader thread as oversubscribed and
/// throttled its members' waits (threadteam-rs#33).
///
/// On any other Linux thread it is the number of CPUs in its affinity mask,
/// i.e. the same set the pin section places participants on. Elsewhere
/// [`std::thread::available_parallelism`]. Zero is never returned.
pub(crate) fn available() -> usize {
    #[cfg(target_os = "linux")]
    if let Some(placed) = PLACED.with(core::cell::Cell::get) {
        let n = count(&placed.wide);
        if n > 0 {
            return n;
        }
    }
    #[cfg(target_os = "linux")]
    if let Some(set) = read_here() {
        let n = count(&set);
        if n > 0 {
            return n;
        }
    }
    std::thread::available_parallelism().map_or(1, |n| n.get())
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

/// Pins the calling thread onto the `k`-th CPU of `set`.
///
/// Called by a member for itself (`k >= 1`); the leader never calls it. Never
/// wraps: `k >= |set|` leaves the thread where it was (the root scope does not
/// pin at all in that case, see `exec::pin_scope`, so this is only a second
/// line of defence). It does not look at the current mask first: it asks the
/// kernel for the single CPU `set[k]` wherever the thread happens to be, so a
/// thread that was outside `set` is moved into it. Best effort: an empty set,
/// or a refused syscall (the CPU is not in the caller's permitted cpuset),
/// leaves the thread where it was.
pub(crate) fn pin_to(set: &CpuMask, k: usize) {
    if let Some(cpu) = nth(set, k) {
        set_here_to(cpu);
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

/// The thread that created a [`Team`](crate::Team) (`Pool::into_team*`), its
/// leader, pinned to `S[0]` for the life of the team.
///
/// A root scope's pin ends with the scope, and `into_team` opens one only to
/// acknowledge, so a team driven by tokens that never open a scope
/// (`fork_sized`, `for_ranges`, `broadcast`) would dispatch with its members on
/// `S[1..]` and its leader on its whole mask. The team holds this pin instead:
/// one `sched_setaffinity` when the team is created, one when it is dropped or
/// turned back into its pool, nothing per token or per section. A root scope
/// opened on this thread runs under this pin and places the members in this
/// `S` instead of reading the (now one-CPU) mask. `TeamExec::external` is the
/// only call that widens it, for the duration of one call.
///
/// The pin belongs to the creating thread. A `Team` is `Send` and `Sync`: a
/// token minted on another thread dispatches from a leader that is not placed,
/// and a team dropped on another thread restores nothing (that thread was not
/// moved), so the creating thread keeps `{S[0]}`.
pub(crate) struct TeamPin {
    pin: core::mem::ManuallyDrop<LeaderPin>,
    thread: std::thread::ThreadId,
}

impl TeamPin {
    /// Pins the calling thread onto `S[0]` of its own mask `S` when `S` has a
    /// CPU for each of the `participants`; `None` (and nothing changed) when
    /// it has not, off Linux, or when the kernel refuses.
    pub(crate) fn enter(participants: usize) -> Option<TeamPin> {
        let set = read_here()?;
        if count(&set) < participants {
            return None;
        }
        let pin = LeaderPin::enter(&set, nth(&set, 0)?)?;
        Some(TeamPin {
            pin: core::mem::ManuallyDrop::new(pin),
            thread: std::thread::current().id(),
        })
    }

    /// `S`, when the calling thread is the one this team pinned.
    pub(crate) fn set_here(&self) -> Option<&CpuMask> {
        (std::thread::current().id() == self.thread).then_some(&self.pin.wide)
    }
}

impl Drop for TeamPin {
    fn drop(&mut self) {
        if std::thread::current().id() == self.thread {
            // SAFETY: dropped once, here, and never used afterwards.
            unsafe { core::mem::ManuallyDrop::drop(&mut self.pin) };
        }
    }
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
    let _restore = placed.and_then(|p| set_here(&p.wide).then_some(Restore(p.home)));
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

    /// Number of CPUs in the set.
    pub(crate) fn count(mask: &CpuMask) -> usize {
        // SAFETY: only counts the bits of a valid set.
        let n = unsafe { libc::CPU_COUNT(&mask.set) };
        n.max(0) as usize
    }

    /// Index of the `k`-th set CPU; `None` when the set has `k` or fewer CPUs
    /// (never wrap: two participants on one CPU is the failure the placement
    /// exists to prevent).
    pub(crate) fn nth(mask: &CpuMask, k: usize) -> Option<usize> {
        if k >= count(mask) {
            return None;
        }
        let mut want = k;
        for cpu in 0..SETSIZE {
            // SAFETY: reads one bit of a valid set.
            if unsafe { libc::CPU_ISSET(cpu, &mask.set) } {
                if want == 0 {
                    return Some(cpu);
                }
                want -= 1;
            }
        }
        None
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

    pub(crate) fn count(_mask: &CpuMask) -> usize {
        0
    }

    pub(crate) fn nth(_mask: &CpuMask, _k: usize) -> Option<usize> {
        None
    }
}
