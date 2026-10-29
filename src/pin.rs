//! Opt-in participant placement on the calling thread's CPU set.
//!
//! With [`PoolBuilder::pin_participants`](crate::PoolBuilder::pin_participants)
//! on, the leader of a root team scope reads its own affinity mask `S` and every
//! **member** pins itself onto one CPU of `S`: participant `k >= 1` onto the
//! `k`-th set CPU of `S`, wrapping modulo `|S|` when `k >= |S|`. The leader
//! (participant 0, the calling thread) is **never pinned**: its mask is never
//! changed, not even for the scope, and `S[0]` is left to it without enforcing
//! it. A vendor library creates its thread team from the thread that calls it,
//! and a team created from a one-CPU thread silently confines the whole library
//! team to that CPU (tenferro-rs `PERFORMANCE_TIPS.md`), so the calling thread
//! keeps its CPU set. A member saves its previous mask in its own slot and
//! restores it itself before it stops being reserved.
//!
//! Every affinity change is made by the thread that is being moved, with
//! `sched_setaffinity(0, ...)`, so no thread ever changes another thread's
//! placement and there is no thread-id bookkeeping. Linux only: on other
//! targets the option is accepted and does nothing. Nothing here allocates: the
//! masks are `cpu_set_t` values on the stack and the `k`-th set CPU is found by
//! scanning one.
//!
//! The same CPU set decides whether a root scope is *oversubscribed* (see
//! [`available`]): a team of more participants than the caller's mask has CPUs
//! parks its members early instead of spinning (see `awake.rs`), and
//! [`oversubscribed`] is what the root-scope entries use.

use core::sync::atomic::{AtomicBool, Ordering};

#[cfg(target_os = "linux")]
pub(crate) use sys::count;
#[cfg(not(target_os = "linux"))]
#[allow(unused_imports)]
pub(crate) use sys::count;
pub(crate) use sys::{nth, read_here, set_here, set_here_to, CpuMask};

/// Number of CPUs a root team scope may spread its participants over.
///
/// On Linux the number of CPUs in the calling thread's affinity mask, i.e. the
/// same set the pin section places participants on. Elsewhere
/// [`std::thread::available_parallelism`]. Zero is never returned.
pub(crate) fn available() -> usize {
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
/// has more participants than the calling thread's CPU set has CPUs.
///
/// Evaluated on the thread that enters the root scope, whether or not pinning
/// is on. For a [`Team`](crate::Team) that is the thread that mints the root
/// token *and* again the thread that opens each team scope through it, so a
/// token that travels to a thread with a different CPU set is paced by that
/// thread's set.
pub(crate) fn oversubscribed(participants: usize) -> bool {
    participants > available()
}

/// Pins the calling thread onto the `k`-th CPU of `set` (wrapping).
///
/// Called by a member for itself (`k >= 1`); the leader never calls it. Best
/// effort: an empty set (never produced by [`available`]), a refused syscall, or
/// the calling thread already being outside `set` leaves the thread where it
/// was.
pub(crate) fn pin_to(set: &CpuMask, k: usize) {
    if let Some(cpu) = nth(set, k) {
        set_here_to(cpu);
    }
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

    /// Whether a pin is still in effect (a restore failed or is pending).
    pub(crate) fn is_pinned(&self) -> bool {
        self.pinned.load(Ordering::Relaxed)
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

    /// Index of the `k`-th set CPU, wrapping modulo the set size; `None` for an
    /// empty set (never pin to an empty mask).
    pub(crate) fn nth(mask: &CpuMask, k: usize) -> Option<usize> {
        let n = count(mask);
        if n == 0 {
            return None;
        }
        let mut want = k % n;
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
