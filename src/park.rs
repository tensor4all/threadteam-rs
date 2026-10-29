//! The park/wake handshake of reserved members, isolated so that it can be
//! model-checked with `loom` and driven by test hooks.
//!
//! Members that found no work park here; a leader that changed the state they
//! wait on (a generation, a reservation state, the lent flag) wakes them. The
//! rules:
//!
//! 1. A member raises `parked` (an acquire-release increment), takes `lock`,
//!    re-checks the state it waits on, and waits on the condition variable only
//!    while the state is unchanged; it lowers `parked` when it leaves.
//! 2. A leader changes the state (SeqCst), then reads `parked` with an
//!    acquire-release `fetch_add(0)`. If it is nonzero, the leader takes and
//!    drops `lock` and notifies everybody.
//!
//! Either the member's re-check sees the new state, or the leader sees
//! `parked`; and the leader's lock handoff orders its notification after a
//! member that passed its re-check has entered the wait. The argument uses the
//! modification order of `parked` (both sides are read-modify-writes) and release
//! sequences, not a total order of SeqCst accesses, because `loom` does not model
//! the latter and the exact code is model-checked. With nobody parked a wake is
//! one uncontended atomic read-modify-write.

use crate::sync::{AtomicBool, AtomicUsize, Condvar, Mutex, Ordering};

/// Places where a test can pause a thread (feature `test-hooks`).
#[doc(hidden)]
#[allow(dead_code)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Point {
    /// Member: `parked` raised, lock not yet taken.
    ParkRaised,
    /// Member: lock held, state re-checked and still unchanged, wait not yet entered.
    ParkChecked,
    /// Leader: state changed, `parked` not yet read.
    WakeBegin,
}

/// Deliberate protocol breakages, to show that the hook tests can fail
/// (feature `test-hooks`).
#[doc(hidden)]
#[allow(dead_code)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mutation {
    None,
    /// The member does not re-check the state before its first wait.
    SkipRecheck,
    /// The leader notifies without taking the lock first.
    NoLockHandoff,
}

#[inline]
fn hook(_point: Point) {
    #[cfg(feature = "test-hooks")]
    crate::test_hooks::fire(_point);
}

#[inline]
#[cfg_attr(not(feature = "parallel"), allow(dead_code))]
fn skip_recheck() -> bool {
    #[cfg(feature = "test-hooks")]
    {
        crate::test_hooks::mutation() == Mutation::SkipRecheck
    }
    #[cfg(not(feature = "test-hooks"))]
    {
        false
    }
}

#[inline]
fn no_lock_handoff() -> bool {
    #[cfg(feature = "test-hooks")]
    {
        crate::test_hooks::mutation() == Mutation::NoLockHandoff
    }
    #[cfg(not(feature = "test-hooks"))]
    {
        false
    }
}

/// Parked members of one pool.
#[doc(hidden)]
pub struct Parker {
    lock: Mutex<()>,
    cv: Condvar,
    /// Members inside (or entering) `park_while`.
    parked: AtomicUsize,
}

impl Default for Parker {
    fn default() -> Self {
        Self::new()
    }
}

impl Parker {
    /// A parker. The lock and the condition variable are touched once here:
    /// some platforms allocate them lazily on first use, and no later park or
    /// wake should allocate.
    pub fn new() -> Self {
        let p = Parker {
            lock: Mutex::new(()),
            cv: Condvar::new(),
            parked: AtomicUsize::new(0),
        };
        drop(p.lock.lock().unwrap_or_else(|e| e.into_inner()));
        p.cv.notify_all();
        p
    }

    /// Sleeps while `still_waiting()` holds. The state it reads must be
    /// changed by the leader with SeqCst before `wake`.
    #[cfg_attr(not(feature = "parallel"), allow(dead_code))]
    pub fn park_while(&self, still_waiting: impl Fn() -> bool) {
        self.parked.fetch_add(1, Ordering::AcqRel);
        hook(Point::ParkRaised);
        let mut guard = self.lock.lock().unwrap_or_else(|e| e.into_inner());
        let mut first = true;
        while (first && skip_recheck()) || still_waiting() {
            first = false;
            hook(Point::ParkChecked);
            guard = self.cv.wait(guard).unwrap_or_else(|e| e.into_inner());
        }
        drop(guard);
        self.parked.fetch_sub(1, Ordering::AcqRel);
    }

    /// Wakes every parked member; call after changing the state they wait on.
    pub fn wake(&self) {
        hook(Point::WakeBegin);
        // A read-modify-write, not a load: it joins the modification order of
        // `parked`, so either it follows a member's increment (and sees it) or
        // that increment reads from it and then sees the new state. This relies
        // on release sequences only, not on a total order of SeqCst accesses
        // (which `loom` does not model).
        if self.parked.fetch_add(0, Ordering::AcqRel) != 0 {
            if !no_lock_handoff() {
                drop(self.lock.lock().unwrap_or_else(|e| e.into_inner()));
            }
            self.cv.notify_all();
        }
    }
}

// ---------------------------------------------------------------------------
// Staged waits (`docs/design/waiting.md`): libgomp's waits. A bounded spin of
// polls, a few yields, then a sleep on a `Parker`. The join and the barrier of
// a section use them through [`Completion`] and [`Barrier`], which are the
// exact code `loom` checks.

/// Spin budget, in polls, of a wait in a statically oversubscribed scope (more
/// participants than CPUs in the caller's set). libgomp's throttled spin count
/// has the same value.
pub(crate) const THROTTLED_POLLS: u32 = 100;

/// Yields between the spin and the sleep. Each lets a thread queued on the same
/// CPU run without a futex round trip.
const YIELD_POLLS: u32 = if cfg!(loom) { 0 } else { 16 };

/// What a waiting thread does after one [`Waiter::step`].
#[doc(hidden)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Step {
    /// It polled (spun or yielded); poll the condition again.
    Poll,
    /// The spin and the yields are spent: sleep until woken.
    Sleep,
}

/// One wait episode: up to `budget` spin polls (one PAUSE, or YIELD on
/// aarch64, each), then [`YIELD_POLLS`] yields, then sleep.
///
/// The budget is a count, as libgomp's `GOMP_SPINCOUNT`: a waiter that is
/// descheduled does not spend it while it waits for its CPU. Nothing in the
/// poll loop reads a clock.
#[doc(hidden)]
pub struct Waiter {
    budget: u32,
    polls: u32,
    yields: u32,
    slept: bool,
}

impl Waiter {
    /// A wait episode with a spin of `budget` polls.
    pub fn new(budget: u32) -> Self {
        Waiter {
            budget,
            polls: 0,
            yields: 0,
            slept: false,
        }
    }

    /// Whether the episode reached its sleep phase.
    pub fn slept(&self) -> bool {
        self.slept
    }

    /// One poll of the episode; [`Step::Sleep`] once the spin and the yields
    /// are spent (and on every call after that).
    #[inline]
    pub fn step(&mut self) -> Step {
        if self.polls < self.budget {
            self.polls += 1;
            std::hint::spin_loop();
            return Step::Poll;
        }
        if self.yields < YIELD_POLLS {
            self.yields += 1;
            std::thread::yield_now();
            return Step::Poll;
        }
        self.slept = true;
        Step::Sleep
    }
}

#[repr(align(128))]
struct Pad<T>(T);

/// The dispatched shares of a section that have not finished. The leader's
/// join waits on it; the member that finishes the last share wakes the leader.
#[doc(hidden)]
pub struct Completion {
    remaining: Pad<AtomicUsize>,
}

impl Completion {
    /// `n` shares outstanding.
    pub fn new(n: usize) -> Self {
        Completion {
            remaining: Pad(AtomicUsize::new(n)),
        }
    }

    /// Whether every share has finished.
    pub fn is_done(&self) -> bool {
        self.remaining.0.load(Ordering::Acquire) == 0
    }

    /// One share finished; the last one wakes a sleeping leader through `sync`.
    ///
    /// # Safety
    ///
    /// `this` is valid when called. It is not dereferenced after the decrement,
    /// because the leader may free it as soon as it observes zero; `sync` must
    /// outlive the call independently of `this`.
    pub unsafe fn finish(this: *const Completion, sync: Option<&Parker>) {
        // SeqCst: the state change of the `Parker` protocol (rule 2).
        if (*this).remaining.0.fetch_sub(1, Ordering::SeqCst) == 1 {
            if let Some(sync) = sync {
                sync.wake();
            }
        }
    }

    /// Waits until every share has finished: spin, yield, then sleep on `sync`.
    pub fn wait(&self, sync: Option<&Parker>, waiter: &mut Waiter) {
        while !self.is_done() {
            if waiter.step() == Step::Sleep {
                match sync {
                    Some(sync) => sync.park_while(|| self.remaining.0.load(Ordering::SeqCst) != 0),
                    None => std::hint::spin_loop(),
                }
            }
        }
    }
}

/// The sense-reversing barrier of a section's participants.
#[doc(hidden)]
pub struct Barrier {
    size: usize,
    arrived: Pad<AtomicUsize>,
    epoch: Pad<AtomicUsize>,
}

impl Barrier {
    /// A barrier over `size` participants.
    pub fn new(size: usize) -> Self {
        Barrier {
            size,
            arrived: Pad(AtomicUsize::new(0)),
            epoch: Pad(AtomicUsize::new(0)),
        }
    }

    /// Waits until all `size` participants arrived, or `cancelled` is raised
    /// (then `false`). The last arrival opens the barrier and wakes sleepers
    /// through `sync`; whoever raises `cancelled` must wake `sync` too.
    pub fn arrive(
        &self,
        cancelled: &AtomicBool,
        sync: Option<&Parker>,
        waiter: &mut Waiter,
    ) -> bool {
        if cancelled.load(Ordering::Acquire) {
            return false;
        }
        if self.size == 1 {
            return true;
        }
        let epoch = self.epoch.0.load(Ordering::Acquire);
        if self.arrived.0.fetch_add(1, Ordering::AcqRel) + 1 == self.size {
            self.arrived.0.store(0, Ordering::Relaxed);
            // SeqCst: the state change of the `Parker` protocol (rule 2).
            self.epoch.0.store(epoch.wrapping_add(1), Ordering::SeqCst);
            if let Some(sync) = sync {
                sync.wake();
            }
            return true;
        }
        loop {
            if self.epoch.0.load(Ordering::Acquire) != epoch {
                return true;
            }
            if cancelled.load(Ordering::Acquire) {
                return false;
            }
            if waiter.step() == Step::Sleep {
                match sync {
                    Some(sync) => sync.park_while(|| {
                        self.epoch.0.load(Ordering::SeqCst) == epoch
                            && !cancelled.load(Ordering::SeqCst)
                    }),
                    None => std::hint::spin_loop(),
                }
            }
        }
    }
}
