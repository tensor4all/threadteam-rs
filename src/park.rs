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

use crate::sync::{AtomicUsize, Condvar, Mutex, Ordering};

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
