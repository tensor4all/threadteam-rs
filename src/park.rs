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
#![cfg_attr(not(feature = "parallel"), allow(dead_code))]

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

    /// Whether every share has finished, read with the ordering of the
    /// `Parker` protocol (for predicates evaluated under its lock).
    pub fn is_done_now(&self) -> bool {
        self.remaining.0.load(Ordering::SeqCst) == 0
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

// ---------------------------------------------------------------------------
// The native scheduler's shared state (`docs/design/own-threads.md`): a slot's
// state word, the pool's queue of stack batches, and its broadcast list. The
// exact code is what `loom` checks.

/// The slot is reserved by a team.
pub(crate) const RESERVED: usize = 1;
/// The reserved member is lent to the pool's queue (`Exec::external`).
pub(crate) const LENT: usize = 2;
/// The worker runs a task or a broadcast share.
pub(crate) const BUSY: usize = 4;
/// The pool is being dropped: the worker acknowledges and leaves.
pub(crate) const DETACH: usize = 8;

/// The state word of one worker slot: bits [`RESERVED`], [`LENT`], [`BUSY`],
/// [`DETACH`]. A team reserves a slot only from `0`; a worker takes work only
/// from `0` (free) or `RESERVED|LENT` (lent), before it removes anything from a
/// queue; ending work clears `BUSY` only, releasing clears `RESERVED|LENT`
/// only, so neither erases the other's bits or `DETACH`.
#[doc(hidden)]
pub struct SlotState(AtomicUsize);

impl Default for SlotState {
    fn default() -> Self {
        Self::new()
    }
}

impl SlotState {
    /// A free slot.
    pub fn new() -> Self {
        SlotState(AtomicUsize::new(0))
    }

    /// The current bits.
    pub fn load(&self) -> usize {
        self.0.load(Ordering::Acquire)
    }

    /// Reserves a free, idle slot for a team (`0 -> RESERVED`).
    pub fn try_reserve(&self) -> bool {
        self.0
            .compare_exchange(0, RESERVED, Ordering::AcqRel, Ordering::Acquire)
            .is_ok()
    }

    /// The worker takes work: `0 -> BUSY`, or `RESERVED|LENT -> RESERVED|LENT|BUSY`.
    pub fn try_take(&self) -> bool {
        let cur = self.0.load(Ordering::Acquire);
        if cur != 0 && cur != RESERVED | LENT {
            return false;
        }
        self.0
            .compare_exchange(cur, cur | BUSY, Ordering::AcqRel, Ordering::Acquire)
            .is_ok()
    }

    /// The worker finished its work (clears `BUSY` only).
    pub fn end_busy(&self) {
        self.0.fetch_and(!BUSY, Ordering::SeqCst);
    }

    /// The team releases the slot (clears `RESERVED|LENT` only); a worker still
    /// running a lent task stays `BUSY` and is claimable only once it finishes.
    pub fn release(&self) {
        self.0.fetch_and(!(RESERVED | LENT), Ordering::SeqCst);
    }

    /// Lends a reserved member to the pool's queue, or takes it back.
    pub fn set_lent(&self, on: bool) {
        if on {
            self.0.fetch_or(LENT, Ordering::SeqCst);
        } else {
            self.0.fetch_and(!LENT, Ordering::SeqCst);
        }
    }

    /// The pool is being dropped.
    pub fn detach(&self) {
        self.0.fetch_or(DETACH, Ordering::SeqCst);
    }
}

/// A batch of `tasks` tasks published on a pool's [`Queue`]: a stack object of
/// its owner, linked into the queue until its last index is claimed, alive
/// until its last task completed (the owner waits for that).
#[doc(hidden)]
pub struct Batch {
    run: unsafe fn(*const (), usize),
    data: *const (),
    tasks: usize,
    /// Next unclaimed index; read and written under the queue lock only.
    next: AtomicUsize,
    /// Next batch in the queue; under the queue lock only.
    link: core::cell::Cell<*const Batch>,
    /// Tasks not yet completed.
    pub done: Completion,
    /// Whom the last completion wakes (the owner's own parker, or the pool's).
    waker: *const std::sync::Arc<Parker>,
    /// First panic payload, boxed twice so that it fits an `AtomicPtr`.
    panic: core::sync::atomic::AtomicPtr<Box<dyn std::any::Any + Send>>,
}

// SAFETY: `data` points to a `Sync` closure of the owner that outlives the
// batch's tasks; `waker` points to an `Arc` the owner keeps alive while it
// waits; `link` is touched under the queue lock only.
unsafe impl Sync for Batch {}
unsafe impl Send for Batch {}

impl Batch {
    /// A batch of `tasks` calls `run(data, index)`, waking `waker` when the
    /// last one completes.
    ///
    /// # Safety
    ///
    /// `run` and `data` belong together and stay valid until the batch is
    /// done; `waker` outlives the owner's wait for the batch.
    pub unsafe fn new(
        run: unsafe fn(*const (), usize),
        data: *const (),
        tasks: usize,
        waker: *const std::sync::Arc<Parker>,
    ) -> Self {
        Batch {
            run,
            data,
            tasks,
            next: AtomicUsize::new(0),
            link: core::cell::Cell::new(core::ptr::null()),
            done: Completion::new(tasks),
            waker,
            panic: core::sync::atomic::AtomicPtr::new(core::ptr::null_mut()),
        }
    }

    /// Runs claimed task `index` and completes it; a panic is caught and kept.
    ///
    /// # Safety
    ///
    /// `this` is a batch whose task `index` the caller claimed. It is not
    /// dereferenced after the final decrement.
    pub unsafe fn run_claimed(this: *const Batch, index: usize) {
        let b = &*this;
        let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| (b.run)(b.data, index)));
        if let Err(p) = r {
            let fresh = Box::into_raw(Box::new(p));
            if b.panic
                .compare_exchange(
                    core::ptr::null_mut(),
                    fresh,
                    core::sync::atomic::Ordering::AcqRel,
                    core::sync::atomic::Ordering::Acquire,
                )
                .is_err()
            {
                drop(Box::from_raw(fresh));
            }
        }
        // Keep the waker alive across the final decrement: after it the owner
        // may return and free the batch.
        let waker = std::sync::Arc::clone(&*b.waker);
        Completion::finish(core::ptr::addr_of!((*this).done), Some(&waker));
    }

    /// The first panic of a task, once the batch is done.
    pub fn take_panic(&self) -> Option<Box<dyn std::any::Any + Send>> {
        let p = self
            .panic
            .swap(core::ptr::null_mut(), core::sync::atomic::Ordering::AcqRel);
        // SAFETY: a non-null value was stored by `run_claimed` from
        // `Box::into_raw` and `swap` hands it to exactly one taker.
        (!p.is_null()).then(|| *unsafe { Box::from_raw(p) })
    }
}

impl Drop for Batch {
    fn drop(&mut self) {
        drop(self.take_panic());
    }
}

/// A pool's queue of [`Batch`]es: an intrusive list under a lock, newest
/// first. Claims happen under the lock, and the claim of a batch's last index
/// unlinks it, so nobody reaches a batch after it was unlinked; its owner frees
/// it only once all of its tasks completed.
#[doc(hidden)]
pub struct Queue {
    head: Mutex<QueueHead>,
    /// Bumped (SeqCst) on every publication: what sleeping workers watch.
    gen: AtomicUsize,
    /// Linked batches, for a lock-free emptiness hint.
    linked: AtomicUsize,
}

struct QueueHead(*const Batch);
// SAFETY: the pointer is only dereferenced under the lock (see `Queue`).
unsafe impl Send for QueueHead {}

impl Default for Queue {
    fn default() -> Self {
        Self::new()
    }
}

impl Queue {
    /// An empty queue. The lock is touched once here: some platforms allocate
    /// it lazily, and no later operation should allocate.
    pub fn new() -> Self {
        let q = Queue {
            head: Mutex::new(QueueHead(core::ptr::null())),
            gen: AtomicUsize::new(0),
            linked: AtomicUsize::new(0),
        };
        drop(q.head.lock().unwrap_or_else(|e| e.into_inner()));
        q
    }

    /// The publication counter.
    pub fn gen(&self) -> usize {
        self.gen.load(Ordering::SeqCst)
    }

    /// Whether any batch has unclaimed tasks (a hint).
    pub fn has_work(&self) -> bool {
        self.linked.load(Ordering::Acquire) != 0
    }

    /// Links `b` (with at least one task); the caller then wakes the workers.
    pub fn publish(&self, b: &Batch) {
        debug_assert!(b.tasks > 0);
        let mut head = self.head.lock().unwrap_or_else(|e| e.into_inner());
        b.link.set(head.0);
        head.0 = b as *const Batch;
        self.linked.fetch_add(1, Ordering::AcqRel);
        drop(head);
        self.gen.fetch_add(1, Ordering::SeqCst);
    }

    /// Claims the next index of the newest batch; the caller must run it with
    /// [`Batch::run_claimed`].
    pub fn claim(&self) -> Option<(*const Batch, usize)> {
        let mut head = self.head.lock().unwrap_or_else(|e| e.into_inner());
        let b = head.0;
        if b.is_null() {
            return None;
        }
        // SAFETY: a linked batch is alive (see `Queue`).
        let batch = unsafe { &*b };
        let i = batch.next.load(Ordering::Relaxed);
        batch.next.store(i + 1, Ordering::Relaxed);
        if i + 1 == batch.tasks {
            head.0 = batch.link.get();
            self.linked.fetch_sub(1, Ordering::AcqRel);
        }
        Some((b, i))
    }

    /// Claims the next index of the caller's own batch `b`, wherever it is in
    /// the list; `None` once all of its indices are claimed.
    pub fn claim_own(&self, b: &Batch) -> Option<usize> {
        let mut head = self.head.lock().unwrap_or_else(|e| e.into_inner());
        let i = b.next.load(Ordering::Relaxed);
        if i >= b.tasks {
            return None;
        }
        b.next.store(i + 1, Ordering::Relaxed);
        if i + 1 == b.tasks {
            // Unlink `b`: walk the (short) list to its predecessor.
            let me = b as *const Batch;
            if head.0 == me {
                head.0 = b.link.get();
            } else {
                let mut cur = head.0;
                // SAFETY: linked batches are alive; `b` is linked (it still
                // had unclaimed indices).
                unsafe {
                    while (*cur).link.get() != me {
                        cur = (*cur).link.get();
                    }
                    (*cur).link.set(b.link.get());
                }
            }
            self.linked.fetch_sub(1, Ordering::AcqRel);
        }
        Some(i)
    }
}

/// A broadcast published on a pool's [`BroadcastList`]: share `i` for every
/// worker `i < budget`. A stack object of its caller, linked until its caller
/// unlinks it after all shares completed.
#[doc(hidden)]
pub struct Broadcast {
    run: unsafe fn(*const (), usize),
    data: *const (),
    budget: usize,
    /// Publication number; written under the list lock before linking.
    seq: core::cell::Cell<u64>,
    link: core::cell::Cell<*const Broadcast>,
    /// Shares not yet completed.
    pub done: Completion,
    waker: *const std::sync::Arc<Parker>,
    panic: core::sync::atomic::AtomicPtr<Box<dyn std::any::Any + Send>>,
}

// SAFETY: as for `Batch`; `seq` and `link` are touched under the list lock.
unsafe impl Sync for Broadcast {}
unsafe impl Send for Broadcast {}

impl Broadcast {
    /// A broadcast of `run(data, i)` to workers `0..budget`.
    ///
    /// # Safety
    ///
    /// As [`Batch::new`].
    pub unsafe fn new(
        run: unsafe fn(*const (), usize),
        data: *const (),
        budget: usize,
        waker: *const std::sync::Arc<Parker>,
    ) -> Self {
        Broadcast {
            run,
            data,
            budget,
            seq: core::cell::Cell::new(0),
            link: core::cell::Cell::new(core::ptr::null()),
            done: Completion::new(budget),
            waker,
            panic: core::sync::atomic::AtomicPtr::new(core::ptr::null_mut()),
        }
    }

    /// Runs worker `i`'s share and completes it; a panic is caught and kept.
    ///
    /// # Safety
    ///
    /// `this` is a linked broadcast that targets `i` and whose share `i` the
    /// caller took; it is not dereferenced after the final decrement.
    pub unsafe fn run_share(this: *const Broadcast, i: usize) {
        let b = &*this;
        let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| (b.run)(b.data, i)));
        if let Err(p) = r {
            let fresh = Box::into_raw(Box::new(p));
            if b.panic
                .compare_exchange(
                    core::ptr::null_mut(),
                    fresh,
                    core::sync::atomic::Ordering::AcqRel,
                    core::sync::atomic::Ordering::Acquire,
                )
                .is_err()
            {
                drop(Box::from_raw(fresh));
            }
        }
        let waker = std::sync::Arc::clone(&*b.waker);
        Completion::finish(core::ptr::addr_of!((*this).done), Some(&waker));
    }

    /// The first panic of a share, once the broadcast is done.
    pub fn take_panic(&self) -> Option<Box<dyn std::any::Any + Send>> {
        let p = self
            .panic
            .swap(core::ptr::null_mut(), core::sync::atomic::Ordering::AcqRel);
        // SAFETY: as in `Batch::take_panic`.
        (!p.is_null()).then(|| *unsafe { Box::from_raw(p) })
    }
}

impl Drop for Broadcast {
    fn drop(&mut self) {
        drop(self.take_panic());
    }
}

/// A pool's broadcasts in publication order. Every worker runs the broadcasts
/// that target it in that order (its own sequence cursor), so concurrent
/// broadcasts reach every worker in the same order. Publication and unlinking
/// take the lock; completion does not, so nobody waits while holding it.
#[doc(hidden)]
pub struct BroadcastList {
    inner: Mutex<BroadcastHead>,
    /// Bumped (SeqCst) on every publication.
    gen: AtomicUsize,
}

struct BroadcastHead {
    head: *const Broadcast,
    tail: *const Broadcast,
    next_seq: u64,
}
// SAFETY: the pointers are only dereferenced under the lock.
unsafe impl Send for BroadcastHead {}

impl Default for BroadcastList {
    fn default() -> Self {
        Self::new()
    }
}

impl BroadcastList {
    /// An empty list (the lock is touched once, as in [`Queue::new`]).
    pub fn new() -> Self {
        let l = BroadcastList {
            inner: Mutex::new(BroadcastHead {
                head: core::ptr::null(),
                tail: core::ptr::null(),
                next_seq: 0,
            }),
            gen: AtomicUsize::new(0),
        };
        drop(l.inner.lock().unwrap_or_else(|e| e.into_inner()));
        l
    }

    /// The publication counter.
    pub fn gen(&self) -> usize {
        self.gen.load(Ordering::SeqCst)
    }

    /// Appends `b`; the caller then wakes the workers.
    pub fn publish(&self, b: &Broadcast) {
        let mut l = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        b.seq.set(l.next_seq);
        l.next_seq += 1;
        b.link.set(core::ptr::null());
        let me = b as *const Broadcast;
        if l.tail.is_null() {
            l.head = me;
        } else {
            // SAFETY: linked broadcasts are alive.
            unsafe { (*l.tail).link.set(me) };
        }
        l.tail = me;
        drop(l);
        self.gen.fetch_add(1, Ordering::SeqCst);
    }

    /// The next broadcast for worker `i` at or after its cursor `seq`, which
    /// moves past every broadcast it inspects. The caller runs the returned
    /// share with [`Broadcast::run_share`].
    pub fn next_for(&self, i: usize, seq: &mut u64) -> Option<*const Broadcast> {
        let l = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        let mut cur = l.head;
        while !cur.is_null() {
            // SAFETY: linked broadcasts are alive.
            let b = unsafe { &*cur };
            if b.seq.get() >= *seq {
                *seq = b.seq.get() + 1;
                if i < b.budget {
                    return Some(cur);
                }
            }
            cur = b.link.get();
        }
        None
    }

    /// Whether a broadcast at or after `seq` exists (a hint, lock-free reads
    /// of the counter only).
    pub fn has_after(&self, seq: u64) -> bool {
        (self.gen.load(Ordering::Acquire) as u64) > seq
    }

    /// Unlinks `b` once all of its shares completed.
    pub fn unlink(&self, b: &Broadcast) {
        let mut l = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        let me = b as *const Broadcast;
        let mut prev: *const Broadcast = core::ptr::null();
        let mut cur = l.head;
        while cur != me {
            prev = cur;
            // SAFETY: `b` is linked, so the walk reaches it; links are alive.
            cur = unsafe { (*cur).link.get() };
        }
        let next = b.link.get();
        if prev.is_null() {
            l.head = next;
        } else {
            // SAFETY: as above.
            unsafe { (*prev).link.set(next) };
        }
        if l.tail == me {
            l.tail = prev;
        }
    }
}
