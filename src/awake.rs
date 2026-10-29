//! Reserved awake members: per-member dispatch slots, sections, acquisition.
//!
//! A *member* is a worker of the pool that runs a long-lived loop as one
//! Rayon job. While the loop runs, the worker serves no other Rayon job.
//! Each worker index owns one cache-line padded [`Slot`]:
//!
//! ```text
//! ABSENT --claim job--> RESERVED <--claim (CAS)-- IDLE
//!    ^                     |  release (store)      ^ |
//!    +---- idle timeout ---+-----------------------+ +-- YIELD (inside a Rayon job)
//! ```
//!
//! * `ABSENT`: no loop on this worker.
//! * `RESERVED`: owned by one leader; the member waits for `gen` (spin, yield,
//!   then park) and runs the section the leader published. It neither runs
//!   Rayon jobs nor leaves.
//! * `IDLE`: loop alive but unreserved; polls for the policy's idle time or
//!   the pool's spin budget, whichever ends first, then leaves (`ABSENT`).
//!   Claimable by a compare-and-swap, with no waiting.
//! * `YIELD`: an idle member inside `rayon::yield_now`; not claimable.
//!
//! A leader publishes a [`Section`] (a stack object) by writing its address
//! and an argument into the member's slot and bumping `gen`; the member runs
//! it and decrements the section's completion counter. Nothing is allocated
//! per dispatch. A member that parked sleeps on the [`Parker`] of its own slot,
//! and a leader wakes only the slots it changed: a section narrower than the
//! team never disturbs the members it does not use (a pool-wide parker woke
//! every sleeper on every publish, and the unused ones went back to sleep: a
//! futex round per section that grew with the number of unused members).
//!
//! No thread-local state is used: a member's identity is its slot index.

use crate::error::Refused;
use crate::park::Parker;
use crate::policy::WaitPolicy;
use core::sync::atomic::{AtomicBool, AtomicPtr, AtomicU32, AtomicU64, AtomicUsize, Ordering};
use std::any::Any;
use std::time::Duration;

/// Records a trace event for a member (`Some(slot)`) or a leader (`None`);
/// nothing without the feature `trace`.
macro_rules! trace {
    ($aw:expr, $id:expr, $kind:ident) => {
        #[cfg(feature = "trace")]
        $aw.trace.record($id, crate::trace::Kind::$kind);
    };
}

const ABSENT: usize = 0;
const IDLE: usize = 1;
#[cfg_attr(not(feature = "parallel"), allow(dead_code))]
const YIELD: usize = 2;
const RESERVED: usize = 3;

/// Pure spin for a while, then yield the CPU between polls.
///
/// Used where progress needs cooperation with other threads on the same CPUs:
/// `acquire`, the cold claim, and the loops that are not reserved dispatch
/// polls (`IDLE`, and the states a loop is leaving). Reserved members, and a
/// leader or participant waiting on a section, wait in staged episodes instead
/// ([`crate::park::Waiter`]).
#[inline]
#[cfg_attr(not(feature = "parallel"), allow(dead_code))]
pub(crate) fn backoff(spins: &mut u32) {
    if *spins < (1 << 14) {
        *spins += 1;
        std::hint::spin_loop();
    } else {
        std::thread::yield_now();
    }
}

/// One member's dispatch slot.
#[repr(align(128))]
struct Slot {
    state: AtomicUsize,
    /// Bumped by the current leader per published section.
    gen: AtomicUsize,
    ptr: AtomicPtr<Section>,
    arg: AtomicUsize,
    /// Idle policy written by the releasing leader before `IDLE` is stored.
    idle_spin_ns: AtomicU64,
    idle_yield: AtomicBool,
    /// Spin budget, in polls, of this reserved member's waits for work (see
    /// `docs/design/waiting.md`); written by the leader at root-scope entry.
    spin_budget: AtomicU32,
    /// Set by the leader while the member's thread is lent to Rayon
    /// (`Exec::external` on a team token).
    lent: AtomicBool,
    /// The mask this member saved before pinning itself; only the member thread
    /// of this slot touches it (see [`crate::pin::SavedMask`]).
    #[cfg_attr(not(feature = "parallel"), allow(dead_code))]
    saved: crate::pin::SavedMask,
    /// Where this member sleeps once its spin and yields are spent; woken by
    /// the leader after it changed `gen`, `state` or `lent` of this slot.
    park: Parker,
}

impl Slot {
    fn new() -> Self {
        Slot {
            state: AtomicUsize::new(ABSENT),
            gen: AtomicUsize::new(0),
            ptr: AtomicPtr::new(core::ptr::null_mut()),
            arg: AtomicUsize::new(0),
            idle_spin_ns: AtomicU64::new(0),
            idle_yield: AtomicBool::new(false),
            spin_budget: AtomicU32::new(crate::park::THROTTLED_POLLS),
            lent: AtomicBool::new(false),
            saved: crate::pin::SavedMask::new(),
            park: Parker::new(),
        }
    }
}

/// The waiting state of one pool (`docs/design/waiting.md`).
pub(crate) struct Waits {
    /// Where leaders sleep in `join` and participants in `barrier`. One per
    /// pool: a member may wake it after its last decrement, when the section
    /// may already be gone, and the pool outlives every member loop.
    pub(crate) sync: Parker,
    /// Recent scheduling pressure seen by this pool's waiters.
    pub(crate) pressure: crate::park::Pressure,
    /// Spin budget of a wait, in polls (`PoolBuilder::spin_polls`).
    pub(crate) spin_polls: u32,
}

/// Shared state of the awake members of one pool.
pub(crate) struct Awake {
    slots: Box<[Slot]>,
    /// What the joins and barriers of this pool's sections sleep on, and the
    /// pool's spin budget and scheduling pressure (`docs/design/waiting.md`).
    pub(crate) waits: Waits,
    pub(crate) shutdown: AtomicBool,
    /// Bumped by `Pool::release_idle`; members that started earlier leave when idle.
    pub(crate) release_epoch: AtomicUsize,
    pub(crate) defaults: WaitPolicy,
    pub(crate) costs: crate::cost::DispatchCosts,
    /// Opt-in participant pinning (`PoolBuilder::pin_participants`).
    pub(crate) pin: bool,
    #[cfg_attr(not(feature = "parallel"), allow(dead_code))]
    pub(crate) acquire_timeout: Duration,
    #[cfg(feature = "trace")]
    pub(crate) trace: crate::trace::Trace,
}

impl Awake {
    pub(crate) fn new(
        width: usize,
        defaults: WaitPolicy,
        acquire_timeout: Duration,
        costs: crate::cost::DispatchCosts,
        pin: bool,
        spin_polls: u32,
    ) -> Self {
        Awake {
            slots: (0..width).map(|_| Slot::new()).collect(),
            waits: Waits {
                sync: Parker::new(),
                pressure: crate::park::Pressure::new(),
                spin_polls,
            },
            shutdown: AtomicBool::new(false),
            release_epoch: AtomicUsize::new(0),
            defaults,
            costs,
            pin,
            acquire_timeout,
            #[cfg(feature = "trace")]
            trace: crate::trace::Trace::new(width),
        }
    }

    /// Workers currently running a member loop, reserved or idle.
    pub(crate) fn resident(&self) -> usize {
        self.slots
            .iter()
            .filter(|s| s.state.load(Ordering::Acquire) != ABSENT)
            .count()
    }

    /// Hands members back: they become idle under `policy`.
    pub(crate) fn release(&self, ids: &[usize], policy: WaitPolicy) {
        let ns = u64::try_from(policy.idle_spin().as_nanos()).unwrap_or(u64::MAX);
        for &id in ids {
            let s = &self.slots[id];
            s.idle_spin_ns.store(ns, Ordering::Relaxed);
            s.idle_yield
                .store(policy.idle_runs_rayon(), Ordering::Relaxed);
            s.state.store(IDLE, Ordering::SeqCst);
            s.park.wake();
        }
    }

    /// Spin budget, in polls, of a wait in a root scope: the pool's budget, or
    /// the throttled one when the scope is oversubscribed.
    pub(crate) fn spin_budget(&self, over: bool) -> u32 {
        if over {
            crate::park::THROTTLED_POLLS.min(self.waits.spin_polls)
        } else {
            self.waits.spin_polls
        }
    }

    /// The spin budget a member slot currently has (tests).
    #[cfg(all(test, feature = "parallel", target_os = "linux"))]
    pub(crate) fn slot_spin_budget(&self, id: usize) -> u32 {
        self.slots[id].spin_budget.load(Ordering::Relaxed)
    }

    /// Sets the spin budget of the given reserved members' waits for work.
    ///
    /// A root team scope writes it once at entry ([`Self::spin_budget`] of its
    /// pressure). Nothing else changes it, so a member keeps the value of the
    /// scope it was reserved for.
    pub(crate) fn set_spin_budget(&self, ids: &[usize], polls: u32) {
        for &id in ids {
            self.slots[id].spin_budget.store(polls, Ordering::Relaxed);
        }
    }

    /// Restores the saved mask of member slot `id`, on the calling thread.
    ///
    /// Only the worker whose index is `id` may call this (`sched::unpin_worker`
    /// does), and that worker owns the slot's saved mask (see
    /// [`crate::pin::SavedMask`]). It is used when a pinned participant's own
    /// thread is about to run work that is not its team's (`Exec::external`,
    /// for example in the member that leads a `fork` child), so the foreign
    /// work does not keep the team's placement. The member re-pins only at the
    /// next pin section.
    #[cfg_attr(not(feature = "parallel"), allow(dead_code))]
    pub(crate) fn unpin_member(&self, id: usize) {
        if let Some(slot) = self.slots.get(id) {
            slot.saved.restore();
        }
    }

    /// Wakes every parked member (see [`Parker`]); a test hook, the entries wake
    /// only the slots they changed.
    #[cfg_attr(not(feature = "test-hooks"), allow(dead_code))]
    pub(crate) fn wake_parked(&self) {
        for s in &*self.slots {
            s.park.wake();
        }
    }

    /// Publishes `section` to its `n` targets (position `p` gets argument `p + 1`).
    ///
    /// The caller leads every target (its slot is `RESERVED` for the caller)
    /// and keeps `section` alive until its completion counter reaches zero.
    /// Each target's generation is bumped and its own parker woken (one atomic
    /// read-modify-write on its slot when it does not sleep); members the
    /// section does not address are not touched.
    pub(crate) fn publish(&self, section: &Section) {
        let ptr = (section as *const Section).cast_mut();
        for pos in 0..section.dispatched {
            let s = &self.slots[section.slot(pos)];
            s.arg.store(pos + 1, Ordering::Relaxed);
            s.ptr.store(ptr, Ordering::Relaxed);
            // fetch_add: always the latest value, whichever thread led before.
            s.gen.fetch_add(1, Ordering::SeqCst);
        }
        // Wake after every generation is bumped, so that a spinning member
        // starts without waiting for the wake checks of the slots after it.
        for pos in 0..section.dispatched {
            self.slots[section.slot(pos)].park.wake();
        }
        trace!(self, None, Publish);
    }

    /// Lends (or takes back) the members' threads to Rayon: while lent, a
    /// reserved member runs the pool's jobs instead of idling.
    #[cfg_attr(not(feature = "parallel"), allow(dead_code))]
    pub(crate) fn lend(&self, ids: &[usize], on: bool) {
        for &id in ids {
            let s = &self.slots[id];
            s.lent.store(on, Ordering::SeqCst);
            if on {
                s.park.wake();
            }
        }
    }

    /// Reserves `need` members, or refuses without waiting for a Rayon job.
    ///
    /// Idle resident members are claimed with a compare-and-swap. Missing ones
    /// are requested with claim jobs; the leader waits at most the pool's
    /// acquisition timeout for *idle workers to wake up*, never for a worker
    /// that is busy: when the members are not all there by then the claim is
    /// closed, every tentative member is released and `Unavailable` returned.
    pub(crate) fn acquire(
        self: &std::sync::Arc<Self>,
        pool: &crate::pool::Pool,
        need: usize,
        policy: WaitPolicy,
        spin_budget: u32,
    ) -> Result<Reservation, Refused> {
        let mut ids = Vec::with_capacity(need);
        // Idle members are claimed with a compare-and-swap. A member that is
        // inside `yield_now` is not claimable at that instant, but it is only
        // polling unless it found a job; retry for at most the timeout while
        // such members exist, instead of treating them as busy at once.
        let started = std::time::Instant::now();
        loop {
            let mut yielding = false;
            for (i, s) in self.slots.iter().enumerate() {
                if ids.len() == need {
                    break;
                }
                match s
                    .state
                    .compare_exchange(IDLE, RESERVED, Ordering::AcqRel, Ordering::Relaxed)
                {
                    Ok(_) => ids.push(i),
                    Err(YIELD) => yielding = true,
                    Err(_) => {}
                }
            }
            if ids.len() == need || !yielding || started.elapsed() >= self.acquire_timeout {
                break;
            }
            std::hint::spin_loop();
        }
        let missing = need - ids.len();
        if missing > 0 {
            self.claim_cold(pool, missing, &mut ids);
            if ids.len() < need {
                self.release(&ids, policy);
                return Err(Refused::Unavailable);
            }
        }
        self.set_spin_budget(&ids, spin_budget);
        Ok(Reservation {
            awake: std::sync::Arc::clone(self),
            ids,
            policy,
        })
    }

    #[cfg(feature = "parallel")]
    fn claim_cold(
        self: &std::sync::Arc<Self>,
        pool: &crate::pool::Pool,
        missing: usize,
        ids: &mut Vec<usize>,
    ) {
        use std::sync::Arc;
        use std::time::Instant;
        let acq = Arc::new(Acq::new(missing));
        for _ in 0..missing {
            let (aw, a) = (Arc::clone(self), Arc::clone(&acq));
            pool.inner.spawn(move || claim_job(&aw, &a));
        }
        let deadline = Instant::now() + self.acquire_timeout;
        let mut spins = 0;
        while acq.joined() < missing && Instant::now() < deadline {
            backoff(&mut spins);
        }
        let n = acq.close();
        for pos in 0..n {
            let mut spins = 0;
            loop {
                let id = acq.ids[pos].load(Ordering::Acquire);
                if id != usize::MAX {
                    ids.push(id);
                    break;
                }
                backoff(&mut spins);
            }
        }
    }

    #[cfg(not(feature = "parallel"))]
    fn claim_cold(
        self: &std::sync::Arc<Self>,
        _pool: &crate::pool::Pool,
        _missing: usize,
        _ids: &mut Vec<usize>,
    ) {
    }
}

/// Members reserved for one leader; released on drop, also on unwind.
pub(crate) struct Reservation {
    awake: std::sync::Arc<Awake>,
    pub(crate) ids: Vec<usize>,
    policy: WaitPolicy,
}

impl Drop for Reservation {
    fn drop(&mut self) {
        // Every section was joined before its entry returned, so nothing is
        // outstanding here.
        self.awake.release(&self.ids, self.policy);
    }
}

/// One acquisition: how many claim jobs found a worker, and which.
#[cfg(feature = "parallel")]
struct Acq {
    joined: AtomicUsize,
    ids: Vec<AtomicUsize>,
}

#[cfg(feature = "parallel")]
impl Acq {
    const CLOSED: usize = 1 << (usize::BITS - 1);

    fn new(capacity: usize) -> Self {
        Acq {
            joined: AtomicUsize::new(0),
            ids: (0..capacity)
                .map(|_| AtomicUsize::new(usize::MAX))
                .collect(),
        }
    }

    fn joined(&self) -> usize {
        self.joined.load(Ordering::Acquire) & !Self::CLOSED
    }

    /// A claim job asks for a position; `None` when closed or full.
    fn join(&self) -> Option<usize> {
        let mut cur = self.joined.load(Ordering::Acquire);
        loop {
            if cur & Self::CLOSED != 0 || cur >= self.ids.len() {
                return None;
            }
            match self.joined.compare_exchange_weak(
                cur,
                cur + 1,
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => return Some(cur),
                Err(now) => cur = now,
            }
        }
    }

    /// Closes the claim; later jobs do nothing. Returns the joined count.
    fn close(&self) -> usize {
        self.joined.fetch_or(Self::CLOSED, Ordering::AcqRel) & !Self::CLOSED
    }
}

/// Runs on a pool worker: becomes a member for the acquisition, or does nothing.
#[cfg(feature = "parallel")]
fn claim_job(aw: &std::sync::Arc<Awake>, acq: &std::sync::Arc<Acq>) {
    let Some(w) = rayon::current_thread_index() else {
        return;
    };
    let Some(slot) = aw.slots.get(w) else {
        return;
    };
    if slot
        .state
        .compare_exchange(ABSENT, RESERVED, Ordering::AcqRel, Ordering::Relaxed)
        .is_err()
    {
        // Stolen by a resident member that is serving Rayon (idle_runs_rayon):
        // its thread cannot host a second loop. Hand the job back to the pool
        // unless the acquisition is already over.
        if acq.joined.load(Ordering::Acquire) & Acq::CLOSED == 0 {
            let (aw, acq) = (std::sync::Arc::clone(aw), std::sync::Arc::clone(acq));
            rayon::spawn(move || claim_job(&aw, &acq));
        }
        return;
    }
    let seen = slot.gen.load(Ordering::Acquire);
    let epoch = aw.release_epoch.load(Ordering::Relaxed);
    match acq.join() {
        Some(pos) => {
            acq.ids[pos].store(w, Ordering::Release);
            member_loop(aw, w, seen, epoch);
        }
        None => slot.state.store(ABSENT, Ordering::Release),
    }
}

/// The member loop. Returns when the idle policy, `release_idle` or shutdown
/// returns the thread to Rayon.
///
/// A reserved member waits for work in episodes (`docs/design/waiting.md`): an
/// episode starts when it first finds nothing to do after its last share (or
/// after it started, or was taken back from Rayon), spins for the slot's spin
/// budget in polls, yields a few times, then parks and stays reserved. The
/// budget is a count, so a member that is descheduled does not spend it while
/// it waits for its CPU; a long gap between its clock reads ends the spin at
/// once (see [`crate::park::Pressure`]).
///
/// An unreserved (`IDLE`) member returns to Rayon after the policy's
/// `idle_spin` of wall time or after the pool's spin budget of polls, whichever
/// comes first, so a team released between calls does not keep every CPU busy
/// for the whole `idle_spin`.
#[cfg(feature = "parallel")]
fn member_loop(aw: &Awake, id: usize, mut seen: usize, epoch: usize) {
    use crate::park::{Step, Waiter};
    use std::time::Instant;
    let slot = &aw.slots[id];
    let mut idle_since: Option<Instant> = None;
    let mut idle_polls = 0u64;
    let mut waiter: Option<Waiter<'_>> = None;
    let mut was_lent = false;
    let mut reserved = true;
    let mut spins = 0u32;
    trace!(aw, Some(id), Enter);
    loop {
        let state = slot.state.load(Ordering::Acquire);
        // Pinning ends with the reservation, and it ends on this thread: a
        // member restores its own mask before it leaves `RESERVED` and before
        // it serves Rayon work (a lend restores it below).
        if state == RESERVED {
            reserved = true;
        } else if reserved {
            reserved = false;
            slot.saved.restore();
        }
        match state {
            RESERVED if slot.lent.load(Ordering::Acquire) => {
                if !was_lent {
                    was_lent = true;
                    // The thread serves the pool's jobs now: undo our pin first.
                    slot.saved.restore();
                    trace!(aw, Some(id), LentBegin);
                }
                idle_since = None;
                waiter = None;
                // Serve the pool's jobs; resume team service when taken back.
                if !matches!(rayon::yield_now(), Some(rayon::Yield::Executed)) {
                    for _ in 0..32 {
                        std::hint::spin_loop();
                    }
                }
            }
            RESERVED => {
                if was_lent {
                    was_lent = false;
                    trace!(aw, Some(id), LentEnd);
                }
                idle_since = None;
                let g = slot.gen.load(Ordering::Acquire);
                if g == seen {
                    let w = waiter.get_or_insert_with(|| {
                        Waiter::new(
                            slot.spin_budget.load(Ordering::Relaxed),
                            Some(&aw.waits.pressure),
                        )
                    });
                    if w.step() == Step::Sleep {
                        trace!(aw, Some(id), ParkEnter);
                        park_member(slot, seen);
                        trace!(aw, Some(id), ParkExit);
                        waiter = None;
                    }
                } else {
                    seen = g;
                    waiter = None;
                    let ptr = slot.ptr.load(Ordering::Relaxed);
                    let arg = slot.arg.load(Ordering::Relaxed);
                    // SAFETY: the leader keeps the section alive until its
                    // completion counter reaches zero, which happens only
                    // after the call below decremented it.
                    if unsafe { (*ptr).pin } {
                        // A pin section: this member saves its previous mask
                        // here, before it pins itself in its share.
                        slot.saved.save_if_unpinned();
                    }
                    trace!(aw, Some(id), SectionBegin);
                    // SAFETY: as above; `aw` outlives this loop, so the wake
                    // after the last decrement does not touch the section.
                    unsafe { Section::execute_member(ptr, arg) };
                    trace!(aw, Some(id), SectionEnd);
                }
            }
            IDLE => {
                waiter = None;
                if idle_since.is_none() {
                    idle_polls = 0;
                }
                let since = *idle_since.get_or_insert_with(Instant::now);
                idle_polls += 1;
                let spin_ns = slot.idle_spin_ns.load(Ordering::Relaxed);
                let leave = aw.shutdown.load(Ordering::Relaxed)
                    || aw.release_epoch.load(Ordering::Relaxed) != epoch
                    || idle_polls > u64::from(aw.waits.spin_polls)
                    || since.elapsed().as_nanos() >= u128::from(spin_ns);
                if leave {
                    if slot
                        .state
                        .compare_exchange(IDLE, ABSENT, Ordering::AcqRel, Ordering::Relaxed)
                        .is_ok()
                    {
                        trace!(aw, Some(id), Leave);
                        return;
                    }
                    continue; // claimed meanwhile
                }
                if slot.idle_yield.load(Ordering::Relaxed)
                    && slot
                        .state
                        .compare_exchange(IDLE, YIELD, Ordering::AcqRel, Ordering::Relaxed)
                        .is_ok()
                {
                    let ran = rayon::yield_now();
                    slot.state.store(IDLE, Ordering::Release);
                    if !matches!(ran, Some(rayon::Yield::Executed)) {
                        // Nothing to run: stay claimable for a moment.
                        for _ in 0..64 {
                            std::hint::spin_loop();
                        }
                    }
                } else {
                    backoff(&mut spins);
                }
            }
            _ => backoff(&mut spins),
        }
    }
}

/// Sleeps (still reserved) until a section is published, the reservation ends
/// or the thread is lent. The handshake lives in [`Parker`].
#[cfg(feature = "parallel")]
fn park_member(slot: &Slot, seen: usize) {
    slot.park.park_while(|| {
        slot.gen.load(Ordering::SeqCst) == seen
            && slot.state.load(Ordering::SeqCst) == RESERVED
            && !slot.lent.load(Ordering::SeqCst)
    });
}

/// A unit of dispatched work, living on the leader's stack.
pub(crate) struct Section {
    call: unsafe fn(*const (), &Section, usize),
    data: *const (),
    size: usize,
    /// Number of dispatched targets and the map from target position to slot.
    dispatched: usize,
    slot_of: unsafe fn(*const (), usize) -> usize,
    slot_data: *const (),
    /// The root scope this section belongs to has more participants than the
    /// caller's CPU set has CPUs. Nested sections of the same root inherit it;
    /// it is never recomputed from a child's size.
    pub(crate) over: bool,
    /// This is the *pin* section of a root scope (the extra section a root
    /// scope runs when participant pinning is on): a member that sees it saves
    /// its own mask before its share pins it. The CPU set itself lives in the
    /// section's closure, which captures it.
    #[cfg_attr(not(feature = "parallel"), allow(dead_code))]
    pin: bool,
    /// Dispatched participants that have not finished.
    done: crate::park::Completion,
    barrier: crate::park::Barrier,
    cancelled: crate::sync::AtomicBool,
    /// The pool's waiting state; null when the section has no pool (nothing is
    /// dispatched, and its single participant never waits).
    waits: *const Waits,
    /// First panic payload, boxed twice so that it fits an `AtomicPtr`.
    /// (A `Mutex` would allocate on first use on some platforms.)
    panic: AtomicPtr<Box<dyn Any + Send>>,
}

// SAFETY: `data` points to a `Sync` closure that outlives the section; `waits`
// points to the pool's `Waits` (in `Awake`, which outlives every section and
// member loop) and is only read; every other field is an atomic, a plain flag
// written before the section is published (and read only after a member
// observed the publishing `gen` with acquire ordering), or a function pointer.
unsafe impl Sync for Section {}

/// The geometry of a section: its shares, its participants, and the flags of
/// the root scope it belongs to.
#[derive(Clone, Copy)]
pub(crate) struct Geometry {
    /// Number of dispatched targets (participants minus the leader).
    pub(crate) dispatched: usize,
    /// Number of participants, leader included.
    pub(crate) size: usize,
    /// The root scope is oversubscribed.
    pub(crate) over: bool,
    /// This is the pin section of a root scope (see `Section::pin`).
    pub(crate) pin: bool,
}

impl Section {
    pub(crate) fn new(
        call: unsafe fn(*const (), &Section, usize),
        data: *const (),
        slot_of: unsafe fn(*const (), usize) -> usize,
        slot_data: *const (),
        geometry: Geometry,
        waits: Option<&Waits>,
    ) -> Self {
        let Geometry {
            dispatched,
            size,
            over,
            pin,
        } = geometry;
        Section {
            call,
            data,
            size,
            dispatched,
            slot_of,
            slot_data,
            over,
            pin,
            done: crate::park::Completion::new(dispatched),
            barrier: crate::park::Barrier::new(size),
            cancelled: crate::sync::AtomicBool::new(false),
            waits: waits.map_or(core::ptr::null(), |w| w as *const Waits),
            panic: AtomicPtr::new(core::ptr::null_mut()),
        }
    }

    /// Slot of the target at `pos`.
    fn slot(&self, pos: usize) -> usize {
        // SAFETY: `slot_of` and `slot_data` were paired by the creator.
        unsafe { (self.slot_of)(self.slot_data, pos) }
    }

    /// Calls the section's function on the leader thread.
    ///
    /// # Safety
    ///
    /// `call` and `data` were built together by the creator of the section.
    pub(crate) unsafe fn call_here(&self, arg: usize) {
        (self.call)(self.data, self, arg);
    }

    /// Runs a dispatched share on a member thread and signals completion.
    ///
    /// # Safety
    ///
    /// `ptr` is the section published for this member; it stays alive until
    /// the completion counter reaches zero. It is not used after the final
    /// decrement.
    #[cfg_attr(not(feature = "parallel"), allow(dead_code))]
    unsafe fn execute_member(ptr: *const Section, arg: usize) {
        // The pool's waiting state outlives the section: read it now, use it
        // after the decrement that may let the leader free the section.
        let waits = (*ptr).waits;
        {
            let sec = &*ptr;
            let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| sec.call_here(arg)));
            if let Err(p) = r {
                sec.fail(p);
            }
        }
        crate::park::Completion::finish(
            core::ptr::addr_of!((*ptr).done),
            waits.as_ref().map(|w| &w.sync),
        );
    }

    /// The pool's waiting state, if the section has a pool.
    fn waits(&self) -> Option<&Waits> {
        // SAFETY: `waits` is null or points into the pool's `Awake`, which
        // outlives the section.
        unsafe { self.waits.as_ref() }
    }

    /// A wait episode of this section's join or barrier.
    fn waiter(&self) -> crate::park::Waiter<'_> {
        match self.waits() {
            Some(w) => crate::park::Waiter::new(
                if self.over {
                    crate::park::THROTTLED_POLLS.min(w.spin_polls)
                } else {
                    w.spin_polls
                },
                Some(&w.pressure),
            ),
            None => crate::park::Waiter::new(u32::MAX, None),
        }
    }

    /// Records a participant's panic and cancels barriers.
    pub(crate) fn fail(&self, payload: Box<dyn Any + Send>) {
        // SeqCst, then wake: barrier waiters may sleep on the pool's parker.
        self.cancelled.store(true, Ordering::SeqCst);
        if let Some(w) = self.waits() {
            w.sync.wake();
        }
        let fresh = Box::into_raw(Box::new(payload));
        if self
            .panic
            .compare_exchange(
                core::ptr::null_mut(),
                fresh,
                Ordering::AcqRel,
                Ordering::Acquire,
            )
            .is_err()
        {
            // SAFETY: `fresh` came from `Box::into_raw` just above and was not stored.
            drop(unsafe { Box::from_raw(fresh) });
        }
    }

    pub(crate) fn take_panic(&self) -> Option<Box<dyn Any + Send>> {
        let p = self.panic.swap(core::ptr::null_mut(), Ordering::AcqRel);
        // SAFETY: a non-null value was stored by `fail` from `Box::into_raw`
        // and `swap` hands it to exactly one taker.
        (!p.is_null()).then(|| *unsafe { Box::from_raw(p) })
    }

    pub(crate) fn size(&self) -> usize {
        self.size
    }

    pub(crate) fn is_cancelled(&self) -> bool {
        self.cancelled.load(Ordering::Acquire)
    }

    /// Waits until every dispatched participant has finished: a staged wait
    /// (spin, yield, then sleep on the pool's parker until the last member's
    /// decrement wakes it), throttled when the root scope is oversubscribed.
    pub(crate) fn join(&self) {
        if self.done.is_done() {
            return;
        }
        let mut waiter = self.waiter();
        self.done.wait(self.waits().map(|w| &w.sync), &mut waiter);
    }

    /// Sense-reversing barrier over the section's participants; `false` when
    /// a participant failed, so that peers can leave instead of waiting. Waits
    /// like [`Self::join`].
    pub(crate) fn barrier(&self) -> bool {
        let mut waiter = self.waiter();
        self.barrier
            .arrive(&self.cancelled, self.waits().map(|w| &w.sync), &mut waiter)
    }
}

/// Joins a section's dispatched work when dropped, also on unwind.
pub(crate) struct JoinGuard<'a>(pub(crate) &'a Section);

impl Drop for JoinGuard<'_> {
    fn drop(&mut self) {
        self.0.join();
    }
}

impl Drop for Section {
    fn drop(&mut self) {
        drop(self.take_panic());
    }
}

/// Lends reserved members to Rayon while alive.
#[cfg_attr(not(feature = "parallel"), allow(dead_code))]
pub(crate) struct LendGuard<'a> {
    awake: &'a Awake,
    ids: &'a [usize],
}

impl<'a> LendGuard<'a> {
    #[cfg_attr(not(feature = "parallel"), allow(dead_code))]
    pub(crate) fn new(awake: &'a Awake, ids: &'a [usize]) -> Self {
        awake.lend(ids, true);
        LendGuard { awake, ids }
    }
}

impl Drop for LendGuard<'_> {
    fn drop(&mut self) {
        self.awake.lend(self.ids, false);
    }
}
