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
//! * `RESERVED`: owned by one leader; the member spins on `gen` and runs the
//!   section the leader published. It neither runs Rayon jobs nor leaves.
//! * `IDLE`: loop alive but unreserved; spins for the policy's idle time, then
//!   leaves (`ABSENT`). Claimable by a compare-and-swap, with no waiting.
//! * `YIELD`: an idle member inside `rayon::yield_now`; not claimable.
//!
//! A leader publishes a [`Section`] (a stack object) by writing its address
//! and an argument into the member's slot and bumping `gen`; the member runs
//! it and decrements the section's completion counter. Nothing is allocated
//! per dispatch; a member that parked is woken by one condition-variable
//! broadcast.
//!
//! No thread-local state is used: a member's identity is its slot index.

use crate::error::Refused;
use crate::park::Parker;
use crate::policy::WaitPolicy;
use core::sync::atomic::{AtomicBool, AtomicPtr, AtomicU64, AtomicUsize, Ordering};
use std::any::Any;
use std::time::Duration;

const ABSENT: usize = 0;
const IDLE: usize = 1;
#[cfg_attr(not(feature = "parallel"), allow(dead_code))]
const YIELD: usize = 2;
const RESERVED: usize = 3;

/// Alignment that keeps a hot atomic on its own cache line pair (128 bytes
/// covers Apple Silicon and adjacent-line prefetch on x86).
#[repr(align(128))]
pub(crate) struct Padded<T>(pub(crate) T);

/// Pure spin for a while, then yield the CPU between polls.
#[inline]
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
    /// How long a reserved member spins without work before it parks.
    park_ns: AtomicU64,
    /// Set by the leader while the member's thread is lent to Rayon
    /// (`Exec::external` on a team token).
    lent: AtomicBool,
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
            park_ns: AtomicU64::new(u64::MAX),
            lent: AtomicBool::new(false),
        }
    }
}

/// Shared state of the awake members of one pool.
pub(crate) struct Awake {
    slots: Box<[Slot]>,
    /// Parked members sleep here; a leader that changed their state wakes them.
    pub(crate) park: Parker,
    pub(crate) shutdown: AtomicBool,
    /// Bumped by `Pool::release_idle`; members that started earlier leave when idle.
    pub(crate) release_epoch: AtomicUsize,
    pub(crate) defaults: WaitPolicy,
    #[cfg_attr(not(feature = "parallel"), allow(dead_code))]
    pub(crate) acquire_timeout: Duration,
}

impl Awake {
    pub(crate) fn new(width: usize, defaults: WaitPolicy, acquire_timeout: Duration) -> Self {
        Awake {
            slots: (0..width).map(|_| Slot::new()).collect(),
            park: Parker::new(),
            shutdown: AtomicBool::new(false),
            release_epoch: AtomicUsize::new(0),
            defaults,
            acquire_timeout,
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
        }
        self.wake_parked();
    }

    /// Wakes every parked member (see [`Parker`]). Called after the state a
    /// member waits on (`gen`, `state`, `lent`) changed with SeqCst.
    pub(crate) fn wake_parked(&self) {
        self.park.wake();
    }

    /// Publishes `section` to its `n` targets (position `p` gets argument `p + 1`).
    ///
    /// The caller leads every target (its slot is `RESERVED` for the caller)
    /// and keeps `section` alive until its completion counter reaches zero.
    /// Every generation is bumped, then parked members are woken by one
    /// broadcast (a single atomic load when nobody sleeps).
    pub(crate) fn publish(&self, section: &Section) {
        let ptr = (section as *const Section).cast_mut();
        for pos in 0..section.dispatched {
            let s = &self.slots[section.slot(pos)];
            s.arg.store(pos + 1, Ordering::Relaxed);
            s.ptr.store(ptr, Ordering::Relaxed);
            // fetch_add: always the latest value, whichever thread led before.
            s.gen.fetch_add(1, Ordering::SeqCst);
        }
        self.wake_parked();
    }

    /// Lends (or takes back) the members' threads to Rayon: while lent, a
    /// reserved member runs the pool's jobs instead of idling.
    #[cfg_attr(not(feature = "parallel"), allow(dead_code))]
    pub(crate) fn lend(&self, ids: &[usize], on: bool) {
        for &id in ids {
            self.slots[id].lent.store(on, Ordering::SeqCst);
        }
        if on {
            self.wake_parked();
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
        park_spin: Duration,
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
        let ns = u64::try_from(park_spin.as_nanos()).unwrap_or(u64::MAX);
        for &id in &ids {
            self.slots[id].park_ns.store(ns, Ordering::Relaxed);
        }
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
#[cfg(feature = "parallel")]
fn member_loop(aw: &Awake, id: usize, mut seen: usize, epoch: usize) {
    use std::time::Instant;
    let slot = &aw.slots[id];
    let mut idle_since: Option<Instant> = None;
    let mut reserved_since: Option<Instant> = None;
    let mut spins = 0u32;
    let mut polls = 0u32;
    loop {
        match slot.state.load(Ordering::Acquire) {
            RESERVED if slot.lent.load(Ordering::Acquire) => {
                idle_since = None;
                reserved_since = None;
                // Serve the pool's jobs; resume team service when taken back.
                if !matches!(rayon::yield_now(), Some(rayon::Yield::Executed)) {
                    for _ in 0..32 {
                        std::hint::spin_loop();
                    }
                }
            }
            RESERVED => {
                idle_since = None;
                let g = slot.gen.load(Ordering::Acquire);
                if g == seen {
                    backoff(&mut spins);
                    polls = polls.wrapping_add(1);
                    if polls & 0xFF == 0 {
                        let since = *reserved_since.get_or_insert_with(Instant::now);
                        let park_ns = slot.park_ns.load(Ordering::Relaxed);
                        if since.elapsed().as_nanos() >= u128::from(park_ns) {
                            park_member(aw, slot, seen);
                            reserved_since = None;
                            spins = 0;
                        }
                    }
                } else {
                    seen = g;
                    spins = 0;
                    reserved_since = None;
                    let ptr = slot.ptr.load(Ordering::Relaxed);
                    let arg = slot.arg.load(Ordering::Relaxed);
                    // SAFETY: the leader keeps the section alive until its
                    // completion counter reaches zero, which happens only
                    // after this call decremented it.
                    unsafe { Section::execute_member(ptr, arg) };
                }
            }
            IDLE => {
                let since = *idle_since.get_or_insert_with(Instant::now);
                let spin_ns = slot.idle_spin_ns.load(Ordering::Relaxed);
                let leave = aw.shutdown.load(Ordering::Relaxed)
                    || aw.release_epoch.load(Ordering::Relaxed) != epoch
                    || since.elapsed().as_nanos() >= u128::from(spin_ns);
                if leave {
                    if slot
                        .state
                        .compare_exchange(IDLE, ABSENT, Ordering::AcqRel, Ordering::Relaxed)
                        .is_ok()
                    {
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
fn park_member(aw: &Awake, slot: &Slot, seen: usize) {
    aw.park.park_while(|| {
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
    /// Dispatched participants that have not finished.
    remaining: Padded<AtomicUsize>,
    arrived: Padded<AtomicUsize>,
    epoch: Padded<AtomicUsize>,
    cancelled: AtomicBool,
    /// First panic payload, boxed twice so that it fits an `AtomicPtr`.
    /// (A `Mutex` would allocate on first use on some platforms.)
    panic: AtomicPtr<Box<dyn Any + Send>>,
}

// SAFETY: `data` points to a `Sync` closure that outlives the section; every
// other field is an atomic or a mutex.
unsafe impl Sync for Section {}

impl Section {
    pub(crate) fn new(
        call: unsafe fn(*const (), &Section, usize),
        data: *const (),
        size: usize,
        dispatched: usize,
        slot_of: unsafe fn(*const (), usize) -> usize,
        slot_data: *const (),
    ) -> Self {
        Section {
            call,
            data,
            size,
            dispatched,
            slot_of,
            slot_data,
            remaining: Padded(AtomicUsize::new(dispatched)),
            arrived: Padded(AtomicUsize::new(0)),
            epoch: Padded(AtomicUsize::new(0)),
            cancelled: AtomicBool::new(false),
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
        let sec = &*ptr;
        let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| sec.call_here(arg)));
        if let Err(p) = r {
            sec.fail(p);
        }
        (*ptr).remaining.0.fetch_sub(1, Ordering::Release);
    }

    /// Records a participant's panic and cancels barriers.
    pub(crate) fn fail(&self, payload: Box<dyn Any + Send>) {
        self.cancelled.store(true, Ordering::Release);
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

    /// Waits until every dispatched participant has finished.
    pub(crate) fn join(&self) {
        let mut spins = 0;
        while self.remaining.0.load(Ordering::Acquire) != 0 {
            backoff(&mut spins);
        }
    }

    /// Sense-reversing barrier over the section's participants; `false` when
    /// a participant failed, so that peers can leave instead of waiting.
    pub(crate) fn barrier(&self) -> bool {
        if self.is_cancelled() {
            return false;
        }
        if self.size == 1 {
            return true;
        }
        let epoch = self.epoch.0.load(Ordering::Acquire);
        if self.arrived.0.fetch_add(1, Ordering::AcqRel) + 1 == self.size {
            self.arrived.0.store(0, Ordering::Relaxed);
            self.epoch.0.store(epoch.wrapping_add(1), Ordering::Release);
            return true;
        }
        let mut spins = 0;
        loop {
            if self.epoch.0.load(Ordering::Acquire) != epoch {
                return true;
            }
            if self.is_cancelled() {
                return false;
            }
            backoff(&mut spins);
        }
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
