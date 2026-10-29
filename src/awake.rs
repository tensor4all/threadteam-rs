//! The workers of a pool: their slots, the native loop, sections, acquisition.
//!
//! Every leased worker runs [`serve`] for its slot until the pool detaches it
//! (`docs/design/own-threads.md`). A slot's state word ([`crate::park::SlotState`])
//! has the bits `RESERVED`, `LENT`, `BUSY`, `DETACH`:
//!
//! * `0`: free. The worker serves the pool's broadcast shares and queued tasks
//!   (winning `BUSY` before it takes one) and otherwise waits (spin, yield,
//!   park). A team claims it by a compare-and-swap from `0`.
//! * `RESERVED`: a team member. It waits for `gen` (spin, yield, park) and runs
//!   the sections its leader publishes; it serves nothing else.
//! * `RESERVED|LENT`: a member lent to the pool's queue (`Exec::external`).
//! * `BUSY`: running a task or a share; not claimable.
//! * `DETACH`: the pool is being dropped; the worker acknowledges and leaves.
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
#![cfg_attr(not(feature = "parallel"), allow(dead_code))]

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

#[cfg(feature = "parallel")]
use crate::park::{BUSY, DETACH};
use crate::park::{LENT, RESERVED};

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

/// One worker's slot: its state word, its team-member dispatch state, and
/// where it sleeps.
#[repr(align(128))]
pub(crate) struct Slot {
    /// `RESERVED`/`LENT`/`BUSY`/`DETACH` (see [`crate::park::SlotState`]).
    pub(crate) state: crate::park::SlotState,
    /// Bumped by the current leader per published section.
    gen: AtomicUsize,
    ptr: AtomicPtr<Section>,
    arg: AtomicUsize,
    /// Idle policy written by the releasing leader before the slot is freed.
    idle_spin_ns: AtomicU64,
    /// Spin budget, in polls, of this reserved member's waits for work (see
    /// `docs/design/waiting.md`); written by the leader at root-scope entry.
    spin_budget: AtomicU32,
    /// The mask this member saved before pinning itself; only the worker
    /// thread of this slot touches it (see [`crate::pin::SavedMask`]).
    #[cfg_attr(not(feature = "parallel"), allow(dead_code))]
    pub(crate) saved: crate::pin::SavedMask,
    /// Where this worker sleeps: idle, as a member between sections, or while
    /// it waits for a batch (`help_while`). An `Arc`, so that a completion can
    /// keep it alive across its final decrement.
    pub(crate) park: std::sync::Arc<Parker>,
    /// The worker thread leased to this slot (set when it attaches).
    pub(crate) thread: std::sync::OnceLock<std::thread::ThreadId>,
    /// The worker's broadcast cursor (only the worker touches it).
    bseq: AtomicU64,
    /// Set while the worker waits for a batch and serves its pool meanwhile.
    pub(crate) helping: AtomicBool,
    /// Broadcast shares on this worker's stack (only the worker touches it).
    /// While one runs, the worker takes no other share: a later broadcast run
    /// above an unfinished one could wait at its barrier for a worker that
    /// waits at the earlier one's.
    in_bcast: AtomicUsize,
    /// A released member that has not gone to sleep since (it still spins,
    /// claimable without a wake): what `Pool::resident_members` counts besides
    /// reserved members.
    lingering: AtomicBool,
    /// The worker attached to its lease / acknowledged its detach.
    pub(crate) attached: AtomicBool,
    pub(crate) acked: AtomicBool,
}

impl Slot {
    fn new() -> Self {
        Slot {
            state: crate::park::SlotState::new(),
            gen: AtomicUsize::new(0),
            ptr: AtomicPtr::new(core::ptr::null_mut()),
            arg: AtomicUsize::new(0),
            idle_spin_ns: AtomicU64::new(u64::MAX),
            spin_budget: AtomicU32::new(crate::park::THROTTLED_POLLS),
            saved: crate::pin::SavedMask::new(),
            park: std::sync::Arc::new(Parker::new()),
            thread: std::sync::OnceLock::new(),
            bseq: AtomicU64::new(0),
            helping: AtomicBool::new(false),
            in_bcast: AtomicUsize::new(0),
            lingering: AtomicBool::new(false),
            attached: AtomicBool::new(false),
            acked: AtomicBool::new(false),
        }
    }
}

/// The waiting state of one pool (`docs/design/waiting.md`).
pub(crate) struct Waits {
    /// Where leaders sleep in `join` and participants in `barrier`. One per
    /// pool: a member may wake it after its last decrement, when the section
    /// may already be gone, and the pool outlives every member loop. Also
    /// where a plain outside thread waits for a batch of this pool.
    pub(crate) sync: std::sync::Arc<Parker>,
    /// Spin budget of a wait, in polls (`PoolBuilder::spin_polls`).
    pub(crate) spin_polls: u32,
    /// Diagnostics (`Pool::wait_counters`): sleeps of leaders in joins, of
    /// participants in barriers, and parks of members waiting for work.
    pub(crate) join_sleeps: AtomicU64,
    pub(crate) barrier_sleeps: AtomicU64,
    pub(crate) member_parks: AtomicU64,
}

/// Shared state of the awake members of one pool.
pub(crate) struct Awake {
    pub(crate) slots: Box<[Slot]>,
    /// The pool's queue of task batches (plain `fork`, `for_ranges`, `external`).
    pub(crate) queue: crate::park::Queue,
    /// The pool's broadcasts.
    pub(crate) bcast: crate::park::BroadcastList,
    /// What the joins and barriers of this pool's sections sleep on, and the
    /// pool's spin budget (`docs/design/waiting.md`).
    pub(crate) waits: Waits,
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
            queue: crate::park::Queue::new(),
            bcast: crate::park::BroadcastList::new(),
            waits: Waits {
                sync: std::sync::Arc::new(Parker::new()),
                spin_polls,
                join_sleeps: AtomicU64::new(0),
                barrier_sleeps: AtomicU64::new(0),
                member_parks: AtomicU64::new(0),
            },
            release_epoch: AtomicUsize::new(0),
            defaults,
            costs,
            pin,
            acquire_timeout,
            #[cfg(feature = "trace")]
            trace: crate::trace::Trace::new(width),
        }
    }

    /// Members kept awake for teams: reserved ones, and released ones that
    /// have not gone to sleep since.
    pub(crate) fn resident(&self) -> usize {
        self.slots
            .iter()
            .filter(|s| s.state.load() & RESERVED != 0 || s.lingering.load(Ordering::Acquire))
            .count()
    }

    /// Hands members back: they become free under `policy`.
    pub(crate) fn release(&self, ids: &[usize], policy: WaitPolicy) {
        let ns = u64::try_from(policy.idle_spin().as_nanos()).unwrap_or(u64::MAX);
        for &id in ids {
            let s = &self.slots[id];
            s.idle_spin_ns.store(ns, Ordering::Relaxed);
            s.lingering.store(true, Ordering::Release);
            s.state.release();
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
    /// A root team scope writes it once at entry ([`Self::spin_budget`] of
    /// whether it is oversubscribed). Nothing else changes it, so a member keeps the value of the
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

    /// Lends (or takes back) the members' threads to the pool's queue: while
    /// lent, a reserved member runs the pool's tasks instead of idling. Taking
    /// it back lets a task it is running finish; the member serves its team
    /// again after that.
    #[cfg_attr(not(feature = "parallel"), allow(dead_code))]
    pub(crate) fn lend(&self, ids: &[usize], on: bool) {
        for &id in ids {
            let s = &self.slots[id];
            s.state.set_lent(on);
            s.park.wake();
        }
    }

    /// Wakes the workers that can take work of the queue or a broadcast: free
    /// ones, lent members and workers helping while they wait.
    pub(crate) fn wake_for_work(&self) {
        for s in &*self.slots {
            let st = s.state.load();
            if st == 0 || st == RESERVED | LENT || s.helping.load(Ordering::SeqCst) {
                s.park.wake();
            }
        }
    }

    /// The slot of the calling thread in this pool, if it is one of its
    /// workers (a scan of the slots: entry points only).
    pub(crate) fn slot_of_current(&self) -> Option<usize> {
        let me = std::thread::current().id();
        self.slots.iter().position(|s| s.thread.get() == Some(&me))
    }

    /// Reserves `need` free workers by compare-and-swap, waiting at most the
    /// pool's acquisition timeout for busy ones to become free; on a shortfall
    /// every tentative member is released and `Unavailable` returned. A worker
    /// that runs a task is not free, so a reservation never shares a thread
    /// with a running task.
    pub(crate) fn acquire(
        self: &std::sync::Arc<Self>,
        need: usize,
        policy: WaitPolicy,
        spin_budget: u32,
    ) -> Result<Reservation, Refused> {
        let mut ids = Vec::with_capacity(need);
        let me = self.slot_of_current();
        let started = std::time::Instant::now();
        let mut spins = 0u32;
        loop {
            // Members of an earlier team that are still awake first (warm,
            // claimable without a wake-up), then any other free worker.
            let order = self
                .slots
                .iter()
                .enumerate()
                .filter(|(_, s)| s.lingering.load(Ordering::Acquire))
                .chain(
                    self.slots
                        .iter()
                        .enumerate()
                        .filter(|(_, s)| !s.lingering.load(Ordering::Acquire)),
                );
            for (i, s) in order {
                if ids.len() == need {
                    break;
                }
                if Some(i) != me
                    && !ids.contains(&i)
                    && s.attached.load(Ordering::Acquire)
                    && s.state.try_reserve()
                {
                    s.lingering.store(false, Ordering::Release);
                    ids.push(i);
                }
            }
            if ids.len() == need || started.elapsed() >= self.acquire_timeout {
                break;
            }
            backoff(&mut spins);
        }
        if ids.len() < need {
            self.release(&ids, policy);
            return Err(Refused::Unavailable);
        }
        self.set_spin_budget(&ids, spin_budget);
        Ok(Reservation {
            awake: std::sync::Arc::clone(self),
            ids,
            policy,
        })
    }

    /// Runs one broadcast share or one queued task on worker `id`, if there is
    /// one; the caller holds the right to run work (it won `BUSY`, or it is a
    /// worker waiting for a batch). `true` when something ran.
    #[cfg(feature = "parallel")]
    pub(crate) fn run_one(&self, id: usize) -> bool {
        let slot = &self.slots[id];
        let mut seq = slot.bseq.load(Ordering::Relaxed);
        if slot.in_bcast.load(Ordering::Relaxed) == 0 && self.bcast.has_after(seq) {
            let next = self.bcast.next_for(id, &mut seq);
            slot.bseq.store(seq, Ordering::Relaxed);
            if let Some(b) = next {
                slot.in_bcast.fetch_add(1, Ordering::Relaxed);
                // SAFETY: a linked broadcast that targets `id`; its caller
                // keeps it alive until this share completed. `run_share`
                // catches a panic of the share.
                unsafe { crate::park::Broadcast::run_share(b, id) };
                slot.in_bcast.fetch_sub(1, Ordering::Relaxed);
                return true;
            }
        }
        if let Some((b, i)) = self.queue.claim() {
            // SAFETY: index `i` of `b` was claimed here.
            unsafe { crate::park::Batch::run_claimed(b, i) };
            return true;
        }
        false
    }

    /// Whether worker `id` has work it could run now (a hint).
    #[cfg(feature = "parallel")]
    fn work_for(&self, id: usize) -> bool {
        let slot = &self.slots[id];
        self.queue.has_work()
            || (slot.in_bcast.load(Ordering::Relaxed) == 0
                && self.bcast.has_after(slot.bseq.load(Ordering::Relaxed)))
    }

    /// A free worker or a lent member takes one piece of work: it wins `BUSY`
    /// first, so a team cannot reserve it meanwhile, then runs it.
    #[cfg(feature = "parallel")]
    fn try_work(&self, id: usize) -> bool {
        if !self.work_for(id) {
            return false;
        }
        let slot = &self.slots[id];
        if !slot.state.try_take() {
            return false;
        }
        let ran = self.run_one(id);
        slot.state.end_busy();
        ran
    }

    /// Waits until `done` reports completion. A worker of this pool (`me`)
    /// serves this pool's queue and broadcasts meanwhile, so nested and
    /// cross-pool waits always make progress; it sleeps on its own parker,
    /// which the batch's completion and new work of this pool wake.
    #[cfg(feature = "parallel")]
    pub(crate) fn help_while(&self, me: usize, done: impl Fn() -> bool) {
        let slot = &self.slots[me];
        // Waits nest (a task run while helping waits in turn): the outer wait
        // stays wakeable after the inner one returns.
        let outer = slot.helping.swap(true, Ordering::SeqCst);
        let mut waiter = crate::park::Waiter::new(self.waits.spin_polls);
        while !done() {
            if self.run_one(me) {
                waiter = crate::park::Waiter::new(self.waits.spin_polls);
                continue;
            }
            if waiter.step() == crate::park::Step::Sleep {
                slot.park.park_while(|| !done() && !self.work_for(me));
                waiter = crate::park::Waiter::new(self.waits.spin_polls);
            }
        }
        slot.helping.store(outer, Ordering::SeqCst);
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

/// The native loop of worker `id` of a pool, from its attachment until the
/// pool detaches it (`DETACH`, while it is not busy). See
/// `docs/design/own-threads.md` and `docs/design/waiting.md`.
///
/// * Reserved (and not lent): a team member. It waits for its slot's `gen`
///   in staged episodes (spin for the slot's budget, yield, park) and runs the
///   sections its leader publishes; it serves nothing else.
/// * Free, or a lent member: it takes broadcast shares and queued tasks,
///   winning `BUSY` before it removes anything, and otherwise waits in a staged
///   episode (the policy's idle spin) and parks until work, a reservation or a
///   detach arrives.
///
/// A member restores its own affinity mask when it stops being reserved, and
/// before it serves foreign work as a lent member.
#[cfg(feature = "parallel")]
pub(crate) fn serve(aw: &Awake, id: usize) {
    use crate::park::{Step, Waiter};
    let slot = &aw.slots[id];
    let mut seen = slot.gen.load(Ordering::Acquire);
    // Claimable only from now on: a leader that reserves this slot publishes
    // its first section after the baseline above.
    slot.attached.store(true, Ordering::SeqCst);
    let mut waiter: Option<Waiter> = None;
    let mut was_lent = false;
    let mut reserved = false;
    let mut epoch = aw.release_epoch.load(Ordering::Relaxed);
    // Whether the last iteration served as a member; a wait episode does not
    // carry over between the member's and the free worker's waits.
    let mut as_member = false;
    trace!(aw, Some(id), Enter);
    loop {
        let st = slot.state.load();
        // Pinning ends with the reservation, and it ends on this thread.
        if st & RESERVED != 0 {
            reserved = true;
        } else if reserved {
            reserved = false;
            slot.saved.restore();
        }
        if st & DETACH != 0 && st & BUSY == 0 {
            trace!(aw, Some(id), Leave);
            return;
        }
        if st & RESERVED != 0 && st & LENT == 0 {
            if !as_member {
                as_member = true;
                waiter = None;
            }
            if was_lent {
                was_lent = false;
                waiter = None;
                trace!(aw, Some(id), LentEnd);
            }
            let g = slot.gen.load(Ordering::Acquire);
            if g == seen {
                let w = waiter
                    .get_or_insert_with(|| Waiter::new(slot.spin_budget.load(Ordering::Relaxed)));
                if w.step() == Step::Sleep {
                    aw.waits.member_parks.fetch_add(1, Ordering::Relaxed);
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
                // completion counter reaches zero, which happens only after
                // the call below decremented it.
                if unsafe { (*ptr).pin } {
                    // A pin section: save the previous mask before pinning.
                    slot.saved.save_if_unpinned();
                }
                trace!(aw, Some(id), SectionBegin);
                // SAFETY: as above; `aw` outlives this loop, so the wake after
                // the last decrement does not touch the section.
                unsafe { Section::execute_member(ptr, arg) };
                trace!(aw, Some(id), SectionEnd);
            }
            continue;
        }
        // Free (`0`) or lent (`RESERVED|LENT`): serve the pool.
        if as_member {
            as_member = false;
            waiter = None;
        }
        if st & LENT != 0 && !was_lent {
            was_lent = true;
            // The thread serves foreign work now: undo our pin first.
            slot.saved.restore();
            trace!(aw, Some(id), LentBegin);
        }
        if aw.try_work(id) {
            waiter = None;
            continue;
        }
        let now = aw.release_epoch.load(Ordering::Relaxed);
        if now != epoch {
            // `Pool::release_idle`: stop spinning now.
            epoch = now;
            waiter = Some(Waiter::new(0));
        }
        let w = waiter.get_or_insert_with(|| {
            let idle = slot.idle_spin_ns.load(Ordering::Relaxed);
            Waiter::new(if idle == 0 { 0 } else { aw.waits.spin_polls })
        });
        if w.step() == Step::Sleep {
            slot.lingering.store(false, Ordering::Release);
            let qgen = aw.queue.gen();
            let bgen = aw.bcast.gen();
            slot.park.park_while(|| {
                slot.state.load() == st
                    && aw.queue.gen() == qgen
                    && aw.bcast.gen() == bgen
                    && !aw.work_for(id)
            });
            waiter = None;
        }
    }
}

/// Sleeps (still reserved) until a section is published, the reservation ends,
/// the member is lent, or the pool detaches it. The handshake lives in
/// [`Parker`].
#[cfg(feature = "parallel")]
fn park_member(slot: &Slot, seen: usize) {
    slot.park
        .park_while(|| slot.gen.load(Ordering::SeqCst) == seen && slot.state.load() == RESERVED);
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
            waits.as_ref().map(|w| &*w.sync),
        );
    }

    /// The pool's waiting state, if the section has a pool.
    fn waits(&self) -> Option<&Waits> {
        // SAFETY: `waits` is null or points into the pool's `Awake`, which
        // outlives the section.
        unsafe { self.waits.as_ref() }
    }

    /// A wait episode of this section's join or barrier.
    fn waiter(&self) -> crate::park::Waiter {
        match self.waits() {
            Some(w) => crate::park::Waiter::new(if self.over {
                crate::park::THROTTLED_POLLS.min(w.spin_polls)
            } else {
                w.spin_polls
            }),
            None => crate::park::Waiter::new(u32::MAX),
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
        self.done.wait(self.waits().map(|w| &*w.sync), &mut waiter);
        if waiter.slept() {
            if let Some(w) = self.waits() {
                w.join_sleeps.fetch_add(1, Ordering::Relaxed);
            }
        }
    }

    /// Sense-reversing barrier over the section's participants; `false` when
    /// a participant failed, so that peers can leave instead of waiting. Waits
    /// like [`Self::join`].
    pub(crate) fn barrier(&self) -> bool {
        let mut waiter = self.waiter();
        let passed =
            self.barrier
                .arrive(&self.cancelled, self.waits().map(|w| &*w.sync), &mut waiter);
        if waiter.slept() {
            if let Some(w) = self.waits() {
                w.barrier_sleeps.fetch_add(1, Ordering::Relaxed);
            }
        }
        passed
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
