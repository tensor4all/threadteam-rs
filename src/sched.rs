//! The only module that talks to the scheduler.
//!
//! No thread-local state is read or written here; placement decisions use only
//! the pool handle and the identity of the calling thread that the scheduler
//! reports.

use crate::pool::Pool;
use crate::track::Track;

/// Whether the calling thread is a worker of `pool`.
#[cfg(feature = "parallel")]
pub(crate) fn in_pool(pool: &Pool) -> bool {
    pool.awake.slot_of_current().is_some()
}

#[cfg(not(feature = "parallel"))]
pub(crate) fn in_pool(_pool: &Pool) -> bool {
    false
}

/// Restores the calling thread's own affinity mask when it is a reserved,
/// pinned member of `pool`.
///
/// A participant can be the thread that runs work which is not its team's: the
/// member that leads a `fork` child and calls
/// [`Exec::external`](crate::Exec::external), or a token used from inside a
/// share. Such a thread must not keep the team's single-CPU placement while it
/// serves that work, and only it may change its own affinity, so it restores
/// the mask it saved before pinning (see `pin::SavedMask`). A no-op when the
/// calling thread is not a worker of `pool` or is not pinned; the member
/// re-pins at the next pin section.
#[cfg(feature = "parallel")]
pub(crate) fn unpin_worker(pool: &Pool) {
    if let Some(worker) = pool.awake.slot_of_current() {
        pool.awake.unpin_member(worker);
    }
}

/// Who waits for a batch of `pool`, and how (`docs/design/own-threads.md`).
#[cfg(feature = "parallel")]
enum Waiter {
    /// A worker of `pool` itself (slot index): it runs its own batch's tasks
    /// first, then serves the pool while it waits.
    Own(usize),
    /// A worker of another pool: it serves its own pool while it waits, so a
    /// cross-pool chain (A -> B -> A) always makes progress.
    Foreign(std::sync::Arc<crate::awake::Awake>, usize),
    /// A plain thread: it sleeps on the pool's parker.
    Outside,
}

#[cfg(feature = "parallel")]
fn waiter_of(pool: &Pool) -> Waiter {
    if let Some(i) = pool.awake.slot_of_current() {
        Waiter::Own(i)
    } else if let Some((aw, i)) = crate::worker::current_worker() {
        Waiter::Foreign(aw, i)
    } else {
        Waiter::Outside
    }
}

#[cfg(feature = "parallel")]
impl Waiter {
    /// The parker the last completion wakes.
    fn parker<'a>(&'a self, pool: &'a Pool) -> &'a std::sync::Arc<crate::park::Parker> {
        match self {
            Waiter::Own(i) => &pool.awake.slots[*i].park,
            Waiter::Foreign(aw, i) => &aw.slots[*i].park,
            Waiter::Outside => &pool.awake.waits.sync,
        }
    }

    /// Waits until `done` holds.
    fn wait(&self, pool: &Pool, done: &crate::park::Completion) {
        match self {
            Waiter::Own(i) => pool.awake.help_while(*i, || done.is_done_now()),
            Waiter::Foreign(aw, i) => aw.help_while(*i, || done.is_done_now()),
            Waiter::Outside => {
                let mut w = crate::park::Waiter::new(pool.awake.waits.spin_polls);
                done.wait(Some(&pool.awake.waits.sync), &mut w);
            }
        }
    }
}

/// Runs `task(0..tasks)` on `pool` and waits for all of them.
///
/// The tasks form one batch on the caller's stack, published on the pool's
/// queue, so a steady-state call allocates nothing.
///
/// * Caller is a worker of `pool`: it runs the batch's unclaimed tasks itself,
///   then serves the pool until the others finish.
/// * Caller is a worker of another pool: the pool's workers run the tasks; the
///   caller serves its own pool meanwhile.
/// * Caller is a plain thread: every task runs on a worker of `pool` and the
///   caller sleeps until the last one finishes.
///
/// A panicking task does not stop the others; the first panic is resumed once
/// all of them finished.
#[cfg(feature = "parallel")]
pub(crate) fn run_tasks<F: Fn(usize) + Sync>(pool: &Pool, tasks: usize, track: &Track, task: F) {
    if tasks == 0 {
        return;
    }
    unsafe fn call<F: Fn(usize)>(data: *const (), i: usize) {
        // SAFETY: `data` is the `&F` of `run_tasks`, alive until the batch is done.
        (*data.cast::<F>())(i)
    }
    let waiter = waiter_of(pool);
    if !matches!(waiter, Waiter::Own(_)) {
        track.note_install();
    }
    // SAFETY: `call::<F>` and `task` belong together; both outlive the batch,
    // which this function waits for; the waker is kept alive by `waiter` (or
    // the pool) until the wait below returns.
    let batch = unsafe {
        crate::park::Batch::new(
            call::<F>,
            (&task as *const F).cast(),
            tasks,
            waiter.parker(pool),
        )
    };
    pool.awake.queue.publish(&batch);
    pool.awake.wake_for_work();
    if let Waiter::Own(_) = waiter {
        while let Some(k) = pool.awake.queue.claim_own(&batch) {
            // SAFETY: index `k` of our own batch was claimed here.
            unsafe { crate::park::Batch::run_claimed(&batch, k) };
        }
    }
    waiter.wait(pool, &batch.done);
    if let Some(payload) = batch.take_panic() {
        std::panic::resume_unwind(payload);
    }
}

#[cfg(not(feature = "parallel"))]
pub(crate) fn run_tasks<F: Fn(usize) + Sync>(_pool: &Pool, tasks: usize, _track: &Track, task: F) {
    for k in 0..tasks {
        task(k);
    }
}

/// Runs `f` on a worker of `pool`: inline when the caller is one, otherwise as
/// a one-task batch while the caller waits.
#[cfg(feature = "parallel")]
pub(crate) fn enter<R: Send>(pool: &Pool, track: &Track, f: impl FnOnce() -> R + Send) -> R {
    if in_pool(pool) {
        return f();
    }
    // No lock: a `Mutex` allocates on its first lock on some platforms
    // (macOS), and this runs on every call.
    let job = Handoff(core::cell::UnsafeCell::new(Some(f)));
    let out = Handoff(core::cell::UnsafeCell::new(None));
    run_tasks(pool, 1, track, |_| {
        // SAFETY: the batch has one index, claimed exactly once (under the
        // queue lock), so this body runs once; the caller reads `out` only
        // after the batch's completion, which orders this write before it.
        let f = unsafe { job.take() };
        if let Some(f) = f {
            let r = f();
            // SAFETY: as above.
            unsafe { out.put(r) };
        }
    });
    let r = out.0.into_inner();
    r.expect("the task ran")
}

/// A value passed into and out of the single task of [`enter`]'s batch: that
/// task is its only user until the batch completes.
#[cfg(feature = "parallel")]
struct Handoff<T>(core::cell::UnsafeCell<Option<T>>);

// SAFETY: one thread at a time touches the value: the task while it runs, the
// caller after the batch completed (see `enter`); `T: Send` lets it move.
#[cfg(feature = "parallel")]
unsafe impl<T: Send> Sync for Handoff<T> {}

#[cfg(feature = "parallel")]
impl<T> Handoff<T> {
    /// # Safety
    /// No other thread accesses the value meanwhile.
    unsafe fn take(&self) -> Option<T> {
        (*self.0.get()).take()
    }

    /// # Safety
    /// As [`take`](Self::take).
    unsafe fn put(&self, v: T) {
        *self.0.get() = Some(v);
    }
}

/// Runs `f(i)` for every worker index `i < budget` at once, each on worker
/// `i` of `pool`. The caller is never a worker of `pool` (`Exec::broadcast`
/// refuses that); it waits, serving its own pool if it is a worker of one.
#[cfg(feature = "parallel")]
pub(crate) fn broadcast(pool: &Pool, track: &Track, budget: usize, f: impl Fn(usize) + Sync) {
    track.note_install();
    let task = |i: usize| {
        let _live = track.task();
        f(i);
    };
    unsafe fn call<F: Fn(usize)>(data: *const (), i: usize) {
        // SAFETY: `data` is the `&F` of `broadcast`, alive until it is done.
        (*data.cast::<F>())(i)
    }
    fn data_of<F>(f: &F) -> *const () {
        (f as *const F).cast()
    }
    fn call_of<F: Fn(usize)>(_: &F) -> unsafe fn(*const (), usize) {
        call::<F>
    }
    let waiter = waiter_of(pool);
    // SAFETY: as in `run_tasks`; the broadcast is unlinked below, after all of
    // its shares completed and before it goes out of scope.
    let b = unsafe {
        crate::park::Broadcast::new(call_of(&task), data_of(&task), budget, waiter.parker(pool))
    };
    pool.awake.bcast.publish(&b);
    pool.awake.wake_for_work();
    waiter.wait(pool, &b.done);
    pool.awake.bcast.unlink(&b);
    if let Some(payload) = b.take_panic() {
        std::panic::resume_unwind(payload);
    }
}
