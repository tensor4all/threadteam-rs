//! The only module that talks to the scheduler.
//!
//! None of the crate's own thread-local state is read or written here:
//! placement decisions use only the pool handle and the identity of the calling
//! thread that the scheduler reports. A worker of another pool that waits for a
//! submission hands out its `Thread` handle, so that the job can wake it.

#[cfg(feature = "parallel")]
use crate::park::Latch;
use crate::pool::Pool;
#[cfg(feature = "parallel")]
use crate::sync::thread::{self, Thread};
use crate::track::Track;
#[cfg(feature = "parallel")]
use std::cell::UnsafeCell;
#[cfg(feature = "parallel")]
use std::panic::{catch_unwind, resume_unwind, AssertUnwindSafe};

/// Whether the calling thread is a worker of `pool`.
#[cfg(feature = "parallel")]
pub(crate) fn in_pool(pool: &Pool) -> bool {
    pool.inner().current_thread_index().is_some()
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
    if let Some(worker) = pool.inner().current_thread_index() {
        pool.awake.unpin_member(worker);
    }
}

/// Runs `task(0..tasks)` concurrently on `pool` and waits for all of them.
///
/// The tasks form a recursive binary `rayon::join` tree, whose jobs live on the
/// stack, entered through [`enter`]: the caller is placed and counted as
/// [`enter`] says, and the tree itself allocates nothing. `rayon::join` is only
/// reached on a worker of `pool`, so it never touches the global registry.
#[cfg(feature = "parallel")]
pub(crate) fn run_tasks<F: Fn(usize) + Sync>(pool: &Pool, tasks: usize, track: &Track, task: F) {
    enter(pool, track, || join_tree(0, tasks, &task));
}

/// Runs `task(lo..hi)` as a balanced join tree. Must run on a pool worker.
#[cfg(feature = "parallel")]
fn join_tree<F: Fn(usize) + Sync>(lo: usize, hi: usize, task: &F) {
    if hi - lo == 1 {
        task(lo);
    } else {
        let mid = lo + (hi - lo) / 2;
        rayon::join(|| join_tree(lo, mid, task), || join_tree(mid, hi, task));
    }
}

#[cfg(not(feature = "parallel"))]
pub(crate) fn run_tasks<F: Fn(usize) + Sync>(_pool: &Pool, tasks: usize, _track: &Track, task: F) {
    for k in 0..tasks {
        task(k);
    }
}

/// Runs `f` on a worker of `pool` and returns its result.
///
/// * Caller is a worker of `pool`: `f` runs on the caller. Nothing is entered.
/// * Caller is a worker of another Rayon pool: [`submit_wait`]; the caller
///   blocks without running its own pool's jobs.
/// * Caller is a plain thread: one `ThreadPool::install` (a stack job); the
///   caller sleeps until it finishes.
///
/// The last two leave the calling thread and count one entry on `track`.
#[cfg(feature = "parallel")]
pub(crate) fn enter<R: Send>(pool: &Pool, track: &Track, f: impl FnOnce() -> R + Send) -> R {
    if in_pool(pool) {
        f()
    } else if rayon::current_thread_index().is_some() {
        track.note_install();
        submit_wait(pool, f)
    } else {
        track.note_install();
        pool.inner().install(f)
    }
}

/// Runs `f` on a worker of `pool` for a caller that is a worker of another
/// Rayon pool, and blocks the caller on a [`Latch`] until `f` has returned or
/// panicked; a panic is re-raised on the caller.
///
/// `ThreadPool::install` would wait inside Rayon, which runs the caller's own
/// pool's jobs on the caller's stack meanwhile, so an outer `par_iter` whose
/// items each enter `pool` would nest until the stack overflows
/// (https://github.com/tensor4all/threadteam-rs/issues/89). Rayon offers no
/// entry whose wait runs nothing and that borrows the caller's stack: every
/// `ThreadPool` entry that waits for its work waits that way, and the spawn
/// entries, which do not wait, take only `'static` jobs. So the job is spawned with the borrow erased to a raw pointer, and the
/// caller waits for it here. `spawn` boxes the job: one allocation per call.
///
/// Because the caller runs nothing while it waits, `f` must not wait, directly
/// or through another thread, for a job of the caller's pool (the precondition
/// on [`Exec`](crate::Exec)'s placement).
#[cfg(feature = "parallel")]
fn submit_wait<F: FnOnce() -> R + Send, R: Send>(pool: &Pool, f: F) -> R {
    let submit = Submit {
        f: UnsafeCell::new(Some(f)),
        out: UnsafeCell::new(None),
        latch: Latch::new(),
    };
    let job = Job {
        submit: (&submit as *const Submit<F, R>).cast(),
        run: run::<F, R>,
        waiter: thread::current(),
    };
    // From here until the wait returns, the job may use `submit`; leaving this
    // frame by unwinding would free it under the job, so it aborts instead.
    let abort = AbortOnUnwind;
    pool.inner().spawn(move || {
        let job = job;
        // SAFETY: `job.submit` points to the `Submit<F, R>` of the frame above
        // (the type `job.run` was instantiated with), which stays alive, with
        // nothing of it touched but the latch (whose flag and owner `wait`
        // only reads), until `latch.wait()` below returns; that happens only
        // after `run` set the latch, its last access to the frame. Nothing in
        // between can unwind out of the frame (`abort`). `job.waiter` is that
        // frame's thread.
        unsafe { (job.run)(job.submit, job.waiter) }
    });
    submit.latch.wait();
    std::mem::forget(abort);
    match submit.out.into_inner() {
        Some(Ok(value)) => value,
        Some(Err(payload)) => resume_unwind(payload),
        None => unreachable!("the job sets the latch only after it stored its result"),
    }
}

/// The caller's side of a [`submit_wait`]: the work, the slot for its result,
/// and the latch the caller waits on. It lives in the caller's frame.
#[cfg(feature = "parallel")]
struct Submit<F, R> {
    f: UnsafeCell<Option<F>>,
    out: UnsafeCell<Option<std::thread::Result<R>>>,
    latch: Latch,
}

/// The spawned job of a [`submit_wait`], with the types of its `Submit` erased
/// so that it is `'static`.
#[cfg(feature = "parallel")]
struct Job {
    submit: *const (),
    run: unsafe fn(*const (), Thread),
    waiter: Thread,
}

// SAFETY: `Job` hands the job thread a pointer to a `Submit<F, R>` whose `F`
// and `R` are `Send` (bounds of `submit_wait`); the job takes `F`, stores an
// `R` (or a panic payload, which is `Send`) and sets the latch, while the
// owning thread touches nothing of the `Submit` but the latch, whose fields it
// only reads, until the latch is set.
#[cfg(feature = "parallel")]
unsafe impl Send for Job {}

/// Runs the work of a [`Submit`], stores its result or its panic, then sets the
/// latch.
///
/// # Safety
///
/// `submit` points to a live `Submit<F, R>` whose owner waits on its latch and
/// touches nothing else of it until the latch is set; `waiter` is the owner's
/// thread. Called once per `Submit`.
#[cfg(feature = "parallel")]
unsafe fn run<F: FnOnce() -> R + Send, R: Send>(submit: *const (), waiter: Thread) {
    let submit = submit.cast::<Submit<F, R>>();
    let f = (*(*submit).f.get()).take();
    let result = catch_unwind(AssertUnwindSafe(move || {
        (f.expect("a submission runs once"))()
    }));
    *(*submit).out.get() = Some(result);
    Latch::set(std::ptr::addr_of!((*submit).latch), waiter);
}

/// Aborts the process when dropped; forgotten once the job no longer uses the
/// caller's frame.
#[cfg(feature = "parallel")]
struct AbortOnUnwind;

#[cfg(feature = "parallel")]
impl Drop for AbortOnUnwind {
    fn drop(&mut self) {
        std::process::abort();
    }
}

/// Runs `f(i)` for every worker index `i < budget` at once, or reports that the
/// caller already is a worker of `pool`.
///
/// The wait is Rayon's own: a caller that is a worker of another pool runs its
/// own pool's jobs while it waits
/// (https://github.com/tensor4all/threadteam-rs/issues/90).
#[cfg(feature = "parallel")]
pub(crate) fn broadcast(pool: &Pool, track: &Track, budget: usize, f: impl Fn(usize) + Sync) {
    // No gate is needed: rayon pushes the per-worker jobs of one broadcast
    // under a single lock, so every worker sees concurrent broadcasts in the
    // same order and two of them cannot each hold part of the workers. A gate
    // would deadlock a foreign-pool caller that steals a second broadcast onto
    // its own stack while waiting.
    track.note_install();
    pool.inner().broadcast(|ctx| {
        let i = ctx.index();
        if i < budget {
            let _live = track.task();
            f(i);
        }
    });
}
