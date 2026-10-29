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
/// stack, so a steady-state call allocates nothing.
///
/// * Caller is a worker of `pool`: the tree runs directly on the caller.
///   Nothing is installed.
/// * Caller is a plain thread or a worker of another pool: one
///   `ThreadPool::install` (a stack job) around the tree; every task runs on a
///   worker of `pool` and the caller waits.
///
/// `rayon::join` is only reached on a worker of `pool` (inside `install`, or
/// when the caller already is one), so it never touches the global registry.
#[cfg(feature = "parallel")]
pub(crate) fn run_tasks<F: Fn(usize) + Sync>(pool: &Pool, tasks: usize, track: &Track, task: F) {
    if in_pool(pool) {
        join_tree(0, tasks, &task);
    } else {
        track.note_install();
        pool.inner().install(|| join_tree(0, tasks, &task));
    }
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

/// Runs `f` on a worker of `pool`, installing only when the caller is not one.
#[cfg(feature = "parallel")]
pub(crate) fn enter<R: Send>(pool: &Pool, track: &Track, f: impl FnOnce() -> R + Send) -> R {
    if in_pool(pool) {
        f()
    } else {
        track.note_install();
        pool.inner().install(f)
    }
}

/// Runs `f(i)` for every worker index `i < budget` at once, or reports that the
/// caller already is a worker of `pool`.
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
