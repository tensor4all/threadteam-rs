//! The worker threads and the process-wide reserve (`docs/design/own-threads.md`).
//!
//! A worker is a `std::thread` that serves one pool at a time: a [`Lease`]
//! names the pool's shared state and the worker's slot in it. Between leases a
//! *shared* worker waits, asleep, in the process-wide reserve, so that a pool
//! built after another one was dropped (faer's `with_team` builds one per
//! call) reuses running threads instead of spawning fresh ones. A *dedicated*
//! worker (a pool with a thread name, a start hook, a stack size or a QoS of
//! its own) serves one pool and exits; that pool joins it.
//!
//! No thread-local state is used: a worker is found by its `ThreadId`.

use crate::awake::Awake;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex, OnceLock};
use std::thread::ThreadId;

/// Worker threads spawned by this process so far (shared and dedicated).
static SPAWNED: AtomicUsize = AtomicUsize::new(0);

/// How many worker threads this process has spawned so far: a diagnostic, to
/// check that pools built one after another reuse their workers.
#[doc(hidden)]
pub fn spawned_workers() -> usize {
    SPAWNED.load(Ordering::Relaxed)
}

/// What a worker serves: slot `id` of the pool whose shared state is `aw`,
/// with the leasing caller's CPU set (Linux) applied first.
pub(crate) struct Lease {
    pub(crate) aw: Arc<Awake>,
    pub(crate) id: usize,
    pub(crate) mask: Option<crate::pin::CpuMask>,
}

/// What a waiting worker is told next.
enum Next {
    Lease(Lease),
    /// Exit without serving (a dedicated worker of a pool whose build failed).
    Exit,
}

/// One worker thread's handle in this crate.
pub(crate) struct WorkerCtl {
    next: Mutex<Option<Next>>,
    cv: Condvar,
    /// A worker of the reserve (as opposed to a dedicated one).
    shared: bool,
    /// The pool and slot it serves now, for cross-pool lookups.
    current: Mutex<Option<(Arc<Awake>, usize)>>,
    thread: OnceLock<ThreadId>,
}

impl WorkerCtl {
    fn new(shared: bool) -> Self {
        WorkerCtl {
            next: Mutex::new(None),
            cv: Condvar::new(),
            shared,
            current: Mutex::new(None),
            thread: OnceLock::new(),
        }
    }

    /// Hands the worker its next lease.
    pub(crate) fn assign(&self, lease: Lease) {
        let mut next = self.next.lock().unwrap_or_else(|e| e.into_inner());
        *next = Some(Next::Lease(lease));
        drop(next);
        self.cv.notify_one();
    }

    /// Tells an unleased worker to exit.
    fn exit(&self) {
        let mut next = self.next.lock().unwrap_or_else(|e| e.into_inner());
        *next = Some(Next::Exit);
        drop(next);
        self.cv.notify_one();
    }
}

struct Reserve {
    /// Shared workers waiting for a lease.
    idle: Vec<Arc<WorkerCtl>>,
    /// The most idle workers kept: the largest of the available parallelism
    /// and every width leased so far.
    cap: usize,
    /// Every live worker, for finding the calling thread's own lease.
    live: Vec<Arc<WorkerCtl>>,
}

static RESERVE: OnceLock<Mutex<Reserve>> = OnceLock::new();

fn reserve() -> &'static Mutex<Reserve> {
    RESERVE.get_or_init(|| {
        Mutex::new(Reserve {
            idle: Vec::new(),
            cap: std::thread::available_parallelism().map_or(1, |n| n.get()),
            live: Vec::new(),
        })
    })
}

fn lock(r: &Mutex<Reserve>) -> std::sync::MutexGuard<'_, Reserve> {
    r.lock().unwrap_or_else(|e| e.into_inner())
}

/// The pool and slot the calling thread serves, when it is a worker of any
/// pool (the lookup of a cross-pool wait; entry points only).
pub(crate) fn current_worker() -> Option<(Arc<Awake>, usize)> {
    let r = RESERVE.get()?;
    let me = std::thread::current().id();
    let ctl = lock(r)
        .live
        .iter()
        .find(|c| c.thread.get() == Some(&me))
        .cloned()?;
    let cur = ctl.current.lock().unwrap_or_else(|e| e.into_inner());
    cur.clone()
}

/// How a pool's workers are started.
pub(crate) struct Spawn<'a> {
    pub(crate) stack: usize,
    pub(crate) name: Option<&'a dyn Fn(usize) -> String>,
    pub(crate) on_start: Option<Arc<dyn Fn(usize) + Send + Sync>>,
    pub(crate) qos: Option<crate::policy::Qos>,
    /// Whether these settings are the reserve's (default stack, platform
    /// QoS, no hooks): then the workers are shared.
    pub(crate) shared: bool,
}

/// The workers obtained for a pool, and the join handles of dedicated ones.
pub(crate) type Obtained = (Vec<Arc<WorkerCtl>>, Vec<std::thread::JoinHandle<()>>);

/// Workers for a pool of `n`: idle ones of the reserve and freshly spawned
/// ones (shared), or `n` fresh dedicated ones with their join handles.
pub(crate) fn obtain(n: usize, how: &Spawn<'_>) -> std::io::Result<Obtained> {
    let mut ctls = Vec::with_capacity(n);
    let mut joins = Vec::new();
    if how.shared {
        let mut r = lock(reserve());
        r.cap = r.cap.max(n);
        while ctls.len() < n {
            match r.idle.pop() {
                Some(c) => ctls.push(c),
                None => break,
            }
        }
    }
    while ctls.len() < n {
        let index = ctls.len();
        match spawn(index, how) {
            Ok((ctl, handle)) => {
                ctls.push(ctl);
                if !how.shared {
                    joins.push(handle);
                }
            }
            Err(e) => {
                // Hand back what was taken; dedicated workers spawned so far
                // exit without a lease.
                give_back_unleased(ctls, joins);
                return Err(e);
            }
        }
    }
    Ok((ctls, joins))
}

/// Workers obtained for a pool whose build failed: shared ones go back to the
/// reserve, dedicated ones are stopped and joined.
fn give_back_unleased(ctls: Vec<Arc<WorkerCtl>>, joins: Vec<std::thread::JoinHandle<()>>) {
    for c in ctls {
        if c.shared {
            lock(reserve()).idle.push(c);
        } else {
            c.exit();
        }
    }
    for j in joins {
        let _ = j.join();
    }
}

fn spawn(
    index: usize,
    how: &Spawn<'_>,
) -> std::io::Result<(Arc<WorkerCtl>, std::thread::JoinHandle<()>)> {
    let ctl = Arc::new(WorkerCtl::new(how.shared));
    let mut b = std::thread::Builder::new().stack_size(how.stack);
    if let Some(name) = &how.name {
        b = b.name(name(index));
    }
    let qos = how.qos.or(crate::policy::Qos::PLATFORM_DEFAULT);
    let on_start = how.on_start.clone();
    let c = Arc::clone(&ctl);
    let handle = b.spawn(move || {
        if let Some(q) = qos {
            q.apply_to_current_thread();
        }
        let _ = c.thread.set(std::thread::current().id());
        lock(reserve()).live.push(Arc::clone(&c));
        if let Some(hook) = on_start {
            hook(index);
        }
        worker_main(&c);
        let mut r = lock(reserve());
        if let Some(p) = r.live.iter().position(|w| Arc::ptr_eq(w, &c)) {
            r.live.swap_remove(p);
        }
    })?;
    SPAWNED.fetch_add(1, Ordering::Relaxed);
    Ok((ctl, handle))
}

/// A worker's life: wait for a lease, attach, serve the pool until it is
/// detached, acknowledge, then wait in the reserve (shared) or exit.
fn worker_main(ctl: &Arc<WorkerCtl>) {
    loop {
        let lease = {
            let mut next = ctl.next.lock().unwrap_or_else(|e| e.into_inner());
            loop {
                match next.take() {
                    Some(Next::Lease(l)) => break l,
                    Some(Next::Exit) => return,
                    None => next = ctl.cv.wait(next).unwrap_or_else(|e| e.into_inner()),
                }
            }
        };
        let slot = &lease.aw.slots[lease.id];
        // The CPU set a thread spawned by the leasing caller would inherit.
        if let Some(mask) = &lease.mask {
            let _ = crate::pin::set_here(mask);
        }
        let _ = slot.thread.set(std::thread::current().id());
        *ctl.current.lock().unwrap_or_else(|e| e.into_inner()) =
            Some((Arc::clone(&lease.aw), lease.id));
        // `serve` takes its section baseline before it marks the slot
        // attached (claimable): a section published after that is seen.
        crate::awake::serve(&lease.aw, lease.id);
        // Epilogue: no access to the pool after the acknowledgement.
        slot.saved.restore();
        let clean = !slot.saved.is_pinned();
        *ctl.current.lock().unwrap_or_else(|e| e.into_inner()) = None;
        // Back in the reserve before the acknowledgement, so that a pool built
        // right after this one was dropped finds the worker idle. A new lease
        // only starts once this loop waits for it again, after the
        // acknowledgement below.
        let keep = ctl.shared && clean && {
            let mut r = lock(reserve());
            if r.idle.len() < r.cap {
                r.idle.push(Arc::clone(ctl));
                true
            } else {
                false
            }
        };
        slot.acked.store(true, Ordering::SeqCst);
        lease.aw.waits.sync.wake();
        drop(lease);
        if !keep {
            return;
        }
    }
}
