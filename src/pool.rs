//! The pool type and its builder.

use crate::awake::Awake;
use crate::cost::DispatchCosts;
use crate::error::Error;
use crate::policy::{Qos, WaitPolicy};
use core::fmt;
use core::num::NonZeroUsize;
use core::sync::atomic::Ordering;
use core::time::Duration;
use std::sync::Arc;

/// Default worker stack size: 16 MiB.
///
/// Sized for the recursive LU and divide-and-conquer providers that run on pool
/// workers; the same value tenferro-rs used for its own pool preset.
const DEFAULT_STACK_BYTES: usize = 16 << 20;

/// Default time an acquisition waits for idle workers to wake: 2 ms.
const DEFAULT_ACQUIRE_TIMEOUT: Duration = Duration::from_millis(2);

/// A fixed set of worker threads that [`Exec`](crate::Exec) tokens place work on.
///
/// `Pool` is the only way to make a pool. It wraps the scheduler privately: no
/// scheduler type appears in any signature, there is no constructor from an
/// external pool and no accessor returning one. The pool's owner keeps the
/// `Pool` and mints tokens from it with [`Exec::new`](crate::Exec::new).
///
/// A `Pool` is a unique owner: it is not `Clone` and no method returns a second
/// handle. An engine that wants reserved, always-awake members moves the pool
/// into a [`Team`](crate::Team) with [`into_team`](Pool::into_team); the team owns
/// the pool from then on (and gives it back with
/// [`Team::into_pool`](crate::Team::into_pool)). Sharing a pool between several
/// callers stays outside this crate (for example behind an `Arc` held by the
/// caller), where each call acquires its members with
/// [`Exec::team`](crate::Exec::team) or uses plain tokens.
///
/// Dropping a `Pool` (or the `Team` that owns it) shuts its workers down and
/// joins them: when `drop` returns, every worker thread has exited, so the
/// thread-local destructors of the workers (including state set up by an
/// [`on_start`](PoolBuilder::on_start) hook) have already run. Without the
/// `parallel` feature there are no workers and nothing is joined.
///
/// A pool dropped from one of its own workers cannot wait for itself: that drop
/// signals shutdown and returns without joining, so the workers exit shortly
/// after instead of before the return. Drop the pool from a plain thread when
/// the guarantee matters.
///
/// # Examples
///
/// ```
/// use threadteam::Pool;
///
/// let pool = Pool::builder().threads(2).thread_name(|i| format!("w{i}")).build()?;
/// assert!(pool.width().get() <= 2);
/// # Ok::<(), threadteam::Error>(())
/// ```
pub struct Pool {
    // Field order matters: `inner` is dropped (shutdown signalled) before
    // `joiner` waits for the threads.
    #[cfg(feature = "parallel")]
    pub(crate) inner: rayon::ThreadPool,
    #[cfg(feature = "parallel")]
    _joiner: Joiner,
    pub(crate) width: NonZeroUsize,
    pub(crate) awake: Arc<Awake>,
}

impl Pool {
    /// Starts configuring a pool.
    ///
    /// # Examples
    ///
    /// ```
    /// let pool = threadteam::Pool::builder().threads(1).build()?;
    /// assert_eq!(pool.width().get(), 1);
    /// # Ok::<(), threadteam::Error>(())
    /// ```
    pub fn builder() -> PoolBuilder {
        PoolBuilder::new()
    }

    /// Number of worker threads, the largest budget a token can have.
    ///
    /// Without the `parallel` feature this is always 1.
    ///
    /// # Examples
    ///
    /// ```
    /// let pool = threadteam::Pool::builder().threads(1).build()?;
    /// assert_eq!(pool.width().get(), 1);
    /// # Ok::<(), threadteam::Error>(())
    /// ```
    pub fn width(&self) -> NonZeroUsize {
        self.width
    }

    /// The pool-wide default [`WaitPolicy`] of idle team members.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::time::Duration;
    ///
    /// let pool = threadteam::Pool::builder()
    ///     .threads(1)
    ///     .idle_spin(Duration::from_millis(5))
    ///     .build()?;
    /// assert_eq!(pool.wait_policy().idle_spin(), Duration::from_millis(5));
    /// # Ok::<(), threadteam::Error>(())
    /// ```
    pub fn wait_policy(&self) -> WaitPolicy {
        self.awake.defaults
    }

    /// Asks idle team members to return their threads to Rayon now.
    ///
    /// Members that are idle leave at once; members that are reserved leave
    /// when their reservation ends, so this never revokes a live
    /// [`TeamExec`](crate::TeamExec). The call does not wait for anything,
    /// so it is safe to make from inside a reservation. A team acquired later
    /// starts its members afresh.
    ///
    /// # Examples
    ///
    /// ```
    /// let pool = threadteam::Pool::builder().threads(2).build()?;
    /// pool.release_idle(); // nothing resident: no effect
    /// # Ok::<(), threadteam::Error>(())
    /// ```
    pub fn release_idle(&self) {
        self.awake.release_epoch.fetch_add(1, Ordering::AcqRel);
    }

    /// The [`DispatchCosts`] of this pool's tokens.
    ///
    /// # Examples
    ///
    /// ```
    /// use threadteam::{DispatchCosts, Pool};
    /// let pool = Pool::builder().threads(1).build()?;
    /// assert_eq!(pool.dispatch_costs(), DispatchCosts::host_default());
    /// # Ok::<(), threadteam::Error>(())
    /// ```
    pub fn dispatch_costs(&self) -> DispatchCosts {
        self.awake.costs
    }

    /// Number of workers currently kept awake as team members, reserved or
    /// idle. Zero once idle members returned to Rayon. A diagnostic: the value
    /// can change at any moment.
    ///
    /// # Examples
    ///
    /// ```
    /// let pool = threadteam::Pool::builder().threads(2).build()?;
    /// assert_eq!(pool.resident_members(), 0);
    /// # Ok::<(), threadteam::Error>(())
    /// ```
    pub fn resident_members(&self) -> usize {
        self.awake.resident()
    }
}

impl Drop for Pool {
    fn drop(&mut self) {
        // Idle members leave their loops so that the workers can be joined.
        self.awake.shutdown.store(true, Ordering::Release);
    }
}

/// The worker threads of a pool, joined when it is dropped.
#[cfg(feature = "parallel")]
struct Joiner {
    handles: std::sync::Arc<std::sync::Mutex<Vec<std::thread::JoinHandle<()>>>>,
}

#[cfg(feature = "parallel")]
impl Drop for Joiner {
    fn drop(&mut self) {
        let handles = match self.handles.lock() {
            Ok(mut guard) => std::mem::take(&mut *guard),
            Err(poisoned) => std::mem::take(&mut *poisoned.into_inner()),
        };
        let me = std::thread::current().id();
        // A worker of this pool cannot join itself: skip joining entirely.
        if handles.iter().any(|h| h.thread().id() == me) {
            return;
        }
        for handle in handles {
            // A worker that panicked has still exited; nothing to report.
            let _ = handle.join();
        }
    }
}

impl fmt::Debug for Pool {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Pool").field("width", &self.width).finish()
    }
}

/// Configuration of a [`Pool`].
///
/// # Examples
///
/// ```
/// use threadteam::Pool;
///
/// let pool = Pool::builder()
///     .threads(2)
///     .stack_size(8 << 20)
///     .on_start(|_worker| { /* e.g. pin the thread */ })
///     .build()?;
/// assert!(pool.width().get() >= 1);
/// # Ok::<(), threadteam::Error>(())
/// ```
#[cfg_attr(not(feature = "parallel"), allow(dead_code))]
pub struct PoolBuilder {
    threads: Option<usize>,
    stack_size: usize,
    thread_name: Option<Box<dyn Fn(usize) -> String>>,
    on_start: Option<Box<dyn Fn(usize) + Send + Sync>>,
    wait: WaitPolicy,
    acquire_timeout: Duration,
    qos: Option<Qos>,
    costs: DispatchCosts,
}

impl PoolBuilder {
    fn new() -> Self {
        PoolBuilder {
            threads: None,
            stack_size: DEFAULT_STACK_BYTES,
            thread_name: None,
            on_start: None,
            wait: WaitPolicy::DEFAULT,
            acquire_timeout: DEFAULT_ACQUIRE_TIMEOUT,
            qos: None,
            costs: DispatchCosts::host_default(),
        }
    }

    /// Number of worker threads. Defaults to the available parallelism.
    ///
    /// Zero is rejected by [`build`](Self::build); the pool never falls back to
    /// an environment-dependent default.
    ///
    /// # Examples
    ///
    /// ```
    /// use threadteam::{Error, Pool};
    ///
    /// assert_eq!(Pool::builder().threads(0).build().unwrap_err(), Error::ZeroThreads);
    /// ```
    #[must_use]
    pub fn threads(mut self, n: usize) -> Self {
        self.threads = Some(n);
        self
    }

    /// Names the workers; the closure receives the worker index.
    ///
    /// # Examples
    ///
    /// ```
    /// let pool = threadteam::Pool::builder()
    ///     .threads(1)
    ///     .thread_name(|i| format!("my-pool-{i}"))
    ///     .build()?;
    /// # let _ = pool;
    /// # Ok::<(), threadteam::Error>(())
    /// ```
    #[must_use]
    pub fn thread_name(mut self, f: impl Fn(usize) -> String + 'static) -> Self {
        self.thread_name = Some(Box::new(f));
        self
    }

    /// Worker stack size in bytes. Defaults to 16 MiB, sized for recursive LU.
    ///
    /// # Examples
    ///
    /// ```
    /// let pool = threadteam::Pool::builder().threads(1).stack_size(4 << 20).build()?;
    /// # let _ = pool;
    /// # Ok::<(), threadteam::Error>(())
    /// ```
    #[must_use]
    pub fn stack_size(mut self, bytes: usize) -> Self {
        self.stack_size = bytes;
        self
    }

    /// Hook run on each worker when it starts, with the worker index.
    ///
    /// Typical use is pinning the worker to a core. The hook is not called
    /// without the `parallel` feature, which starts no workers.
    ///
    /// # Examples
    ///
    /// ```
    /// let pool = threadteam::Pool::builder().threads(1).on_start(|_| {}).build()?;
    /// # let _ = pool;
    /// # Ok::<(), threadteam::Error>(())
    /// ```
    #[must_use]
    pub fn on_start(mut self, f: impl Fn(usize) + Send + Sync + 'static) -> Self {
        self.on_start = Some(Box::new(f));
        self
    }

    /// Default spin time of an unreserved idle team member before it returns
    /// its thread to Rayon. Defaults to 200 ms; zero returns it immediately.
    ///
    /// Operations may override it with [`WaitOverride`](crate::WaitOverride).
    /// A reserved member never leaves on a timeout.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::time::Duration;
    ///
    /// let pool = threadteam::Pool::builder().threads(2).idle_spin(Duration::ZERO).build()?;
    /// assert_eq!(pool.wait_policy().idle_spin(), Duration::ZERO);
    /// # Ok::<(), threadteam::Error>(())
    /// ```
    #[must_use]
    pub fn idle_spin(mut self, spin: Duration) -> Self {
        self.wait = WaitPolicy::new(spin, self.wait.idle_runs_rayon());
        self
    }

    /// Default for whether unreserved idle members run Rayon jobs while they
    /// spin. Defaults to `false`: native team execution keeps them for the
    /// next reservation. `true` is the compatibility mode in which an idle
    /// member serves Rayon work (an acquisition then refuses a member that is
    /// inside such a job instead of waiting for it).
    ///
    /// # Examples
    ///
    /// ```
    /// let pool = threadteam::Pool::builder().threads(2).idle_runs_rayon(true).build()?;
    /// assert!(pool.wait_policy().idle_runs_rayon());
    /// # Ok::<(), threadteam::Error>(())
    /// ```
    #[must_use]
    pub fn idle_runs_rayon(mut self, on: bool) -> Self {
        self.wait = WaitPolicy::new(self.wait.idle_spin(), on);
        self
    }

    /// How long [`Exec::team`](crate::Exec::team) waits for idle workers to
    /// wake before it refuses. Defaults to 2 ms.
    ///
    /// This bounds the wait for workers that are *free*; the acquisition does
    /// not wait for a worker that is running a job, it only runs out this
    /// time and refuses. Raise it on a loaded host to reduce spurious refusals.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::time::Duration;
    /// let pool = threadteam::Pool::builder()
    ///     .threads(2)
    ///     .acquire_timeout(Duration::from_millis(10))
    ///     .build()?;
    /// # let _ = pool;
    /// # Ok::<(), threadteam::Error>(())
    /// ```
    #[must_use]
    pub fn acquire_timeout(mut self, timeout: Duration) -> Self {
        self.acquire_timeout = timeout;
        self
    }

    /// Scheduling class of the workers (see [`Qos`]).
    ///
    /// Defaults to [`Qos::UserInitiated`] on macOS and to nothing elsewhere.
    /// The calling thread's class is not changed by this crate; an engine that
    /// wants its leading thread raised too calls
    /// [`raise_current_thread_qos`](crate::raise_current_thread_qos) once on it.
    ///
    /// # Examples
    ///
    /// ```
    /// use threadteam::{Pool, Qos};
    /// let pool = Pool::builder().threads(1).qos(Qos::Default).build()?;
    /// # let _ = pool;
    /// # Ok::<(), threadteam::Error>(())
    /// ```
    #[must_use]
    pub fn qos(mut self, qos: Qos) -> Self {
        self.qos = Some(qos);
        self
    }

    /// Overrides the [`DispatchCosts`] of this pool's tokens.
    ///
    /// Defaults to [`DispatchCosts::host_default`]. The costs feed
    /// [`Exec::dispatch_cost`](crate::Exec::dispatch_cost) and everything built on
    /// it (`pays`, `width_for`); they never change how work is placed.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::num::NonZeroUsize;
    /// use std::time::Duration;
    /// use threadteam::{DispatchCosts, DispatchKind, Exec, Pool, WidthCosts};
    ///
    /// let costs = DispatchCosts::generic()
    ///     .with(DispatchKind::Outside, WidthCosts::flat(Duration::from_micros(7)));
    /// let pool = Pool::builder().threads(2).dispatch_costs(costs).build()?;
    /// let exec = Exec::new(&pool, 2)?;
    /// assert_eq!(exec.dispatch_cost(NonZeroUsize::new(2).unwrap()), Duration::from_micros(7));
    /// # Ok::<(), threadteam::Error>(())
    /// ```
    #[must_use]
    pub fn dispatch_costs(mut self, costs: DispatchCosts) -> Self {
        self.costs = costs;
        self
    }

    /// Starts the workers.
    ///
    /// Without the `parallel` feature no thread is started and the pool has
    /// width 1.
    ///
    /// # Errors
    ///
    /// [`Error::ZeroThreads`] when `threads(0)` was requested, and
    /// [`Error::PoolBuild`] when the workers cannot be started.
    ///
    /// # Examples
    ///
    /// ```
    /// let pool = threadteam::Pool::builder().threads(2).build()?;
    /// assert!(pool.width().get() <= 2);
    /// # Ok::<(), threadteam::Error>(())
    /// ```
    pub fn build(self) -> Result<Pool, Error> {
        if self.threads == Some(0) {
            return Err(Error::ZeroThreads);
        }
        self.build_checked()
    }

    #[cfg(feature = "parallel")]
    fn build_checked(self) -> Result<Pool, Error> {
        let n = self
            .threads
            .unwrap_or_else(|| std::thread::available_parallelism().map_or(1, NonZeroUsize::get));
        let handles = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
        let sink = std::sync::Arc::clone(&handles);
        let qos = self.qos.or(Qos::PLATFORM_DEFAULT);
        let mut builder = rayon::ThreadPoolBuilder::new()
            .num_threads(n)
            .stack_size(self.stack_size)
            .spawn_handler(move |thread| {
                let mut b = std::thread::Builder::new();
                if let Some(name) = thread.name() {
                    b = b.name(name.to_owned());
                }
                if let Some(size) = thread.stack_size() {
                    b = b.stack_size(size);
                }
                let handle = b.spawn(move || {
                    if let Some(q) = qos {
                        q.apply_to_current_thread();
                    }
                    thread.run()
                })?;
                match sink.lock() {
                    Ok(mut guard) => guard.push(handle),
                    Err(poisoned) => poisoned.into_inner().push(handle),
                }
                Ok(())
            });
        if let Some(name) = self.thread_name {
            builder = builder.thread_name(name);
        }
        if let Some(hook) = self.on_start {
            builder = builder.start_handler(hook);
        }
        // On failure `builder` drops its handler, but threads already started
        // are signalled by rayon and detached; join them here for determinism.
        let built = builder.build();
        let inner = match built {
            Ok(inner) => inner,
            Err(e) => {
                drop(Joiner { handles });
                return Err(Error::PoolBuild(e.to_string()));
            }
        };
        let width = NonZeroUsize::new(inner.current_num_threads()).unwrap_or(NonZeroUsize::MIN);
        Ok(Pool {
            inner,
            _joiner: Joiner { handles },
            width,
            awake: Arc::new(Awake::new(
                width.get(),
                self.wait,
                self.acquire_timeout,
                self.costs,
            )),
        })
    }

    #[cfg(not(feature = "parallel"))]
    fn build_checked(self) -> Result<Pool, Error> {
        Ok(Pool {
            width: NonZeroUsize::MIN,
            awake: Arc::new(Awake::new(1, self.wait, self.acquire_timeout, self.costs)),
        })
    }
}

impl fmt::Debug for PoolBuilder {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PoolBuilder")
            .field("threads", &self.threads)
            .field("stack_size", &self.stack_size)
            .field("wait", &self.wait)
            .field("acquire_timeout", &self.acquire_timeout)
            .finish_non_exhaustive()
    }
}

#[cfg(all(test, feature = "parallel"))]
mod tests {
    use super::*;
    use std::sync::{mpsc, Arc};
    use std::time::Duration;

    /// The last reference is dropped on a worker of the pool itself: the drop
    /// must not wait for its own thread.
    #[test]
    fn drop_from_a_worker_of_the_pool_does_not_hang() {
        let pool = Arc::new(Pool::builder().threads(2).build().unwrap());
        let last = Arc::clone(&pool);
        let (gate_tx, gate_rx) = mpsc::channel::<()>();
        let (done_tx, done_rx) = mpsc::channel::<()>();
        pool.inner.spawn(move || {
            gate_rx.recv().unwrap();
            drop(last); // the last reference: runs `Pool::drop` on this worker
            done_tx.send(()).unwrap();
        });
        drop(pool);
        gate_tx.send(()).unwrap();
        done_rx
            .recv_timeout(Duration::from_secs(10))
            .expect("drop on a worker hung");
    }
}
