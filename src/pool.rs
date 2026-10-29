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

/// Whether a pool pins the members of a root team scope by default.
///
/// The one switch for the default of [`PoolBuilder::pin_participants`]: off, as
/// in every compared runtime (libgomp, libomp, OpenBLAS, faer) and as measured
/// on broccoli, where the slow mode of tensor4all/faer#107 disappears only with
/// member placement (arm C: 0 of 20 processes slow, 0 % stacked samples) while it
/// persists with a free CPU (arm B: 7 of 20). Turning it on by default would take
/// CPUs away from other users of the process, fight the caller's `taskset`, and
/// confine any vendor thread team a member creates, so the decision stays per
/// pool.
const PIN_PARTICIPANTS_DEFAULT: bool = false;

/// Default spin budget of a wait, in polls (one PAUSE, or YIELD on aarch64,
/// per poll); see `docs/design/waiting.md`. 300000 on x86-64, libgomp's default
/// `GOMP_SPINCOUNT` ("about 3 ms"; about 7 ms at the 24 ns per poll measured on
/// Zen 3, EPYC 7713P; longer on Intel cores whose PAUSE is slower). With it a
/// team called again 10 ms later on a quiet host starts as fast as 0.1.3
/// (evo); 12288 polls (about 300 µs) made that first section 10.8x slower,
/// because the members had gone to sleep. On aarch64 a YIELD is much cheaper
/// than a PAUSE; 2^21 aims at the same order of time and is not measured yet.
const SPIN_POLLS_DEFAULT: u32 = if cfg!(target_arch = "aarch64") {
    1 << 21
} else {
    300_000
};

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
    /// The leased workers (`docs/design/own-threads.md`).
    #[cfg(feature = "parallel")]
    workers: Vec<Arc<crate::worker::WorkerCtl>>,
    /// Join handles of dedicated workers (empty for shared ones).
    #[cfg(feature = "parallel")]
    joins: std::sync::Mutex<Vec<std::thread::JoinHandle<()>>>,
    pub(crate) width: NonZeroUsize,
    pub(crate) awake: Arc<Awake>,
    /// The private Rayon pool of `Exec::external_rayon` (feature `rayon`).
    #[cfg(feature = "rayon")]
    rayon: std::sync::OnceLock<rayon::ThreadPool>,
}

#[cfg(feature = "rayon")]
impl Pool {
    /// This pool's private Rayon pool, built on first use.
    pub(crate) fn rayon(&self) -> &rayon::ThreadPool {
        self.rayon.get_or_init(|| {
            rayon::ThreadPoolBuilder::new()
                .num_threads(self.width.get())
                .build()
                .expect("a private Rayon pool for external_rayon")
        })
    }
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

    /// Asks idle team members to stop spinning and sleep now.
    ///
    /// Members that are idle sleep at once; members that are reserved stop
    /// spinning after their reservation ends, so this never revokes a live
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

    /// Whether the pool pins the **members** of a root team scope (see
    /// [`PoolBuilder::pin_participants`]); the leader is never pinned.
    ///
    /// # Examples
    ///
    /// ```
    /// let pool = threadteam::Pool::builder().threads(1).build()?;
    /// assert!(!pool.pin_participants()); // off by default
    /// # Ok::<(), threadteam::Error>(())
    /// ```
    pub fn pin_participants(&self) -> bool {
        self.awake.pin
    }

    /// The spin budget of this pool's team waits, in polls
    /// ([`PoolBuilder::spin_polls`]).
    pub fn spin_polls(&self) -> u32 {
        self.awake.waits.spin_polls
    }

    /// How often this pool's waits slept since the pool was built: diagnostics
    /// for the wait benchmarks (`docs/design/waiting.md`).
    #[doc(hidden)]
    pub fn wait_counters(&self) -> WaitCounters {
        use core::sync::atomic::Ordering::Relaxed;
        let w = &self.awake.waits;
        WaitCounters {
            join_sleeps: w.join_sleeps.load(Relaxed),
            barrier_sleeps: w.barrier_sleeps.load(Relaxed),
            member_parks: w.member_parks.load(Relaxed),
        }
    }

    /// Number of workers currently kept awake as team members, reserved or
    /// idle and still awake. Zero once released members went to sleep. A diagnostic: the value
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
        self.awake.shutdown.store(true, Ordering::Release);
        #[cfg(feature = "parallel")]
        {
            // Detach every worker, then wait for their acknowledgements: after
            // its acknowledgement a worker touches nothing of this pool. A
            // worker of this pool that drops it (its own task owned the pool)
            // acknowledges once that task returns; nobody waits for it.
            for s in &*self.awake.slots {
                s.state.detach();
                s.park.wake();
            }
            let me = self.awake.slot_of_current();
            for (i, s) in self.awake.slots.iter().enumerate() {
                // A worker that has not attached yet (a dedicated one still in
                // its start hook) attaches, sees `DETACH` and acknowledges too.
                if Some(i) == me {
                    continue;
                }
                self.awake
                    .waits
                    .sync
                    .park_while(|| !s.acked.load(Ordering::SeqCst));
            }
            let joins = std::mem::take(&mut *self.joins.lock().unwrap_or_else(|e| e.into_inner()));
            let my_thread = std::thread::current().id();
            for j in joins {
                if j.thread().id() != my_thread {
                    // A worker that panicked has still exited.
                    let _ = j.join();
                }
            }
            drop(core::mem::take(&mut self.workers));
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
    pin_participants: bool,
    spin_polls: u32,
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
            pin_participants: PIN_PARTICIPANTS_DEFAULT,
            spin_polls: SPIN_POLLS_DEFAULT,
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

    /// Default idle spin of an unreserved team member before it sleeps.
    /// Defaults to 200 ms; zero puts it to sleep immediately. Any nonzero value
    /// lets it spin for the pool's spin budget ([`spin_polls`](Self::spin_polls)).
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

    /// **No effect** since threadteam runs its own worker threads
    /// (`docs/design/own-threads.md`): an idle worker always serves the pool's
    /// queued tasks. Kept so that existing code compiles.
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
    /// use threadteam::{DispatchCosts, DispatchKind, Exec, HostClass, Pool, WidthCosts};
    ///
    /// let costs = DispatchCosts::for_class(HostClass::Generic)
    ///     .with(DispatchKind::Outside, WidthCosts::flat(Duration::from_micros(7)));
    /// let pool = Pool::builder().threads(2).dispatch_costs(costs).build()?;
    /// let exec = Exec::new(&pool, 2)?;
    /// // Without the `parallel` feature the pool has width 1: no section, no cost.
    /// let expected = if pool.width().get() >= 2 { Duration::from_micros(7) } else { Duration::MAX };
    /// assert_eq!(exec.dispatch_cost(NonZeroUsize::new(2).unwrap()), expected);
    /// # Ok::<(), threadteam::Error>(())
    /// ```
    #[must_use]
    pub fn dispatch_costs(mut self, costs: DispatchCosts) -> Self {
        self.costs = costs;
        self
    }

    /// Pins the **members** of a root team scope onto the calling thread's CPU
    /// set. Off by default; Linux only, where it does something.
    ///
    /// With it on, the leader of a root team scope
    /// ([`Exec::team`](crate::Exec::team), or the root token of a
    /// [`Team`](crate::Team)) reads **its own** affinity mask `S` with
    /// `sched_getaffinity`, and every member pins itself with
    /// `sched_setaffinity` onto one CPU of `S`: member `k` (participant `k`,
    /// counting the leader as 0) onto the `k`-th set CPU of `S`, wrapping modulo
    /// `|S|` when the team is wider than the set. `S[0]` is thereby left to the
    /// leader, without enforcing it.
    ///
    /// **The leader is never pinned.** Its mask is never changed, not even for
    /// the scope: it keeps all of `S` while the members work. A vendor library
    /// (OpenMP, OpenBLAS, a LAPACK provider) creates its thread team from the
    /// thread that calls it, and a team created from a one-CPU thread silently
    /// confines the whole provider team to that CPU — measured at 3-20x slower
    /// dgemm and LU at 4 threads. Engines call such libraries from the
    /// coordinating thread, so that thread must keep its CPU set.
    ///
    /// A member saves its previous mask in its own slot and restores it
    /// *itself* before it stops being reserved: before it serves the pool's
    /// queue again, before it leaves the pool, and
    /// before its thread serves work that is not its team's — while it is lent
    /// by [`Exec::external`](crate::Exec::external), and when it runs such work
    /// itself (a member that leads a `fork` child and calls `external`). No
    /// thread ever changes another thread's affinity.
    ///
    /// # What it is for
    ///
    /// It keeps the members of a team apart. Measured on broccoli (16 threads on
    /// 16 allowed CPUs, 20 processes, tensor4all/faer#107): with the scheduler
    /// free to place the threads, 11 of 20 processes ran a `qr_factor` 512 at
    /// the one-thread time, with 92 % of the samples showing two runnable team
    /// threads on one CPU; with the participants pinned, 0 of 20 did, and no
    /// sample was stacked. Giving the team one spare CPU did not help (7 of 20
    /// slow). That arm pinned every participant including the leader; this crate
    /// leaves the leader on the caller's whole set, which is what the rule above
    /// requires and only relaxes the placement of participant 0.
    ///
    /// # Assumptions and limits
    ///
    /// * **Linux only.** The CPU set is a stack `cpu_set_t` and the placement is
    ///   `sched_setaffinity(0, ...)`. On every other target the option is
    ///   accepted and does nothing.
    /// * **The caller's CPU set is the source of placement**, so the caller's
    ///   mask should belong to this team alone. Two teams whose leaders see the
    ///   same mask place their members on the same CPUs; give each team (in
    ///   practice: each process) its own CPU set (`taskset -c`).
    /// * **Best effort.** A refused syscall (a mask larger than `cpu_set_t`, a
    ///   cpuset that changed under us, a CPU offlined) leaves that thread where
    ///   it was; an empty mask pins nobody; a failed pin or restore never fails
    ///   the scope.
    /// * **A pinned member must not create a foreign thread team.** A member
    ///   that calls OpenMP, OpenBLAS or any library that builds its team from
    ///   the calling thread *directly inside a share* confines that library's
    ///   team to one CPU. Call it through
    ///   [`Exec::external`](crate::Exec::external), which restores the member's
    ///   mask first, or keep this option off.
    /// * Only the members of a team are placed. A token of a
    ///   [`Team`](crate::Team) that never opens a team scope
    ///   (`exec.for_ranges(...)` directly) is not re-pinned; its members keep
    ///   the placement of the last root scope (or of `Pool::into_team`).
    ///
    /// Off by default, consistently with the rule above and with libgomp,
    /// libomp, OpenBLAS and faer, which leave threads unpinned unless asked.
    ///
    /// # Examples
    ///
    /// ```
    /// let pool = threadteam::Pool::builder().threads(2).pin_participants(true).build()?;
    /// assert!(pool.pin_participants());
    /// # let _ = pool;
    /// # Ok::<(), threadteam::Error>(())
    /// ```
    #[must_use]
    pub fn pin_participants(mut self, on: bool) -> Self {
        self.pin_participants = on;
        self
    }

    /// Spin budget of a team wait, in polls of one PAUSE (YIELD on aarch64):
    /// how long a reserved member waiting for work, a leader in a section's
    /// join, or a participant in a barrier spins before it yields a few times
    /// and then sleeps until woken (`docs/design/waiting.md`; the shape of
    /// libgomp's `GOMP_SPINCOUNT`). A count, so a waiter that is descheduled
    /// does not spend it while it waits for its CPU. Zero sleeps after the
    /// yields. An oversubscribed scope uses at most 100 polls. It also bounds
    /// how long an idle worker keeps polling before it sleeps.
    ///
    /// # Examples
    ///
    /// ```
    /// let pool = threadteam::Pool::builder().threads(2).spin_polls(4096).build()?;
    /// assert_eq!(pool.spin_polls(), 4096);
    /// # Ok::<(), threadteam::Error>(())
    /// ```
    #[must_use]
    pub fn spin_polls(mut self, polls: u32) -> Self {
        self.spin_polls = polls;
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
        // Workers of the reserve have the default stack, the platform's QoS
        // and no hooks; any other setting gets dedicated workers.
        let shared = self.thread_name.is_none()
            && self.on_start.is_none()
            && self.stack_size == DEFAULT_STACK_BYTES
            && self.qos.is_none();
        let on_start: Option<Arc<dyn Fn(usize) + Send + Sync>> = self.on_start.map(Arc::from);
        let how = crate::worker::Spawn {
            stack: self.stack_size,
            name: self.thread_name.as_deref(),
            on_start,
            qos: self.qos,
            shared,
        };
        let (workers, joins) =
            crate::worker::obtain(n, &how).map_err(|e| Error::PoolBuild(e.to_string()))?;
        let width = NonZeroUsize::new(n).unwrap_or(NonZeroUsize::MIN);
        let awake = Arc::new(Awake::new(
            width.get(),
            self.wait,
            self.acquire_timeout,
            self.costs,
            self.pin_participants,
            self.spin_polls,
        ));
        // A worker takes the caller's CPU set, as a thread it spawned would.
        let mask = crate::pin::read_here();
        for (id, w) in workers.iter().enumerate() {
            w.assign(crate::worker::Lease {
                aw: Arc::clone(&awake),
                id,
                mask,
            });
        }
        // Workers attach on their own (after a start hook, for dedicated
        // ones); until then a slot is not claimable by a team, and queued work
        // waits for them.
        Ok(Pool {
            workers,
            joins: std::sync::Mutex::new(joins),
            width,
            awake,
            #[cfg(feature = "rayon")]
            rayon: std::sync::OnceLock::new(),
        })
    }

    #[cfg(not(feature = "parallel"))]
    fn build_checked(self) -> Result<Pool, Error> {
        Ok(Pool {
            width: NonZeroUsize::MIN,
            awake: Arc::new(Awake::new(
                1,
                self.wait,
                self.acquire_timeout,
                self.costs,
                self.pin_participants,
                self.spin_polls,
            )),
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
            .field("pin_participants", &self.pin_participants)
            .finish_non_exhaustive()
    }
}

#[cfg(all(test, feature = "parallel"))]
mod tests {
    use super::*;
    use std::sync::{mpsc, Arc};
    use std::time::Duration;

    /// The last reference is dropped on a worker of the pool itself: the drop
    /// must not wait for its own thread (it detaches the others, waits for
    /// their acknowledgements and skips itself; the worker acknowledges and
    /// leaves after its task). Public entries always borrow the pool while
    /// their tasks run, so the task here is published on the pool's queue
    /// directly, as the Rayon version used `spawn`.
    #[test]
    fn drop_from_a_worker_of_the_pool_does_not_hang() {
        for shared in [true, false] {
            let mut b = Pool::builder().threads(2);
            if !shared {
                b = b.thread_name(|i| format!("self-drop-{i}"));
            }
            let pool = Arc::new(b.build().unwrap());
            let aw = Arc::clone(&pool.awake);
            let last = std::sync::Mutex::new(Some(Arc::clone(&pool)));
            let task = |_: usize| {
                // The last reference: runs `Pool::drop` on this worker.
                drop(last.lock().unwrap().take());
            };
            unsafe fn call<F: Fn(usize)>(data: *const (), i: usize) {
                (*data.cast::<F>())(i)
            }
            fn call_of<F: Fn(usize)>(_: &F) -> unsafe fn(*const (), usize) {
                call::<F>
            }
            // SAFETY: `task` and the waker (held by `aw`) outlive the batch,
            // which this test waits for before they go out of scope.
            let batch = unsafe {
                crate::park::Batch::new(
                    call_of(&task),
                    (&task as *const _ as *const ()).cast(),
                    1,
                    &aw.waits.sync,
                )
            };
            aw.queue.publish(&batch);
            aw.wake_for_work();
            drop(pool);
            let (tx, rx) = mpsc::channel::<()>();
            std::thread::scope(|s| {
                s.spawn(|| {
                    let mut w = crate::park::Waiter::new(0);
                    batch.done.wait(Some(&aw.waits.sync), &mut w);
                    tx.send(()).unwrap();
                });
                rx.recv_timeout(Duration::from_secs(10))
                    .expect("drop on a worker hung");
            });
            assert!(
                last.lock().unwrap().is_none(),
                "the pool was dropped by its worker"
            );
        }
    }

    /// A pool built, used and dropped inside a task of another pool: the drop
    /// waits for the inner pool's workers only, never for the outer worker it
    /// runs on (a pool cannot be dropped from one of its own workers: every
    /// task's caller borrows the pool).
    #[test]
    fn a_pool_dropped_inside_a_task_of_another_pool_does_not_hang() {
        let outer = Pool::builder().threads(2).build().unwrap();
        let (done_tx, done_rx) = mpsc::channel::<()>();
        let done_tx = std::sync::Mutex::new(Some(done_tx));
        crate::Exec::new(&outer, 2).unwrap().fork(2, |i, _| {
            if i == 0 {
                let inner = Arc::new(Pool::builder().threads(2).build().unwrap());
                crate::Exec::new(&inner, 2).unwrap().fork(2, |_, _| {});
                drop(inner);
                if let Some(tx) = done_tx.lock().unwrap().take() {
                    tx.send(()).unwrap();
                }
            }
        });
        done_rx
            .recv_timeout(Duration::from_secs(10))
            .expect("a drop inside another pool's task hung");
    }
}

/// Counters of [`Pool::wait_counters`] (a diagnostic for the wait benchmarks).
#[doc(hidden)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct WaitCounters {
    /// Leaders that slept in a section's join.
    pub join_sleeps: u64,
    /// Participants that slept in a barrier.
    pub barrier_sleeps: u64,
    /// Reserved members that parked waiting for work.
    pub member_parks: u64,
}
