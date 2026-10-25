//! The splittable placement token.

use crate::awake::Awake;
#[cfg(feature = "parallel")]
use crate::awake::LendGuard;
use crate::cost::{pays_at, width_for_by, DispatchKind};
use crate::error::{Error, Refused};
use crate::policy::{WaitOverride, WaitPolicy};
use crate::pool::Pool;
use crate::reserve::LeaderGuard;
use crate::sched;
use crate::split;
use crate::team::{split_raw, spread_raw, Plan, TeamExec};
use crate::track::Track;
use core::fmt;
use core::num::NonZeroUsize;
use core::ops::Range;
use core::sync::atomic::{AtomicUsize, Ordering};
use core::time::Duration;

/// `budget` slots of a [`Pool`], or the calling thread alone.
///
/// An `Exec` is the only way kernels start parallel work. It is a *token*:
///
/// * It is neither `Clone` nor `Copy`, and every parallel entry consumes it, so
///   a kernel cannot run two phases at full width at the same time. Use
///   [`reborrow`](Exec::reborrow) to run several phases in sequence.
/// * Only [`Exec::new`] mints a token, and it needs the `&Pool`; a token never
///   hands its pool back out, so a kernel cannot mint a fresh full-budget token.
/// * [`fork`](Exec::fork) divides the budget among its tasks (shares differ by
///   at most one and sum to the budget), so the number of tasks running at once
///   under one root token is at most its budget at every nesting depth.
///
/// A parent waiting for its children keeps no slot.
///
/// # Placement
///
/// Serial, budget one, or one task: the work runs inline on the calling thread
/// and the scheduler is never called, so no global pool is started. The caller
/// is a worker of the pool: the tasks run as a join tree on the caller, nothing is
/// installed. The caller is a plain thread or a worker of a different pool: the
/// pool is entered once and the caller waits; the tasks run on the target pool.
/// No thread-local state is involved at any point.
///
/// # Examples
///
/// ```
/// use std::sync::atomic::{AtomicUsize, Ordering};
/// use threadteam::{Exec, Pool};
///
/// let pool = Pool::builder().threads(2).acquire_timeout(std::time::Duration::from_secs(10)).build()?;
/// let exec = Exec::new(&pool, 2)?;
/// let sum = AtomicUsize::new(0);
/// exec.for_ranges(100, 1.try_into().unwrap(), |r| {
///     sum.fetch_add(r.sum::<usize>(), Ordering::Relaxed);
/// });
/// assert_eq!(sum.into_inner(), 4950);
/// # Ok::<(), threadteam::Error>(())
/// ```
///
/// A token cannot be duplicated:
///
/// ```compile_fail
/// fn needs_clone<T: Clone>() {}
/// needs_clone::<threadteam::Exec<'static>>();
/// ```
///
/// ```compile_fail
/// fn needs_copy<T: Copy>() {}
/// needs_copy::<threadteam::Exec<'static>>();
/// ```
///
/// ```compile_fail
/// let exec = threadteam::Exec::serial();
/// exec.fork(1, |_, _| {});
/// exec.fork(1, |_, _| {}); // `fork` consumed the token
/// ```
#[must_use = "an Exec is a budget of execution slots; pass it to a kernel"]
pub struct Exec<'p> {
    pool: Option<&'p Pool>,
    budget: NonZeroUsize,
    /// Produced by `fork` (directly or through `reborrow`).
    child: bool,
    track: Track,
    /// Authority over reserved members, for tokens minted from a [`Team`](crate::Team).
    team: Option<TeamAuth<'p>>,
    /// Held by the root token of a `Team`: one leader at a time.
    _leader: Option<LeaderGuard<'p>>,
}

/// The reserved members a token may dispatch to; `ids.len() + 1 == budget`.
#[derive(Clone, Copy)]
pub(crate) struct TeamAuth<'p> {
    pub(crate) awake: &'p Awake,
    pub(crate) ids: &'p [usize],
    pub(crate) policy: WaitPolicy,
}

impl Exec<'static> {
    /// A token that runs everything inline on the calling thread.
    ///
    /// It never touches a scheduler, so using it cannot start any pool.
    ///
    /// # Examples
    ///
    /// ```
    /// use threadteam::Exec;
    ///
    /// let exec = Exec::serial();
    /// assert!(exec.is_serial());
    /// let here = std::thread::current().id();
    /// exec.fork(3, |_, _| assert_eq!(std::thread::current().id(), here));
    /// ```
    pub const fn serial() -> Exec<'static> {
        Exec {
            pool: None,
            budget: NonZeroUsize::MIN,
            child: false,
            track: Track::NONE,
            team: None,
            _leader: None,
        }
    }
}

impl<'p> Exec<'p> {
    /// A token of `budget` slots of `pool`, clamped to the pool's width.
    ///
    /// This is the only place a budget is clamped.
    ///
    /// # Errors
    ///
    /// [`Error::ZeroBudget`] when `budget` is zero.
    ///
    /// # Examples
    ///
    /// ```
    /// use threadteam::{Error, Exec, Pool};
    ///
    /// let pool = Pool::builder().threads(2).acquire_timeout(std::time::Duration::from_secs(10)).build()?;
    /// let exec = Exec::new(&pool, 64)?;
    /// assert_eq!(exec.budget(), pool.width());
    /// assert_eq!(Exec::new(&pool, 0).unwrap_err(), Error::ZeroBudget);
    /// # Ok::<(), Error>(())
    /// ```
    pub fn new(pool: &'p Pool, budget: usize) -> Result<Exec<'p>, Error> {
        let budget = NonZeroUsize::new(budget).ok_or(Error::ZeroBudget)?;
        Ok(Exec {
            pool: Some(pool),
            budget: budget.min(pool.width()),
            child: false,
            track: Track::root(),
            team: None,
            _leader: None,
        })
    }

    /// A root token of a [`Team`](crate::Team): `budget` is already clamped and
    /// `ids.len() + 1 == budget`.
    pub(crate) fn from_team(
        pool: &'p Pool,
        auth: TeamAuth<'p>,
        guard: LeaderGuard<'p>,
    ) -> Exec<'p> {
        Exec {
            pool: Some(pool),
            budget: NonZeroUsize::new(auth.ids.len() + 1).unwrap_or(NonZeroUsize::MIN),
            child: false,
            track: Track::root(),
            team: Some(auth),
            _leader: Some(guard),
        }
    }

    /// The number of slots this token stands for.
    ///
    /// # Examples
    ///
    /// ```
    /// assert_eq!(threadteam::Exec::serial().budget().get(), 1);
    /// ```
    pub fn budget(&self) -> NonZeroUsize {
        self.budget
    }

    /// The cost of one parallel section of `width` participants started now on
    /// this token: the steady state (sections back to back, shares that carry
    /// work). [`Duration::MAX`] when no section is possible: a serial token, or
    /// `width < 2`. `width` is clamped to the budget.
    ///
    /// The kind of token is decided at call time:
    ///
    /// * the token has team authority (it comes from a [`Team`](crate::Team), or
    ///   is a child of such a token): [`DispatchKind::Team`];
    /// * a plain token whose caller is a worker of its pool:
    ///   [`DispatchKind::InPool`];
    /// * a plain token whose caller is outside the pool: [`DispatchKind::Outside`].
    ///
    /// The numbers come from the pool's [`DispatchCosts`] (see
    /// [`PoolBuilder::dispatch_costs`](crate::PoolBuilder::dispatch_costs)). The
    /// call allocates nothing and reads no thread-local state of this crate.
    /// The answer depends on the calling thread, so ask it on the thread that
    /// will dispatch.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::num::NonZeroUsize;
    /// use std::time::Duration;
    /// use threadteam::Exec;
    ///
    /// let two = NonZeroUsize::new(2).unwrap();
    /// assert_eq!(Exec::serial().dispatch_cost(two), Duration::MAX);
    /// ```
    pub fn dispatch_cost(&self, width: NonZeroUsize) -> Duration {
        let width = width.min(self.budget);
        if width.get() < 2 {
            return Duration::MAX;
        }
        match (self.team, self.pool) {
            (Some(t), _) => t.awake.costs.cost(DispatchKind::Team, width),
            (None, Some(pool)) => {
                let kind = if sched::in_pool(pool) {
                    DispatchKind::InPool
                } else {
                    DispatchKind::Outside
                };
                pool.awake.costs.cost(kind, width)
            }
            (None, None) => Duration::MAX,
        }
    }

    /// Whether a region of serial time `estimate` that opens `sections` parallel
    /// sections of `width` participants pays on this token: `estimate >=
    /// PAYS * sections * dispatch_cost(width)` (see [`pays_at`]). False for a
    /// serial token.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::time::Duration;
    /// use threadteam::Exec;
    ///
    /// let two = 2.try_into().unwrap();
    /// assert!(!Exec::serial().pays(Duration::from_secs(1), 1, two));
    /// ```
    pub fn pays(&self, estimate: Duration, sections: u32, width: NonZeroUsize) -> bool {
        pays_at(estimate, sections, self.dispatch_cost(width))
    }

    /// The width in `1..=budget` that minimises `sections * D(k) + estimate / k`
    /// for this token, counting only widths at which the region
    /// [`pays`](Self::pays); 1 when none does (see [`width_for_by`]).
    ///
    /// # Examples
    ///
    /// ```
    /// use std::time::Duration;
    /// use threadteam::Exec;
    ///
    /// assert_eq!(Exec::serial().width_for(Duration::from_secs(1), 1).get(), 1);
    /// ```
    pub fn width_for(&self, estimate: Duration, sections: u32) -> NonZeroUsize {
        width_for_by(estimate, sections, self.budget, |k| self.dispatch_cost(k))
    }

    /// Whether no parallelism is possible: the budget is one slot.
    ///
    /// Kernels use this to take their serial path directly.
    ///
    /// # Examples
    ///
    /// ```
    /// assert!(threadteam::Exec::serial().is_serial());
    /// ```
    pub fn is_serial(&self) -> bool {
        self.budget.get() == 1
    }

    /// Narrows the token to at most `max` slots.
    ///
    /// The result has budget `min(budget, max)` and the same pool; the dropped
    /// slots are simply not used, so slots are conserved. Callers narrow before
    /// [`broadcast`](Exec::broadcast) or [`external`](Exec::external) to match a team size chosen
    /// by their own work model. Co-scheduling eligibility is unchanged by
    /// `limit`: a root token stays eligible for `broadcast`, a child produced by
    /// [`fork`](Exec::fork) stays ineligible.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::num::NonZeroUsize;
    /// use threadteam::{Exec, Pool};
    ///
    /// let pool = Pool::builder().threads(4).acquire_timeout(std::time::Duration::from_secs(10)).build()?;
    /// let exec = Exec::new(&pool, 4)?.limit(NonZeroUsize::new(2).unwrap());
    /// assert!(exec.budget().get() <= 2);
    /// # Ok::<(), threadteam::Error>(())
    /// ```
    pub fn limit(self, max: NonZeroUsize) -> Exec<'p> {
        let budget = self.budget.min(max);
        Exec {
            budget,
            team: self.team.map(|t| TeamAuth {
                ids: &t.ids[..budget.get() - 1],
                ..t
            }),
            ..self
        }
    }

    /// A token with the same pool and budget, borrowed from `self`.
    ///
    /// Parallel entries consume the token, but a kernel often runs several
    /// parallel phases in sequence. Each phase takes a reborrow. The parent is
    /// exclusively borrowed and unusable while the child lives, so the slots are
    /// held by one token at a time: this conserves slots exactly as handing the
    /// token over would.
    ///
    /// # Examples
    ///
    /// ```
    /// use threadteam::Exec;
    ///
    /// let mut exec = Exec::serial();
    /// exec.reborrow().fork(2, |_, _| {});
    /// exec.reborrow().fork(2, |_, _| {});
    /// exec.fork(2, |_, _| {});
    /// ```
    pub fn reborrow(&mut self) -> Exec<'_> {
        Exec {
            pool: self.pool,
            budget: self.budget,
            child: self.child,
            track: self.track.alias(),
            team: self.team,
            _leader: None,
        }
    }

    /// Runs `jobs` jobs on `min(jobs, budget)` tasks; job `i` gets a child token.
    ///
    /// The budget is split into one share per task, shares differing by at most
    /// one and summing to the budget. A job's child token carries its task's
    /// share and may be passed to any kernel, which can `fork` again inside it.
    /// When `jobs` exceeds the budget the tasks take job indices from a shared
    /// counter, so the load balances; otherwise job `i` runs on task `i`.
    ///
    /// One task, a serial token, or `jobs <= 1` runs inline on the calling
    /// thread without any scheduler call. Returns when every job has finished.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::sync::atomic::{AtomicUsize, Ordering};
    /// use threadteam::{Exec, Pool};
    ///
    /// let pool = Pool::builder().threads(2).acquire_timeout(std::time::Duration::from_secs(10)).build()?;
    /// let hits: Vec<AtomicUsize> = (0..5).map(|_| AtomicUsize::new(0)).collect();
    /// Exec::new(&pool, 2)?.fork(5, |i, _child| {
    ///     hits[i].fetch_add(1, Ordering::Relaxed);
    /// });
    /// assert!(hits.iter().all(|h| h.load(Ordering::Relaxed) == 1));
    /// # Ok::<(), threadteam::Error>(())
    /// ```
    pub fn fork<F>(self, jobs: usize, f: F)
    where
        F: Fn(usize, Exec<'_>) + Sync,
    {
        if jobs == 0 {
            return;
        }
        let budget = self.budget.get();
        let tasks = jobs.min(budget);
        let _released = self.track.release();
        let next = AtomicUsize::new(0);
        let this = &self;
        let body = |k: usize, team_ids: Option<&[usize]>| {
            let _live = this.track.task();
            let share = split::share(budget, tasks, k);
            let child = || Exec {
                pool: this.pool,
                // `share >= 1` because `tasks <= budget`.
                budget: NonZeroUsize::new(share).unwrap_or(NonZeroUsize::MIN),
                child: true,
                track: this.track.child(),
                team: this
                    .team
                    .zip(team_ids)
                    .map(|(t, ids)| TeamAuth { ids, ..t }),
                _leader: None,
            };
            if jobs <= tasks {
                f(k, child());
            } else {
                loop {
                    let job = next.fetch_add(1, Ordering::Relaxed);
                    if job >= jobs {
                        break;
                    }
                    f(job, child());
                }
            }
        };
        match (self.team, self.pool) {
            (Some(t), _) if tasks > 1 => {
                let plan = Plan::Even {
                    size: budget,
                    children: tasks,
                };
                split_raw(Some(t.awake), t.ids, &plan, |k, ids| body(k, Some(ids)));
            }
            (None, Some(pool)) if tasks > 1 => {
                sched::run_tasks(pool, tasks, &self.track, |k| body(k, None));
            }
            _ => {
                for k in 0..tasks {
                    body(k, self.team.map(|t| t.ids));
                }
            }
        }
    }

    /// Data-parallel loop over `0..len` in contiguous ranges.
    ///
    /// Uses at most `budget` tasks; every range is at least `grain` long except
    /// possibly the last, and together the ranges cover `0..len` exactly once.
    /// When one task suffices (`len <= grain`, a serial token) `f(0..len)` runs
    /// inline on the calling thread without any scheduler call. `len == 0` does
    /// nothing.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::sync::atomic::{AtomicUsize, Ordering};
    /// use threadteam::{Exec, Pool};
    ///
    /// let pool = Pool::builder().threads(2).acquire_timeout(std::time::Duration::from_secs(10)).build()?;
    /// let covered = AtomicUsize::new(0);
    /// Exec::new(&pool, 2)?.for_ranges(10, 4.try_into().unwrap(), |r| {
    ///     covered.fetch_add(r.len(), Ordering::Relaxed);
    /// });
    /// assert_eq!(covered.into_inner(), 10);
    /// # Ok::<(), threadteam::Error>(())
    /// ```
    pub fn for_ranges<F>(self, len: usize, grain: NonZeroUsize, f: F)
    where
        F: Fn(Range<usize>) + Sync,
    {
        if len == 0 {
            return;
        }
        let (chunk, tasks) = split::range_plan(len, grain.get(), self.budget.get());
        let _released = self.track.release();
        let track = &self.track;
        let body = |k: usize| {
            let _live = track.task();
            f(split::chunk_range(len, chunk, k));
        };
        match (self.team, self.pool) {
            (Some(t), _) if tasks > 1 => spread_raw(Some(t.awake), t.ids, tasks, body),
            (None, Some(pool)) if tasks > 1 => sched::run_tasks(pool, tasks, track, body),
            _ => {
                for k in 0..tasks {
                    body(k);
                }
            }
        }
    }

    /// Co-scheduling for barrier-bearing kernels: runs `f(i)` for every
    /// `i in 0..budget` with all participants running concurrently, or refuses.
    ///
    /// A serial or budget-one token runs `f(0)` inline and returns `Ok`.
    /// Otherwise the pool is entered once and every participant runs on its own
    /// worker, so `f` may block on a barrier shared by the `budget`
    /// participants.
    ///
    /// # Errors
    ///
    /// Nothing runs and [`Refused`] is returned when co-scheduling cannot be
    /// guaranteed:
    ///
    /// * [`Refused::ForkChild`]: the token was produced by [`fork`](Exec::fork)
    ///   (or derived from such a token by `reborrow` or `limit`); its slots may
    ///   share workers with siblings. Roots from [`Exec::new`], and their
    ///   `reborrow`s and `limit`s, stay eligible.
    ///
    /// Callers must handle `Refused` by falling back to a barrier-free path.
    /// * [`Refused::CallerInPool`]: the calling thread is a worker of the
    ///   token's pool and would occupy a slot.
    ///
    /// # Guarantee and limits
    ///
    /// Concurrent broadcasts on one pool are queued on every worker in the same
    /// order, so two of them cannot each hold half the workers. The participants
    /// are the pool's workers `0..budget`; if unrelated tasks that never finish
    /// occupy those workers, the broadcast waits for them. Keep barrier kernels on a pool whose workers are not held
    /// by blocking work of another token.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::sync::Barrier;
    /// use threadteam::{Exec, Pool};
    ///
    /// let pool = Pool::builder().threads(2).acquire_timeout(std::time::Duration::from_secs(10)).build()?;
    /// let exec = Exec::new(&pool, 2)?;
    /// let barrier = Barrier::new(exec.budget().get());
    /// exec.broadcast(|_i| {
    ///     barrier.wait(); // every participant is running, or this call refused
    /// }).expect("called from a plain thread on a root token");
    /// # Ok::<(), threadteam::Error>(())
    /// ```
    pub fn broadcast<F>(self, f: F) -> Result<(), Refused>
    where
        F: Fn(usize) + Sync,
    {
        if let Some(refused) = self.refusal() {
            return Err(refused);
        }
        if let Some(t) = self.team {
            // Reserved members are exclusively this token's: every participant
            // runs at once, also for a child produced by `fork`.
            let _released = self.track.release();
            let track = &self.track;
            spread_raw(Some(t.awake), t.ids, self.budget.get(), |i| {
                let _live = track.task();
                f(i);
            });
            return Ok(());
        }
        let pool = match self.pool {
            Some(pool) if self.budget.get() > 1 => pool,
            _ => {
                let _live = self.track.task();
                f(0);
                return Ok(());
            }
        };
        let _released = self.track.release();
        #[cfg(feature = "parallel")]
        sched::broadcast(pool, &self.track, self.budget.get(), f);
        #[cfg(not(feature = "parallel"))]
        {
            let _ = pool;
            f(0);
        }
        Ok(())
    }

    /// Whether [`broadcast`](Exec::broadcast) would run right now, without
    /// consuming the token.
    ///
    /// True exactly when `broadcast` would not return [`Refused`]: the token is
    /// not derived from [`fork`](Exec::fork) and the calling thread is not a
    /// worker of its pool. A serial or budget-one token returns true, because
    /// `broadcast` then runs `f(0)` inline. The answer depends on the calling
    /// thread, so ask it on the thread that will broadcast.
    ///
    /// Use it before building an expensive setup that only a broadcast needs,
    /// instead of paying for the setup and then being refused.
    ///
    /// # Examples
    ///
    /// ```
    /// use threadteam::{Exec, Pool};
    ///
    /// let pool = Pool::builder().threads(2).acquire_timeout(std::time::Duration::from_secs(10)).build()?;
    /// let exec = Exec::new(&pool, 2)?;
    /// assert!(exec.can_broadcast());
    /// exec.fork(1, |_, child| {
    ///     let predicted = child.can_broadcast();
    ///     assert_eq!(child.broadcast(|_| {}).is_ok(), predicted);
    /// });
    /// assert!(Exec::serial().can_broadcast());
    /// # Ok::<(), threadteam::Error>(())
    /// ```
    pub fn can_broadcast(&self) -> bool {
        self.refusal().is_none()
    }

    /// The single refusal rule shared by `broadcast` and `can_broadcast`.
    fn refusal(&self) -> Option<Refused> {
        if self.team.is_some() {
            return None;
        }
        let pool = match self.pool {
            Some(pool) if self.budget.get() > 1 => pool,
            _ => return None,
        };
        if self.child {
            return Some(Refused::ForkChild);
        }
        if sched::in_pool(pool) {
            return Some(Refused::CallerInPool);
        }
        None
    }

    /// Hands the budget as a plain width to a library outside this stack.
    ///
    /// This is the only place a bare width exists. The token is consumed, the
    /// pool is entered once (inline when the token is serial, has a budget of
    /// one, or the caller already is a worker of the pool) and `f` receives the
    /// budget, for example to build `faer::Par::rayon(width)` or to set a vendor
    /// BLAS thread count. Whatever the library schedules with that width runs on
    /// this pool, never on an ambient one. How the library divides the width is
    /// its own responsibility. Without the `parallel` feature, `f` runs inline
    /// and receives a width of one.
    ///
    /// # Examples
    ///
    /// ```
    /// use threadteam::Exec;
    ///
    /// let width = Exec::serial().external(|width| width.get());
    /// assert_eq!(width, 1);
    /// ```
    pub fn external<R: Send>(self, f: impl FnOnce(NonZeroUsize) -> R + Send) -> R {
        #[cfg(feature = "parallel")]
        if let Some(pool) = self.pool {
            if self.budget.get() > 1 {
                let width = self.budget;
                let _released = self.track.release();
                // A team token lends its reserved members to the pool for the
                // call: the library's Rayon work runs on them as well.
                let _lent = self.team.map(|t| LendGuard::new(t.awake, t.ids));
                return sched::enter(pool, &self.track, move || f(width));
            }
        }
        f(NonZeroUsize::MIN)
    }

    /// Reserves `budget - 1` pool members for the scope of `f` and hands `f`
    /// the exclusive [`TeamExec`] authority over them and the calling thread.
    ///
    /// The calling thread is participant 0 (the leader); the members are
    /// pool workers that stay reserved, spinning and serving no Rayon job,
    /// until `f` returns (or unwinds). A serial or budget-one token yields a
    /// team of one without touching the pool.
    ///
    /// Acquisition never waits for a Rayon job that is already running. Idle
    /// members of an earlier team (see [`WaitPolicy`]) are claimed with a
    /// compare-and-swap; missing ones are requested from the pool, and the
    /// call waits at most the pool's acquisition timeout
    /// ([`PoolBuilder::acquire_timeout`](crate::PoolBuilder::acquire_timeout))
    /// for idle workers to wake. If not all members are there by then, every
    /// tentative claim is released and nothing runs. Before `f` is called,
    /// every participant has acknowledged on its own thread.
    ///
    /// The token stays with the caller (this takes `&mut self`), so a refused
    /// acquisition leaves it available for a barrier-free fallback such as
    /// [`fork`](Exec::fork). It is conserved as usual: while `f` runs the token
    /// is borrowed.
    ///
    /// # Errors
    ///
    /// [`Refused::Unavailable`] when the members could not be reserved.
    ///
    /// # Examples
    ///
    /// ```
    /// use threadteam::{Exec, Pool, Refused};
    ///
    /// let pool = Pool::builder().threads(4).acquire_timeout(std::time::Duration::from_secs(10)).build()?;
    /// let mut exec = Exec::new(&pool, 4)?;
    /// match exec.team(|team| team.size().get()) {
    ///     Ok(n) => assert!(n >= 1),
    ///     Err(Refused::Unavailable) => exec.fork(4, |_, _| {}), // barrier-free fallback
    ///     Err(_) => unreachable!(),
    /// }
    /// # Ok::<(), threadteam::Error>(())
    /// ```
    pub fn team<R>(
        &mut self,
        f: impl for<'team> FnOnce(TeamExec<'team>) -> R,
    ) -> Result<R, Refused> {
        self.team_with(WaitOverride::new(), f)
    }

    /// [`team`](Exec::team) with a scoped override of the waiting policy.
    ///
    /// The override applies to this operation only and is inherited by the
    /// children of the team; the pool's defaults are not changed. It governs
    /// the idle interval after this reservation ends and never weakens the
    /// reservation itself.
    ///
    /// # Errors
    ///
    /// [`Refused::Unavailable`] when the members could not be reserved.
    pub fn team_with<R>(
        &mut self,
        over: WaitOverride,
        f: impl for<'team> FnOnce(TeamExec<'team>) -> R,
    ) -> Result<R, Refused> {
        if let Some(t) = self.team {
            // Carried members: the (sub)team is available without acquisition.
            return Ok(f(TeamExec {
                awake: Some(t.awake),
                ids: t.ids,
                policy: t.policy.with(over),
                track: self.track.alias(),
            }));
        }
        let pool = match self.pool {
            Some(pool) if self.budget.get() > 1 => pool,
            _ => {
                let policy = self.pool.map_or(WaitPolicy::DEFAULT, Pool::wait_policy);
                return Ok(f(TeamExec {
                    awake: None,
                    ids: &[],
                    policy: policy.with(over),
                    track: self.track.alias(),
                }));
            }
        };
        let policy = pool.wait_policy().with(over);
        let reservation =
            pool.awake
                .acquire(pool, self.budget.get() - 1, policy, policy.idle_spin())?;
        let mut team = TeamExec {
            awake: Some(&pool.awake),
            ids: &reservation.ids,
            policy,
            track: self.track.alias(),
        };
        team.acknowledge();
        Ok(f(team))
    }

    /// Root id of this token in the [`ledger`](crate::ledger), if it has one.
    ///
    /// Tokens from [`Exec::new`] are roots; children and reborrows report the
    /// root of their origin; [`Exec::serial`] has none.
    ///
    /// Available with the `ledger` cargo feature.
    ///
    /// # Examples
    ///
    /// ```
    /// use threadteam::{Exec, Pool};
    ///
    /// let pool = Pool::builder().threads(1).acquire_timeout(std::time::Duration::from_secs(10)).build()?;
    /// assert!(Exec::new(&pool, 1)?.ledger_root().is_some());
    /// assert!(Exec::serial().ledger_root().is_none());
    /// # Ok::<(), threadteam::Error>(())
    /// ```
    #[cfg(feature = "ledger")]
    pub fn ledger_root(&self) -> Option<u64> {
        self.track.root_id()
    }
}

impl fmt::Debug for Exec<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Exec")
            .field("budget", &self.budget)
            .field("pooled", &self.pool.is_some())
            .field("child", &self.child)
            .field("team", &self.team.is_some())
            .finish()
    }
}
