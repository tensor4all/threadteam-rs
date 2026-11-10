//! The splittable placement token.

use crate::awake::Awake;
#[cfg(feature = "parallel")]
use crate::awake::LendGuard;
use crate::cost::{
    pays_at, width_for_by, width_for_split_by, DispatchKind, SplitCost, TransferRate,
};
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
///   at most one and sum to the budget), and [`fork_sized`](Exec::fork_sized)
///   hands out explicit shares that sum to at most the budget, so the number of
///   tasks running at once under one root token is at most its budget at every
///   nesting depth.
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
    /// The root scope of the token is oversubscribed (see [`TeamExec`]).
    /// Fixed when the root token is minted and carried into `reborrow`,
    /// `limit`, `fork` children and every section.
    pub(crate) over: bool,
    /// Number of members the `Team` reserved, all of which the placement covers.
    /// A root token whose `ids` are fewer (`Team::exec` with a smaller budget,
    /// or `limit`) does not change the placement (see `team_with`).
    pub(crate) reserved: usize,
    /// The `Team`'s placement (`pin::Home`): the CPU no member takes, written by
    /// every root scope over the whole budget and read before each section and
    /// each reborrowed scope, so the leader goes home (`pin::home_pin`). `None`
    /// on a child that a member leads, whose leader must keep the CPU its own
    /// share was given.
    pub(crate) placement: Option<&'p AtomicUsize>,
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
    /// The numbers come from the pool's [`DispatchCosts`](crate::DispatchCosts) (see
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

    /// [`width_for`](Self::width_for) with the transfer term of a section whose
    /// shares read data this thread has just produced: the width in
    /// `1..=budget` that minimises
    /// `sections * D(k) + estimate / k + moved(k) / rate`, where `rate` is
    /// [`TransferRate::host_default`] (see [`width_for_split_by`]).
    ///
    /// Use it where the caller knows the bytes it wrote just before the region.
    /// A region whose input is not caller-resident passes
    /// `resident_bytes: 0` and gets [`width_for`](Self::width_for) exactly.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::time::Duration;
    /// use threadteam::{Exec, SplitCost};
    ///
    /// let split = SplitCost {
    ///     estimate: Duration::from_secs(1),
    ///     sections: 1,
    ///     resident_bytes: 1 << 20,
    /// };
    /// assert_eq!(Exec::serial().width_for_split(split).get(), 1);
    /// ```
    pub fn width_for_split(&self, split: SplitCost) -> NonZeroUsize {
        width_for_split_by(
            split,
            self.budget,
            |k| self.dispatch_cost(k),
            TransferRate::host_default(),
        )
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

    /// Whether this token carries team authority: it was minted by a
    /// [`Team`](crate::Team) or is a child of such a token. Its sections are team
    /// sections ([`DispatchKind::Team`]) and [`team`](Exec::team) returns the
    /// carried (sub)team at once, without acquisition, so a consumer that wants a
    /// [`TeamExec`] (faer's `Par::Team`) asks this first and keeps
    /// [`external`](Exec::external) for plain tokens, whose `team` would reserve
    /// members.
    ///
    /// # Examples
    ///
    /// ```
    /// use threadteam::{Exec, Pool};
    ///
    /// assert!(!Exec::serial().carries_team());
    /// let pool = Pool::builder().threads(2).acquire_timeout(std::time::Duration::from_secs(10)).build()?;
    /// assert!(!Exec::new(&pool, 2)?.carries_team());
    /// if let Ok(team) = pool.into_team(2) {
    ///     assert!(team.exec(2).carries_team());
    /// }
    /// # Ok::<(), threadteam::Error>(())
    /// ```
    pub fn carries_team(&self) -> bool {
        self.team.is_some()
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
        let size = self.budget.get();
        let plan = Plan::Even {
            size,
            children: jobs.min(size),
        };
        self.fork_plan(&plan, jobs, f);
    }

    /// Runs one child per entry of `shares`; child `i` gets a token of exactly
    /// `shares[i]` slots.
    ///
    /// For work of uneven size: give a large item several slots and the small
    /// ones one slot each. Child `i` always runs job `i` (there is no shared
    /// counter), and join, panic propagation and placement are those of
    /// [`fork`](Exec::fork), which is the even special case of this call.
    /// One child, or a serial token, runs inline on the calling thread without
    /// any scheduler call. An empty `shares` does nothing. Returns when every
    /// child has finished.
    ///
    /// **Leftover slots.** When the shares sum to less than the budget, the
    /// remaining slots are given to no child and stay idle for the call: no
    /// task runs on them and, on a team token, their reserved members stay
    /// parked. They remain the parent's: fork through
    /// [`reborrow`](Exec::reborrow) and the parent token has its full budget
    /// again after the join.
    ///
    /// # Errors
    ///
    /// Nothing runs and the token is consumed when
    ///
    /// * a share is zero: [`Error::ZeroBudget`];
    /// * the shares sum to more than the budget:
    ///   [`Error::SharesExceedBudget`], naming the budget and the sum. Shares
    ///   are never clamped.
    ///
    /// # Panics
    ///
    /// Re-raises a panic from any child after every child has finished.
    ///
    /// # Examples
    ///
    /// ```
    /// use threadteam::{Error, Exec, Pool};
    ///
    /// let pool = Pool::builder().threads(4).acquire_timeout(std::time::Duration::from_secs(10)).build()?;
    /// let mut exec = Exec::new(&pool, 4)?;
    /// let budget = exec.budget().get(); // 1 without the `parallel` feature
    /// if budget == 4 {
    ///     // One large item on three slots, one small item on the last slot.
    ///     exec.reborrow().fork_sized(&[3, 1], |i, child| {
    ///         assert_eq!(child.budget().get(), [3, 1][i]);
    ///     })?;
    /// }
    /// let err = exec.fork_sized(&[budget, 1], |_, _| {}).unwrap_err();
    /// assert_eq!(err, Error::SharesExceedBudget { budget, sum: budget + 1 });
    /// # Ok::<(), Error>(())
    /// ```
    pub fn fork_sized<F>(self, shares: &[usize], f: F) -> Result<(), Error>
    where
        F: Fn(usize, Exec<'_>) + Sync,
    {
        if shares.contains(&0) {
            return Err(Error::ZeroBudget);
        }
        let budget = self.budget.get();
        let sum = shares.iter().fold(0usize, |a, &s| a.saturating_add(s));
        if sum > budget {
            return Err(Error::SharesExceedBudget { budget, sum });
        }
        if !shares.is_empty() {
            self.fork_plan(&Plan::Sizes(shares), shares.len(), f);
        }
        Ok(())
    }

    /// [`fork_sized`](Exec::fork_sized) where child `i` also gets exclusive
    /// `&mut` access to `slots[i]`, its own state (a cursor, scratch, an error
    /// slot), with no lock.
    ///
    /// `slots` stays exclusively borrowed by the call; each child derives its
    /// own `&mut slots[i]` from its index right before calling `f`, and
    /// `fork_sized` runs each index exactly once, so no two children ever
    /// hold the same slot. The borrow of `slots` ends when every child has
    /// joined: while the call runs, neither a sibling nor the caller can reach
    /// slot `i`, and the reference a child receives cannot outlive its call
    /// (see `docs/design/placement.md`). Splitting,
    /// placement, join, panic propagation and the share checks are those of
    /// `fork_sized`, which this call goes through.
    ///
    /// # Errors
    ///
    /// Nothing runs, no slot is touched and the token is consumed when
    ///
    /// * `shares.len() != slots.len()`: [`Error::SlotsMismatch`], naming both
    ///   counts (no slot is silently left out);
    /// * otherwise, as [`fork_sized`](Exec::fork_sized): a zero share is
    ///   [`Error::ZeroBudget`], shares summing past the budget are
    ///   [`Error::SharesExceedBudget`].
    ///
    /// # Panics
    ///
    /// Re-raises a panic from any child after every child has finished, as
    /// `fork_sized` does. By then the borrow of `slots` has ended, so a caller
    /// that catches the panic can use every slot again. Each slot holds a valid
    /// `S`: a child's slot holds what the child last wrote, which after a panic
    /// in child `i` may be the middle of an update to slot `i` that the child
    /// did not finish (nothing is rolled back; validity, not consistency, is
    /// what unwinding guarantees, hence the `AssertUnwindSafe` a catching
    /// caller writes).
    ///
    /// # Examples
    ///
    /// ```
    /// use threadteam::{Error, Exec, Pool};
    ///
    /// let pool = Pool::builder().threads(4).acquire_timeout(std::time::Duration::from_secs(10)).build()?;
    /// let exec = Exec::new(&pool, 4)?;
    /// let shares: &[usize] = if exec.budget().get() == 4 { &[3, 1] } else { &[1] };
    /// // One cursor per child, advanced without a lock.
    /// let mut cursors = vec![0usize; shares.len()];
    /// exec.fork_sized_slots(shares, &mut cursors, |i, cursor, child| {
    ///     *cursor += 10 * (i + 1) + child.budget().get();
    /// })?;
    /// assert_eq!(cursors[0], 10 + shares[0]);
    ///
    /// let err = Exec::serial().fork_sized_slots(&[1], &mut cursors[..0], |_, _, _| {});
    /// assert_eq!(err.unwrap_err(), Error::SlotsMismatch { shares: 1, slots: 0 });
    /// # Ok::<(), Error>(())
    /// ```
    ///
    /// A slot reference cannot escape its child:
    ///
    /// ```compile_fail
    /// use std::sync::Mutex;
    /// let mut slots = [0u32];
    /// let kept: Mutex<Option<&mut u32>> = Mutex::new(None);
    /// threadteam::Exec::serial()
    ///     .fork_sized_slots(&[1], &mut slots, |_, slot, _| *kept.lock().unwrap() = Some(slot))
    ///     .unwrap();
    /// ```
    pub fn fork_sized_slots<S, F>(
        self,
        shares: &[usize],
        slots: &mut [S],
        f: F,
    ) -> Result<(), Error>
    where
        S: Send,
        F: Fn(usize, &mut S, Exec<'_>) + Sync,
    {
        if shares.len() != slots.len() {
            return Err(Error::SlotsMismatch {
                shares: shares.len(),
                slots: slots.len(),
            });
        }
        let base = SlotBase(slots.as_mut_ptr());
        self.fork_sized(shares, |i, child| {
            // SAFETY: the slice is not pre-split; this child derives its own
            // element from its index here, right before `f`. Exclusivity rests
            // on `fork_sized` dispatching each `i in 0..shares.len()` exactly
            // once (`fork_plan` with as many jobs as tasks, no shared counter)
            // and on `shares.len() == slots.len()`: `i` is in bounds and no two
            // live references name the same element.
            // `slots` stays exclusively borrowed by this call, and
            // `fork_sized` returns or unwinds only after every child has
            // finished, so the pointee outlives every reference made here; the
            // reference is handed to `f` for one call and its higher-ranked
            // lifetime keeps it from escaping. Safe callers reach this only
            // through the checks above.
            let slot = unsafe { &mut *base.at(i) };
            f(i, slot, child);
        })
    }

    /// The one splitter of `fork` and `fork_sized`: task `k` of `plan` gets a
    /// child token of `plan.child(k).1` slots. More `jobs` than tasks pull job
    /// indices from a shared counter; otherwise job `k` runs on task `k`,
    /// exactly once (`fork_sized_slots` relies on this for disjoint slots).
    fn fork_plan<F>(self, plan: &Plan<'_>, jobs: usize, f: F)
    where
        F: Fn(usize, Exec<'_>) + Sync,
    {
        let tasks = plan.children();
        let _released = self.track.release();
        let next = AtomicUsize::new(0);
        let this = &self;
        let body = |k: usize, team_ids: Option<&[usize]>| {
            let _live = this.track.task();
            let share = plan.child(k).1;
            let child = || Exec {
                pool: this.pool,
                // Every share of a validated plan is at least one.
                budget: NonZeroUsize::new(share).unwrap_or(NonZeroUsize::MIN),
                child: true,
                track: this.track.child(),
                team: this.team.zip(team_ids).map(|(t, ids)| TeamAuth {
                    ids,
                    // Share 0 runs on the calling thread, the team's leader; the
                    // others are led by members, which must not move onto the
                    // team leader's CPU because their share was not given it.
                    placement: if k > 0 { None } else { t.placement },
                    ..t
                }),
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
                // A team token places itself on the CPU no member takes for the
                // section, and has its own mask back when it ends (see
                // `pin::home_pin`). Share 0 runs on the calling thread, the
                // leader; the others are led by members.
                let _home = crate::pin::home_pin(placement_home(t));
                split_raw(Some(t.awake), t.ids, plan, t.over, |k, ids| {
                    body(k, Some(ids));
                });
            }
            (None, Some(pool)) if tasks > 1 => {
                sched::run_tasks(pool, tasks, &self.track, |k| body(k, None));
            }
            _ => {
                for k in 0..tasks {
                    // The members of child `k`, as `split_raw` hands them out.
                    // This arm runs every share on the calling thread, which
                    // leads the team.
                    let (start, size) = plan.child(k);
                    body(k, self.team.map(|t| &t.ids[start..start + size - 1]));
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
            (Some(t), _) if tasks > 1 => {
                let _home = crate::pin::home_pin(placement_home(t));
                spread_raw(Some(t.awake), t.ids, tasks, t.over, body);
            }
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
            let _home = if self.budget.get() > 1 {
                crate::pin::home_pin(placement_home(t))
            } else {
                None
            };
            spread_raw(Some(t.awake), t.ids, self.budget.get(), t.over, |i| {
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
    /// # Placement
    ///
    /// The token's reserved members are **lent** for the call: each restores its
    /// own affinity mask and serves the pool's jobs until the call returns. The
    /// calling thread does the same when it is itself a pinned participant of
    /// this pool (for example a member that leads a [`fork`](Exec::fork) child),
    /// so the foreign work never keeps a team's single-CPU placement. Both
    /// re-pin at the next pin section, that is at the next root team scope.
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
                // The calling thread may itself be a pinned participant of this
                // pool (the member that leads a `fork` child, or a token used
                // from inside a share): it restores its own mask before it
                // serves foreign work, as a lent member does.
                sched::unpin_worker(pool);
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
    /// This call opens a **root team scope**: it reads the calling thread's
    /// affinity mask, which decides two things. With
    /// [`PoolBuilder::pin_participants`](crate::PoolBuilder::pin_participants) on
    /// (the default) it places every member on its own CPU of that mask and pins
    /// the leader to the first CPU of the mask until the scope ends (a vendor
    /// call from the leader goes through [`TeamExec::external`], which widens it
    /// for that call). A scoped acquisition is placed for the scope; a root
    /// token of a [`Team`](crate::Team) with the whole team's budget places the
    /// team's members until a later such scope re-places or unpins them or the
    /// `Team` ends. A root token of a
    /// `Team` with a smaller budget changes no member placement and runs under
    /// the team's current one; its leader is still pinned to the CPU that
    /// placement left free, when the caller's own mask holds it. A mask with
    /// fewer CPUs than
    /// participants places nobody ([`Pool::pin_skips`](crate::Pool::pin_skips) counts it).
    /// Whether or not pinning is on, it records whether the scope is
    /// *oversubscribed* (more participants than CPUs in the mask), which decides
    /// how the members park and how a joining leader waits. See the design
    /// record.
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
            let mut team = TeamExec {
                awake: Some(t.awake),
                ids: t.ids,
                policy: t.policy.with(over),
                track: self.track.alias(),
                over: t.over,
            };
            // Only the root token minted by a `Team` opens a root scope; a
            // child, a reborrow or a sub-team re-pins nothing of its own.
            if self._leader.is_some() {
                // Root-scope entry, on the executing thread: the root pressure
                // and the members' spin budget are decided by *this* thread's CPU
                // set. A token that travels to a thread with another mask is
                // paced by the thread that enters the scope, exactly as its pin
                // section is placed by that thread. Descendants inherit it.
                let oversubscribed = crate::pin::oversubscribed(team.size().get());
                team.over = oversubscribed;
                if let Some(awake) = team.awake {
                    awake.set_spin_budget(team.ids, awake.spin_budget(oversubscribed));
                }
                if let Some(auth) = self.team.as_mut() {
                    auth.over = oversubscribed;
                }
                // The members' placement outlives the scope: they stay pinned
                // until an unpin section or the end of the reservation. The
                // leader's does not: it is pinned for this scope and gets its
                // own mask back when it ends, as it does around a section. The
                // placement is one value for every reserved member, so only a
                // scope over all of them may change it: a token narrower than
                // the team would pin or unpin a prefix and leave the suffix as
                // it was. A narrower root scope therefore runs no pin or unpin
                // section and dispatches under the team's current placement. A
                // scope that ran no section (pinning off, no CPU set) leaves the
                // placement as it was.
                let _pin = if t.ids.len() == t.reserved && team.size().get() > 1 {
                    pin_scope(&mut team, t.placement)
                } else {
                    None
                };
                // The leader is placed whatever the scope's width: a narrower
                // one takes its home from the team's record, as a reborrowed
                // scope does. Left wide it ran on the CPUs its members are
                // pinned to (threadteam-rs#53). A no-op after `pin_scope` placed
                // the leader, which left this thread's `PLACED` record set.
                let _home = if team.size().get() > 1 {
                    crate::pin::home_pin(placement_home(t))
                } else {
                    None
                };
                return Ok(f(team));
            }
            // A reborrow or a child of the root token: this scope places no
            // members (the whole-team scope that set the placement did), but the
            // leader still runs its shares here and is placed for the scope.
            let _home = if team.size().get() > 1 {
                crate::pin::home_pin(placement_home(t))
            } else {
                None
            };
            return Ok(f(team));
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
                    over: false,
                }));
            }
        };
        let policy = pool.wait_policy().with(over);
        // Root-scope entry: the CPU set of this calling thread decides both the
        // placement of the participants and whether the scope is
        // oversubscribed, whatever the pool's pinning setting.
        let oversubscribed = crate::pin::oversubscribed(self.budget.get());
        let reservation = pool.awake.acquire(
            pool,
            self.budget.get() - 1,
            policy,
            pool.awake.spin_budget(oversubscribed),
        )?;
        let mut team = TeamExec {
            awake: Some(&pool.awake),
            ids: &reservation.ids,
            policy,
            track: self.track.alias(),
            over: oversubscribed,
        };
        // The pin (or unpin) section is also the acknowledgement; without
        // pinning the participants acknowledge as before. The guard gives the
        // leader its mask back when the scope ends or unwinds.
        let pin = if team.size().get() > 1 {
            pin_scope(&mut team, None)
        } else {
            None
        };
        if pin.is_none() {
            team.acknowledge();
        }
        let _pin = pin;
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

/// The base of the slots of a [`Exec::fork_sized_slots`] call, shared by
/// its children; each child derives only its own element.
struct SlotBase<S>(*mut S);

// SAFETY: children on other threads each obtain a `&mut S` to a distinct
// element through the shared base (see `fork_sized_slots`), which is sending
// that `S` to the child's thread; `S: Send` makes that sound.
unsafe impl<S: Send> Sync for SlotBase<S> {}

impl<S> SlotBase<S> {
    /// Element `i`; a method so that closures capture the whole `Sync` base,
    /// not its raw-pointer field.
    fn at(&self, i: usize) -> *mut S {
        self.0.wrapping_add(i)
    }
}

/// What a root team scope holds while it is open: the leader's pin to `S[0]`,
/// if it was placed. Dropping it (scope end or unwind) gives `S` back.
struct ScopePin(#[allow(dead_code)] Option<crate::pin::LeaderPin>);

/// Places a root team scope on the calling thread's CPU set `S`: the leader is
/// pinned to `S[0]` for the life of the scope (`pin::LeaderPin`, one
/// `sched_setaffinity` now and one when the returned guard drops), member `k`
/// pins itself onto `P[k]`, the `k`-th CPU of the placement order of `S` under
/// the pool's `PinStrategy` (`pin::order`: round-robin over NUMA nodes by
/// default, `S[k]` under `Pack`).
///
/// When `S` has fewer CPUs than the team has participants nothing is pinned:
/// the skip is counted (`Pool::pin_skips`) and the members restore their own
/// masks, in case an earlier scope of a persistent team pinned them. Wrapping
/// members around `S` puts two of them on one CPU, which measured worse than
/// the scheduler's placement and worse than one thread (evo2, LU f64 512 with
/// 8 participants wrapped onto one CPU: 3.1 ms against 0.89 ms unpinned and
/// 1.7 ms on one thread), so a team that cannot be placed is not placed.
///
/// The placement is the team's (`placement`, the [`Team`](crate::Team)'s
/// `pin::Home`): this scope writes it, so a later section or reborrowed scope of
/// any token of the team puts its leader on the CPU no member takes.
///
/// `None` when participant pinning is off for this pool, when the platform has
/// no CPU set (off Linux), or when the calling thread's mask could not be read:
/// then nothing ran and the caller acknowledges as usual. `Some` when a section
/// ran (pin or unpin), which also served as the acknowledgement; keep the guard
/// alive for the scope.
fn pin_scope(team: &mut TeamExec<'_>, placement: Option<&AtomicUsize>) -> Option<ScopePin> {
    let awake = team.awake.filter(|awake| awake.pin)?;
    // The leader reads its own mask: that set is where the team is placed, and
    // the leader's mask is that set again once the scope has closed.
    let set = crate::pin::read_here()?;
    let cpus = crate::pin::count(&set);
    // This thread just read its own mask, which is the count a later mint on it
    // would need: hand the read to the memo, so a mask another library changed is
    // seen by every mint after this scope without a second kernel call
    // (threadteam-rs#62).
    crate::pin::note_count(cpus);
    if cpus < team.size().get() {
        if let Some(placement) = placement {
            placement.store(0, Ordering::Relaxed);
        }
        team.unpin_section();
        return Some(ScopePin(None));
    }
    // Participant `k`'s CPU under the pool's strategy (`pin::order`). The
    // leader's home is participant 0's, which is `S[0]` under every strategy and
    // which no member takes. It is pinned there before the pin section, so the
    // scope opens with the leader home too, and recorded for the sections and
    // reborrowed scopes that follow.
    let order = crate::pin::Placement::of(&set, awake.pin_strategy, team.size().get());
    let leader = order
        .get(0)
        .and_then(|home| crate::pin::LeaderPin::enter(&set, home));
    if let Some(placement) = placement {
        placement.store(crate::pin::home_of(&order), Ordering::Relaxed);
    }
    team.pin_section(&order);
    Some(ScopePin(leader))
}

/// The placement a team token dispatches under: the [`Team`](crate::Team)'s
/// `pin::Home`, or `0` on a child that a member leads.
fn placement_home(auth: TeamAuth<'_>) -> crate::pin::Home {
    auth.placement
        .map_or(0, |placement| placement.load(Ordering::Relaxed))
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
