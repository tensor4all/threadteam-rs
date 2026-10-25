//! Reserved awake members: the scoped authority [`TeamExec`].

use crate::awake::{Awake, JoinGuard, Section};
use crate::error::Error;
use crate::policy::{WaitOverride, WaitPolicy};
use crate::split;
use crate::track::Track;
use core::fmt;
use core::num::NonZeroUsize;
use std::panic::{catch_unwind, resume_unwind, AssertUnwindSafe};

/// A peer of a team section panicked: barriers return this so that the other
/// participants can leave the section instead of waiting forever.
///
/// # Examples
///
/// ```
/// assert!(threadteam::Cancelled.to_string().contains("panicked"));
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
#[error("a participant of the team section panicked; leave the section")]
pub struct Cancelled;

/// The view one participant has of the section it runs.
///
/// Participant `0` is the leader (the thread that called
/// [`TeamExec::run_with`]); `1..size` are the reserved members. All
/// participants of a section run at the same time on distinct threads, so
/// [`barrier`](Participant::barrier) is safe to use.
pub struct Participant<'a> {
    index: usize,
    section: &'a Section,
}

impl Participant<'_> {
    /// This participant's index in `0..size`; `0` is the leader.
    pub fn index(&self) -> usize {
        self.index
    }

    /// Number of participants of the section.
    pub fn size(&self) -> usize {
        self.section.size()
    }

    /// Waits until every participant reached the barrier.
    ///
    /// # Errors
    ///
    /// [`Cancelled`] when a participant of this section panicked: return from
    /// the closure (propagating the error is fine) so that the team can join
    /// and the panic reach the leader. Use `?` or return early; do not wait
    /// again.
    pub fn barrier(&self) -> Result<(), Cancelled> {
        if self.section.barrier() {
            Ok(())
        } else {
            Err(Cancelled)
        }
    }

    /// Whether a participant of this section has panicked.
    pub fn is_cancelled(&self) -> bool {
        self.section.is_cancelled()
    }
}

/// Exclusive authority over a set of reserved pool members, for one scope.
///
/// Obtained from [`Exec::team`](crate::Exec::team); the `'team` lifetime is
/// the reservation, so the authority cannot escape the callback:
///
/// * It is neither `Clone` nor `Copy`; entries ([`run`](Self::run),
///   [`split`](Self::split)) consume it. Successive phases use
///   [`reborrow`](Self::reborrow), which locks the parent while it lives.
/// * The team is the leader (the calling thread, participant 0) plus
///   `budget - 1` pool workers. Before any section runs, every participant
///   acknowledged on its own thread (a rendezvous), so barriers inside
///   sections cannot deadlock for lack of threads.
/// * Reserved members spin, then park once the park delay (`idle_spin`) has
///   passed since their own last share; they run no unrelated Rayon job and do
///   not leave on a timeout, also between sections and during a long serial
///   phase of the leader. They are returned when the scope ends.
/// * A panic in any participant propagates to the leader after every
///   dispatched share has finished; the reservation is released.
///
/// A team of one (serial token, budget one) runs sections inline.
///
/// # Examples
///
/// ```
/// use std::sync::atomic::{AtomicUsize, Ordering};
/// use threadteam::{Exec, Pool};
///
/// let pool = Pool::builder().threads(4).acquire_timeout(std::time::Duration::from_secs(10)).build()?;
/// let mut exec = Exec::new(&pool, 4)?;
/// let hits = AtomicUsize::new(0);
/// let outcome = exec.team(|mut team| {
///     let size = team.size().get();
///     team.reborrow().run(|_i, _n| {
///         hits.fetch_add(1, Ordering::Relaxed);
///     });
///     team.run_with(|p| {
///         p.barrier().ok(); // all participants are running, or `Cancelled`
///     });
///     size
/// });
/// // `Err(Refused::Unavailable)` would mean: take a barrier-free path.
/// let size = outcome.expect("idle pool"); // 4, or 1 without the `parallel` feature
/// assert_eq!(hits.into_inner(), size);
/// # Ok::<(), threadteam::Error>(())
/// ```
///
/// The authority cannot be duplicated:
///
/// ```compile_fail
/// fn needs_clone<T: Clone>() {}
/// needs_clone::<threadteam::TeamExec<'static>>();
/// ```
///
/// ```compile_fail
/// fn needs_copy<T: Copy>() {}
/// needs_copy::<threadteam::TeamExec<'static>>();
/// ```
///
/// It cannot be used twice, nor escape the scope:
///
/// ```compile_fail
/// let mut exec = threadteam::Exec::serial();
/// exec.team(|team| {
///     team.run(|_, _| {});
///     team.run(|_, _| {}); // `run` consumed the authority
/// });
/// ```
///
/// ```compile_fail
/// let mut exec = threadteam::Exec::serial();
/// let escaped = exec.team(|team| team); // the reservation ends with the scope
/// ```
///
/// The parent is unusable while a reborrow or child reservation lives:
///
/// ```compile_fail
/// let mut exec = threadteam::Exec::serial();
/// exec.team(|mut team| {
///     let child = team.reborrow();
///     team.run(|_, _| {}); // `child` still borrows `team`
///     child.run(|_, _| {});
/// });
/// ```
///
/// ```compile_fail
/// let mut exec = threadteam::Exec::serial();
/// exec.team(|team| {
///     // `split` consumed the parent; the callback cannot reach it.
///     team.split(&[1], |_, _child| {
///         team.run(|_, _| {});
///     })
/// });
/// ```
///
/// A child cannot leave its callback:
///
/// ```compile_fail
/// use std::sync::Mutex;
/// let mut exec = threadteam::Exec::serial();
/// let slot: Mutex<Option<threadteam::TeamExec<'static>>> = Mutex::new(None);
/// exec.team(|team| {
///     let _ = team.split(&[1], |_, child| {
///         *slot.lock().unwrap() = Some(child);
///     });
/// });
/// ```
#[must_use = "a TeamExec is exclusive authority over reserved members; run a section"]
pub struct TeamExec<'team> {
    pub(crate) awake: Option<&'team Awake>,
    /// Pool slots of the members; team index `t >= 1` is `ids[t - 1]`.
    pub(crate) ids: &'team [usize],
    pub(crate) policy: WaitPolicy,
    pub(crate) track: Track,
}

/// How the members of a split are divided.
pub(crate) enum Plan<'a> {
    Sizes(&'a [usize]),
    Even { size: usize, children: usize },
}

impl Plan<'_> {
    pub(crate) fn children(&self) -> usize {
        match self {
            Plan::Sizes(s) => s.len(),
            Plan::Even { children, .. } => *children,
        }
    }

    /// Team index of child `c`'s leader and its size.
    fn child(&self, c: usize) -> (usize, usize) {
        match self {
            Plan::Sizes(s) => (s[..c].iter().sum(), s[c]),
            Plan::Even { size, children } => {
                let q = size / children;
                let r = size % children;
                (c * q + c.min(r), split::share(*size, *children, c))
            }
        }
    }
}

impl<'team> TeamExec<'team> {
    /// Number of participants: the leader plus the reserved members.
    ///
    /// # Examples
    ///
    /// ```
    /// use threadteam::Exec;
    /// Exec::serial().team(|team| assert_eq!(team.size().get(), 1)).unwrap();
    /// ```
    pub fn size(&self) -> NonZeroUsize {
        NonZeroUsize::new(self.ids.len() + 1).unwrap_or(NonZeroUsize::MIN)
    }

    /// The effective waiting policy of this scope.
    pub fn policy(&self) -> WaitPolicy {
        self.policy
    }

    /// The same authority with `over` applied to the scoped policy.
    ///
    /// Children inherit the effective policy. An override never weakens a
    /// reservation: members stay reserved and run no unrelated Rayon job
    /// whatever the policy says. The policy of the *root* reservation governs
    /// the idle interval after it ends; a child's override expires when the
    /// child returns and cannot release the parent's members.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::time::Duration;
    /// use threadteam::{Exec, WaitOverride};
    ///
    /// Exec::serial().team(|team| {
    ///     let team = team.with_wait(WaitOverride::new().idle_spin(Duration::ZERO));
    ///     assert_eq!(team.policy().idle_spin(), Duration::ZERO);
    /// }).unwrap();
    /// ```
    pub fn with_wait(mut self, over: WaitOverride) -> Self {
        self.policy = self.policy.with(over);
        self
    }

    /// Lends the authority for one phase; the parent is locked meanwhile.
    ///
    /// # Examples
    ///
    /// ```
    /// use threadteam::Exec;
    /// Exec::serial().team(|mut team| {
    ///     team.reborrow().run(|_, _| {});
    ///     team.run(|_, _| {});
    /// }).unwrap();
    /// ```
    pub fn reborrow(&mut self) -> TeamExec<'_> {
        TeamExec {
            awake: self.awake,
            ids: self.ids,
            policy: self.policy,
            track: self.track.alias(),
        }
    }

    /// Runs `f(index, size)` on every participant at once and returns when
    /// all have finished.
    ///
    /// The leader runs `f(0, size)` on the calling thread and the members run
    /// the other indices, each on its own thread. A panic in any participant
    /// is re-raised on the leader after all shares finished. Use
    /// [`run_with`](Self::run_with) for a barrier.
    ///
    /// Dispatch writes to the members' slots and allocates nothing.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::sync::atomic::{AtomicUsize, Ordering};
    /// use threadteam::{Exec, Pool};
    ///
    /// let pool = Pool::builder().threads(3).acquire_timeout(std::time::Duration::from_secs(10)).build()?;
    /// let sum = AtomicUsize::new(0);
    /// Exec::new(&pool, 3)?.team(|team| {
    ///     team.run(|i, n| { sum.fetch_add(i + 10 * n, Ordering::Relaxed); });
    /// }).expect("idle pool");
    /// let n = pool.width().get().min(3); // 1 without the `parallel` feature
    /// assert_eq!(sum.into_inner(), (0..n).sum::<usize>() + n * 10 * n);
    /// # Ok::<(), threadteam::Error>(())
    /// ```
    pub fn run<F>(self, f: F)
    where
        F: Fn(usize, usize) + Sync,
    {
        self.run_with(move |p| f(p.index(), p.size()));
    }

    /// Like [`run`](Self::run), with a [`Participant`] that offers a barrier.
    ///
    /// # Panics
    ///
    /// Re-raises the panic of any participant, after every share finished.
    /// A participant that panics cancels the section's barriers, so peers
    /// blocked in [`Participant::barrier`] return `Err(Cancelled)`.
    pub fn run_with<F>(self, f: F)
    where
        F: Fn(&Participant<'_>) + Sync,
    {
        let track = &self.track;
        let _released = track.release();
        let body = |p: &Participant<'_>| {
            let _live = track.task();
            f(p);
        };
        let ids = self.ids;
        section(self.awake, &|pos| ids[pos], ids.len(), ids.len() + 1, &body);
    }

    /// Divides the team into child reservations of the given sizes.
    ///
    /// Child `c` gets `sizes[c]` participants: its leader and its members are
    /// a subset of this team's. Child 0 is led by the calling thread, the
    /// others by one of the parent's members. `f(c, child)` runs on the
    /// child's leader; the child can [`run`](Self::run) sections on its own
    /// members, concurrently with its siblings, or split again. Members the
    /// sizes leave over stay parked. When this call returns, every child has
    /// finished and the members are back in the parent, which was consumed
    /// and is not reachable from the callback.
    ///
    /// # Errors
    ///
    /// [`Error::InvalidSplit`] (nothing runs) when `sizes` is empty, contains
    /// a zero, or sums to more than [`size`](Self::size).
    ///
    /// # Panics
    ///
    /// Re-raises a panic from any child after all children finished.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::sync::atomic::{AtomicUsize, Ordering};
    /// use threadteam::{Exec, Pool};
    ///
    /// let pool = Pool::builder().threads(4).acquire_timeout(std::time::Duration::from_secs(10)).build()?;
    /// let total = AtomicUsize::new(0);
    /// Exec::new(&pool, 4)?.team(|team| {
    ///     let _ = team.split(&[2, 2], |c, child| {
    ///         child.run(|i, n| { total.fetch_add(100 * c + 10 * n + i, Ordering::Relaxed); });
    ///     });
    /// }).expect("idle pool");
    /// // Without the `parallel` feature the pool has width 1 and `split` refuses.
    /// if pool.width().get() == 4 {
    ///     assert_eq!(total.into_inner(), 2 * 20 + 1 + 2 * (100 + 20) + 1);
    /// }
    /// # Ok::<(), threadteam::Error>(())
    /// ```
    pub fn split<F>(self, sizes: &[usize], f: F) -> Result<(), Error>
    where
        F: for<'c> Fn(usize, TeamExec<'c>) + Sync,
    {
        let total = self.ids.len() + 1;
        let sum = sizes.iter().try_fold(0usize, |a, &s| a.checked_add(s));
        if sizes.is_empty() || sizes.contains(&0) || sum.is_none_or(|s| s > total) {
            return Err(Error::InvalidSplit);
        }
        self.split_plan(&Plan::Sizes(sizes), f);
        Ok(())
    }

    /// Divides the team into `children` equal parts (shares differ by at
    /// most one and sum to the team size), like [`Exec::fork`](crate::Exec::fork).
    ///
    /// `children` is clamped to the team size; zero does nothing. See
    /// [`split`](Self::split) for the semantics.
    ///
    /// # Examples
    ///
    /// ```
    /// use threadteam::{Exec, Pool};
    ///
    /// let pool = Pool::builder().threads(4).acquire_timeout(std::time::Duration::from_secs(10)).build()?;
    /// Exec::new(&pool, 4)?.team(|team| {
    ///     team.fork(2, |_c, child| assert!(child.size().get() <= 2));
    /// }).expect("idle pool");
    /// # Ok::<(), threadteam::Error>(())
    /// ```
    pub fn fork<F>(self, children: usize, f: F)
    where
        F: for<'c> Fn(usize, TeamExec<'c>) + Sync,
    {
        let size = self.ids.len() + 1;
        let children = children.min(size);
        if children == 0 {
            return;
        }
        self.split_plan(&Plan::Even { size, children }, f);
    }

    fn split_plan<F>(self, plan: &Plan<'_>, f: F)
    where
        F: for<'c> Fn(usize, TeamExec<'c>) + Sync,
    {
        let track = &self.track;
        let _released = track.release();
        let (awake, ids, policy) = (self.awake, self.ids, self.policy);
        split_raw(awake, ids, plan, |c, child_ids| {
            let _live = track.task();
            let child = TeamExec {
                awake,
                ids: child_ids,
                policy,
                track: track.child(),
            };
            f(c, child);
        });
    }

    /// Rendezvous of every participant on its own thread.
    ///
    /// All participants are inside the section at the same time, so they
    /// are on distinct threads: this is the acknowledgement that precedes
    /// any user section.
    pub(crate) fn acknowledge(&mut self) {
        self.reborrow().run_with(|p| {
            let _ = p.barrier();
        });
    }
}

/// Publishes `body` to `targets`, runs participant 0 on the calling thread,
/// joins every dispatched share, then re-raises the first panic.
///
/// The one implementation of a section: `TeamExec` and team-carrying `Exec`
/// tokens both dispatch through it.
pub(crate) fn section<B, S>(
    awake: Option<&Awake>,
    slot_of: &S,
    dispatched: usize,
    size: usize,
    body: &B,
) where
    B: Fn(&Participant<'_>) + Sync,
    S: Fn(usize) -> usize + Sync,
{
    unsafe fn slot_call<S: Fn(usize) -> usize>(data: *const (), pos: usize) -> usize {
        // SAFETY: `data` is the `&S` passed to `section`, alive for the call.
        (*data.cast::<S>())(pos)
    }
    unsafe fn call<B: Fn(&Participant<'_>)>(data: *const (), sec: &Section, index: usize) {
        // SAFETY: `data` is the `&B` passed to `section`, alive for the call.
        let body = &*data.cast::<B>();
        body(&Participant {
            index,
            section: sec,
        });
    }
    let sec = Section::new(
        call::<B>,
        (body as *const B).cast(),
        size,
        dispatched,
        slot_call::<S>,
        (slot_of as *const S).cast(),
    );
    let guard = JoinGuard(&sec);
    if dispatched > 0 {
        if let Some(awake) = awake {
            awake.publish(&sec);
        }
    }
    // SAFETY: `call::<B>` and `body` were paired just above.
    let own = catch_unwind(AssertUnwindSafe(|| unsafe { sec.call_here(0) }));
    if let Err(payload) = own {
        sec.fail(payload);
    }
    drop(guard); // every dispatched share has finished
    if let Some(payload) = sec.take_panic() {
        resume_unwind(payload);
    }
}

/// Runs `f(c, child_ids)` for every child of `plan` at once: child 0 on the
/// calling thread, child `c` on the first member of its index range.
pub(crate) fn split_raw<F>(awake: Option<&Awake>, ids: &[usize], plan: &Plan<'_>, f: F)
where
    F: Fn(usize, &[usize]) + Sync,
{
    let body = |p: &Participant<'_>| {
        let c = p.index();
        let (start, size) = plan.child(c);
        f(c, &ids[start..start + size - 1]);
    };
    let n = plan.children();
    section(
        awake,
        &|pos| ids[plan.child(pos + 1).0 - 1],
        n - 1,
        ids.len() + 1,
        &body,
    );
}

/// Runs `f(k)` for `k in 0..participants` at once (leader is `k = 0`).
pub(crate) fn spread_raw<F>(awake: Option<&Awake>, ids: &[usize], participants: usize, f: F)
where
    F: Fn(usize) + Sync,
{
    let body = |p: &Participant<'_>| f(p.index());
    section(
        awake,
        &|pos| ids[pos],
        participants - 1,
        participants,
        &body,
    );
}

impl fmt::Debug for TeamExec<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("TeamExec")
            .field("size", &self.size())
            .field("policy", &self.policy)
            .finish()
    }
}
