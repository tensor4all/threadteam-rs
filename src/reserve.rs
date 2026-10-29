//! [`Team`]: members reserved for the lifetime of an owner.

use crate::awake::Reservation;
use crate::error::Refused;
use crate::exec::{Exec, TeamAuth};
use crate::policy::{WaitOverride, WaitPolicy};
use crate::pool::Pool;
use core::fmt;
use core::num::NonZeroUsize;
use core::time::Duration;
use std::sync::{Condvar, Mutex, PoisonError};
use std::thread::ThreadId;

/// Who currently leads the team: at most one thread at a time.
struct Gate {
    owner: Mutex<Option<ThreadId>>,
    free: Condvar,
}

/// Held by the root token of a [`Team`]; dropping the token hands the
/// leadership to the next waiting thread.
pub(crate) struct LeaderGuard<'t>(&'t Gate);

impl Drop for LeaderGuard<'_> {
    fn drop(&mut self) {
        *self.0.owner.lock().unwrap_or_else(PoisonError::into_inner) = None;
        self.0.free.notify_one();
    }
}

/// Pool members reserved for as long as this value lives.
///
/// An owner that has its pool to itself (an engine) moves the pool into a
/// `Team` with [`Pool::into_team`], which reserves `width - 1` workers once,
/// and mints tokens with
/// [`exec`](Team::exec). A minted token is an ordinary [`Exec`] that carries
/// the team's authority:
///
/// * `fork` and `for_ranges` dispatch as team sections: no Rayon entry, no
///   wake-up, no allocation. Children of `fork` carry the sub-team of their
///   share, so nested forks stay on reserved members.
/// * `broadcast` runs on the reserved members (a `fork` child may broadcast
///   too: its members are its own); `can_broadcast` is always true.
/// * `team` returns the carried (sub)team at once, without acquisition.
/// * `external` hands the width as on any token and **lends the token's
///   members to the pool for the call**: they run the pool's Rayon jobs (the
///   library's work) instead of idling, and resume team service when the call
///   returns. A lent job that is still running delays the next section until it
///   finishes.
///
/// Reserved members wait for work in staged episodes: they spin for the pool's
/// spin budget ([`PoolBuilder::spin_polls`](crate::PoolBuilder::spin_polls)),
/// yield a few times, then **park** while staying reserved; they do not return
/// to Rayon until the `Team` is dropped (`docs/design/waiting.md`). The next
/// dispatch wakes a parked member through its slot's parker, which costs a
/// syscall on the first dispatch after a park (`benches/wake_probe.rs`) and one
/// atomic read-modify-write for a warm one.
///
/// # Serialization of callers
///
/// The calling thread is the leader of every section, so a team has one
/// leader at a time. [`exec`](Team::exec) **blocks** until the previous token
/// (and everything derived from it by `reborrow`) was dropped;
/// [`try_exec`](Team::try_exec) returns [`Refused::Busy`] instead. Minting a
/// second token on the thread that still holds one would wait for itself, so
/// `exec` panics in that case. No thread-local state is used.
///
/// A `Team` **owns** its [`Pool`] and has no lifetime: an engine stores a `Team`
/// (or a plain `Pool` when it wants no team). `Pool` is not `Clone`, no method
/// hands out a second handle, and [`pool`](Team::pool) only borrows. Dropping
/// the `Team` releases the members and then drops the pool, which joins the
/// workers; [`into_pool`](Team::into_pool) releases the members and gives the
/// pool back instead. Tokens borrow the `Team`, so none outlives it.
///
/// # Examples
///
/// ```
/// use std::num::NonZeroUsize;
/// use std::sync::atomic::{AtomicUsize, Ordering};
/// use threadteam::Pool;
///
/// let pool = Pool::builder().threads(4).acquire_timeout(std::time::Duration::from_secs(10)).build()?;
/// let team = pool.into_team(4).expect("an idle pool");
/// let sum = AtomicUsize::new(0);
/// let exec = team.exec(4); // blocks while another thread leads
/// exec.for_ranges(100, NonZeroUsize::MIN, |r| {
///     sum.fetch_add(r.len(), Ordering::Relaxed);
/// });
/// assert_eq!(sum.into_inner(), 100);
/// # Ok::<(), threadteam::Error>(())
/// ```
pub struct Team {
    // Field order matters: the members are released before the pool is dropped.
    reservation: Reservation,
    policy: WaitPolicy,
    gate: Gate,
    pool: Pool,
}

/// [`Pool::into_team`] could not reserve the members: the pool is handed back.
///
/// # Examples
///
/// ```
/// let pool = threadteam::Pool::builder().threads(2).build()?;
/// match pool.into_team(2) {
///     Ok(team) => drop(team),
///     Err(e) => {
///         let _reason = e.reason();
///         let pool = e.into_pool(); // fall back to plain tokens
///         assert!(pool.width().get() >= 1);
///     }
/// }
/// # Ok::<(), threadteam::Error>(())
/// ```
pub struct IntoTeamError {
    pool: Pool,
    reason: Refused,
}

impl IntoTeamError {
    /// Why the members could not be reserved.
    pub fn reason(&self) -> Refused {
        self.reason
    }

    /// The pool, unchanged.
    pub fn into_pool(self) -> Pool {
        self.pool
    }
}

impl fmt::Debug for IntoTeamError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("IntoTeamError")
            .field("reason", &self.reason)
            .finish_non_exhaustive()
    }
}

impl fmt::Display for IntoTeamError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "cannot turn the pool into a team: {}", self.reason)
    }
}

impl std::error::Error for IntoTeamError {}

impl Pool {
    /// Moves the pool into a [`Team`] that reserves `width - 1` members for as
    /// long as it lives.
    ///
    /// `width` counts the leader and is clamped to the pool width; zero is
    /// treated as one. Acquisition never waits for a running Rayon job (see
    /// [`Exec::team`](crate::Exec::team)). Members park after the pool's spin
    /// budget without work ([`PoolBuilder::spin_polls`](crate::PoolBuilder::spin_polls)).
    ///
    /// # Errors
    ///
    /// [`IntoTeamError`] when the members could not all be reserved; it gives
    /// the pool back with the [`Refused`] reason so the caller falls back to
    /// plain tokens.
    ///
    /// # Examples
    ///
    /// ```
    /// let pool = threadteam::Pool::builder()
    ///     .threads(3)
    ///     .acquire_timeout(std::time::Duration::from_secs(10))
    ///     .build()?;
    /// let team = pool.into_team(3).expect("an idle pool");
    /// assert_eq!(team.size().get(), team.pool().width().get().min(3));
    /// # Ok::<(), threadteam::Error>(())
    /// ```
    pub fn into_team(self, width: usize) -> Result<Team, IntoTeamError> {
        self.into_team_with(width, WaitOverride::new())
    }

    /// [`into_team`](Pool::into_team) with an override of the wait policy of
    /// the reservation. The reserved members' waits for work use the pool's spin
    /// budget, not `idle_spin` (`docs/design/waiting.md`); the override is kept
    /// for the policy the team reports.
    ///
    /// # Errors
    ///
    /// As [`into_team`](Pool::into_team).
    pub fn into_team_with(self, width: usize, over: WaitOverride) -> Result<Team, IntoTeamError> {
        let width = width.clamp(1, self.width().get());
        let policy = self.wait_policy().with(over);
        // On drop the members leave at once.
        let release = WaitPolicy::new(Duration::ZERO, false);
        let reservation =
            match self
                .awake
                .acquire(&self, width - 1, release, self.awake.spin_budget(false))
            {
                Ok(r) => r,
                Err(reason) => return Err(IntoTeamError { pool: self, reason }),
            };
        let team = Team {
            reservation,
            policy,
            gate: Gate {
                owner: Mutex::new(None),
                free: Condvar::new(),
            },
            pool: self,
        };
        if width > 1 {
            // Every participant acknowledges on its own thread.
            let mut token = team.exec(width);
            let _ = token.team(|mut t| t.acknowledge());
        }
        Ok(team)
    }
}

impl Team {
    /// The owned pool, borrowed (for `width`, `resident_members`, or plain
    /// tokens next to the team). It cannot be cloned or moved out of a `&Team`.
    pub fn pool(&self) -> &Pool {
        &self.pool
    }

    /// Releases the members and gives the pool back.
    pub fn into_pool(self) -> Pool {
        let Team {
            reservation, pool, ..
        } = self;
        drop(reservation);
        pool
    }

    /// Number of participants: the leader plus the reserved members.
    pub fn size(&self) -> NonZeroUsize {
        NonZeroUsize::new(self.reservation.ids.len() + 1).unwrap_or(NonZeroUsize::MIN)
    }

    /// A token of `budget` slots (clamped to `1..=size`) carrying the team's
    /// authority. Blocks until no other token of this team lives.
    ///
    /// # Panics
    ///
    /// When the calling thread already holds a token of this team, because
    /// waiting would never end. Use [`try_exec`](Team::try_exec) or
    /// `reborrow` the token you hold.
    ///
    /// # Examples
    ///
    /// ```
    /// let pool = threadteam::Pool::builder().threads(2).acquire_timeout(std::time::Duration::from_secs(10)).build()?;
    /// let team = pool.into_team(2).expect("an idle pool");
    /// {
    ///     let exec = team.exec(2);
    ///     assert!(team.try_exec(2).is_err()); // this thread leads already
    ///     drop(exec);
    /// }
    /// assert!(team.try_exec(2).is_ok());
    /// # Ok::<(), threadteam::Error>(())
    /// ```
    pub fn exec(&self, budget: usize) -> Exec<'_> {
        let me = std::thread::current().id();
        let mut owner = self
            .gate
            .owner
            .lock()
            .unwrap_or_else(PoisonError::into_inner);
        loop {
            match *owner {
                None => break,
                Some(id) if id == me => {
                    panic!(
                        "Team::exec called by the thread that already holds a token of this team"
                    )
                }
                Some(_) => {
                    owner = self
                        .gate
                        .free
                        .wait(owner)
                        .unwrap_or_else(PoisonError::into_inner);
                }
            }
        }
        *owner = Some(me);
        drop(owner);
        self.token(budget)
    }

    /// Like [`exec`](Team::exec) without waiting.
    ///
    /// # Errors
    ///
    /// [`Refused::Busy`] when a token of this team lives, on this thread or another.
    pub fn try_exec(&self, budget: usize) -> Result<Exec<'_>, Refused> {
        let mut owner = self
            .gate
            .owner
            .lock()
            .unwrap_or_else(PoisonError::into_inner);
        if owner.is_some() {
            return Err(Refused::Busy);
        }
        *owner = Some(std::thread::current().id());
        drop(owner);
        Ok(self.token(budget))
    }

    fn token(&self, budget: usize) -> Exec<'_> {
        let budget = budget.clamp(1, self.size().get());
        let ids = &self.reservation.ids[..budget - 1];
        // Root-scope entry of the token: `exec` is serialized (one leader at a
        // time), so the calling thread's CPU set decides here whether the scope
        // is oversubscribed. When it is, the members' waits get the throttled
        // spin budget; otherwise the pool's. `Exec::team_with` on the minted token repeats
        // this on the thread that enters each scope, so a token that travels to
        // a thread with a different CPU set is paced by that set.
        let oversubscribed = crate::pin::oversubscribed(budget);
        self.pool
            .awake
            .set_spin_budget(ids, self.pool.awake.spin_budget(oversubscribed));
        let auth = TeamAuth {
            awake: &self.pool.awake,
            ids,
            policy: self.policy,
            over: oversubscribed,
        };
        Exec::from_team(&self.pool, auth, LeaderGuard(&self.gate))
    }
}

impl fmt::Debug for Team {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Team").field("size", &self.size()).finish()
    }
}
