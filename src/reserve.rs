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
/// An owner that has its pool to itself (an engine) reserves `width - 1`
/// workers once with [`Pool::reserve`] and mints tokens with
/// [`exec`](Team::exec). A minted token is an ordinary [`Exec`] that carries
/// the team's authority:
///
/// * `fork` and `for_ranges` dispatch as team sections: no Rayon entry, no
///   wake-up, no allocation. Children of `fork` carry the sub-team of their
///   share, so nested forks stay on reserved members.
/// * `broadcast` runs on the reserved members (a `fork` child may broadcast
///   too: its members are its own); `can_broadcast` is always true.
/// * `team` returns the carried (sub)team at once, without acquisition.
/// * `external` hands the width as on any token. The members spin or park, so
///   a library started through it runs on the pool's *other* workers only.
///
/// Reserved members spin for the reservation's `idle_spin`, then **park**
/// while staying reserved: they do not return to Rayon until the `Team` is
/// dropped. The next dispatch unparks them, which costs a syscall for the
/// first dispatch after a park (measured by `benches/team_dispatch.rs`) and
/// nothing for a warm one.
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
/// Dropping the `Team` returns the members to Rayon immediately (the pool
/// still joins its threads when it is dropped). Tokens borrow the `Team`, so
/// none outlives it.
///
/// # Examples
///
/// ```
/// use std::num::NonZeroUsize;
/// use std::sync::atomic::{AtomicUsize, Ordering};
/// use threadteam::Pool;
///
/// let pool = Pool::builder().threads(4).build()?;
/// let team = pool.reserve(4).expect("an idle pool");
/// let sum = AtomicUsize::new(0);
/// let exec = team.exec(4); // blocks while another thread leads
/// exec.for_ranges(100, NonZeroUsize::MIN, |r| {
///     sum.fetch_add(r.len(), Ordering::Relaxed);
/// });
/// assert_eq!(sum.into_inner(), 100);
/// # Ok::<(), threadteam::Error>(())
/// ```
pub struct Team<'p> {
    pool: &'p Pool,
    reservation: Reservation<'p>,
    policy: WaitPolicy,
    gate: Gate,
}

impl Pool {
    /// Reserves `width - 1` members for as long as the returned [`Team`] lives.
    ///
    /// `width` counts the leader and is clamped to the pool width; zero is
    /// treated as one. Acquisition never waits for a running Rayon job (see
    /// [`Exec::team`](crate::Exec::team)). Members park after the pool's
    /// `idle_spin` without work.
    ///
    /// # Errors
    ///
    /// [`Refused::Unavailable`] when the members could not all be reserved.
    ///
    /// # Examples
    ///
    /// ```
    /// let pool = threadteam::Pool::builder().threads(3).build()?;
    /// let team = pool.reserve(3).expect("an idle pool");
    /// assert_eq!(team.size().get(), pool.width().get().min(3));
    /// # Ok::<(), threadteam::Error>(())
    /// ```
    pub fn reserve(&self, width: usize) -> Result<Team<'_>, Refused> {
        self.reserve_with(width, WaitOverride::new())
    }

    /// [`reserve`](Pool::reserve) with an override of the park delay
    /// (`idle_spin`) of the reserved members.
    ///
    /// # Errors
    ///
    /// [`Refused::Unavailable`] when the members could not all be reserved.
    pub fn reserve_with(&self, width: usize, over: WaitOverride) -> Result<Team<'_>, Refused> {
        let width = width.clamp(1, self.width().get());
        let policy = self.wait_policy().with(over);
        // On drop the members leave at once.
        let release = WaitPolicy::new(Duration::ZERO, false);
        let reservation = self
            .awake
            .acquire(self, width - 1, release, policy.idle_spin())?;
        let team = Team {
            pool: self,
            reservation,
            policy,
            gate: Gate {
                owner: Mutex::new(None),
                free: Condvar::new(),
            },
        };
        if width > 1 {
            // Every participant acknowledges on its own thread.
            let mut token = team.exec(width);
            let _ = token.team(|mut t| t.acknowledge());
        }
        Ok(team)
    }
}

impl<'p> Team<'p> {
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
    /// let pool = threadteam::Pool::builder().threads(2).build()?;
    /// let team = pool.reserve(2).expect("an idle pool");
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
        let auth = TeamAuth {
            awake: &self.pool.awake,
            ids: &self.reservation.ids[..budget - 1],
            policy: self.policy,
        };
        Exec::from_team(self.pool, auth, LeaderGuard(&self.gate))
    }
}

impl fmt::Debug for Team<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Team").field("size", &self.size()).finish()
    }
}
