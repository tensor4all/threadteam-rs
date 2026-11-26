//! Error types of the crate.

/// Failure to build a [`Pool`](crate::Pool) or an [`Exec`](crate::Exec).
///
/// The enum is `#[non_exhaustive]`: new refusals may be added in a minor
/// release, so match with a wildcard arm.
///
/// # Examples
///
/// ```
/// use threadteam::{Error, Exec, Pool};
///
/// let pool = Pool::builder().threads(1).build()?;
/// assert_eq!(Exec::new(&pool, 0).unwrap_err(), Error::ZeroBudget);
/// # Ok::<(), Error>(())
/// ```
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[non_exhaustive]
pub enum Error {
    /// A token was requested with a budget of zero slots.
    #[error("an execution budget must be at least one slot")]
    ZeroBudget,
    /// A pool was requested with zero worker threads.
    #[error("a pool needs at least one worker thread")]
    ZeroThreads,
    /// The operating system or the scheduler refused to start the workers.
    #[error("failed to start the pool workers: {0}")]
    PoolBuild(String),
    /// [`TeamExec::split`](crate::TeamExec::split) was given sizes that are
    /// empty, contain a zero, or sum to more than the team size.
    #[error("child team sizes must be non-empty, non-zero and sum to at most the team size")]
    InvalidSplit,
    /// [`Exec::fork_sized`](crate::Exec::fork_sized) was given shares that sum
    /// to more than the token's budget. Nothing ran; the shares are not
    /// clamped.
    #[error("fork shares sum to {sum} slots, more than the token's budget of {budget}")]
    SharesExceedBudget {
        /// The budget of the token that was asked to fork.
        budget: usize,
        /// The sum of the requested shares (saturated at `usize::MAX`).
        sum: usize,
    },
    /// [`Exec::fork_sized_slots`](crate::Exec::fork_sized_slots) was given a
    /// different number of shares and state slots. Nothing ran; no slot is
    /// ignored and no share is dropped.
    #[error("{shares} fork shares but {slots} state slots; there must be one slot per share")]
    SlotsMismatch {
        /// The number of requested shares.
        shares: usize,
        /// The number of state slots.
        slots: usize,
    },
}

/// A [`broadcast`](crate::Exec::broadcast) that cannot guarantee co-scheduling.
///
/// Nothing was run when this is returned. The caller falls back to a kernel
/// that does not need all participants at once.
///
/// # Examples
///
/// ```
/// use threadteam::Refused;
///
/// assert!(Refused::ForkChild.to_string().contains("fork"));
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
#[non_exhaustive]
pub enum Refused {
    /// The token was produced by [`fork`](crate::Exec::fork) (or by a
    /// [`reborrow`](crate::Exec::reborrow) of such a token): its slots may share
    /// workers with sibling tasks, so its participants could wait on each other.
    #[error("the token is a child produced by fork; its slots may share workers with siblings")]
    ForkChild,
    /// The caller is already a worker of the token's pool, so it would occupy
    /// one of the slots the participants need.
    #[error("the caller is a worker of the token's pool and would occupy a participant slot")]
    CallerInPool,
    /// [`Exec::team`](crate::Exec::team) could not reserve every member
    /// without waiting for a Rayon job that is already running.
    ///
    /// Nothing ran and no member stays claimed. The caller takes a
    /// barrier-free path.
    #[error("the team members could not all be reserved without waiting for a running Rayon job")]
    Unavailable,
    /// [`Team::try_exec`](crate::Team::try_exec): another thread holds the
    /// team's leader token.
    #[error("another thread is the leader of this team")]
    Busy,
}

/// Why [`Exec::calibrate_dispatch_costs`](crate::Exec::calibrate_dispatch_costs)
/// installed no row.
///
/// Every variant leaves the pool's table as it was: nothing is installed, no
/// partial row either, and [`Pool::dispatch_costs`](crate::Pool::dispatch_costs)
/// returns what it returned before the call.
///
/// The enum is `#[non_exhaustive]`: match with a wildcard arm.
///
/// # Examples
///
/// ```
/// use std::time::Duration;
/// use threadteam::{CalibrationError, Exec};
///
/// let err = Exec::serial().calibrate_dispatch_costs(Duration::from_secs(1));
/// assert_eq!(err, Err(CalibrationError::Serial));
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
#[non_exhaustive]
pub enum CalibrationError {
    /// The token has no section to measure: [`Exec::serial`](crate::Exec::serial),
    /// or a token of one slot (every token of a pool of width one is). Fails
    /// without running anything and without touching any table.
    #[error("a serial token has no parallel section to calibrate")]
    Serial,
    /// An earlier call on this pool installed a row. Calibration is one-shot;
    /// the installed row stays.
    #[error("the pool's team row has already been calibrated")]
    AlreadyCalibrated,
    /// The pool was built with
    /// [`PoolBuilder::dispatch_costs`](crate::PoolBuilder::dispatch_costs): the
    /// owner's row takes precedence. Fails without running anything and without
    /// touching the table.
    #[error("the pool was built with explicit dispatch costs, which calibration does not replace")]
    ExplicitCosts,
    /// The budget cannot hold the minimum measurement; refused before any
    /// section ran.
    #[error("a calibration needs a budget of at least {needed:?}")]
    BudgetTooSmall {
        /// The smallest budget this token's ladder accepts.
        needed: core::time::Duration,
    },
    /// The budget ran out before the measurement at `width` finished.
    #[error("the calibration budget ran out while measuring width {width}")]
    BudgetExhausted {
        /// The ladder width that was being measured.
        width: core::num::NonZeroUsize,
    },
    /// [`Exec::team`](crate::Exec::team) could not reserve the members
    /// ([`Refused::Unavailable`]).
    #[error("the team members could not be reserved for the calibration")]
    TeamUnavailable,
    /// The measured cost at `width` is outside `0 < measured <= limit`, where
    /// `limit` is ten times the default team row at that width: a cost that
    /// large is a wake after the members parked or a disturbed run, not the
    /// steady state the row stands for.
    #[error("measured {measured:?} at width {width}, outside (0, {limit:?}]")]
    Rejected {
        /// The ladder width whose measurement failed (the first one that did).
        width: core::num::NonZeroUsize,
        /// The median cost per section; zero for a value at or below zero.
        measured: core::time::Duration,
        /// Ten times the default team row at `width`.
        limit: core::time::Duration,
    },
}
