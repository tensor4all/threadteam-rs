//! Waiting policy of unreserved idle team members.

use core::time::Duration;

/// How an idle member behaves after its reservation ended.
///
/// A member that is *reserved* neither runs unrelated Rayon jobs nor leaves
/// on a timeout, whatever the policy says: a policy only governs the idle
/// interval after the root reservation of a team ends. The interval is
/// measured from that moment, not from the last time the member computed.
///
/// * `idle_spin`: how long an idle member keeps spinning, ready to be claimed
///   again without a Rayon wake-up, before it returns its thread to Rayon.
///   Zero returns it immediately.
/// * `idle_runs_rayon`: whether the spinning member runs queued Rayon jobs in
///   the meantime. When it does, a job it picked up runs to completion
///   (including nested work) before the member is claimable again, and an
///   acquisition that finds the member in such a job refuses instead of
///   waiting for it.
///
/// While a member spins with `idle_runs_rayon == false`, its thread serves no
/// Rayon job of the pool for up to `idle_spin`.
///
/// # Examples
///
/// ```
/// use std::time::Duration;
/// use threadteam::{WaitOverride, WaitPolicy};
///
/// let base = WaitPolicy::default();
/// assert_eq!(base.idle_spin(), Duration::from_millis(200));
/// assert!(!base.idle_runs_rayon());
///
/// let tuned = base.with(WaitOverride::new().idle_spin(Duration::ZERO));
/// assert_eq!(tuned.idle_spin(), Duration::ZERO);
/// assert!(!tuned.idle_runs_rayon()); // not overridden: inherited
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WaitPolicy {
    idle_spin: Duration,
    idle_runs_rayon: bool,
}

impl WaitPolicy {
    /// The pool default: spin for 200 ms, run no Rayon jobs while idle.
    pub const DEFAULT: WaitPolicy = WaitPolicy {
        idle_spin: Duration::from_millis(200),
        idle_runs_rayon: false,
    };

    /// A policy with both values given.
    pub const fn new(idle_spin: Duration, idle_runs_rayon: bool) -> Self {
        WaitPolicy {
            idle_spin,
            idle_runs_rayon,
        }
    }

    /// Spin time of an unreserved idle member before it returns to Rayon.
    pub const fn idle_spin(&self) -> Duration {
        self.idle_spin
    }

    /// Whether an unreserved idle member runs Rayon jobs while it spins.
    pub const fn idle_runs_rayon(&self) -> bool {
        self.idle_runs_rayon
    }

    /// This policy with the fields set in `over` replaced.
    #[must_use]
    pub fn with(self, over: WaitOverride) -> Self {
        WaitPolicy {
            idle_spin: over.idle_spin.unwrap_or(self.idle_spin),
            idle_runs_rayon: over.idle_runs_rayon.unwrap_or(self.idle_runs_rayon),
        }
    }
}

impl Default for WaitPolicy {
    fn default() -> Self {
        WaitPolicy::DEFAULT
    }
}

/// A partial [`WaitPolicy`] for one operation: unset fields are inherited.
///
/// # Examples
///
/// ```
/// use std::time::Duration;
/// use threadteam::{Exec, Pool, WaitOverride};
///
/// let pool = Pool::builder().threads(2).build()?;
/// let mut exec = Exec::new(&pool, 2)?;
/// exec.team_with(WaitOverride::new().idle_spin(Duration::ZERO), |team| {
///     assert_eq!(team.policy().idle_spin(), Duration::ZERO);
/// })
/// .expect("an idle pool grants the team");
/// # Ok::<(), threadteam::Error>(())
/// ```
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct WaitOverride {
    pub(crate) idle_spin: Option<Duration>,
    pub(crate) idle_runs_rayon: Option<bool>,
}

impl WaitOverride {
    /// An override that changes nothing.
    pub const fn new() -> Self {
        WaitOverride {
            idle_spin: None,
            idle_runs_rayon: None,
        }
    }

    /// Overrides the idle spin time.
    #[must_use]
    pub const fn idle_spin(mut self, spin: Duration) -> Self {
        self.idle_spin = Some(spin);
        self
    }

    /// Overrides whether idle members run Rayon jobs.
    #[must_use]
    pub const fn idle_runs_rayon(mut self, on: bool) -> Self {
        self.idle_runs_rayon = Some(on);
        self
    }
}
