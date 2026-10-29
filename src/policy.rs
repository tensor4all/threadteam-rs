//! Waiting policy of unreserved idle team members.

use core::time::Duration;

/// How an idle member behaves after its reservation ended.
///
/// A member that is *reserved* neither runs unrelated tasks nor leaves
/// on a timeout, whatever the policy says: a policy only governs the idle
/// interval after the root reservation of a team ends. The interval is
/// measured from that moment, not from the last time the member computed.
///
/// * `idle_spin`: whether an idle member keeps spinning, ready to be claimed
///   again without a wake-up, before it sleeps. Zero puts it to sleep at once;
///   any other value lets it spin for the pool's spin budget of polls
///   ([`PoolBuilder::spin_polls`](crate::PoolBuilder::spin_polls)).
/// * `idle_runs_rayon`: **no effect** since threadteam runs its own worker
///   threads (`docs/design/own-threads.md`): an idle worker always serves the
///   pool's queued tasks, and a worker that runs one is not claimable until it
///   finishes. Kept so that existing code compiles.
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
    /// The pool default: an idle spin of 200 ms (bounded by the spin budget).
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

    /// Spin time of an unreserved idle member before it sleeps.
    pub const fn idle_spin(&self) -> Duration {
        self.idle_spin
    }

    /// The `idle_runs_rayon` setting (no effect; see [`WaitPolicy`]).
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
/// let pool = Pool::builder().threads(2).acquire_timeout(std::time::Duration::from_secs(10)).build()?;
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

    /// Overrides `idle_runs_rayon` (no effect; see [`WaitPolicy`]).
    #[must_use]
    pub const fn idle_runs_rayon(mut self, on: bool) -> Self {
        self.idle_runs_rayon = Some(on);
        self
    }
}

/// Scheduling class of the pool's worker threads (macOS quality of service).
///
/// Applied to each worker when it starts, with `pthread_set_qos_class_self_np`
/// on Apple targets; on other targets it has no effect. No platform type
/// appears here.
///
/// * The builder default is [`Qos::UserInitiated`] on Apple targets: numerical
///   workers should not run at the system's default class.
/// * [`Qos::Default`] asks for the system default class explicitly.
/// * The **leading (calling) thread** takes part in every team section, so its
///   class matters as much as the workers'. `threadteam` never changes the
///   caller's QoS; set it where the thread is created.
///
/// # Examples
///
/// ```
/// use threadteam::{Pool, Qos};
///
/// let pool = Pool::builder().threads(2).qos(Qos::UserInteractive).build()?;
/// # let _ = pool;
/// # Ok::<(), threadteam::Error>(())
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum Qos {
    /// The system default class (`QOS_CLASS_DEFAULT`).
    Default,
    /// `QOS_CLASS_UTILITY`: long-running work the user is not waiting on.
    Utility,
    /// `QOS_CLASS_USER_INITIATED`: work the user started and waits for.
    UserInitiated,
    /// `QOS_CLASS_USER_INTERACTIVE`: the highest class.
    UserInteractive,
}

impl Qos {
    /// What a pool uses when [`PoolBuilder::qos`](crate::PoolBuilder::qos) was not called.
    #[cfg_attr(not(feature = "parallel"), allow(dead_code))]
    pub(crate) const PLATFORM_DEFAULT: Option<Qos> = if cfg!(target_vendor = "apple") {
        Some(Qos::UserInitiated)
    } else {
        None
    };

    /// Sets the calling thread's class (a no-op off Apple targets).
    #[cfg(feature = "parallel")]
    pub(crate) fn apply_to_current_thread(self) {
        self.apply_to_current_thread_always();
    }

    #[allow(dead_code)] // used by `raise_current_thread_qos` on Apple targets
    pub(crate) fn apply_to_current_thread_always(self) {
        #[cfg(target_vendor = "apple")]
        {
            let class = match self {
                Qos::Default => libc::qos_class_t::QOS_CLASS_DEFAULT,
                Qos::Utility => libc::qos_class_t::QOS_CLASS_UTILITY,
                Qos::UserInitiated => libc::qos_class_t::QOS_CLASS_USER_INITIATED,
                Qos::UserInteractive => libc::qos_class_t::QOS_CLASS_USER_INTERACTIVE,
            };
            // SAFETY: plain libpthread call on the current thread; a failure
            // (nonzero return) leaves the thread's class unchanged.
            unsafe {
                libc::pthread_set_qos_class_self_np(class, 0);
            }
        }
        #[cfg(not(target_vendor = "apple"))]
        let _ = self;
    }
}

/// Raises the calling thread's [`Qos`] to at least `qos`; never lowers it.
///
/// The current class is read first (`pthread_get_qos_class_np`) and the call
/// does nothing when it already is `qos` or higher, so a thread the application
/// runs at `UserInteractive` stays there. A no-op on non-Apple targets.
///
/// An engine calls this once for the thread that will lead team sections: the
/// leader computes alongside the members, so a leader at a lower class than the
/// workers slows every section. `threadteam` itself never changes the QoS of a
/// caller; whether to call this is the engine's choice (and can be offered as an
/// opt-out).
///
/// # Examples
///
/// ```
/// use threadteam::{raise_current_thread_qos, Qos};
///
/// raise_current_thread_qos(Qos::UserInitiated);
/// ```
pub fn raise_current_thread_qos(qos: Qos) {
    #[cfg(target_vendor = "apple")]
    {
        use libc::qos_class_t as C;
        let rank = |c: C| match c {
            C::QOS_CLASS_USER_INTERACTIVE => 5,
            C::QOS_CLASS_USER_INITIATED => 4,
            C::QOS_CLASS_DEFAULT => 3,
            C::QOS_CLASS_UTILITY => 2,
            C::QOS_CLASS_BACKGROUND => 1,
            _ => 0,
        };
        let wanted = match qos {
            Qos::Default => C::QOS_CLASS_DEFAULT,
            Qos::Utility => C::QOS_CLASS_UTILITY,
            Qos::UserInitiated => C::QOS_CLASS_USER_INITIATED,
            Qos::UserInteractive => C::QOS_CLASS_USER_INTERACTIVE,
        };
        let mut current = C::QOS_CLASS_UNSPECIFIED;
        let mut relative = 0;
        // SAFETY: reads the calling thread's class into two local variables.
        let ok = unsafe {
            libc::pthread_get_qos_class_np(libc::pthread_self(), &mut current, &mut relative)
        } == 0;
        if !ok || rank(current) < rank(wanted) {
            qos.apply_to_current_thread_always();
        }
    }
    #[cfg(not(target_vendor = "apple"))]
    let _ = qos;
}
