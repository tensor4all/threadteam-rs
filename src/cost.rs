//! Dispatch costs of the three kinds of token and the rule that uses them.
//!
//! A token states what one parallel section costs it; a kernel states how long
//! its region takes on one thread; the region runs in parallel when the serial
//! time is at least [`PAYS`] times the dispatch cost of every section it opens.
//! The design record is `docs/design/thresholds.md`.

use core::num::NonZeroUsize;
use core::time::Duration;

/// The one multiple of the stack: a region runs in parallel when its serial
/// estimate is at least `PAYS * sections * dispatch_cost`.
///
/// At the threshold a region that scales three times on six threads is 1.5 times
/// faster (`3 / (3/3 + 1)`) and one that scales only twice is not slower.
pub const PAYS: f64 = 3.0;

/// Which kind of token a dispatch cost belongs to.
///
/// The kind is read from the token and the calling thread when the cost is
/// asked for.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DispatchKind {
    /// A token with authority over reserved members (team): a section is a
    /// publication to spinning members plus a join.
    Team,
    /// A plain token used by a worker of its own pool: the join tree runs on the
    /// caller, nothing is installed.
    InPool,
    /// A plain token used by a thread outside the pool: one install, a wake of
    /// the workers and a wait.
    Outside,
}

/// Dispatch cost of one kind of token at the measured widths 2, 4 and 6.
///
/// The model: linear interpolation between the anchors, the value at width 2
/// below it, and above width 6 the width 6 value held flat (the Linux data at
/// width 8 is below the width 6 value, so flat is conservative and simple).
/// The model is evaluated once, when the value is built (`new` and `flat` are
/// `const`), into a table of nanoseconds, so a lookup is an index. Width 1 never
/// reaches the model: a token of one slot has no parallel section, see
/// [`Exec::dispatch_cost`](crate::Exec::dispatch_cost).
///
/// # Examples
///
/// ```
/// use std::num::NonZeroUsize;
/// use std::time::Duration;
/// use threadteam::WidthCosts;
///
/// let us = Duration::from_micros;
/// let c = WidthCosts::new([us(3), us(2), us(1)]);
/// assert_eq!(c.at(NonZeroUsize::new(2).unwrap()), us(3));
/// assert_eq!(c.at(NonZeroUsize::new(3).unwrap()), Duration::from_nanos(2500));
/// assert_eq!(c.at(NonZeroUsize::new(12).unwrap()), us(1));
/// assert_eq!(c.min(), us(1));
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WidthCosts {
    /// Nanoseconds at widths 1..=6 (index `w - 1`), the model evaluated once when
    /// the table is built; every wider token reads the last entry.
    ns: [u64; TABLE],
}

/// The last anchor: the model is flat from here on, so a table to this width
/// holds the whole model.
const TABLE: usize = 6;

const ANCHORS: [u64; 3] = [2, 4, 6];

const fn to_ns(d: Duration) -> u64 {
    let n = d.as_nanos();
    if n > u64::MAX as u128 {
        u64::MAX
    } else {
        n as u64
    }
}

/// The model at width `w` from the anchor values; the reference the table is
/// built from (linear between anchors, flat outside).
const fn model_ns(anchors: &[u64; 3], w: u64) -> u64 {
    if w <= ANCHORS[0] {
        anchors[0]
    } else if w >= ANCHORS[2] {
        // Flat above the last anchor.
        anchors[2]
    } else {
        let i = (w > ANCHORS[1]) as usize;
        let (w0, w1) = (ANCHORS[i], ANCHORS[i + 1]);
        let (c0, c1) = (anchors[i] as i128, anchors[i + 1] as i128);
        let v = c0 + (c1 - c0) * (w - w0) as i128 / (w1 - w0) as i128;
        if v < 0 {
            0
        } else {
            v as u64
        }
    }
}

impl WidthCosts {
    /// The costs measured at widths 2, 4 and 6.
    pub const fn new(at_2_4_6: [Duration; 3]) -> Self {
        let anchors = [to_ns(at_2_4_6[0]), to_ns(at_2_4_6[1]), to_ns(at_2_4_6[2])];
        let mut ns = [0u64; TABLE];
        let mut k = 0;
        while k < TABLE {
            ns[k] = model_ns(&anchors, k as u64 + 1);
            k += 1;
        }
        WidthCosts { ns }
    }

    /// The same cost at every anchor (the model of an unmeasured host).
    pub const fn flat(cost: Duration) -> Self {
        WidthCosts::new([cost; 3])
    }

    /// Nanoseconds of a section of `width` participants: one index.
    #[inline]
    pub(crate) fn ns_at(&self, width: NonZeroUsize) -> u64 {
        self.ns[(width.get() - 1).min(TABLE - 1)]
    }

    /// The cost of a section of `width` participants, by the model above.
    #[inline]
    pub fn at(&self, width: NonZeroUsize) -> Duration {
        Duration::from_nanos(self.ns_at(width))
    }

    /// The cheapest cost over all widths (the model is evaluated at every width,
    /// so this is right also for a table whose cost rises with width). A region
    /// whose serial estimate is below `PAYS * min()` pays at no width, which is
    /// the cheap rejection before a per-width search.
    #[inline]
    pub fn min(&self) -> Duration {
        let mut m = self.ns[0];
        let mut k = 1;
        while k < TABLE {
            if self.ns[k] < m {
                m = self.ns[k];
            }
            k += 1;
        }
        Duration::from_nanos(m)
    }
}

/// Dispatch costs of a pool's three kinds of token.
///
/// [`host_default`](DispatchCosts::host_default) holds measured values for the
/// host classes it knows (Apple silicon on macOS, QoS `UserInitiated`, the pool
/// default; x86_64 Linux) and a conservative generic table for every other host. A pool owner
/// that measured its own host overrides them with
/// [`PoolBuilder::dispatch_costs`](crate::PoolBuilder::dispatch_costs).
///
/// The values are *steady state*: sections back to back, with shares that carry
/// work. A section after the members parked costs more (about 27 us on the
/// measured host; after a gap of a millisecond or more 6 us on a team, and 30 to
/// 45 us for a plain token from outside). That penalty is documented, not part
/// of the values: a rule that read the state would keep sparse callers serial
/// and never warm the team.
///
/// # Examples
///
/// ```
/// use std::num::NonZeroUsize;
/// use std::time::Duration;
/// use threadteam::{DispatchCosts, DispatchKind, WidthCosts};
///
/// let costs = DispatchCosts::host_default()
///     .with(DispatchKind::Outside, WidthCosts::flat(Duration::from_micros(50)));
/// let w6 = NonZeroUsize::new(6).unwrap();
/// assert_eq!(costs.cost(DispatchKind::Outside, w6), Duration::from_micros(50));
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DispatchCosts {
    team: WidthCosts,
    in_pool: WidthCosts,
    outside: WidthCosts,
}

const fn us_f(x: f64) -> Duration {
    // Microseconds with one decimal, for the tables below.
    Duration::from_nanos((x * 1000.0) as u64)
}

impl DispatchCosts {
    /// Costs of the three kinds.
    pub fn new(team: WidthCosts, in_pool: WidthCosts, outside: WidthCosts) -> Self {
        DispatchCosts {
            team,
            in_pool,
            outside,
        }
    }

    /// Measured on an Apple M5 Max, 6 threads, QoS `UserInitiated`, from the
    /// back to back sections whose shares carry 10 us of work
    /// (`docs/design/thresholds-data/`), in microseconds at widths 2, 4, 6:
    /// team 3.0, 2.3, 1.4; in pool 6.0, 11, 5.5; outside 14, 18, 19.
    pub fn apple_m5() -> Self {
        DispatchCosts {
            team: WidthCosts::new([us_f(3.0), us_f(2.3), us_f(1.4)]),
            in_pool: WidthCosts::new([us_f(6.0), us_f(11.0), us_f(5.5)]),
            outside: WidthCosts::new([us_f(14.0), us_f(18.0), us_f(19.0)]),
        }
    }

    /// Measured on an AMD Ryzen AI 9 HX 470 under Linux, 8 threads on cores
    /// 4-11, boost off (2.0 GHz), from the back to back sections whose shares
    /// carry 10 us of work (`faer-benchmark`
    /// `data/probes/20261026-awake-team-linux`, `docs/design/thresholds.md`
    /// section 3.6), in microseconds at widths 2, 4, 6: team 0.6, 0.7, 0.9;
    /// in pool 3.0, 3.6, 3.5; outside 9.7, 10.1, 8.7. The team row is the one
    /// measured after the per-member parker fix. Width 8 measured 0.98 us on a
    /// team, 8 % above the flat model (0.9).
    pub fn linux_x86() -> Self {
        DispatchCosts {
            team: WidthCosts::new([us_f(0.6), us_f(0.7), us_f(0.9)]),
            in_pool: WidthCosts::new([us_f(3.0), us_f(3.6), us_f(3.5)]),
            outside: WidthCosts::new([us_f(9.7), us_f(10.1), us_f(8.7)]),
        }
    }

    /// The conservative table of a host that is not measured: 3 us for a team,
    /// 15 us in the pool and 40 us from outside, at every width. It errs towards
    /// staying serial.
    pub fn generic() -> Self {
        DispatchCosts {
            team: WidthCosts::flat(us_f(3.0)),
            in_pool: WidthCosts::flat(us_f(15.0)),
            outside: WidthCosts::flat(us_f(40.0)),
        }
    }

    /// The table of this host class: [`apple_m5`](Self::apple_m5) on macOS with
    /// an aarch64 CPU, [`linux_x86`](Self::linux_x86) on Linux with an x86_64
    /// CPU, [`generic`](Self::generic) elsewhere.
    pub fn host_default() -> Self {
        #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
        {
            Self::apple_m5()
        }
        #[cfg(all(target_os = "linux", target_arch = "x86_64"))]
        {
            Self::linux_x86()
        }
        #[cfg(not(any(
            all(target_os = "macos", target_arch = "aarch64"),
            all(target_os = "linux", target_arch = "x86_64")
        )))]
        {
            Self::generic()
        }
    }

    /// These costs with the table of `kind` replaced.
    #[must_use]
    pub fn with(mut self, kind: DispatchKind, costs: WidthCosts) -> Self {
        match kind {
            DispatchKind::Team => self.team = costs,
            DispatchKind::InPool => self.in_pool = costs,
            DispatchKind::Outside => self.outside = costs,
        }
        self
    }

    /// The cheapest cost of `kind` over all widths ([`WidthCosts::min`]): a
    /// region below `PAYS * min_cost(kind)` pays at no width of that kind.
    pub fn min_cost(&self, kind: DispatchKind) -> Duration {
        match kind {
            DispatchKind::Team => self.team.min(),
            DispatchKind::InPool => self.in_pool.min(),
            DispatchKind::Outside => self.outside.min(),
        }
    }

    /// The cost of one section of `width` participants on a token of `kind`.
    #[inline]
    pub fn cost(&self, kind: DispatchKind, width: NonZeroUsize) -> Duration {
        match kind {
            DispatchKind::Team => self.team.at(width),
            DispatchKind::InPool => self.in_pool.at(width),
            DispatchKind::Outside => self.outside.at(width),
        }
    }
}

impl Default for DispatchCosts {
    fn default() -> Self {
        DispatchCosts::host_default()
    }
}

/// [`PAYS`] as the integer the comparison uses (checked equal in the tests).
const PAYS_INT: u128 = 3;

/// `est_ns >= PAYS * sections * cost_ns` in integers; `cost_ns` must not be the
/// saturated serial cost.
#[inline]
fn pays_ns(est_ns: u64, sections: u32, cost_ns: u64) -> bool {
    u128::from(est_ns) >= PAYS_INT * u128::from(sections) * u128::from(cost_ns)
}

/// The one rule: parallel exactly when
/// `estimate >= PAYS * sections * dispatch_cost`, compared in integer
/// nanoseconds.
///
/// `dispatch_cost` comes from [`Exec::dispatch_cost`](crate::Exec::dispatch_cost)
/// (or a consumer's own token type that delegates to it). `Duration::MAX`
/// (a serial token) never pays; `sections == 0` always pays (nothing to
/// dispatch). `sections` is the real number of dispatches the region performs,
/// not a tuning knob.
///
/// # Examples
///
/// ```
/// use std::time::Duration;
/// use threadteam::pays_at;
///
/// let d = Duration::from_micros(2);
/// assert!(pays_at(Duration::from_micros(6), 1, d));
/// assert!(!pays_at(Duration::from_micros(5), 1, d));
/// assert!(!pays_at(Duration::from_micros(6), 2, d)); // two sections need 12 us
/// assert!(!pays_at(Duration::from_secs(1), 1, Duration::MAX));
/// ```
pub fn pays_at(estimate: Duration, sections: u32, dispatch_cost: Duration) -> bool {
    if dispatch_cost == Duration::MAX {
        return false;
    }
    if sections == 0 {
        return true;
    }
    pays_ns(to_ns(estimate), sections, to_ns(dispatch_cost))
}

/// The width in `1..=budget` that minimises `sections * D(k) + estimate / k`,
/// where `D` is `cost` and a width only counts when the region [`pays_at`] it;
/// 1 when no width does. Ties go to the smaller width. Allocation-free.
///
/// # Examples
///
/// ```
/// use std::num::NonZeroUsize;
/// use std::time::Duration;
/// use threadteam::width_for_by;
///
/// let budget = NonZeroUsize::new(6).unwrap();
/// let flat = |_| Duration::from_micros(2);
/// assert_eq!(width_for_by(Duration::from_micros(1), 1, budget, flat).get(), 1);
/// assert_eq!(width_for_by(Duration::from_millis(1), 1, budget, flat).get(), 6);
/// ```
pub fn width_for_by(
    estimate: Duration,
    sections: u32,
    budget: NonZeroUsize,
    cost: impl Fn(NonZeroUsize) -> Duration,
) -> NonZeroUsize {
    let est_ns = to_ns(estimate);
    let est = est_ns as f64;
    let mut best = NonZeroUsize::MIN;
    let mut best_obj = est;
    for k in 2..=budget.get() {
        let Some(width) = NonZeroUsize::new(k) else {
            continue;
        };
        let d = cost(width);
        if d == Duration::MAX {
            continue;
        }
        let d_ns = to_ns(d);
        if sections != 0 && !pays_ns(est_ns, sections, d_ns) {
            continue;
        }
        let obj = f64::from(sections) * d_ns as f64 + est / k as f64;
        if obj < best_obj {
            best = width;
            best_obj = obj;
        }
    }
    best
}

#[cfg(test)]
mod tests {
    use super::*;

    fn nz(n: usize) -> NonZeroUsize {
        NonZeroUsize::new(n).unwrap()
    }
    fn us(n: u64) -> Duration {
        Duration::from_micros(n)
    }

    #[test]
    fn anchors_are_returned_exactly() {
        let c = DispatchCosts::apple_m5();
        assert_eq!(c.cost(DispatchKind::Team, nz(2)), us_f(3.0));
        assert_eq!(c.cost(DispatchKind::Team, nz(4)), us_f(2.3));
        assert_eq!(c.cost(DispatchKind::Team, nz(6)), us_f(1.4));
        assert_eq!(c.cost(DispatchKind::Outside, nz(6)), us_f(19.0));
    }

    #[test]
    fn interpolation_and_extrapolation() {
        let c = WidthCosts::new([us(3), us(2), us(1)]);
        assert_eq!(c.at(nz(1)), us(3));
        assert_eq!(c.at(nz(3)), Duration::from_nanos(2500));
        assert_eq!(c.at(nz(5)), Duration::from_nanos(1500));
        // Flat above the last anchor.
        assert_eq!(c.at(nz(8)), us(1));
        assert_eq!(c.at(nz(12)), us(1));
        // Non-monotone anchors interpolate without underflow.
        let n = WidthCosts::new([us(1), us(5), us(0)]);
        assert_eq!(n.at(nz(5)), Duration::from_nanos(2500));
    }

    /// The model as it was before the table: i128 interpolation on every call.
    fn old_at(ns: [u64; 3], w: u64) -> u64 {
        if w <= 2 {
            ns[0]
        } else if w >= 6 {
            ns[2]
        } else {
            let i = usize::from(w > 4);
            let (w0, w1) = ([2u64, 4, 6][i], [2u64, 4, 6][i + 1]);
            let (c0, c1) = (i128::from(ns[i]), i128::from(ns[i + 1]));
            (c0 + (c1 - c0) * i128::from(w - w0) / i128::from(w1 - w0)).max(0) as u64
        }
    }

    #[test]
    fn table_equals_the_interpolation_for_every_width() {
        let tables = [
            [3000, 2300, 1400],
            [2800, 2200, 1700],
            [14000, 18000, 19000],
            [9500, 10100, 8800],
            [3100, 3600, 3500],
            [1000, 5000, 0],
            [0, 0, 0],
            [7, 3, 11],
        ];
        for ns in tables {
            let c = WidthCosts::new(ns.map(Duration::from_nanos));
            for w in 1..=64u64 {
                assert_eq!(
                    c.at(nz(w as usize)),
                    Duration::from_nanos(old_at(ns, w)),
                    "{ns:?} width {w}"
                );
            }
        }
        for d in [
            DispatchCosts::apple_m5(),
            DispatchCosts::linux_x86(),
            DispatchCosts::generic(),
        ] {
            for kind in [
                DispatchKind::Team,
                DispatchKind::InPool,
                DispatchKind::Outside,
            ] {
                let m = (1..=64).map(|w| d.cost(kind, nz(w))).min().unwrap();
                assert_eq!(d.min_cost(kind), m);
            }
        }
    }

    #[test]
    fn min_is_right_for_a_rising_table_and_a_falling_one() {
        let rising = WidthCosts::new([us(2), us(5), us(9)]);
        assert_eq!(rising.min(), us(2));
        let falling = WidthCosts::new([us(9), us(5), us(2)]);
        assert_eq!(falling.min(), us(2));
        let dip = WidthCosts::new([us(9), us(1), us(9)]);
        assert_eq!(dip.min(), us(1));
        assert_eq!(WidthCosts::flat(us(4)).min(), us(4));
    }

    #[test]
    fn pays_matches_the_float_rule() {
        assert_eq!(PAYS, PAYS_INT as f64);
        for cost in [1u64, 700, 1700, 2800, 19_000] {
            for sections in [1u32, 2, 5] {
                let edge = 3 * u64::from(sections) * cost;
                for est in [edge - 1, edge, edge + 1] {
                    let f = est as f64 >= PAYS * f64::from(sections) * cost as f64;
                    assert_eq!(
                        pays_at(
                            Duration::from_nanos(est),
                            sections,
                            Duration::from_nanos(cost)
                        ),
                        f
                    );
                }
            }
        }
    }

    #[test]
    fn pays_is_the_multiple_of_the_cost() {
        let d = us(2);
        assert!(pays_at(us(6), 1, d));
        assert!(!pays_at(Duration::from_nanos(5999), 1, d));
        assert!(pays_at(us(12), 2, d));
        assert!(!pays_at(us(11), 2, d));
        assert!(!pays_at(us(1_000_000), 1, Duration::MAX));
        assert!(pays_at(Duration::ZERO, 0, d));
    }

    #[test]
    fn width_for_minimises_the_model() {
        let budget = nz(6);
        let d = |k: NonZeroUsize| us(2) * k.get() as u32;
        // 2 us * k + est / k: for est = 200 us the optimum is k = 10 clamped to
        // the budget; for est = 50 us it is k = 5 (obj 20) against k = 4 (20.5)
        // and k = 6 (20.3).
        assert_eq!(width_for_by(us(200), 1, budget, d).get(), 6);
        assert_eq!(width_for_by(us(50), 1, budget, d).get(), 5);
        // Below the multiple at every width: serial.
        assert_eq!(width_for_by(us(5), 1, budget, d).get(), 1);
        // More sections move the optimum down.
        assert!(width_for_by(us(200), 4, budget, d).get() < 6);
        // A serial cost never pays.
        assert_eq!(
            width_for_by(us(1_000_000), 1, budget, |_| Duration::MAX).get(),
            1
        );
        assert_eq!(width_for_by(us(1_000_000), 1, nz(1), d).get(), 1);
    }
}
