//! Dispatch costs of the three kinds of token and the rule that uses them.
//!
//! A token states what one parallel section costs it; a kernel states how long
//! its region takes on one thread; the region runs in parallel when the serial
//! time is at least [`PAYS`] times the dispatch cost of every section it opens.
//! The design record is `docs/design/thresholds.md`.

use core::num::NonZeroUsize;
use core::time::Duration;
use std::sync::OnceLock;

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

/// Dispatch cost of one kind of token, stored as the measured anchors.
///
/// The model at width `w`: the value at the first anchor below it, linear
/// interpolation between two anchors, and **linear extrapolation above the last
/// anchor** `w_m`, `ns[w_m] * w / w_m`, saturating at `u64::MAX`. Extrapolation
/// never falls below the last anchor and no width above it is charged a flat
/// cost: a wide section pays at least in proportion to its width, which is the
/// conservative reading of the measurements.
///
/// The anchors are the widths a row was measured at. [`new`](Self::new) and
/// [`flat`](Self::flat) carry widths 2, 4 and 6;
/// [`from_anchors`](Self::from_anchors) takes any strictly increasing widths from
/// 2 to `u16::MAX`, at most [`MAX_ANCHORS`](Self::MAX_ANCHORS) of them, so a row
/// measured on a many-core host replaces the extrapolation where it has data.
/// The values are integer nanoseconds and a lookup scans at most
/// [`MAX_ANCHORS`](Self::MAX_ANCHORS) entries, so it allocates nothing and the
/// comparisons of the rule stay exact. Width 1 never reaches the model: a token
/// of one slot has no parallel section, see
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
/// assert_eq!(c.at(NonZeroUsize::new(6).unwrap()), us(1));
/// // Above the last anchor (6) the cost grows with the width.
/// assert_eq!(c.at(NonZeroUsize::new(12).unwrap()), us(2));
/// assert_eq!(c.at(NonZeroUsize::new(48).unwrap()), us(8));
/// assert_eq!(c.min(), us(1));
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WidthCosts {
    /// The measured widths, strictly increasing, in `widths[..len]`.
    widths: [u16; CAPACITY],
    /// The cost in nanoseconds at each anchor, in `ns[..len]`.
    ns: [u64; CAPACITY],
    /// The number of anchors. The slots past it are zeroed by construction, so
    /// the derived `PartialEq` compares only the anchors.
    len: u8,
}

/// The capacity of a row: enough for a measurement every power of two from 2 to
/// 4096 plus the widths a many-core campaign reports.
const CAPACITY: usize = 12;

const fn to_ns(d: Duration) -> u64 {
    let n = d.as_nanos();
    if n > u64::MAX as u128 {
        u64::MAX
    } else {
        n as u64
    }
}

/// Linear interpolation between two anchors in `i128`, clamped at zero as the
/// model it replaces was.
#[inline]
fn interpolate(c0: u64, c1: u64, w0: u16, w1: u16, w: usize) -> u64 {
    let (c0, c1) = (i128::from(c0), i128::from(c1));
    let v = c0 + (c1 - c0) * (w - usize::from(w0)) as i128 / i128::from(w1 - w0);
    if v < 0 {
        0
    } else {
        v as u64
    }
}

impl WidthCosts {
    /// The largest number of anchors a row holds.
    pub const MAX_ANCHORS: usize = CAPACITY;

    /// The costs measured at widths 2, 4 and 6.
    pub const fn new(at_2_4_6: [Duration; 3]) -> Self {
        Self::store(&[(2, at_2_4_6[0]), (4, at_2_4_6[1]), (6, at_2_4_6[2])])
    }

    /// The same cost at every anchor 2, 4 and 6 (the model of an unmeasured
    /// host). Above width 6 the row grows with the width like every other row; a
    /// host that measured equal costs out to a wide width says so with
    /// [`from_anchors`](Self::from_anchors).
    pub const fn flat(cost: Duration) -> Self {
        WidthCosts::new([cost; 3])
    }

    /// A row from measured anchors `(width, cost)`: widths 2 to `u16::MAX`,
    /// strictly increasing, at most [`MAX_ANCHORS`](Self::MAX_ANCHORS) of them.
    ///
    /// `None` for empty input, for more than [`MAX_ANCHORS`](Self::MAX_ANCHORS)
    /// anchors, for a width below 2 or above `u16::MAX`, and for widths that are
    /// not strictly increasing. No panic, no allocation, usable in a `const`
    /// item.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::num::NonZeroUsize;
    /// use std::time::Duration;
    /// use threadteam::WidthCosts;
    ///
    /// let us = Duration::from_micros;
    /// let a = WidthCosts::from_anchors(&[(2, us(3)), (4, us(2)), (8, us(1))]).unwrap();
    /// assert_eq!(a.at(NonZeroUsize::new(6).unwrap()), Duration::from_nanos(1500));
    /// // Above the last anchor (8) the cost grows with the width.
    /// assert_eq!(a.at(NonZeroUsize::new(16).unwrap()), us(2));
    /// assert!(WidthCosts::from_anchors(&[]).is_none());
    /// assert!(WidthCosts::from_anchors(&[(1, us(3))]).is_none());
    /// ```
    pub const fn from_anchors(anchors: &[(usize, Duration)]) -> Option<Self> {
        let n = anchors.len();
        if n == 0 || n > CAPACITY {
            return None;
        }
        let mut k = 0;
        while k < n {
            let w = anchors[k].0;
            if w < 2 || w > u16::MAX as usize {
                return None;
            }
            if k > 0 && w <= anchors[k - 1].0 {
                return None;
            }
            k += 1;
        }
        Some(Self::store(anchors))
    }

    /// Write the anchors into the fixed-capacity arrays; unused slots stay zero
    /// so that the derived `PartialEq` sees only the anchors. The caller
    /// validates.
    const fn store(anchors: &[(usize, Duration)]) -> Self {
        let mut widths = [0u16; CAPACITY];
        let mut ns = [0u64; CAPACITY];
        let mut k = 0;
        while k < CAPACITY {
            if k < anchors.len() {
                widths[k] = anchors[k].0 as u16;
                ns[k] = to_ns(anchors[k].1);
            }
            k += 1;
        }
        WidthCosts {
            widths,
            ns,
            len: anchors.len() as u8,
        }
    }

    /// Nanoseconds of a section of `width` participants: a forward scan over at
    /// most `len` anchors.
    #[inline]
    pub(crate) fn ns_at(&self, width: NonZeroUsize) -> u64 {
        let w = width.get();
        let len = usize::from(self.len);
        if w <= usize::from(self.widths[0]) {
            return self.ns[0];
        }
        let mut i = 0;
        while i + 1 < len {
            let w1 = self.widths[i + 1];
            if w <= usize::from(w1) {
                return interpolate(self.ns[i], self.ns[i + 1], self.widths[i], w1, w);
            }
            i += 1;
        }
        // Above the last anchor: at least linear in the width, saturating.
        let v = u128::from(self.ns[len - 1]) * w as u128 / u128::from(self.widths[len - 1]);
        if v > u64::MAX as u128 {
            u64::MAX
        } else {
            v as u64
        }
    }

    /// The cost of a section of `width` participants, by the model above.
    #[inline]
    pub fn at(&self, width: NonZeroUsize) -> Duration {
        Duration::from_nanos(self.ns_at(width))
    }

    /// The cheapest cost over all widths: the minimum over the anchors (neither
    /// interpolation nor the extrapolation can undershoot the anchor they start
    /// from). A region whose serial estimate is below `PAYS * min()` pays at no
    /// width, which is the cheap rejection before a per-width search.
    #[inline]
    pub fn min(&self) -> Duration {
        let len = usize::from(self.len);
        let mut m = self.ns[0];
        let mut k = 1;
        while k < len {
            if self.ns[k] < m {
                m = self.ns[k];
            }
            k += 1;
        }
        Duration::from_nanos(m)
    }
}

/// The class of the machine, keyed by the kernel dispatch it will take, not by
/// the CPU model. Detected once per process by [`HostClass::detect`].
///
/// The same enum and the same `detect` are kept in sync by hand in
/// `strided-rs`, `tcontract-rs` and `faer`'s cost tables: each crate owns a
/// table of its own kernels keyed by this class.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum HostClass {
    /// `x86_64` with AVX-512F at run time (`x86_64-avx512`).
    X86_64Avx512,
    /// `x86_64` with AVX2 and FMA, without AVX-512F (`x86_64-avx2`).
    X86_64Avx2,
    /// Any other `x86_64` (`x86_64-scalar`).
    X86_64Scalar,
    /// `aarch64` on macOS (`aarch64-apple`).
    Aarch64Apple,
    /// Any other `aarch64` (`aarch64`).
    Aarch64,
    /// Everything else (`generic`).
    Generic,
}

impl HostClass {
    /// The class of this host, detected at the first call and cached
    /// (`OnceLock`; `is_x86_feature_detected!` on x86_64).
    pub fn detect() -> Self {
        static CLASS: OnceLock<HostClass> = OnceLock::new();
        *CLASS.get_or_init(Self::detect_uncached)
    }

    #[cfg(target_arch = "x86_64")]
    fn detect_uncached() -> Self {
        if std::arch::is_x86_feature_detected!("avx512f") {
            HostClass::X86_64Avx512
        } else if std::arch::is_x86_feature_detected!("avx2")
            && std::arch::is_x86_feature_detected!("fma")
        {
            HostClass::X86_64Avx2
        } else {
            HostClass::X86_64Scalar
        }
    }

    #[cfg(all(target_arch = "aarch64", target_os = "macos"))]
    fn detect_uncached() -> Self {
        HostClass::Aarch64Apple
    }

    #[cfg(all(target_arch = "aarch64", not(target_os = "macos")))]
    fn detect_uncached() -> Self {
        HostClass::Aarch64
    }

    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    fn detect_uncached() -> Self {
        HostClass::Generic
    }

    /// The class name used in the docs and the tables: `x86_64-avx512`,
    /// `x86_64-avx2`, `x86_64-scalar`, `aarch64-apple`, `aarch64`, `generic`.
    pub const fn name(self) -> &'static str {
        match self {
            HostClass::X86_64Avx512 => "x86_64-avx512",
            HostClass::X86_64Avx2 => "x86_64-avx2",
            HostClass::X86_64Scalar => "x86_64-scalar",
            HostClass::Aarch64Apple => "aarch64-apple",
            HostClass::Aarch64 => "aarch64",
            HostClass::Generic => "generic",
        }
    }
}

/// Dispatch costs of a pool's three kinds of token.
///
/// [`for_class`](DispatchCosts::for_class) holds measured rows for the host
/// classes it knows (`x86_64-avx512`; `aarch64-apple`, QoS `UserInitiated`, the
/// pool default) and a conservative generic row for every other class;
/// [`host_default`](DispatchCosts::host_default) picks the row of the detected
/// class. A pool owner that measured its own host overrides them with
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

    /// The row of a host class. Both the dispatch costs of a row and the rates
    /// of the consumer crates' tables for the same class come from one machine
    /// in one clock state, so a class is never mixed across crates' tables.
    ///
    /// * [`HostClass::X86_64Avx512`]: measured on an AMD Ryzen AI 9 HX 470
    ///   under Linux, 8 threads on cores 4-11, boost off (2.0 GHz), from the
    ///   back to back sections whose shares carry 10 us of work (`faer-benchmark`
    ///   `data/probes/20261026-awake-team-linux`, `docs/design/thresholds.md`
    ///   section 3.6), in microseconds at widths 2, 4, 6: team 0.6, 0.7, 0.9;
    ///   in pool 3.0, 3.6, 3.5; outside 9.7, 10.1, 8.7. The team row is the one
    ///   measured after the per-member parker fix and carries the measured width
    ///   8 anchor 0.98 us; in pool and outside keep the three widths of the same
    ///   run (their width 8 values are from the pre-fix run and are not used).
    /// * [`HostClass::Aarch64Apple`]: measured on an Apple M5 Max, 6 threads,
    ///   QoS `UserInitiated`, from the back to back sections whose shares
    ///   carry 10 us of work (`docs/design/thresholds-data/`), at widths 2, 4,
    ///   6: team 3.0, 2.3, 1.4; in pool 6.0, 11, 5.5; outside 14, 18, 19.
    /// * Every other class has no measurement and takes the conservative
    ///   generic row: 3 us for a team, 15 us in the pool and 40 us from
    ///   outside at the anchors 2, 4 and 6, and above width 6 the width
    ///   proportion (32, 160 and 427 us at width 64). It errs towards staying
    ///   serial.
    ///
    /// Above the largest anchor of a row the cost grows with the width
    /// ([`WidthCosts`]); [`WidthCosts::from_anchors`] replaces that
    /// extrapolation where a host has measured anchors beyond width 6.
    pub fn for_class(class: HostClass) -> Self {
        match class {
            HostClass::X86_64Avx512 => DispatchCosts {
                team: WidthCosts::from_anchors(&[
                    (2, us_f(0.6)),
                    (4, us_f(0.7)),
                    (6, us_f(0.9)),
                    (8, us_f(0.98)),
                ])
                .expect("the avx512 team row has four valid anchors"),
                in_pool: WidthCosts::new([us_f(3.0), us_f(3.6), us_f(3.5)]),
                outside: WidthCosts::new([us_f(9.7), us_f(10.1), us_f(8.7)]),
            },
            HostClass::Aarch64Apple => DispatchCosts {
                team: WidthCosts::new([us_f(3.0), us_f(2.3), us_f(1.4)]),
                in_pool: WidthCosts::new([us_f(6.0), us_f(11.0), us_f(5.5)]),
                outside: WidthCosts::new([us_f(14.0), us_f(18.0), us_f(19.0)]),
            },
            HostClass::X86_64Avx2
            | HostClass::X86_64Scalar
            | HostClass::Aarch64
            | HostClass::Generic => DispatchCosts {
                team: WidthCosts::flat(us_f(3.0)),
                in_pool: WidthCosts::flat(us_f(15.0)),
                outside: WidthCosts::flat(us_f(40.0)),
            },
        }
    }

    /// The row of the host class this process runs on
    /// (`for_class(HostClass::detect())`).
    pub fn host_default() -> Self {
        Self::for_class(HostClass::detect())
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
    /// Measured anchors keep their values, widths between two anchors are
    /// interpolated, and widths above the largest anchor of the row grow with
    /// the width ([`WidthCosts`]).
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
        let c = DispatchCosts::for_class(HostClass::Aarch64Apple);
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
        // Above the last anchor the cost grows with the width.
        assert_eq!(c.at(nz(8)), Duration::from_nanos(1333));
        assert_eq!(c.at(nz(12)), us(2));
        // Non-monotone anchors interpolate without underflow.
        let n = WidthCosts::new([us(1), us(5), us(0)]);
        assert_eq!(n.at(nz(5)), Duration::from_nanos(2500));
        assert_eq!(n.at(nz(12)), Duration::ZERO);
    }

    #[test]
    fn a_flat_row_extrapolates_in_proportion_to_the_width() {
        let c = WidthCosts::flat(us(1));
        assert_eq!(c.at(nz(6)), us(1));
        assert_eq!(c.at(nz(12)), us(2));
        assert_eq!(c.at(nz(24)), us(4));
        assert_eq!(c.at(nz(48)), us(8));
        assert_eq!(c.at(nz(64)), Duration::from_nanos(10_666));
        assert_eq!(c.min(), us(1));
    }

    /// Above the last anchor `w_m` the cost never falls and is at least
    /// `cost(w_m) * w / w_m`.
    #[test]
    fn above_the_last_anchor_is_non_decreasing_and_at_least_linear() {
        let rows: [[Duration; 3]; 5] = [
            [us_f(0.6), us_f(0.7), us_f(0.9)],
            [us_f(3.0), us_f(3.6), us_f(3.5)],
            [us_f(9.7), us_f(10.1), us_f(8.7)],
            [us(3), us(2), us(1)],
            [us(1), us(5), us(0)],
        ];
        for row in rows {
            let c = WidthCosts::new(row);
            let cm = u128::from(to_ns(c.at(nz(6))));
            let mut prev = c.at(nz(6));
            for w in 7..=64u64 {
                let v = c.at(nz(w as usize));
                assert!(v >= prev, "{row:?}: width {w} fell");
                assert!(
                    u128::from(to_ns(v)) >= cm * u128::from(w) / 6,
                    "{row:?}: width {w} below the linear bound"
                );
                prev = v;
            }
        }
    }

    #[test]
    fn anchors_beyond_six_interpolate_and_extrapolate() {
        let c = WidthCosts::from_anchors(&[
            (2, Duration::from_nanos(8000)),
            (4, Duration::from_nanos(4000)),
            (8, Duration::from_nanos(2000)),
            (16, Duration::from_nanos(1000)),
            (32, Duration::from_nanos(500)),
            (64, Duration::from_nanos(500)),
        ])
        .unwrap();
        let ns = |w: usize| c.at(nz(w)).as_nanos() as u64;
        assert_eq!(ns(2), 8000);
        assert_eq!(ns(3), 6000); // between 2 and 4
        assert_eq!(ns(6), 3000); // between 4 and 8, past the old last anchor
        assert_eq!(ns(12), 1500); // between 8 and 16
        assert_eq!(ns(24), 750); // between 16 and 32
        assert_eq!(ns(48), 500); // between 32 and 64
        assert_eq!(ns(64), 500); // the last anchor
        assert_eq!(ns(96), 750); // above the last anchor
        assert_eq!(ns(128), 1000);
        assert_eq!(c.min(), Duration::from_nanos(500));
    }

    #[test]
    fn from_anchors_rejects_invalid_rows() {
        assert!(WidthCosts::from_anchors(&[]).is_none());
        assert!(WidthCosts::from_anchors(&[(0, us(1))]).is_none());
        assert!(WidthCosts::from_anchors(&[(1, us(1)), (2, us(1))]).is_none());
        assert!(WidthCosts::from_anchors(&[(2, us(1)), (2, us(1))]).is_none());
        assert!(WidthCosts::from_anchors(&[(4, us(1)), (2, us(1))]).is_none());
        let too_wide = [(usize::from(u16::MAX) + 1, us(1))];
        assert!(WidthCosts::from_anchors(&too_wide).is_none());
        assert!(WidthCosts::from_anchors(&[(usize::from(u16::MAX), us(1))]).is_some());
        // MAX_ANCHORS anchors fit, one more is rejected.
        let mut many = [(0usize, Duration::ZERO); WidthCosts::MAX_ANCHORS + 1];
        for (k, a) in many.iter_mut().enumerate() {
            *a = (k + 2, us(1));
        }
        assert!(WidthCosts::from_anchors(&many[..WidthCosts::MAX_ANCHORS]).is_some());
        assert!(WidthCosts::from_anchors(&many).is_none());
    }

    /// `from_anchors` is usable in a `const` item and the unused capacity does
    /// not take part in equality.
    #[test]
    fn from_anchors_is_const_and_capacity_is_invisible() {
        const ROW: WidthCosts = match WidthCosts::from_anchors(&[
            (2, Duration::from_micros(3)),
            (4, Duration::from_micros(2)),
            (8, Duration::from_micros(1)),
        ]) {
            Some(c) => c,
            None => panic!("valid anchors"),
        };
        assert_eq!(WidthCosts::MAX_ANCHORS, 12);
        assert_eq!(ROW.at(nz(6)), Duration::from_nanos(1500));
        assert_eq!(ROW.at(nz(16)), us(2));
        assert_eq!(ROW.min(), us(1));

        let three = WidthCosts::from_anchors(&[(2, us(3)), (4, us(2)), (6, us(1))]).unwrap();
        assert_eq!(three, WidthCosts::new([us(3), us(2), us(1)]));
        let four =
            WidthCosts::from_anchors(&[(2, us(3)), (4, us(2)), (6, us(1)), (8, us(1))]).unwrap();
        assert_ne!(three, four);
    }

    #[test]
    fn a_duration_max_row_never_pays() {
        // `Duration::MAX` saturates to `u64::MAX` nanoseconds in a row.
        let cap = Duration::from_nanos(u64::MAX);
        let c = WidthCosts::flat(Duration::MAX);
        assert_eq!(c.at(nz(6)), cap);
        assert_eq!(c.at(nz(64)), cap); // the extrapolation saturates
        assert_eq!(c.min(), cap);
        assert!(!pays_at(cap, 1, c.at(nz(64))));
        assert!(!pays_at(cap, 2, c.at(nz(64))));
        let mixed = WidthCosts::from_anchors(&[(2, Duration::ZERO), (4, Duration::MAX)]).unwrap();
        assert_eq!(mixed.at(nz(64)), cap);
        assert_eq!(mixed.min(), Duration::ZERO);
    }

    /// A wide region selects a useful interior width above six, not always six
    /// or the budget.
    #[test]
    fn a_wide_region_selects_an_interior_width_above_six() {
        let row = WidthCosts::flat(us(1));
        let d = |k: NonZeroUsize| row.at(k);
        let budget = nz(64);
        // obj(k) = D(k) + 24 us / k with D(k) = 1 us * k / 6 above 6: obj(12) =
        // 2 + 2 = 4 us against obj(6) = 1 + 4 = 5 us, obj(11) = 1.833 + 2.18 and
        // obj(13) = 2.167 + 1.85.
        assert_eq!(width_for_by(us(24), 1, budget, d).get(), 12);
        // A large enough estimate pushes the optimum to the budget.
        assert_eq!(width_for_by(us(1_000_000), 1, budget, d).get(), 64);
        // A region that does not pay at width 6 does not pay at 64 either.
        let below = Duration::from_nanos(3 * 1000 - 1);
        assert!(!pays_at(below, 1, row.at(nz(6))));
        assert!(!pays_at(below, 1, row.at(nz(64))));
        assert_eq!(width_for_by(below, 1, budget, d).get(), 1);
    }

    /// The model as it was before the anchors were stored: i128 interpolation on
    /// every call, flat above width 6.
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
    fn rows_keep_the_old_values_inside_the_anchors() {
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
            for w in 1..=6u64 {
                assert_eq!(
                    c.at(nz(w as usize)),
                    Duration::from_nanos(old_at(ns, w)),
                    "{ns:?} width {w}"
                );
            }
        }
        // The minimum over the anchors is the minimum over all widths: neither
        // interpolation nor the extrapolation can undershoot its endpoint.
        for d in [
            DispatchCosts::for_class(HostClass::Aarch64Apple),
            DispatchCosts::for_class(HostClass::X86_64Avx512),
            DispatchCosts::for_class(HostClass::Generic),
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

    /// The measured rows at widths 1 to 6 as the pre-anchor model gave them, so
    /// this change moves no decision inside the anchors.
    #[test]
    fn measured_rows_are_pinned_at_widths_one_to_six() {
        let avx512 = DispatchCosts::for_class(HostClass::X86_64Avx512);
        let apple = DispatchCosts::for_class(HostClass::Aarch64Apple);
        let generic = DispatchCosts::for_class(HostClass::Generic);
        for (costs, kind, expected) in [
            (
                avx512,
                DispatchKind::Team,
                [600u64, 600, 650, 700, 800, 900],
            ),
            (
                avx512,
                DispatchKind::InPool,
                [3000, 3000, 3300, 3600, 3550, 3500],
            ),
            (
                avx512,
                DispatchKind::Outside,
                [9700, 9700, 9900, 10100, 9400, 8700],
            ),
            (
                apple,
                DispatchKind::Team,
                [3000, 3000, 2650, 2300, 1850, 1400],
            ),
            (
                apple,
                DispatchKind::InPool,
                [6000, 6000, 8500, 11000, 8250, 5500],
            ),
            (
                apple,
                DispatchKind::Outside,
                [14000, 14000, 16000, 18000, 18500, 19000],
            ),
            (generic, DispatchKind::Team, [3000; 6]),
            (generic, DispatchKind::InPool, [15000; 6]),
            (generic, DispatchKind::Outside, [40000; 6]),
        ] {
            for (k, want) in expected.into_iter().enumerate() {
                assert_eq!(
                    costs.cost(kind, nz(k + 1)).as_nanos() as u64,
                    want,
                    "{kind:?} width {}",
                    k + 1
                );
            }
        }
        // The measured width-8 team anchor replaces the extrapolation there;
        // width 7 interpolates between the width-6 and width-8 team anchors.
        assert_eq!(avx512.cost(DispatchKind::Team, nz(8)), us_f(0.98));
        assert_eq!(
            avx512.cost(DispatchKind::Team, nz(7)),
            Duration::from_nanos(940)
        );
        assert_eq!(
            avx512.cost(DispatchKind::Team, nz(64)),
            Duration::from_nanos(7_840)
        );
        // The width-8 values in pool and outside come from the pre-fix run and
        // are not anchors: the extrapolation applies to them.
        assert_eq!(
            avx512.cost(DispatchKind::InPool, nz(8)),
            Duration::from_nanos(4666)
        );
        assert_eq!(
            avx512.cost(DispatchKind::Outside, nz(8)),
            Duration::from_nanos(11_600)
        );
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
