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
    /// The multi-node term, `step + per_participant * w` nanoseconds, charged on
    /// top of the row when a section spans two or more NUMA nodes; zero unless
    /// installed with [`with_multi_node`](Self::with_multi_node).
    node_step: u64,
    node_per_participant: u64,
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
    // 64-bit fast path (a 128-bit division is a library call); the same
    // truncating division, so the value is identical to the 128-bit one.
    let dw = (w - usize::from(w0)) as i64;
    if let (Ok(a), Ok(b)) = (i64::try_from(c0), i64::try_from(c1)) {
        if let Some(t) = (b - a).checked_mul(dw) {
            let v = a + t / i64::from(w1 - w0);
            return if v < 0 { 0 } else { v as u64 };
        }
    }
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
            node_step: 0,
            node_per_participant: 0,
        }
    }

    /// This row with a multi-node term: a section whose participants span two
    /// or more NUMA nodes costs `step + per_participant * width` on top of the
    /// row. One step for every node count from two up; the row itself stays the
    /// single-node cost. The rows this crate builds carry no term (both zero).
    ///
    /// The node count is that of the placement this crate made: a team token
    /// counts the nodes its pinned participants are on (see
    /// [`Exec::dispatch_cost`](crate::Exec::dispatch_cost)); an unplaced team and
    /// a plain token count one node.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::num::NonZeroUsize;
    /// use std::time::Duration;
    /// use threadteam::WidthCosts;
    ///
    /// let us = Duration::from_micros;
    /// let n = |k| NonZeroUsize::new(k).unwrap();
    /// let row = WidthCosts::flat(us(1)).with_multi_node(us(2), Duration::from_nanos(100));
    /// assert_eq!(row.at_nodes(n(4), n(1)), us(1));
    /// assert_eq!(row.at_nodes(n(4), n(2)), Duration::from_nanos(3400));
    /// assert_eq!(row.at_nodes(n(4), n(4)), Duration::from_nanos(3400));
    /// ```
    #[must_use]
    pub const fn with_multi_node(self, step: Duration, per_participant: Duration) -> Self {
        WidthCosts {
            node_step: to_ns(step),
            node_per_participant: to_ns(per_participant),
            ..self
        }
    }

    /// Whether a multi-node term is installed: only then is a node count worth
    /// reading.
    #[inline]
    pub(crate) fn has_multi_node(&self) -> bool {
        self.node_step != 0 || self.node_per_participant != 0
    }

    /// [`ns_at`](Self::ns_at) on `nodes` NUMA nodes.
    #[inline]
    pub(crate) fn ns_at_nodes(&self, width: NonZeroUsize, nodes: NonZeroUsize) -> u64 {
        let row = self.ns_at(width);
        if nodes.get() < 2 {
            return row;
        }
        let slope = self.node_per_participant.saturating_mul(width.get() as u64);
        row.saturating_add(self.node_step).saturating_add(slope)
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
        // Above the last anchor: at least linear in the width, saturating
        // (64-bit when the product fits, as above).
        let (last, w_m) = (self.ns[len - 1], u64::from(self.widths[len - 1]));
        if let Some(p) = last.checked_mul(w as u64) {
            return p / w_m;
        }
        let v = u128::from(last) * w as u128 / u128::from(w_m);
        if v > u64::MAX as u128 {
            u64::MAX
        } else {
            v as u64
        }
    }

    /// The cost of a section of `width` participants on one NUMA node, by the
    /// model above: [`at_nodes(width, 1)`](Self::at_nodes).
    #[inline]
    pub fn at(&self, width: NonZeroUsize) -> Duration {
        Duration::from_nanos(self.ns_at(width))
    }

    /// The cost of a section of `width` participants spanning `nodes` NUMA
    /// nodes: [`at`](Self::at), plus the multi-node term of
    /// [`with_multi_node`](Self::with_multi_node) when `nodes >= 2`.
    #[inline]
    pub fn at_nodes(&self, width: NonZeroUsize, nodes: NonZeroUsize) -> Duration {
        Duration::from_nanos(self.ns_at_nodes(width, nodes))
    }

    /// The cheapest cost over all widths and node counts: the minimum over the
    /// anchors (an interpolated value lies between its two anchors, the
    /// extrapolation never falls below the last anchor, and the multi-node term
    /// is never negative). A region whose serial estimate is below `PAYS * min()` pays at no
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
/// pool default; the team kind of `x86_64-avx2`) and a conservative generic row
/// for everything else;
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

/// A measured row built at compile time: an invalid anchor list fails the
/// build, never a caller.
const fn anchored(anchors: &[(usize, Duration)]) -> WidthCosts {
    match WidthCosts::from_anchors(anchors) {
        Some(costs) => costs,
        None => panic!("invalid measured anchors"),
    }
}

/// `x86_64-avx512` team row (post-fix run, with its width 8 anchor).
const AVX512_TEAM: WidthCosts = anchored(&[
    (2, us_f(0.6)),
    (4, us_f(0.7)),
    (6, us_f(0.9)),
    (8, us_f(0.98)),
]);

/// `x86_64-avx2` team row: broccoli (AMD EPYC 7702P, Zen2), pool 64,
/// `team/load10us-minus-10us`, nanoseconds rounded to the nearest
/// (`docs/design/thresholds-data/raw/broccoli-dispatch-20261029.csv`).
const AVX2_TEAM: WidthCosts = anchored(&[
    (2, Duration::from_nanos(1423)),
    (4, Duration::from_nanos(1992)),
    (6, Duration::from_nanos(2329)),
    (8, Duration::from_nanos(2738)),
    (12, Duration::from_nanos(3558)),
    (16, Duration::from_nanos(4076)),
    (24, Duration::from_nanos(4292)),
    (32, Duration::from_nanos(4838)),
    (48, Duration::from_nanos(6402)),
    (64, Duration::from_nanos(9333)),
]);

impl DispatchCosts {
    /// Costs of the three kinds.
    pub fn new(team: WidthCosts, in_pool: WidthCosts, outside: WidthCosts) -> Self {
        DispatchCosts {
            team,
            in_pool,
            outside,
        }
    }

    /// The row of a host class. For `x86_64-avx512` and `aarch64-apple` the
    /// dispatch costs and the rates of the consumer crates' tables for the same
    /// class come from one machine in one clock state. The `x86_64-avx2` team
    /// row is the exception: it is measured (on a Zen2 host, below) while the
    /// consumer crates' rates for that class are still their generic rows, so
    /// the gate compares a measured dispatch cost with a conservative rate.
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
    /// * [`HostClass::X86_64Avx2`]: the team row measured on `broccoli`, an
    ///   AMD EPYC 7702P (Zen2, 64 cores, SMT off), boost off, governor
    ///   `performance`, pool 64 pinned to CPUs 0-63, with the Linux port of
    ///   `benches/dispatch_cost.rs` in `faer-benchmark` (back to back sections
    ///   whose shares carry 10 us of work, threadteam 0.1.0;
    ///   tensor4all/threadteam-rs#6, `docs/design/thresholds.md` section 3.7), in microseconds at widths 2, 4, 6, 8,
    ///   12, 16, 24, 32, 48, 64: 1.42, 1.99, 2.33, 2.74, 3.56, 4.08, 4.29,
    ///   4.84, 6.40, 9.33. In pool and outside deliberately keep the generic
    ///   row. The class covers every AVX2 host, also small desktops, so the
    ///   team row is a representative default for the class, not a guarantee
    ///   for a given topology; a pool owner overrides it with
    ///   [`PoolBuilder::dispatch_costs`](crate::PoolBuilder::dispatch_costs).
    /// * Every other class has no measurement and takes the conservative
    ///   generic row: 3 us for a team, 15 us in the pool and 40 us from
    ///   outside at the anchors 2, 4 and 6, and above width 6 the width
    ///   proportion (32, 160 and 427 us at width 64). It errs towards staying
    ///   serial.
    ///
    /// Above the largest anchor of a row the cost grows with the width
    /// ([`WidthCosts`]); [`WidthCosts::from_anchors`] replaces that
    /// extrapolation where a host has measured anchors beyond width 6.
    ///
    /// No row carries a multi-node term ([`WidthCosts::with_multi_node`]): the
    /// rows are single-node costs, and a pool owner on a multi-node host installs
    /// a measured term (`docs/design/thresholds.md`, threadteam-rs#68).
    pub fn for_class(class: HostClass) -> Self {
        match class {
            HostClass::X86_64Avx512 => DispatchCosts {
                team: AVX512_TEAM,
                in_pool: WidthCosts::new([us_f(3.0), us_f(3.6), us_f(3.5)]),
                outside: WidthCosts::new([us_f(9.7), us_f(10.1), us_f(8.7)]),
            },
            HostClass::Aarch64Apple => DispatchCosts {
                team: WidthCosts::new([us_f(3.0), us_f(2.3), us_f(1.4)]),
                in_pool: WidthCosts::new([us_f(6.0), us_f(11.0), us_f(5.5)]),
                outside: WidthCosts::new([us_f(14.0), us_f(18.0), us_f(19.0)]),
            },
            HostClass::X86_64Avx2 => DispatchCosts {
                team: AVX2_TEAM,
                in_pool: WidthCosts::flat(us_f(15.0)),
                outside: WidthCosts::flat(us_f(40.0)),
            },
            HostClass::X86_64Scalar | HostClass::Aarch64 | HostClass::Generic => DispatchCosts {
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

    /// The team row, for the token paths that add the node count of their
    /// placement ([`WidthCosts::with_multi_node`]).
    #[inline]
    pub(crate) fn team_row(&self) -> &WidthCosts {
        &self.team
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

    /// The cost of one section of `width` participants on a token of `kind`, on
    /// one NUMA node ([`WidthCosts::at`]). Measured anchors keep their values,
    /// widths between two anchors are interpolated, and widths above the largest
    /// anchor of the row grow with the width ([`WidthCosts`]). The team tokens add
    /// the multi-node term of their placement themselves
    /// ([`Exec::dispatch_cost`](crate::Exec::dispatch_cost)).
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

/// How far the shares of a parallel section have to reach for their input.
///
/// The width model [`width_for_by`] charges a section `D(k) + estimate / k`,
/// where `estimate` is the serial time measured on data the calling thread can
/// reach at its own speed. When the caller has just produced that data, the
/// members of a team fork do not find it in their own caches: they pull it out
/// of the caller's, and that transfer is not in `estimate / k`. `SplitCost`
/// states how many bytes are in that state, so a width can charge it.
///
/// A region with `resident_bytes == 0` is exactly [`width_for_by`]: the term is
/// zero at every width and [`width_for_split_by`] then returns the same width.
///
/// The evidence is `benches/resident_split.rs` (threadteam-rs#27): the members
/// pull concurrently from one source, so the section pays **one** transfer of
/// the moved bytes, at about the same rate with 2 and with 23 pullers, while
/// their summed busy grows like `k - 1`. Charging one transfer per member would
/// over-price wide sections by that factor.
///
/// # Examples
///
/// ```
/// use std::time::Duration;
/// use threadteam::SplitCost;
///
/// let split = SplitCost {
///     estimate: Duration::from_micros(120),
///     sections: 1,
///     resident_bytes: 1 << 20,
/// };
/// assert_eq!(split.with_resident_bytes(0).resident_bytes, 0);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SplitCost {
    /// Serial time of the region on data resident in the caller's caches.
    pub estimate: Duration,
    /// The real number of dispatches the region performs (`pays_at`).
    pub sections: u32,
    /// Bytes the caller wrote just before the first section.
    pub resident_bytes: u64,
}

impl SplitCost {
    /// `self` with a different number of caller-resident bytes.
    #[must_use]
    pub const fn with_resident_bytes(self, resident_bytes: u64) -> Self {
        SplitCost {
            resident_bytes,
            ..self
        }
    }
}

/// The aggregate rate at which the members of a team fork pull data out of the
/// caller's caches, in integer nanoseconds per KiB.
///
/// One row per [`HostClass`], measured with `benches/resident_split.rs`: the
/// moved bytes (`resident_bytes * (k - 1) / k`) over the busy time of the
/// slowest member at `k >= 4` on one die keeps one rate per host while the
/// source is resident. An unmeasured class takes the slowest measured row,
/// [`per_kib(41)`](Self::per_kib) (about 25 GB/s aggregate), so the transfer is
/// never under-charged on a host that has not been calibrated; a pool owner
/// that measured its own host overrides it with [`per_kib`](Self::per_kib).
///
/// # Examples
///
/// ```
/// use std::time::Duration;
/// use threadteam::TransferRate;
///
/// let rate = TransferRate::per_kib(8);
/// assert_eq!(rate.time(128 * 1024), Duration::from_nanos(1024));
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct TransferRate {
    /// Nanoseconds to move one KiB, aggregate over the pullers.
    ns_per_kib: u64,
}

impl TransferRate {
    /// A row measured on one host, in nanoseconds per KiB of moved bytes.
    ///
    /// One KiB at 100 GB/s is 10.2 ns; at 25 GB/s, 41 ns.
    pub const fn per_kib(ns_per_kib: u64) -> Self {
        Self { ns_per_kib }
    }

    /// The row of a host class.
    ///
    /// * [`HostClass::Aarch64Apple`]: 8 ns/KiB, measured on an Apple M5 Max on
    ///   2026-11-03 with `benches/resident_split.rs` (hot, mode `D`, widths 4
    ///   and 8, 384 KiB to 4 MiB: 5.8 to 8.4 ns/KiB, the widest point of the
    ///   smallest size being the slowest).
    /// * [`HostClass::X86_64Avx512`]: 41 ns/KiB, from the same bench on a
    ///   2 x Xeon Platinum 9242, one die, source hot at 392 KiB: the pullers
    ///   drain the moved bytes at 20 to 26 GB/s whatever their number
    ///   (threadteam-rs#27).
    /// * Every other class: that 41 ns/KiB default, see [`Self`].
    pub fn for_class(class: HostClass) -> Self {
        match class {
            HostClass::Aarch64Apple => Self::per_kib(8),
            HostClass::X86_64Avx512
            | HostClass::X86_64Avx2
            | HostClass::X86_64Scalar
            | HostClass::Aarch64
            | HostClass::Generic => Self::per_kib(41),
        }
    }

    /// The row of the host class this process runs on
    /// (`for_class(HostClass::detect())`).
    pub fn host_default() -> Self {
        Self::for_class(HostClass::detect())
    }

    /// The nanoseconds to move one KiB.
    pub const fn ns_per_kib(&self) -> u64 {
        self.ns_per_kib
    }

    /// The time to move `bytes` at this rate, saturating at `u64::MAX`
    /// nanoseconds.
    pub fn time(&self, bytes: u64) -> Duration {
        Duration::from_nanos(self.ns_of_bytes(bytes))
    }

    /// [`time`](Self::time) in nanoseconds.
    #[inline]
    pub(crate) fn ns_of_bytes(&self, bytes: u64) -> u64 {
        let ns = u128::from(bytes) * u128::from(self.ns_per_kib) / 1024;
        if ns > u64::MAX as u128 {
            u64::MAX
        } else {
            ns as u64
        }
    }
}

/// The width in `1..=budget` that minimises
/// `sections * D(k) + estimate / k + moved(k) / rate`, where `moved(k)` is
/// `resident_bytes * (k - 1) / k` of the caller-resident source and `D` is
/// `dispatch`; 1 when no width does. Ties go to the smaller width.
/// Allocation-free.
///
/// The transfer is charged once per section, at one aggregate rate, and a width
/// only counts when the region [`pays_at`] that section: the transfer is *not*
/// part of the pay decision, so a region that does not pay stays serial instead
/// of being made parallel by work the split itself creates. At `k = 1` the term
/// is zero (there is no member to move anything to), and
/// `resident_bytes == 0` makes this function identical to [`width_for_by`].
///
/// # Examples
///
/// ```
/// use std::num::NonZeroUsize;
/// use std::time::Duration;
/// use threadteam::{width_for_by, width_for_split_by, SplitCost, TransferRate};
///
/// let budget = NonZeroUsize::new(6).unwrap();
/// let dispatch = |_| Duration::from_micros(2);
/// let est = Duration::from_millis(1);
/// // 1 GiB the caller has just written: at 41 ns/KiB the transfer alone is
/// // 42 ms, so no width pays for itself.
/// let split = SplitCost { estimate: est, sections: 1, resident_bytes: 1 << 30 };
/// assert_eq!(width_for_split_by(split, budget, dispatch, TransferRate::per_kib(41)).get(), 1);
/// // The same region from cold data has nothing to move.
/// assert_eq!(
///     width_for_split_by(split.with_resident_bytes(0), budget, dispatch, TransferRate::per_kib(41)),
///     width_for_by(est, 1, budget, dispatch),
/// );
/// ```
pub fn width_for_split_by(
    split: SplitCost,
    budget: NonZeroUsize,
    dispatch: impl Fn(NonZeroUsize) -> Duration,
    rate: TransferRate,
) -> NonZeroUsize {
    let est_ns = to_ns(split.estimate);
    let est = est_ns as f64;
    let mut best = NonZeroUsize::MIN;
    let mut best_obj = est;
    for k in 2..=budget.get() {
        let Some(width) = NonZeroUsize::new(k) else {
            continue;
        };
        let d = dispatch(width);
        if d == Duration::MAX {
            continue;
        }
        let d_ns = to_ns(d);
        if split.sections != 0 && !pays_ns(est_ns, split.sections, d_ns) {
            continue;
        }
        let moved = u128::from(split.resident_bytes) * (k as u128 - 1) / k as u128;
        let moved = if moved > u128::from(u64::MAX) {
            u64::MAX
        } else {
            moved as u64
        };
        let transfer = rate.ns_of_bytes(moved) as f64;
        let obj = f64::from(split.sections) * d_ns as f64 + est / k as f64 + transfer;
        if obj < best_obj {
            best = width;
            best_obj = obj;
        }
    }
    best
}

#[cfg(test)]
mod tests;
