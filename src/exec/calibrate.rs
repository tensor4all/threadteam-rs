//! Calibration of a running pool's team row (threadteam-rs#92): the shipped
//! team probe of `benches/dispatch_cost.rs` (`team/load10us-minus-10us`), run
//! inside the library under a wall-time budget.

use super::{Exec, TeamAuth};
use crate::cost::WidthCosts;
use crate::error::CalibrationError;
use crate::policy::WaitOverride;
use core::num::NonZeroUsize;
use core::sync::atomic::AtomicUsize;
use core::time::Duration;
use std::time::Instant;

/// Work in every share of a measured section; the dispatch cost is the section
/// time minus this. An empty share can finish on the caller before a member
/// wakes, so a loaded share is what the shipped rows measure.
const LOAD_NS: u64 = 10_000;
/// Timed batches per width; the measurement is their median.
const BATCHES: u32 = 9;
/// The fewest sections in one batch, whatever the budget.
const MIN_ITERS: u32 = 100;
/// A measurement above this multiple of the default row is rejected
/// ([`CalibrationError::Rejected`]).
const K: u32 = 10;
/// The ladder below the token's budget, which is added as the last width (the
/// widths of the shipped probe).
const LADDER: [usize; 3] = [2, 4, 6];

/// Where the time of a calibration comes from: the wall clock, or a scripted
/// one in the tests.
pub(crate) trait Sampler {
    /// Time since the calibration started.
    fn elapsed(&self) -> Duration;
    /// Runs `iters` back-to-back sections of `width` participants on the team
    /// token `token`, each share spinning for [`LOAD_NS`], and returns their
    /// total time.
    fn batch(&mut self, token: &mut Exec<'_>, width: NonZeroUsize, iters: u32) -> Duration;
}

/// The wall clock and real sections.
struct Wall(Instant);

impl Sampler for Wall {
    fn elapsed(&self) -> Duration {
        self.0.elapsed()
    }

    fn batch(&mut self, token: &mut Exec<'_>, width: NonZeroUsize, iters: u32) -> Duration {
        let start = Instant::now();
        for _ in 0..iters {
            token
                .reborrow()
                .limit(width)
                .for_ranges(width.get(), NonZeroUsize::MIN, |_| spin_for(LOAD_NS));
        }
        start.elapsed()
    }
}

fn spin_for(ns: u64) {
    let start = Instant::now();
    while start.elapsed().as_nanos() < u128::from(ns) {
        core::hint::spin_loop();
    }
}

/// The ladder of a token of `top` slots: 2, 4 and 6 up to `top`, then `top`
/// (capped at the widest anchor a row holds). Strictly increasing from 2.
fn ladder(top: NonZeroUsize) -> ([NonZeroUsize; LADDER.len() + 1], usize) {
    let top = top.get().min(usize::from(u16::MAX));
    let mut widths = [NonZeroUsize::MIN; LADDER.len() + 1];
    let mut len = 0;
    for w in LADDER.into_iter().filter(|&w| w < top).chain([top]) {
        if let Some(w) = NonZeroUsize::new(w) {
            widths[len] = w;
            len += 1;
        }
    }
    (widths, len)
}

/// The time one measured section is expected to take on the default row.
fn per_call(default: &WidthCosts, width: NonZeroUsize) -> Duration {
    Duration::from_nanos(LOAD_NS).saturating_add(default.at(width))
}

/// Measures every width of `widths` on the team token `token` and checks it
/// against `default`: the anchors of the row, in ladder order.
fn ladder_anchors(
    sampler: &mut impl Sampler,
    token: &mut Exec<'_>,
    widths: &[NonZeroUsize],
    default: &WidthCosts,
    budget: Duration,
) -> Result<[(usize, Duration); LADDER.len() + 1], CalibrationError> {
    // One batch is at most budget / (widths * (batches + 1)).
    let slot = budget / (widths.len() as u32 * (BATCHES + 1));
    let mut anchors = [(0usize, Duration::ZERO); LADDER.len() + 1];
    for (anchor, &w) in anchors.iter_mut().zip(widths) {
        let iters = slot.as_nanos() / per_call(default, w).as_nanos().max(1);
        let iters = u32::try_from(iters).unwrap_or(u32::MAX).max(MIN_ITERS);
        let measured = measure(&mut *sampler, &mut *token, w, iters, budget)?;
        let limit = default.at(w).saturating_mul(K);
        if measured.is_zero() || measured > limit {
            return Err(CalibrationError::Rejected {
                width: w,
                measured,
                limit,
            });
        }
        *anchor = (w.get(), measured);
    }
    Ok(anchors)
}

/// The median over [`BATCHES`] of the mean cost per section minus the load,
/// after a warm-up of `iters / 10 + 10` sections (as `per_call_ns` in the
/// bench). The deadline is checked before every batch, warm-up included, so
/// the call overruns `budget` by at most one batch. Zero for a value at or
/// below zero.
fn measure(
    sampler: &mut impl Sampler,
    token: &mut Exec<'_>,
    width: NonZeroUsize,
    iters: u32,
    budget: Duration,
) -> Result<Duration, CalibrationError> {
    let exhausted = CalibrationError::BudgetExhausted { width };
    if sampler.elapsed() >= budget {
        return Err(exhausted);
    }
    sampler.batch(token, width, iters / 10 + 10);
    let mut ns = [0f64; BATCHES as usize];
    for slot in &mut ns {
        if sampler.elapsed() >= budget {
            return Err(exhausted);
        }
        let t = sampler.batch(token, width, iters);
        *slot = t.as_nanos() as f64 / f64::from(iters) - LOAD_NS as f64;
    }
    ns.sort_by(f64::total_cmp);
    let median = ns[ns.len() / 2].round();
    Ok(if median > 0.0 {
        Duration::from_nanos(median as u64)
    } else {
        Duration::ZERO
    })
}

impl Exec<'_> {
    /// Measures this pool's team row at the ladder widths up to
    /// [`budget`](Self::budget) within the wall time `budget` and installs it
    /// for every team-priced read of the pool.
    ///
    /// For a library that holds only a token: the class row of
    /// [`DispatchCosts::host_default`](crate::DispatchCosts::host_default) can
    /// be off by several times on a given host, and this replaces it with the
    /// host's own measurement instead of a second cost model.
    ///
    /// # Method
    ///
    /// The shipped team probe (`benches/dispatch_cost.rs`,
    /// `team/load10us-minus-10us`): back-to-back
    /// `reborrow().limit(w).for_ranges(w, 1, ..)` calls on a team token, whose
    /// `w` shares each spin for 10 us, minus those 10 us, at the widths 2, 4
    /// and 6 up to the token's budget plus the budget itself. Per width a
    /// warm-up, then the median of 9 batches of the mean nanoseconds per call.
    /// Each call is the full [`for_ranges`](Self::for_ranges) entry a consumer
    /// pays: the leader's home check, the section and its join, and the
    /// ledger bookkeeping when that feature is on.
    ///
    /// A token that carries a team measures on itself. A plain token reserves
    /// its members for the call as [`team`](Self::team) does (the root scope
    /// places them when pinning is on) and measures on a team token over that
    /// reservation, which records the scope's placement the way a
    /// [`Team`](crate::Team)'s whole-team scope does, so each call takes the
    /// path of a placed team token. Above the largest measured width the row
    /// extrapolates as
    /// [`WidthCosts`] documents; calibrate from a token whose budget is the
    /// pool's width to cover the whole pool.
    ///
    /// # Budget
    ///
    /// `budget` bounds the whole call. The minimum reserves, at every width, 10
    /// slots of 100 sections at the default row's cost: the 9 timed batches of
    /// at least 100 sections and the warm-up (which runs `iters / 10 + 10`
    /// sections, 20 at the minimum). A budget below it is refused before any
    /// section runs. Otherwise every width gets an
    /// equal share; the deadline is checked before every batch, so the call
    /// overruns by at most one batch.
    ///
    /// # What is installed
    ///
    /// All or nothing: on `Ok` the returned row replaces the team row of
    /// [`Pool::dispatch_costs`](crate::Pool::dispatch_costs) and every
    /// team-priced read of the pool: [`dispatch_cost`](Self::dispatch_cost),
    /// [`pays`](Self::pays) and `width_for*` of team tokens (tokens with team
    /// authority, including those that existed before the call) and of every
    /// [`TeamExec`](crate::TeamExec); on `Err` nothing is installed. A plain
    /// token's own reads price from the in-pool and outside rows, which are not
    /// calibrated and do not move. A decision already in flight finishes on the
    /// row it read; a decision never mixes the two rows.
    ///
    /// A measurement is accepted when `0 < measured <= 10 * default` at every
    /// width, where `default` is the pool's default team row
    /// ([`DispatchCosts::for_class`](crate::DispatchCosts::for_class)); the
    /// multiple is fixed.
    ///
    /// # Errors
    ///
    /// [`CalibrationError`], and the table is unchanged with each of them:
    /// [`Serial`](CalibrationError::Serial) for [`Exec::serial`] and every
    /// token of one slot, [`ExplicitCosts`](CalibrationError::ExplicitCosts)
    /// for a pool built with
    /// [`PoolBuilder::dispatch_costs`](crate::PoolBuilder::dispatch_costs)
    /// (neither runs a section); [`AlreadyCalibrated`](CalibrationError::AlreadyCalibrated)
    /// after an earlier success; [`BudgetTooSmall`](CalibrationError::BudgetTooSmall),
    /// [`BudgetExhausted`](CalibrationError::BudgetExhausted),
    /// [`TeamUnavailable`](CalibrationError::TeamUnavailable) and
    /// [`Rejected`](CalibrationError::Rejected).
    ///
    /// # Examples
    ///
    /// ```
    /// use std::num::NonZeroUsize;
    /// use std::time::Duration;
    /// use threadteam::{DispatchCosts, DispatchKind, Exec, Pool};
    ///
    /// let pool = Pool::builder().threads(2).acquire_timeout(Duration::from_secs(10)).build()?;
    /// let mut exec = Exec::new(&pool, 2)?;
    /// let two = NonZeroUsize::new(2).unwrap();
    /// match exec.calibrate_dispatch_costs(Duration::from_millis(500)) {
    ///     Ok(row) => assert_eq!(pool.dispatch_costs().cost(DispatchKind::Team, two), row.at(two)),
    ///     // Rejected, out of budget, or a pool of width 1 without `parallel`.
    ///     Err(_) => assert_eq!(pool.dispatch_costs(), DispatchCosts::host_default()),
    /// }
    /// # Ok::<(), threadteam::Error>(())
    /// ```
    pub fn calibrate_dispatch_costs(
        &mut self,
        budget: Duration,
    ) -> Result<WidthCosts, CalibrationError> {
        self.calibrate_by(budget, &mut Wall(Instant::now()))
    }

    /// [`calibrate_dispatch_costs`](Self::calibrate_dispatch_costs) on the time
    /// of `sampler`.
    pub(crate) fn calibrate_by(
        &mut self,
        budget: Duration,
        sampler: &mut impl Sampler,
    ) -> Result<WidthCosts, CalibrationError> {
        let pool = match self.pool {
            Some(pool) if self.budget.get() >= 2 => pool,
            _ => return Err(CalibrationError::Serial),
        };
        let awake = &*pool.awake;
        let default = *awake.calibration_base()?;
        let (widths, len) = ladder(self.budget);
        let widths = &widths[..len];
        let needed = widths.iter().fold(Duration::ZERO, |sum, &w| {
            sum.saturating_add(per_call(&default, w).saturating_mul((BATCHES + 1) * MIN_ITERS))
        });
        if budget < needed {
            return Err(CalibrationError::BudgetTooSmall { needed });
        }
        let anchors = if self.team.is_some() {
            // A token with team authority measures on itself: the sections are
            // the ones its own consumers dispatch.
            ladder_anchors(sampler, self, widths, &default, budget)?
        } else {
            // A plain token reserves members for the call, and the sections run
            // on a team token over that reservation. The scope records its home
            // as a `Team`'s root scope over the whole budget does, so the token's
            // `for_ranges` takes the path of a placed team token.
            let home = AtomicUsize::new(0);
            let top = self.budget;
            self.team_scope(WaitOverride::new(), Some(&home), |team| {
                let awake = match team.awake {
                    Some(awake) if team.size() == top => awake,
                    _ => return Err(CalibrationError::TeamUnavailable),
                };
                let mut token = Exec {
                    pool: Some(pool),
                    budget: top,
                    child: false,
                    track: team.track.alias(),
                    team: Some(TeamAuth {
                        awake,
                        ids: team.ids,
                        policy: team.policy,
                        over: team.over,
                        reserved: team.ids.len(),
                        placement: Some(&home),
                        lead_node: crate::pin::NO_NODE,
                        owner: crate::pin::NO_OWNER,
                    }),
                    _leader: None,
                };
                ladder_anchors(sampler, &mut token, widths, &default, budget)
            })
            .map_err(|_| CalibrationError::TeamUnavailable)??
        };
        // The ladder is 2, 4, 6 below the budget and then the budget, capped at
        // u16::MAX, with the budget at least 2: at most four strictly increasing
        // widths in 2..=u16::MAX, which `from_anchors` always accepts.
        let row = WidthCosts::from_anchors(&anchors[..len])
            .expect("a ladder of at most four strictly increasing widths from 2 to u16::MAX");
        awake.install_team_row(row)?;
        Ok(row)
    }
}

#[cfg(all(test, feature = "parallel"))]
mod tests;
