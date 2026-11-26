//! Calibration contracts (threadteam-rs#92) on a scripted clock: every outcome
//! is decided by the sampler, never by the wall clock, except the one smoke
//! test, which accepts any outcome.

use super::{Sampler, K, LOAD_NS};
use crate::cost::{pays_at, width_for_by, DispatchCosts, DispatchKind, SplitCost, WidthCosts};
use crate::error::CalibrationError;
use crate::team::TeamExec;
use crate::{Exec, Pool};
use core::num::NonZeroUsize;
use core::time::Duration;
use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

/// Far above what any ladder of these pools needs.
const BUDGET: Duration = Duration::from_secs(60);
const ESTIMATES: [Duration; 5] = [
    Duration::from_micros(1),
    Duration::from_micros(5),
    Duration::from_micros(20),
    Duration::from_micros(100),
    Duration::from_millis(1),
];

fn nz(n: usize) -> NonZeroUsize {
    NonZeroUsize::new(n).unwrap()
}

fn ns(n: i64) -> Duration {
    Duration::from_nanos(n as u64)
}

fn pool(threads: usize) -> Pool {
    Pool::builder()
        .threads(threads)
        .acquire_timeout(Duration::from_secs(10))
        .build()
        .unwrap()
}

/// The default team row in integer nanoseconds at `w`.
fn default_ns(w: usize) -> i64 {
    DispatchCosts::host_default()
        .cost(DispatchKind::Team, nz(w))
        .as_nanos() as i64
}

/// Twice the default row: accepted, and different from it.
fn twice(w: usize) -> i64 {
    2 * default_ns(w)
}

/// Sections that cost `LOAD_NS + overhead(width)` each, on a clock that stands
/// still unless `exhaust_at` names a width, whose first batch moves it past
/// every budget.
struct Scripted<F> {
    overhead: F,
    exhaust_at: Option<usize>,
    batches: usize,
    clock: Duration,
}

fn scripted<F: Fn(usize) -> i64>(overhead: F) -> Scripted<F> {
    Scripted {
        overhead,
        exhaust_at: None,
        batches: 0,
        clock: Duration::ZERO,
    }
}

impl<F: Fn(usize) -> i64> Sampler for Scripted<F> {
    fn elapsed(&self) -> Duration {
        self.clock
    }

    fn batch(&mut self, _team: &mut TeamExec<'_>, width: NonZeroUsize, iters: u32) -> Duration {
        self.batches += 1;
        if self.exhaust_at == Some(width.get()) {
            self.clock = Duration::MAX;
        }
        let per = (LOAD_NS as i64 + (self.overhead)(width.get())).max(0) as u64;
        Duration::from_nanos(per * u64::from(iters))
    }
}

/// The row a token of `size` slots measured at `overhead` installs.
fn row_of(size: usize, overhead: impl Fn(usize) -> i64) -> WidthCosts {
    let anchors: Vec<_> = [2, 4, 6]
        .into_iter()
        .filter(|&w| w < size)
        .chain([size])
        .map(|w| (w, ns(overhead(w))))
        .collect();
    WidthCosts::from_anchors(&anchors).unwrap()
}

/// Every reader of one token answers from `row`: the cost, `pays` and
/// `width_for` equal `pays_at` / `width_for_by` over `row.at`.
fn reads_row(
    row: &WidthCosts,
    size: usize,
    cost: &dyn Fn(NonZeroUsize) -> Duration,
    pays: &dyn Fn(Duration, u32, NonZeroUsize) -> bool,
    width_for: &dyn Fn(Duration, u32) -> NonZeroUsize,
) {
    let at = |k: NonZeroUsize| {
        let k = k.min(nz(size));
        if k.get() < 2 {
            Duration::MAX
        } else {
            row.at(k)
        }
    };
    for w in (1..=size + 2).map(nz) {
        assert_eq!(cost(w), at(w), "width {w}");
        for est in ESTIMATES {
            assert_eq!(pays(est, 1, w), pays_at(est, 1, at(w)), "{est:?} at {w}");
        }
    }
    for est in ESTIMATES {
        for sections in [1, 2] {
            assert_eq!(
                width_for(est, sections),
                width_for_by(est, sections, nz(size), at),
                "{est:?} x {sections}"
            );
        }
    }
}

fn team_reads_row(team: &TeamExec<'_>, row: &WidthCosts) {
    reads_row(
        row,
        team.size().get(),
        &|w| team.dispatch_cost(w),
        &|e, s, w| team.pays(e, s, w),
        &|e, s| team.width_for(e, s),
    );
    let split = SplitCost {
        estimate: Duration::from_micros(100),
        sections: 1,
        resident_bytes: 0,
    };
    assert_eq!(
        team.width_for_split(split),
        team.width_for(split.estimate, 1)
    );
}

fn exec_reads_row(exec: &Exec<'_>, row: &WidthCosts) {
    reads_row(
        row,
        exec.budget().get(),
        &|w| exec.dispatch_cost(w),
        &|e, s, w| exec.pays(e, s, w),
        &|e, s| exec.width_for(e, s),
    );
    let split = SplitCost {
        estimate: Duration::from_micros(100),
        sections: 1,
        resident_bytes: 0,
    };
    assert_eq!(
        exec.width_for_split(split),
        exec.width_for(split.estimate, 1)
    );
}

/// Test 1: a pool that is never calibrated prices from the class row.
#[test]
fn an_uncalibrated_pool_keeps_the_default_row() {
    let pool = pool(4);
    let default = DispatchCosts::host_default();
    assert_eq!(pool.dispatch_costs(), default);
    let mut exec = Exec::new(&pool, 4).unwrap();
    exec.team(|team| team_reads_row(&team, default.row(DispatchKind::Team)))
        .expect("idle pool");
}

/// Test 2 and the maintainer's case: after `Ok(row)` every reader answers from
/// `row`, a token that existed before the call included; the plain rows stay.
#[test]
fn an_installed_row_reaches_every_reader() {
    let default = DispatchCosts::host_default();

    // A plain token minted before the call, calibrated through another token.
    let pool = pool(4);
    let mut earlier = Exec::new(&pool, 4).unwrap();
    let plain_before = earlier.dispatch_cost(nz(4));
    let row = Exec::new(&pool, 4)
        .unwrap()
        .calibrate_by(BUDGET, &mut scripted(twice))
        .unwrap();
    assert_eq!(row, row_of(4, twice));
    assert_ne!(row, *default.row(DispatchKind::Team));
    assert_eq!(pool.dispatch_costs(), default.with(DispatchKind::Team, row));
    assert_eq!(
        earlier.dispatch_cost(nz(4)),
        plain_before,
        "plain row moved"
    );
    earlier
        .team(|team| {
            assert_eq!(team.dispatch_costs(), pool.dispatch_costs());
            team_reads_row(&team, &row);
        })
        .expect("idle pool");

    // A team token minted before the call, calibrated through its reborrow.
    let team = pool_team(4);
    let mut token = team.exec(4);
    let row = token
        .reborrow()
        .calibrate_by(BUDGET, &mut scripted(twice))
        .unwrap();
    exec_reads_row(&token, &row);
    token
        .team(|t| team_reads_row(&t, &row))
        .expect("carried members");
    drop(token);
    exec_reads_row(&team.exec(4), &row);
    assert_eq!(
        team.pool().dispatch_costs(),
        default.with(DispatchKind::Team, row)
    );
}

fn pool_team(threads: usize) -> crate::Team {
    pool(threads)
        .into_team(threads)
        .unwrap_or_else(|_| panic!("idle pool"))
}

/// Test 3: `0 < measured <= K * default` at every width, first failure
/// reported, nothing installed on a rejection.
#[test]
fn the_rejection_boundary_is_k_times_the_default_row() {
    let pool = pool(4);
    let default = DispatchCosts::host_default();
    let limit = |w: usize| i64::from(K) * default_ns(w);
    let mut exec = Exec::new(&pool, 4).unwrap();
    let rejected = |width: usize, measured: i64| -> Result<WidthCosts, CalibrationError> {
        Err(CalibrationError::Rejected {
            width: nz(width),
            measured: ns(measured),
            limit: ns(limit(width)),
        })
    };

    let r = exec.calibrate_by(BUDGET, &mut scripted(|_| 0));
    assert_eq!(r, rejected(2, 0));
    let r = exec.calibrate_by(BUDGET, &mut scripted(|_| -5));
    assert_eq!(r, rejected(2, 0));
    // Over at the last width only: no partial row.
    let over_last = |w: usize| if w == 4 { limit(4) + 1 } else { limit(w) };
    let r = exec.calibrate_by(BUDGET, &mut scripted(over_last));
    assert_eq!(r, rejected(4, limit(4) + 1));
    assert_eq!(pool.dispatch_costs(), default);

    let r = exec.calibrate_by(BUDGET, &mut scripted(limit));
    assert_eq!(r, Ok(row_of(4, limit)));
}

/// Test 4: too small a budget runs nothing; a deadline mid-ladder installs
/// nothing; the reported minimum is enough.
#[test]
fn the_budget_bounds_the_call() {
    let pool = pool(4);
    let default = DispatchCosts::host_default();
    let mut exec = Exec::new(&pool, 4).unwrap();

    let mut s = scripted(|_| 100);
    let Err(CalibrationError::BudgetTooSmall { needed }) =
        exec.calibrate_by(Duration::ZERO, &mut s)
    else {
        panic!("a zero budget is too small");
    };
    assert!(needed > Duration::ZERO);
    let r = exec.calibrate_by(needed - Duration::from_nanos(1), &mut s);
    assert_eq!(r, Err(CalibrationError::BudgetTooSmall { needed }));
    assert_eq!(s.batches, 0, "a refused budget ran a section");

    let mut s = Scripted {
        exhaust_at: Some(4),
        ..scripted(|_| 100)
    };
    let r = exec.calibrate_by(BUDGET, &mut s);
    assert_eq!(r, Err(CalibrationError::BudgetExhausted { width: nz(4) }));
    assert_eq!(pool.dispatch_costs(), default);

    assert_eq!(
        exec.calibrate_by(needed, &mut scripted(|_| 100)),
        Ok(row_of(4, |_| 100))
    );
}

/// Test 5: one-shot; an owner's costs and a serial token are refused without
/// running anything or touching the table.
#[test]
fn calibration_is_one_shot_and_never_replaces_owner_costs() {
    let pool = pool(4);
    let row = Exec::new(&pool, 4)
        .unwrap()
        .calibrate_by(BUDGET, &mut scripted(twice))
        .unwrap();
    let mut s = scripted(|_| 100);
    let r = Exec::new(&pool, 4).unwrap().calibrate_by(BUDGET, &mut s);
    assert_eq!(r, Err(CalibrationError::AlreadyCalibrated));
    assert_eq!(s.batches, 0);
    assert_eq!(*pool.dispatch_costs().row(DispatchKind::Team), row);

    let owned = Pool::builder()
        .threads(4)
        .dispatch_costs(DispatchCosts::host_default())
        .acquire_timeout(Duration::from_secs(10))
        .build()
        .unwrap();
    let r = Exec::new(&owned, 4).unwrap().calibrate_by(BUDGET, &mut s);
    assert_eq!(r, Err(CalibrationError::ExplicitCosts));
    assert_eq!(owned.dispatch_costs(), DispatchCosts::host_default());

    let r = Exec::serial().calibrate_by(BUDGET, &mut s);
    assert_eq!(r, Err(CalibrationError::Serial));
    let fresh = self::pool(4);
    let r = Exec::new(&fresh, 1).unwrap().calibrate_by(BUDGET, &mut s);
    assert_eq!(r, Err(CalibrationError::Serial));
    assert_eq!(fresh.dispatch_costs(), DispatchCosts::host_default());
    assert_eq!(s.batches, 0);
}

/// Test 6: readers on other threads see the old row or the new one, never a
/// mix, and the new one once the install is published.
#[test]
fn readers_never_mix_the_rows() {
    let pool = pool(8);
    let default = DispatchCosts::host_default();
    let old = *default.row(DispatchKind::Team);
    let new = row_of(4, twice);
    let est = Duration::from_micros(20);
    let width = |row: &WidthCosts| {
        width_for_by(est, 1, nz(3), |k| {
            if k.get() < 2 {
                Duration::MAX
            } else {
                row.at(k.min(nz(3)))
            }
        })
    };
    let installed = AtomicBool::new(false);
    std::thread::scope(|s| {
        for _ in 0..2 {
            s.spawn(|| {
                // Two members per reader and three for the calibration: 7 of 8.
                let mut exec = Exec::new(&pool, 3).unwrap();
                exec.team(|team| {
                    let mut after = 0;
                    while after < 1000 {
                        let done = installed.load(Ordering::Acquire);
                        let (c, w, costs) = (
                            team.dispatch_cost(nz(3)),
                            team.width_for(est, 1),
                            pool.dispatch_costs(),
                        );
                        if done {
                            after += 1;
                            assert_eq!(c, new.at(nz(3)));
                            assert_eq!(w, width(&new));
                            assert_eq!(costs, default.with(DispatchKind::Team, new));
                        } else {
                            assert!(c == old.at(nz(3)) || c == new.at(nz(3)));
                            assert!(w == width(&old) || w == width(&new));
                            assert!(
                                costs == default || costs == default.with(DispatchKind::Team, new)
                            );
                        }
                    }
                })
                .expect("idle pool");
            });
        }
        let r = Exec::new(&pool, 4)
            .unwrap()
            .calibrate_by(BUDGET, &mut scripted(twice));
        installed.store(true, Ordering::Release);
        assert_eq!(r, Ok(new));
    });
}

struct Counting;
static ALLOCS: AtomicUsize = AtomicUsize::new(0);

thread_local! {
    /// Only the thread that opted in is counted: the other unit tests run
    /// concurrently and allocate.
    static COUNT_HERE: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

// SAFETY: forwards every call to `System` unchanged; the counter is a side
// effect that does not touch the memory.
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, l: Layout) -> *mut u8 {
        if COUNT_HERE.try_with(|c| c.get()).unwrap_or(false) {
            ALLOCS.fetch_add(1, Ordering::SeqCst);
        }
        System.alloc(l)
    }
    unsafe fn dealloc(&self, p: *mut u8, l: Layout) {
        System.dealloc(p, l)
    }
}

#[global_allocator]
static A: Counting = Counting;

/// Allocations `f` makes on this thread.
fn allocations(f: impl FnOnce()) -> usize {
    COUNT_HERE.with(|c| c.set(true));
    let before = ALLOCS.load(Ordering::SeqCst);
    f();
    let after = ALLOCS.load(Ordering::SeqCst);
    COUNT_HERE.with(|c| c.set(false));
    after - before
}

/// Test 7: the readers allocate nothing once a row is installed.
#[test]
fn reads_after_an_install_do_not_allocate() {
    let team = pool_team(4);
    let mut token = team.exec(4);
    token
        .reborrow()
        .calibrate_by(BUDGET, &mut scripted(twice))
        .unwrap();
    let split = SplitCost {
        estimate: Duration::from_micros(100),
        sections: 1,
        resident_bytes: 1 << 20,
    };
    let reads = |exec: &Exec<'_>| {
        let mut acc = 0usize;
        for i in 0..1000u64 {
            let est = Duration::from_nanos(i * 100);
            acc += exec.width_for(est, 1).get();
            acc += exec.width_for_split(split).get();
            acc += usize::from(exec.pays(est, 2, nz(4)));
            acc += usize::from(exec.dispatch_cost(nz(3)) > Duration::ZERO);
            acc += usize::from(team.pool().dispatch_costs() == DispatchCosts::host_default());
        }
        acc
    };
    let _ = reads(&token); // warm anything lazy
    let n = allocations(|| {
        std::hint::black_box(reads(&token));
    });
    assert_eq!(n, 0, "token reads");
    token
        .team(|t| {
            let n = allocations(|| {
                for i in 0..1000u64 {
                    let est = Duration::from_nanos(i * 100);
                    let _ = t.width_for(est, 1);
                    let _ = t.width_for_split(split);
                    let _ = t.pays(est, 2, nz(4));
                    let _ = t.dispatch_cost(nz(3));
                    let _ = t.dispatch_costs();
                }
            });
            assert_eq!(n, 0, "team reads");
        })
        .expect("carried members");
}

/// Test 8: the wall clock, any outcome; only the install invariant is checked.
#[test]
fn a_real_calibration_installs_what_it_returns_or_nothing() {
    let pool = pool(4);
    let mut exec = Exec::new(&pool, 4).unwrap();
    let default = DispatchCosts::host_default();
    match exec.calibrate_dispatch_costs(Duration::from_millis(300)) {
        Ok(row) => assert_eq!(pool.dispatch_costs(), default.with(DispatchKind::Team, row)),
        Err(_) => assert_eq!(pool.dispatch_costs(), default),
    }
}
