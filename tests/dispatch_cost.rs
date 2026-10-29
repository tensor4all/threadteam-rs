//! The token kind of `dispatch_cost` is decided at call time, and the decision
//! functions are allocation-free. Bounded: a few tokens, no timing.
#![cfg(all(feature = "parallel", not(feature = "ledger")))]

use std::alloc::{GlobalAlloc, Layout, System};
use std::num::NonZeroUsize;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;
use threadteam::{
    pays_at, width_for_by, DispatchCosts, DispatchKind, Exec, HostClass, Pool, WidthCosts, PAYS,
};

struct Counting;
static ALLOCS: AtomicUsize = AtomicUsize::new(0);

thread_local! {
    /// Only the thread that opted in is counted: the other tests of this
    /// binary run concurrently and allocate.
    static COUNT_HERE: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

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

fn nz(n: usize) -> NonZeroUsize {
    NonZeroUsize::new(n).unwrap()
}
fn us(n: u64) -> Duration {
    Duration::from_micros(n)
}

/// Distinct flat costs per kind so that each answer names its kind.
fn costs() -> DispatchCosts {
    DispatchCosts::for_class(HostClass::Generic)
        .with(DispatchKind::Team, WidthCosts::flat(us(1)))
        .with(DispatchKind::InPool, WidthCosts::flat(us(10)))
        .with(DispatchKind::Outside, WidthCosts::flat(us(100)))
}

fn pool(n: usize) -> Pool {
    Pool::builder()
        .threads(n)
        .dispatch_costs(costs())
        .acquire_timeout(Duration::from_secs(10))
        .build()
        .unwrap()
}

#[test]
fn kind_is_decided_at_call_time() {
    let p = pool(4);
    assert_eq!(p.dispatch_costs(), costs());

    // Serial and width one: no section.
    assert_eq!(Exec::serial().dispatch_cost(nz(4)), Duration::MAX);
    let one = Exec::new(&p, 1).unwrap();
    assert_eq!(one.dispatch_cost(nz(4)), Duration::MAX);

    // Plain token from outside the pool.
    let exec = Exec::new(&p, 4).unwrap();
    assert_eq!(exec.dispatch_cost(nz(4)), us(100));
    assert_eq!(exec.dispatch_cost(nz(1)), Duration::MAX);
    // Width is clamped to the budget: width 64 is a width-4 section.
    assert_eq!(exec.dispatch_cost(nz(64)), us(100) * 4 / 4);

    // The same plain token used by a worker of its pool.
    exec.fork(2, |_, child| {
        if child.budget().get() >= 2 {
            assert_eq!(child.dispatch_cost(nz(2)), us(10));
        }
    });
    let exec = Exec::new(&p, 4).unwrap();
    exec.external(|_width| {
        // `external` entered the pool: a token of the pool used here is in pool.
        let inner = Exec::new(&p, 2).unwrap();
        assert_eq!(inner.dispatch_cost(nz(2)), us(10));
    });

    // Team authority.
    let mut exec = Exec::new(&p, 4).unwrap();
    exec.team(|team| {
        assert_eq!(team.dispatch_cost(nz(4)), us(1));
        assert_eq!(team.dispatch_cost(nz(1)), Duration::MAX);
        assert!(team.pays(us(3), 1, nz(4)));
        assert!(!team.pays(us(2), 1, nz(4)));
    })
    .unwrap();

    let team = Pool::builder()
        .threads(4)
        .dispatch_costs(costs())
        .acquire_timeout(Duration::from_secs(10))
        .build()
        .unwrap()
        .into_team(4)
        .ok()
        .expect("idle pool");
    let exec = team.exec(4);
    assert_eq!(exec.dispatch_cost(nz(4)), us(1));
    assert_eq!(exec.width_for(us(1_000), 1).get(), 4);
    assert_eq!(exec.width_for(us(2), 1).get(), 1);
    exec.fork(2, |_, child| {
        // A child of a team token keeps the team authority.
        assert_eq!(child.dispatch_cost(nz(2)), us(1));
    });
}

#[test]
fn pays_and_width_for_follow_the_token() {
    let p = pool(4);
    let exec = Exec::new(&p, 4).unwrap();
    // Outside: 100 us per section, so 300 us is the threshold.
    assert!(exec.pays(us(300), 1, nz(4)));
    assert!(!exec.pays(us(299), 1, nz(4)));
    assert!(!exec.pays(us(300), 2, nz(4)));
    assert_eq!(exec.width_for(us(299), 1).get(), 1);
    assert!(exec.width_for(us(100_000), 1).get() > 1);
    assert_eq!(PAYS, 3.0);
}

#[test]
fn decision_functions_do_not_allocate() {
    let p = pool(4);
    let exec = Exec::new(&p, 4).unwrap();
    // Warm anything lazy before counting.
    let _ = exec.width_for(us(10), 1);
    COUNT_HERE.with(|c| c.set(true));
    let before = ALLOCS.load(Ordering::SeqCst);
    let mut acc = 0usize;
    for i in 0..1000u64 {
        acc += exec.width_for(us(i), 1).get();
        acc += usize::from(exec.pays(us(i), 2, nz(4)));
        acc += usize::from(exec.dispatch_cost(nz(3)) > Duration::ZERO);
    }
    let after = ALLOCS.load(Ordering::SeqCst);
    COUNT_HERE.with(|c| c.set(false));
    assert_eq!(after, before, "acc = {acc}");
}

#[test]
fn width_model_extrapolates_above_the_last_anchor() {
    let c = WidthCosts::new([us(3), us(2), us(1)]);
    assert_eq!(c.at(nz(6)), us(1));
    // Above the last anchor the cost grows with the width, never flat.
    assert_eq!(c.at(nz(8)), Duration::from_nanos(1333));
    assert_eq!(c.at(nz(64)), Duration::from_nanos(10_666));
    // Linux row: the team carries the measured width-8 anchor; in pool and
    // outside have no anchor there and extrapolate from width 6.
    let l = DispatchCosts::for_class(HostClass::X86_64Avx512);
    assert_eq!(l.cost(DispatchKind::Team, nz(8)), Duration::from_nanos(980));
    assert_eq!(
        l.cost(DispatchKind::InPool, nz(8)),
        Duration::from_nanos(4666)
    );
    assert_eq!(
        l.cost(DispatchKind::Outside, nz(8)),
        Duration::from_nanos(11_600)
    );
    // A team GEMM 64^3 (about 9.7 us) pays at width 8.
    assert!(pays_at(
        Duration::from_nanos(9_700),
        1,
        l.cost(DispatchKind::Team, nz(8))
    ));
}

/// A row measured up to width 64 replaces the extrapolation where it has data.
#[test]
fn anchors_beyond_six_reach_the_decision_functions() {
    let row = WidthCosts::from_anchors(&[
        (2, us(8)),
        (4, us(4)),
        (8, us(2)),
        (16, us(1)),
        (32, Duration::from_nanos(500)),
        (64, Duration::from_nanos(500)),
    ])
    .expect("valid anchors");
    let costs = DispatchCosts::for_class(HostClass::Generic).with(DispatchKind::Team, row);
    assert_eq!(costs.cost(DispatchKind::Team, nz(6)), us(3));
    assert_eq!(
        costs.cost(DispatchKind::Team, nz(64)),
        Duration::from_nanos(500)
    );
    // 24 us: obj(6) = 3 + 4 = 7 us against obj(64) = 0.5 + 0.375 us, and the
    // anchors above six are what the search reads.
    assert_eq!(
        width_for_by(us(24), 1, nz(64), |k| costs.cost(DispatchKind::Team, k)).get(),
        64
    );
    assert_eq!(
        costs.min_cost(DispatchKind::Team),
        Duration::from_nanos(500)
    );
}

#[test]
fn class_rows_and_host_default() {
    let ns = |x: f64| Duration::from_nanos((x * 1000.0) as u64);
    let l = DispatchCosts::for_class(HostClass::X86_64Avx512);
    assert_eq!(l.cost(DispatchKind::Team, nz(2)), ns(0.6));
    assert_eq!(l.cost(DispatchKind::Team, nz(4)), ns(0.7));
    assert_eq!(l.cost(DispatchKind::InPool, nz(4)), ns(3.6));
    assert_eq!(l.cost(DispatchKind::Outside, nz(2)), ns(9.7));
    let m = DispatchCosts::for_class(HostClass::Aarch64Apple);
    assert_eq!(m.cost(DispatchKind::Team, nz(2)), ns(3.0));
    assert_eq!(m.cost(DispatchKind::InPool, nz(4)), ns(11.0));
    assert_eq!(m.cost(DispatchKind::Outside, nz(6)), ns(19.0));
    // Classes without a measurement take the generic row: 3, 15, 40 us at the
    // anchors 2, 4, 6 and proportional to the width above them.
    for class in [
        HostClass::X86_64Avx2,
        HostClass::X86_64Scalar,
        HostClass::Aarch64,
        HostClass::Generic,
    ] {
        let g = DispatchCosts::for_class(class);
        assert_eq!(g, DispatchCosts::for_class(HostClass::Generic), "{class:?}");
        // The three anchors of the generic row are equal; above the last anchor
        // the cost still grows with the width.
        for (w, team, in_pool, outside) in [
            (2, 3_000, 15_000, 40_000),
            (4, 3_000, 15_000, 40_000),
            (6, 3_000, 15_000, 40_000),
            (8, 4_000, 20_000, 53_333),
            (64, 32_000, 160_000, 426_666),
        ] {
            assert_eq!(
                g.cost(DispatchKind::Team, nz(w)),
                Duration::from_nanos(team),
                "{class:?} team width {w}"
            );
            assert_eq!(
                g.cost(DispatchKind::InPool, nz(w)),
                Duration::from_nanos(in_pool),
                "{class:?} in pool width {w}"
            );
            assert_eq!(
                g.cost(DispatchKind::Outside, nz(w)),
                Duration::from_nanos(outside),
                "{class:?} outside width {w}"
            );
        }
    }
    // The default is the row of the detected class, and detection is stable.
    let c = HostClass::detect();
    assert_eq!(c, HostClass::detect());
    assert_eq!(DispatchCosts::host_default(), DispatchCosts::for_class(c));
    #[cfg(target_arch = "x86_64")]
    assert!(matches!(
        c,
        HostClass::X86_64Avx512 | HostClass::X86_64Avx2 | HostClass::X86_64Scalar
    ));
    #[cfg(all(target_arch = "aarch64", target_os = "macos"))]
    assert_eq!(c, HostClass::Aarch64Apple);
    #[cfg(all(target_arch = "aarch64", not(target_os = "macos")))]
    assert_eq!(c, HostClass::Aarch64);
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    assert_eq!(c, HostClass::Generic);
}

#[test]
fn class_names() {
    assert_eq!(HostClass::X86_64Avx512.name(), "x86_64-avx512");
    assert_eq!(HostClass::X86_64Avx2.name(), "x86_64-avx2");
    assert_eq!(HostClass::X86_64Scalar.name(), "x86_64-scalar");
    assert_eq!(HostClass::Aarch64Apple.name(), "aarch64-apple");
    assert_eq!(HostClass::Aarch64.name(), "aarch64");
    assert_eq!(HostClass::Generic.name(), "generic");
}
