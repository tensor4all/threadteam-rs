//! The token kind of `dispatch_cost` is decided at call time, and the decision
//! functions are allocation-free. Bounded: a few tokens, no timing.
#![cfg(all(feature = "parallel", not(feature = "ledger")))]

use std::alloc::{GlobalAlloc, Layout, System};
use std::num::NonZeroUsize;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;
use threadteam::{pays_at, DispatchCosts, DispatchKind, Exec, Pool, WidthCosts, PAYS};

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
    DispatchCosts::generic()
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
fn width_model_is_flat_above_the_last_anchor() {
    let c = WidthCosts::new([us(3), us(2), us(1)]);
    assert_eq!(c.at(nz(6)), us(1));
    assert_eq!(c.at(nz(8)), us(1));
    assert_eq!(c.at(nz(64)), us(1));
    // Linux row: width 8 measured below width 6, the model holds width 6.
    let l = DispatchCosts::linux_x86();
    for (kind, w6) in [
        (DispatchKind::Team, 900),
        (DispatchKind::InPool, 3_500),
        (DispatchKind::Outside, 8_700),
    ] {
        assert_eq!(l.cost(kind, nz(8)), Duration::from_nanos(w6));
    }
    // A team GEMM 64^3 (about 9.7 us) pays at width 8.
    assert!(pays_at(
        Duration::from_nanos(9_700),
        1,
        l.cost(DispatchKind::Team, nz(8))
    ));
}

#[test]
fn linux_x86_row_and_host_default() {
    let l = DispatchCosts::linux_x86();
    let ns = |x: f64| Duration::from_nanos((x * 1000.0) as u64);
    assert_eq!(l.cost(DispatchKind::Team, nz(2)), ns(0.6));
    assert_eq!(l.cost(DispatchKind::Team, nz(4)), ns(0.7));
    assert_eq!(l.cost(DispatchKind::InPool, nz(4)), ns(3.6));
    assert_eq!(l.cost(DispatchKind::Outside, nz(2)), ns(9.7));
    #[cfg(all(target_os = "linux", target_arch = "x86_64"))]
    assert_eq!(DispatchCosts::host_default(), l);
    #[cfg(not(any(
        all(target_os = "linux", target_arch = "x86_64"),
        all(target_os = "macos", target_arch = "aarch64")
    )))]
    assert_eq!(DispatchCosts::host_default(), DispatchCosts::generic());
}
