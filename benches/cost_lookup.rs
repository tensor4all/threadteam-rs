//! Cost of the dispatch-cost lookups themselves: `Exec::dispatch_cost`, `Exec::pays`,
//! `Exec::width_for`, `DispatchCosts::min_cost` and a by-value copy of `DispatchCosts`, on a
//! team token of a pool of 8 (what a kernel calls per GEMM and per gate). The `width_for`
//! row takes a budget of 64 through `width_for_by`, so it reads the anchors up to 64 whatever
//! the pool size.
//!
//! `cargo bench --bench cost_lookup`

use std::hint::black_box;
use std::num::NonZeroUsize;
use std::time::{Duration, Instant};
use threadteam::{DispatchCosts, DispatchKind};

fn ns_per_call(mut f: impl FnMut(u64) -> u64) -> f64 {
    let iters = 2_000_000u64;
    let mut best = f64::MAX;
    for _ in 0..9 {
        let t = Instant::now();
        let mut acc = 0u64;
        for i in 0..iters {
            acc = acc.wrapping_add(f(i));
        }
        black_box(acc);
        best = best.min(t.elapsed().as_nanos() as f64 / iters as f64);
    }
    best
}

fn main() {
    let costs = black_box(DispatchCosts::host_default());
    let w8 = NonZeroUsize::new(8).unwrap();
    let budget8 = black_box(w8);
    let budget64 = black_box(NonZeroUsize::new(64).unwrap());
    // Varying input so nothing is hoisted: 5 us .. 1 ms.
    let est = |i: u64| Duration::from_nanos(5_000 + (i % 1024) * 1_000);
    let w = |i: u64| NonZeroUsize::new(2 + (i % 7) as usize).unwrap();
    let kind = |i: u64| match i % 3 {
        0 => DispatchKind::Team,
        1 => DispatchKind::InPool,
        _ => DispatchKind::Outside,
    };
    println!(
        "at (width 2..8)       {:>7.2} ns",
        ns_per_call(|i| costs.cost(DispatchKind::Team, w(i)).as_nanos() as u64)
    );
    println!(
        "pays                  {:>7.2} ns",
        ns_per_call(|i| {
            let d = costs.cost(DispatchKind::Team, w(i));
            threadteam::pays_at(est(i), 1, d) as u64
        })
    );
    println!(
        "width_for (budget 8)  {:>7.2} ns",
        ns_per_call(|i| {
            threadteam::width_for_by(est(i), 1, budget8, |k| costs.cost(DispatchKind::Team, k))
                .get() as u64
        })
    );
    println!(
        "width_for (budget 64) {:>7.2} ns",
        ns_per_call(|i| {
            threadteam::width_for_by(est(i), 1, budget64, |k| costs.cost(DispatchKind::Team, k))
                .get() as u64
        })
    );
    println!(
        "min_cost              {:>7.2} ns",
        ns_per_call(|i| costs.min_cost(kind(i)).as_nanos() as u64)
    );
    println!(
        "copy DispatchCosts    {:>7.2} ns",
        ns_per_call(|i| {
            // A by-value copy, as `TeamExec::dispatch_costs` and a consumer's
            // shape hand it on; read a field so the copy cannot be elided.
            let copy: DispatchCosts = black_box(costs);
            copy.cost(DispatchKind::Team, w(i)).as_nanos() as u64
        })
    );

    #[cfg(feature = "parallel")]
    token_rows(costs);
}

/// The lookups on a real team token: the kind is read at call time, so these are
/// the paths a kernel takes.
#[cfg(feature = "parallel")]
fn token_rows(costs: DispatchCosts) {
    use threadteam::{Pool, WaitOverride};

    let pool = Pool::builder()
        .threads(8)
        .dispatch_costs(costs)
        .acquire_timeout(Duration::from_secs(10))
        .build()
        .expect("pool");
    // Park the reserved members at once: the timing loop must not compete with
    // idle spinners on a host of fewer cores.
    let team = pool
        .into_team_with(8, WaitOverride::new().idle_spin(Duration::ZERO))
        .expect("idle pool");
    let exec = team.exec(8);
    let w = |i: u64| NonZeroUsize::new(2 + (i % 7) as usize).unwrap();
    let est = |i: u64| Duration::from_nanos(5_000 + (i % 1024) * 1_000);
    println!(
        "Exec::dispatch_cost   {:>7.2} ns",
        ns_per_call(|i| exec.dispatch_cost(w(i)).as_nanos() as u64)
    );
    println!(
        "Exec::width_for (b 8) {:>7.2} ns",
        ns_per_call(|i| exec.width_for(est(i), 1).get() as u64)
    );
    println!(
        "Exec::pays            {:>7.2} ns",
        ns_per_call(|i| exec.pays(est(i), 1, w(i)) as u64)
    );
}
