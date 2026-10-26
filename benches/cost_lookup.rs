//! Cost of the dispatch-cost lookups themselves: `Exec::dispatch_cost`, `Exec::pays` and
//! `Exec::width_for` on a team token of a pool of 8 (what a kernel calls per GEMM and per gate).
//!
//! `cargo bench --bench cost_lookup`

use std::hint::black_box;
use std::num::NonZeroUsize;
use std::time::{Duration, Instant};
use threadteam::{DispatchCosts, DispatchKind, WidthCosts};

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
    let budget = black_box(w8);
    // Varying input so nothing is hoisted: 5 us .. 1 ms.
    let est = |i: u64| Duration::from_nanos(5_000 + (i % 1024) * 1_000);
    let w = |i: u64| NonZeroUsize::new(2 + (i % 7) as usize).unwrap();
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
            threadteam::width_for_by(est(i), 1, budget, |k| costs.cost(DispatchKind::Team, k)).get()
                as u64
        })
    );
    let _ = WidthCosts::flat(Duration::ZERO);
}
