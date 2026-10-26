//! Dispatch cost of one parallel section for the three kinds of token, at widths 2, 4 and 6.
//!
//! The numbers behind `docs/design/thresholds.md`: the time `Exec::for_ranges(w, 1, noop)`
//! takes, which is the cost a kernel pays to fan out over `w` participants and join, with no work
//! in the shares.
//!
//! * `team`: a token from `Team::exec(w)` (reserved members, awake or parked);
//! * `plain/outside`: a plain token used by a thread that is not a worker of its pool
//!   (install, wake, join), measured back to back (`hot`) and after a 1 ms gap (`gap`);
//! * `plain/in-pool`: a plain token used by a worker of its own pool (spawn into the current
//!   scope, join);
//! * `external`: `Exec::external` with an empty body (the entry a faer or BLAS call pays).
//!
//! `team` is also measured after gaps of 1 ms, 10 ms and 300 ms (members spinning, then parked
//! after the default 200 ms idle spin).
//!
//! `cargo bench --bench dispatch_cost -- [csv-path]`
//!
//! `DISPATCH_TEAM_ONLY=<n>` measures only the team rows, on a pool of `n` threads and widths
//! 2, 4, 6 and `n` (hot, after 1 ms, and a hot loop of 10 us shares), about 4 s per width; `DISPATCH_WIDTH=<w>` keeps one width.

use std::hint::black_box;
use std::io::Write;
use std::num::NonZeroUsize;
use std::time::{Duration, Instant};
use threadteam::{raise_current_thread_qos, Exec, Pool, Qos, WaitOverride};

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(f64::total_cmp);
    v[v.len() / 2]
}

/// Median over `batches` of the mean ns per call.
fn per_call_ns(batches: usize, iters: usize, mut f: impl FnMut()) -> f64 {
    for _ in 0..iters / 10 + 10 {
        f();
    }
    median(
        (0..batches)
            .map(|_| {
                let t = Instant::now();
                for _ in 0..iters {
                    f();
                }
                t.elapsed().as_nanos() as f64 / iters as f64
            })
            .collect(),
    )
}

/// Median ns of one call issued after sleeping `gap` (the call alone is timed).
fn after_gap_ns(gap: Duration, reps: usize, mut f: impl FnMut()) -> f64 {
    let mut v = Vec::with_capacity(reps);
    for _ in 0..3 {
        std::thread::sleep(gap);
        f();
    }
    for _ in 0..reps {
        if gap >= Duration::from_millis(1) {
            std::thread::sleep(gap);
        } else {
            let t = Instant::now();
            while t.elapsed() < gap {
                std::hint::spin_loop();
            }
        }
        let t = Instant::now();
        f();
        v.push(t.elapsed().as_nanos() as f64);
    }
    median(v)
}

/// Busy-wait for `ns` nanoseconds (the load of one share in the `load` rows).
fn spin_for(ns: u64) {
    let t = Instant::now();
    while (t.elapsed().as_nanos() as u64) < ns {
        std::hint::spin_loop();
    }
}

/// Load per share of the `load` rows: the overhead of a section with real work in every share is the
/// call time minus this (it includes the wait for every participant to start, which an empty body
/// does not: an empty share can finish on the caller before another worker has woken).
const LOAD_NS: u64 = 10_000;

fn build(threads: usize) -> Pool {
    Pool::builder()
        .threads(threads)
        .qos(Qos::UserInitiated)
        .acquire_timeout(Duration::from_secs(30))
        .build()
        .unwrap()
}

fn main() {
    let csv_path = std::env::args().nth(1).filter(|a| !a.starts_with("--"));
    raise_current_thread_qos(Qos::UserInitiated);
    let one = NonZeroUsize::MIN;
    let mut rows: Vec<(String, usize, f64)> = Vec::new();
    let mut put = |name: &str, w: usize, ns: f64| {
        println!("{name:<28} w={w} {ns:>10.0} ns");
        rows.push((name.to_string(), w, ns));
    };

    if let Some(n) = std::env::var("DISPATCH_TEAM_ONLY")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
    {
        // `DISPATCH_PARK_MS=<ms>` overrides the park delay (default 200 ms).
        let over = match std::env::var("DISPATCH_PARK_MS")
            .ok()
            .and_then(|v| v.parse::<u64>().ok())
        {
            Some(ms) => WaitOverride::new().idle_spin(Duration::from_millis(ms)),
            None => WaitOverride::new(),
        };
        let tsize = std::env::var("DISPATCH_TEAM")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(n);
        let team = build(n).into_team_with(tsize, over).expect("team");
        let mut widths = vec![2usize, 4, 6];
        if !widths.contains(&n) {
            widths.push(n);
        }
        let only = std::env::var("DISPATCH_WIDTH")
            .ok()
            .and_then(|v| v.parse::<usize>().ok());
        for w in widths
            .into_iter()
            .filter(|&w| w <= n && only.is_none_or(|o| o == w))
        {
            let mut tok = team.exec(w);
            let call = |tok: &mut Exec<'_>| {
                tok.reborrow().for_ranges(w, one, |_| {
                    black_box(());
                })
            };
            // Hot for long enough (0.4 s) that members the width does not use have parked.
            put("team/hot", w, per_call_ns(9, 100_000, || call(&mut tok)));
            put(
                "team/gap1ms",
                w,
                after_gap_ns(Duration::from_millis(1), 60, || call(&mut tok)),
            );
            put(
                "team/load10us-minus-10us",
                w,
                per_call_ns(9, 10_000, || {
                    tok.reborrow().for_ranges(w, one, |_| spin_for(LOAD_NS))
                }) - LOAD_NS as f64,
            );
        }
        return;
    }

    // Team token: pool of 6, one persistent team, tokens of width 2/4/6 minted from it.
    {
        let team = build(6).into_team(6).expect("team of 6");
        for w in [2usize, 4, 6] {
            let mut tok = team.exec(w);
            let call = |tok: &mut Exec<'_>| {
                tok.reborrow().for_ranges(w, one, |_| {
                    black_box(());
                })
            };
            put("team/hot", w, per_call_ns(9, 200_000, || call(&mut tok)));
            for (label, gap) in [("1ms", 1u64), ("10ms", 10), ("300ms", 300)] {
                let reps = if gap >= 300 { 12 } else { 60 };
                put(
                    &format!("team/gap{label}"),
                    w,
                    after_gap_ns(Duration::from_millis(gap), reps, || call(&mut tok)),
                );
            }
            put(
                "team/load10us-minus-10us",
                w,
                per_call_ns(9, 20_000, || {
                    tok.reborrow().for_ranges(w, one, |_| spin_for(LOAD_NS))
                }) - LOAD_NS as f64,
            );
            // an empty `fork` for comparison with the plain token's fork
            put(
                "team/fork-hot",
                w,
                per_call_ns(9, 200_000, || {
                    tok.reborrow().fork(w, |_, _| {
                        black_box(());
                    })
                }),
            );
            drop(tok);
        }
    }

    // Plain token on a pool of 6.
    {
        let pool = build(6);
        for w in [2usize, 4, 6] {
            let mut tok = Exec::new(&pool, w).unwrap();
            put(
                "plain/outside-hot",
                w,
                per_call_ns(9, 20_000, || {
                    tok.reborrow().for_ranges(w, one, |_| {
                        black_box(());
                    })
                }),
            );
            for (label, gap) in [("1ms", 1u64), ("10ms", 10)] {
                put(
                    &format!("plain/outside-gap{label}"),
                    w,
                    after_gap_ns(Duration::from_millis(gap), 100, || {
                        tok.reborrow().for_ranges(w, one, |_| {
                            black_box(());
                        })
                    }),
                );
            }
            put(
                "plain/outside-load10us-minus-10us",
                w,
                per_call_ns(9, 5_000, || {
                    tok.reborrow().for_ranges(w, one, |_| spin_for(LOAD_NS))
                }) - LOAD_NS as f64,
            );
            put(
                "external/outside-hot",
                w,
                per_call_ns(9, 20_000, || {
                    tok.reborrow().external(|n| {
                        black_box(n);
                    })
                }),
            );
            put(
                "external/outside-gap1ms",
                w,
                after_gap_ns(Duration::from_millis(1), 100, || {
                    tok.reborrow().external(|n| {
                        black_box(n);
                    })
                }),
            );
            // from inside the pool: the loop runs in `external`, each call forks from a worker
            let inner = tok.reborrow().external(|_| {
                let mut t = Exec::new(&pool, w).unwrap();
                per_call_ns(9, 20_000, || {
                    t.reborrow().for_ranges(w, one, |_| {
                        black_box(());
                    })
                })
            });
            put("plain/in-pool", w, inner);
            let loaded = tok.reborrow().external(|_| {
                let mut t = Exec::new(&pool, w).unwrap();
                per_call_ns(9, 5_000, || {
                    t.reborrow().for_ranges(w, one, |_| spin_for(LOAD_NS))
                })
            });
            put(
                "plain/in-pool-load10us-minus-10us",
                w,
                loaded - LOAD_NS as f64,
            );
        }
    }

    if let Some(p) = csv_path {
        let mut f = std::fs::File::create(p).unwrap();
        writeln!(f, "kind,width,ns").unwrap();
        for (k, w, ns) in rows {
            writeln!(f, "{k},{w},{ns:.1}").unwrap();
        }
    }
}
