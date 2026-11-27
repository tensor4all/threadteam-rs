//! Cost of a small entry made from a worker of another pool, against the same
//! entry from a plain thread (https://github.com/tensor4all/threadteam-rs/issues/89).
//!
//! The entry is `Exec::new(&pool, 4)?.for_ranges(64, 1, ..)` with an empty body
//! on a 4-thread pool, and `Exec::new(&pool, 4)?.external(..)` with an empty
//! body. A plain thread enters with `ThreadPool::install`; a worker of another
//! pool submits the work and blocks on a latch. The pricing row
//! `DispatchKind::Outside` covers both callers.
//!
//! Each printed row is one entry. Each time in it is the median over 9 batches
//! of the mean time per call, after a warm-up; the plain-thread and the
//! foreign-worker times are measured one after the other on the same pool.
//!
//! `cargo bench --bench foreign_entry`

use std::hint::black_box;
use std::num::NonZeroUsize;
use std::sync::Mutex;
use std::time::Instant;
use threadteam::{Exec, Pool};

const WIDTH: usize = 4;
const BATCHES: usize = 9;
const ITERS: usize = 20_000;

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(f64::total_cmp);
    v[v.len() / 2]
}

/// Median over `BATCHES` of the mean ns per call.
fn per_call_ns(f: impl Fn()) -> f64 {
    for _ in 0..ITERS / 10 {
        f();
    }
    median(
        (0..BATCHES)
            .map(|_| {
                let t = Instant::now();
                for _ in 0..ITERS {
                    f();
                }
                t.elapsed().as_nanos() as f64 / ITERS as f64
            })
            .collect(),
    )
}

fn for_ranges(pool: &Pool) {
    Exec::new(pool, WIDTH)
        .unwrap()
        .for_ranges(64, NonZeroUsize::MIN, |r| {
            black_box(r);
        });
}

fn external(pool: &Pool) {
    let w = Exec::new(pool, WIDTH)
        .unwrap()
        .external(|w| black_box(w.get()));
    assert_eq!(w, WIDTH);
}

/// Measures every entry from the calling thread.
fn row(pool: &Pool) -> [f64; 2] {
    [
        per_call_ns(|| for_ranges(pool)),
        per_call_ns(|| external(pool)),
    ]
}

fn main() {
    let pool = Pool::builder().threads(WIDTH).build().unwrap();
    let other = Pool::builder()
        .threads(2)
        .thread_name(|i| format!("other-{i}"))
        .build()
        .unwrap();

    let plain = row(&pool);
    // Task 0 of a fork on `other` runs on one of its workers: a worker of
    // another pool from `pool`'s point of view. Task 1 returns at once.
    let foreign = Mutex::new(None);
    Exec::new(&other, 2).unwrap().fork(2, |i, _| {
        if i == 0 {
            let name = std::thread::current().name().map(str::to_owned);
            assert!(name.is_some_and(|n| n.starts_with("other-")));
            *foreign.lock().unwrap() = Some(row(&pool));
        }
    });
    let foreign = foreign.into_inner().unwrap().unwrap();

    println!("entry                        plain thread ns   worker of another pool ns   ratio");
    for (k, name) in ["for_ranges(64, 1), budget 4", "external, budget 4"]
        .into_iter()
        .enumerate()
    {
        println!(
            "{name:<28} {:>15.0}   {:>25.0}   {:>5.2}",
            plain[k],
            foreign[k],
            foreign[k] / plain[k]
        );
    }
}
