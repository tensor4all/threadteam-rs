//! Entries made on a worker of another Rayon pool
//! (https://github.com/tensor4all/threadteam-rs/issues/89).
//!
//! Such a caller submits the work and blocks without running its own pool's
//! jobs, so an outer `par_iter` whose items each enter the target pool keeps
//! the nesting depth of its items at one.
#![cfg(feature = "parallel")]

use rayon::prelude::*;
use std::cell::Cell;
use std::num::NonZeroUsize;
use std::panic::{catch_unwind, panic_any, AssertUnwindSafe};
use std::sync::atomic::{AtomicUsize, Ordering};
use threadteam::{Exec, Pool};

/// Target width and budget. Fixed, not the host's CPU count: a budget of one
/// runs inline and would never reach the path under test.
const WIDTH: usize = 4;

/// Items of the outer loop; the issue's reproduction overflowed the stack at
/// this count.
const ITEMS: usize = 20_000;

thread_local! {
    static DEPTH: Cell<usize> = const { Cell::new(0) };
}

fn target() -> Pool {
    Pool::builder().threads(WIDTH).build().unwrap()
}

/// Runs `entry` once per item of an outer `par_iter` on the global pool and
/// returns the largest number of items that were open at once on one thread.
fn max_nesting(entry: impl Fn(&Pool) + Sync) -> usize {
    let pool = target();
    let max = AtomicUsize::new(0);
    (0..ITEMS).into_par_iter().for_each(|_| {
        assert!(
            rayon::current_thread_index().is_some(),
            "the outer item runs on a worker of the global pool"
        );
        let depth = DEPTH.with(|d| {
            d.set(d.get() + 1);
            d.get()
        });
        max.fetch_max(depth, Ordering::Relaxed);
        entry(&pool);
        DEPTH.with(|d| d.set(d.get() - 1));
    });
    max.into_inner()
}

#[test]
fn for_ranges_from_a_foreign_worker_does_not_nest() {
    let depth = max_nesting(|pool| {
        Exec::new(pool, WIDTH)
            .unwrap()
            .for_ranges(64, NonZeroUsize::MIN, |r| {
                std::hint::black_box(r);
            });
    });
    assert_eq!(depth, 1);
}

#[test]
fn fork_from_a_foreign_worker_does_not_nest() {
    let depth = max_nesting(|pool| {
        Exec::new(pool, WIDTH).unwrap().fork(WIDTH, |i, _| {
            std::hint::black_box(i);
        });
    });
    assert_eq!(depth, 1);
}

#[test]
fn fork_sized_slots_from_a_foreign_worker_does_not_nest() {
    let depth = max_nesting(|pool| {
        let mut slots = [0usize; WIDTH];
        Exec::new(pool, WIDTH)
            .unwrap()
            .fork_sized_slots(&[1; WIDTH], &mut slots, |i, slot, _| *slot = i + 1)
            .unwrap();
        assert_eq!(slots, [1, 2, 3, 4]);
    });
    assert_eq!(depth, 1);
}

#[test]
fn external_from_a_foreign_worker_does_not_nest() {
    let depth = max_nesting(|pool| {
        let width = Exec::new(pool, WIDTH).unwrap().external(|w| w.get());
        assert_eq!(width, WIDTH);
    });
    assert_eq!(depth, 1);
}

/// Runs `f` on a worker of a fresh two-thread Rayon pool, which is not a
/// worker of any `threadteam` pool.
fn on_foreign_worker<R: Send>(f: impl FnOnce() -> R + Send) -> R {
    let outer = rayon::ThreadPoolBuilder::new()
        .num_threads(2)
        .build()
        .unwrap();
    outer.install(|| {
        assert!(rayon::current_thread_index().is_some());
        f()
    })
}

#[derive(Debug, PartialEq)]
struct Payload(usize);

#[test]
fn a_panic_in_a_foreign_worker_entry_reaches_the_caller() {
    let pool = target();
    on_foreign_worker(|| {
        // `for_ranges`: one of four tasks panics; the other three finish before
        // the caller sees the panic.
        let finished = AtomicUsize::new(0);
        let caught = catch_unwind(AssertUnwindSafe(|| {
            Exec::new(&pool, WIDTH)
                .unwrap()
                .for_ranges(WIDTH, NonZeroUsize::MIN, |r| {
                    if r.start == 2 {
                        panic_any(Payload(2));
                    }
                    std::thread::sleep(std::time::Duration::from_millis(5));
                    finished.fetch_add(1, Ordering::SeqCst);
                });
        }))
        .unwrap_err();
        assert_eq!(caught.downcast_ref::<Payload>(), Some(&Payload(2)));
        assert_eq!(finished.load(Ordering::SeqCst), WIDTH - 1);

        // `external`: the callback's own panic.
        let caught = catch_unwind(AssertUnwindSafe(|| {
            Exec::new(&pool, WIDTH)
                .unwrap()
                .external(|_| -> usize { panic_any(Payload(9)) })
        }))
        .unwrap_err();
        assert_eq!(caught.downcast_ref::<Payload>(), Some(&Payload(9)));

        // The pool still runs every task of a further call.
        let covered = AtomicUsize::new(0);
        Exec::new(&pool, WIDTH)
            .unwrap()
            .for_ranges(64, NonZeroUsize::MIN, |r| {
                covered.fetch_add(r.len(), Ordering::SeqCst);
            });
        assert_eq!(covered.into_inner(), 64);
    });
}

#[test]
fn external_from_a_foreign_worker_returns_its_value() {
    let pool = target();
    let text = on_foreign_worker(|| {
        Exec::new(&pool, WIDTH)
            .unwrap()
            .external(|w| format!("width {w}"))
    });
    assert_eq!(text, "width 4");
}

#[test]
fn fork_sized_slots_from_a_foreign_worker_writes_each_slot_once() {
    let pool = target();
    let slots = on_foreign_worker(|| {
        let mut slots = [0usize; WIDTH];
        Exec::new(&pool, WIDTH)
            .unwrap()
            .fork_sized_slots(&[1; WIDTH], &mut slots, |_, slot, _| *slot += 1)
            .unwrap();
        slots
    });
    assert_eq!(slots, [1; WIDTH]);
}
