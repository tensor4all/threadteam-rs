//! Steady-state entries allocate nothing (without the ledger).
#![cfg(all(feature = "parallel", not(feature = "ledger")))]

use std::alloc::{GlobalAlloc, Layout, System};
use std::num::NonZeroUsize;
use std::sync::atomic::{AtomicUsize, Ordering};
use threadteam::{Exec, Pool};

struct Counting;
static ALLOCS: AtomicUsize = AtomicUsize::new(0);

unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, l: Layout) -> *mut u8 {
        ALLOCS.fetch_add(1, Ordering::SeqCst);
        System.alloc(l)
    }
    unsafe fn dealloc(&self, p: *mut u8, l: Layout) {
        System.dealloc(p, l)
    }
    unsafe fn realloc(&self, p: *mut u8, l: Layout, n: usize) -> *mut u8 {
        ALLOCS.fetch_add(1, Ordering::SeqCst);
        System.realloc(p, l, n)
    }
}

#[global_allocator]
static A: Counting = Counting;

fn entries(pool: &Pool) {
    let sink = AtomicUsize::new(0);
    let grain = NonZeroUsize::new(1).unwrap();
    Exec::new(pool, 3).unwrap().fork(7, |i, _| {
        sink.fetch_add(i, Ordering::Relaxed);
    });
    Exec::new(pool, 3).unwrap().for_ranges(100, grain, |r| {
        sink.fetch_add(r.len(), Ordering::Relaxed);
    });
    let w = Exec::new(pool, 3).unwrap().external(|w| w.get());
    assert_eq!(w, 3);
    if !threadteam::Exec::new(pool, 3).unwrap().can_broadcast() {
        return;
    }
    Exec::new(pool, 3)
        .unwrap()
        .broadcast(|i| {
            sink.fetch_add(i, Ordering::Relaxed);
        })
        .unwrap();
    assert!(sink.load(Ordering::Relaxed) > 0);
}

/// Largest allocation count over `rounds` windows of one call each, after
/// warm-up: strict, every window counts (the native scheduler publishes stack
/// batches, so there is no amortized queue allocation to tolerate).
fn measure(rounds: usize, f: impl Fn()) -> usize {
    for _ in 0..20 {
        f();
    }
    (0..rounds)
        .map(|_| {
            let before = ALLOCS.load(Ordering::SeqCst);
            f();
            ALLOCS.load(Ordering::SeqCst) - before
        })
        .max()
        .unwrap_or(0)
}

// One test: the counter is process-global.
#[test]
fn fork_for_ranges_external_do_not_allocate() {
    let pool = Pool::builder().threads(3).build().unwrap();
    let plain = measure(10, || entries(&pool));
    assert_eq!(plain, 0, "plain-thread entries allocated");
    // The same after the workers went to sleep: waking them allocates nothing.
    let asleep = measure(5, || {
        std::thread::sleep(std::time::Duration::from_millis(20));
        let before = ALLOCS.load(Ordering::SeqCst);
        entries(&pool);
        assert_eq!(
            ALLOCS.load(Ordering::SeqCst) - before,
            0,
            "entries after a park allocated"
        );
    });
    let _ = asleep;

    // Inside the pool: a worker of `pool` enters again, which installs nothing,
    // so every window is strict.
    let inside = AtomicUsize::new(usize::MAX);
    Exec::new(&pool, 3).unwrap().fork(2, |i, _| {
        if i == 0 {
            inside.store(measure(10, || entries(&pool)), Ordering::SeqCst);
        }
    });
    assert_eq!(
        inside.load(Ordering::SeqCst),
        0,
        "in-pool entries allocated"
    );

    // Participant pinning on: the extra pin section a root team scope runs must
    // not allocate either. The window contains the acquisition's id `Vec`, the
    // one allocation of an acquisition documented in the design record, so it
    // is compared against the same window with pinning off rather than
    // against zero.
    let team_scope_window = |pin: bool| {
        let pool = Pool::builder()
            .threads(3)
            .pin_participants(pin)
            .idle_spin(std::time::Duration::from_millis(1))
            .acquire_timeout(std::time::Duration::from_secs(5))
            .build()
            .unwrap();
        let scope = |pool: &Pool| {
            let mut exec = Exec::new(pool, 3).unwrap();
            let _live = exec.team(|mut team| team.reborrow().run(|_, _| {}));
        };
        measure(10, || scope(&pool))
    };
    let off = team_scope_window(false);
    let on = team_scope_window(true);
    assert!(off <= 1, "a team scope allocates at most the id Vec: {off}");
    assert_eq!(on, off, "participant pinning added an allocation");
}
