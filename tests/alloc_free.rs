//! Steady-state entries allocate nothing (without the ledger), except the one
//! job that an entry from a worker of another pool submits.
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

fn fork(pool: &Pool) {
    let sink = AtomicUsize::new(0);
    Exec::new(pool, 3).unwrap().fork(7, |i, _| {
        sink.fetch_add(i, Ordering::Relaxed);
    });
    assert_eq!(sink.into_inner(), 21);
}

fn for_ranges(pool: &Pool) {
    let sink = AtomicUsize::new(0);
    Exec::new(pool, 3)
        .unwrap()
        .for_ranges(100, NonZeroUsize::MIN, |r| {
            sink.fetch_add(r.len(), Ordering::Relaxed);
        });
    assert_eq!(sink.into_inner(), 100);
}

fn fork_sized(pool: &Pool) {
    let sink = AtomicUsize::new(0);
    Exec::new(pool, 3)
        .unwrap()
        .fork_sized(&[2, 1], |_, child| {
            sink.fetch_add(child.budget().get(), Ordering::Relaxed);
        })
        .unwrap();
    assert_eq!(sink.into_inner(), 3);
}

fn fork_sized_slots(pool: &Pool) {
    // Per-child state on the stack, no lock: nothing to allocate.
    let mut cursors = [0usize; 3];
    Exec::new(pool, 3)
        .unwrap()
        .fork_sized_slots(&[1, 1, 1], &mut cursors, |i, c, _| *c += i + 1)
        .unwrap();
    assert_eq!(cursors, [1, 2, 3]);
}

fn external(pool: &Pool) {
    let w = Exec::new(pool, 3).unwrap().external(|w| w.get());
    assert_eq!(w, 3);
}

/// Every parallel entry but `broadcast` (whose Rayon call collects its jobs into
/// a `Vec`), each called once with a budget of three.
const ENTRIES: [(&str, fn(&Pool)); 5] = [
    ("fork", fork),
    ("fork_sized", fork_sized),
    ("for_ranges", for_ranges),
    ("fork_sized_slots", fork_sized_slots),
    ("external", external),
];

fn entries(pool: &Pool) {
    for (_, entry) in ENTRIES {
        entry(pool);
    }
}

/// Smallest allocation count over `rounds` windows of one call each, after warm-up.
///
/// From a plain thread every entry pushes one job into rayon's injector queue,
/// whose block allocation (one per 63 pushes with crossbeam-deque 0.8) is
/// amortized and outside this crate; a per-call allocation shows up in every
/// window, so the minimum separates the two.
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
        .min()
        .unwrap_or(0)
}

// One test: the counter is process-global.
#[test]
fn entries_allocate_nothing_but_the_job_a_foreign_worker_submits() {
    let pool = Pool::builder().threads(3).build().unwrap();
    let plain = measure(10, || entries(&pool));
    assert_eq!(plain, 0, "plain-thread entries allocated");

    // On a worker of another pool: each entry submits one job, which
    // `ThreadPool::spawn` boxes, and allocates nothing else. Each entry is
    // measured on its own, so one allocating twice cannot hide behind another
    // allocating nothing.
    let other = Pool::builder()
        .threads(2)
        .thread_name(|i| format!("other-{i}"))
        .build()
        .unwrap();
    let foreign: [AtomicUsize; ENTRIES.len()] = Default::default();
    Exec::new(&other, 2).unwrap().fork(2, |i, _| {
        if i == 0 {
            let name = std::thread::current().name().map(str::to_owned);
            assert!(name.is_some_and(|n| n.starts_with("other-")));
            for ((_, entry), count) in ENTRIES.iter().zip(&foreign) {
                count.store(measure(10, || entry(&pool)), Ordering::SeqCst);
            }
        }
    });
    for ((name, _), count) in ENTRIES.iter().zip(&foreign) {
        assert_eq!(
            count.load(Ordering::SeqCst),
            1,
            "`{name}` from a worker of another pool: one job and nothing else"
        );
    }

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
