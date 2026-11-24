//! Ledger-backed invariants: concurrency per root token and pool entries.
#![cfg(all(feature = "parallel", feature = "ledger"))]

use std::num::NonZeroUsize;
use std::sync::Barrier;
use std::time::Duration;
use threadteam::{ledger, Exec, Pool};

fn pool(n: usize) -> Pool {
    Pool::builder()
        .threads(n)
        .thread_name(|i| format!("ledger-{i}"))
        .build()
        .unwrap()
}

fn nap() {
    std::thread::sleep(Duration::from_millis(2));
}

#[test]
fn nested_fork_never_exceeds_the_root_budget() {
    let pool = pool(8);
    for budget in [1, 2, 3, 5, 8] {
        let exec = Exec::new(&pool, budget).unwrap();
        let root = exec.ledger_root().unwrap();
        exec.fork(5, |_, l1| {
            l1.fork(4, |_, l2| {
                l2.fork(3, |_, mut l3| {
                    nap();
                    l3.reborrow().for_ranges(10, NonZeroUsize::MIN, |_| nap());
                    l3.fork(2, |_, _| nap());
                });
            });
        });
        assert!(ledger::max_live(root) <= budget, "budget {budget}");
        assert_eq!(ledger::live(root), 0);
    }
}

#[test]
fn saturating_flat_fork_reaches_the_budget() {
    let pool = pool(8);
    let exec = Exec::new(&pool, 8).unwrap();
    let root = exec.ledger_root().unwrap();
    let barrier = Barrier::new(8);
    exec.fork(8, |_, _| {
        barrier.wait();
    });
    assert_eq!(ledger::max_live(root), 8);
}

#[test]
fn saturating_three_level_fork_reaches_exactly_the_budget() {
    let pool = pool(8);
    let exec = Exec::new(&pool, 8).unwrap();
    let root = exec.ledger_root().unwrap();
    let barrier = Barrier::new(8);
    exec.fork(2, |_, l1| {
        l1.fork(2, |_, l2| {
            l2.fork(2, |_, _| {
                barrier.wait();
            });
        });
    });
    assert_eq!(ledger::max_live(root), 8);
}

#[test]
fn for_ranges_and_broadcast_are_counted() {
    let pool = pool(4);
    let exec = Exec::new(&pool, 4).unwrap();
    let root = exec.ledger_root().unwrap();
    let barrier = Barrier::new(4);
    exec.for_ranges(4, NonZeroUsize::MIN, |_| {
        barrier.wait();
    });
    assert_eq!(ledger::max_live(root), 4);

    let exec = Exec::new(&pool, 4).unwrap();
    let root = exec.ledger_root().unwrap();
    let barrier = Barrier::new(4);
    exec.broadcast(|_| {
        barrier.wait();
    })
    .unwrap();
    assert_eq!(ledger::max_live(root), 4);
}

#[test]
fn same_pool_reentry_does_not_install() {
    let pool = pool(4);
    // Entering from a plain thread installs exactly once per entry.
    let outer = Exec::new(&pool, 2).unwrap();
    let outer_root = outer.ledger_root().unwrap();
    outer.fork(2, |_, _| {
        // Now on a worker of `pool`: nested entries of any root must not install.
        let inner = Exec::new(&pool, 2).unwrap();
        let inner_root = inner.ledger_root().unwrap();
        inner.fork(2, |_, _| {});
        let inner = Exec::new(&pool, 2).unwrap();
        let inner_root2 = inner.ledger_root().unwrap();
        inner.for_ranges(8, NonZeroUsize::MIN, |_| {});
        assert_eq!(ledger::installs(inner_root), 0);
        assert_eq!(ledger::installs(inner_root2), 0);
    });
    assert_eq!(ledger::installs(outer_root), 1);
}

/// One parallel entry made on a token it consumes.
type Entry = Box<dyn FnOnce(Exec<'_>)>;

/// Runs `entry` on a fresh budget-3 root token of `pool` and returns that
/// root's `(installs, max_live)`.
fn counts(pool: &Pool, entry: impl FnOnce(Exec<'_>)) -> (usize, usize) {
    let exec = Exec::new(pool, 3).unwrap();
    let root = exec.ledger_root().unwrap();
    entry(exec);
    (ledger::installs(root), ledger::max_live(root))
}

#[test]
fn a_foreign_worker_entry_counts_one_install_and_stays_in_budget() {
    let pool = pool(4);
    let outer = rayon::ThreadPoolBuilder::new()
        .num_threads(2)
        .build()
        .unwrap();
    outer.install(|| {
        // On a worker of another pool: every entry leaves the calling thread
        // once, through `run_tasks` (`for_ranges`, `fork`, `fork_sized`,
        // `fork_sized_slots`), `enter` (`external`) or Rayon's `broadcast`.
        assert!(rayon::current_thread_index().is_some());
        let entries: [(&str, Entry); 6] = [
            (
                "for_ranges",
                Box::new(|e| e.for_ranges(12, NonZeroUsize::MIN, |_| nap())),
            ),
            ("fork", Box::new(|e| e.fork(6, |_, _| nap()))),
            (
                "fork_sized",
                Box::new(|e| e.fork_sized(&[2, 1], |_, _| nap()).unwrap()),
            ),
            (
                "fork_sized_slots",
                Box::new(|e| {
                    let mut slots = [0usize; 2];
                    e.fork_sized_slots(&[2, 1], &mut slots, |_, slot, _| *slot += 1)
                        .unwrap();
                    assert_eq!(slots, [1, 1]);
                }),
            ),
            (
                "external",
                Box::new(|e| assert_eq!(e.external(|w| w.get()), 3)),
            ),
            ("broadcast", Box::new(|e| e.broadcast(|_| nap()).unwrap())),
        ];
        for (name, entry) in entries {
            let (installs, max_live) = counts(&pool, entry);
            assert_eq!(installs, 1, "{name}");
            assert!(max_live <= 3, "{name}: {max_live} tasks at once");
        }
    });
}

#[test]
fn a_broadcast_counts_one_install() {
    let pool = pool(4);
    let (installs, _) = counts(&pool, |e| e.broadcast(|_| {}).unwrap());
    assert_eq!(installs, 1);
}

#[test]
fn inline_entries_never_install() {
    let pool = pool(4);
    let exec = Exec::new(&pool, 4).unwrap();
    let root = exec.ledger_root().unwrap();
    exec.fork(1, |_, child| child.for_ranges(1, NonZeroUsize::MIN, |_| {}));
    assert_eq!(ledger::installs(root), 0);

    let exec = Exec::new(&pool, 1).unwrap();
    let root = exec.ledger_root().unwrap();
    exec.fork(9, |_, _| {});
    assert_eq!(ledger::installs(root), 0);
}
