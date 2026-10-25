//! Dispatch inside a reservation allocates nothing (without the ledger).
#![cfg(all(feature = "parallel", not(feature = "ledger")))]

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;
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

fn allocs_in(f: impl FnOnce()) -> usize {
    let before = ALLOCS.load(Ordering::SeqCst);
    f();
    ALLOCS.load(Ordering::SeqCst) - before
}

// One test: the counter is process-global.
#[test]
fn team_dispatch_does_not_allocate_after_warm_up() {
    let pool = Pool::builder()
        .threads(4)
        .idle_spin(Duration::from_secs(5))
        .acquire_timeout(Duration::from_secs(5))
        .build()
        .unwrap();
    let mut exec = Exec::new(&pool, 4).unwrap();
    let sink = AtomicUsize::new(0);
    exec.team(|mut team| {
        let phases = |team: &mut threadteam::TeamExec<'_>| {
            team.reborrow().run(|i, _| {
                sink.fetch_add(i, Ordering::Relaxed);
            });
            team.reborrow().run_with(|p| {
                p.barrier().unwrap();
                sink.fetch_add(p.index(), Ordering::Relaxed);
            });
            team.reborrow()
                .split(&[2, 2], |c, child| {
                    child.run(|i, _| {
                        sink.fetch_add(i + c, Ordering::Relaxed);
                    });
                })
                .unwrap();
            team.reborrow().fork(3, |_, child| child.run(|_, _| {}));
        };
        for _ in 0..50 {
            phases(&mut team); // warm-up
        }
        for round in 0..20 {
            assert_eq!(allocs_in(|| phases(&mut team)), 0, "round {round}");
        }
    })
    .unwrap();
    assert!(sink.load(Ordering::Relaxed) > 0);

    // Team-minted tokens: fork / for_ranges / broadcast / team, warm and
    // right after members parked (unpark must not allocate either).
    let pool = Pool::builder()
        .threads(4)
        .idle_spin(Duration::from_millis(1))
        .acquire_timeout(Duration::from_secs(5))
        .build()
        .unwrap();
    let team = pool.into_team(4).unwrap();
    let mut exec = team.exec(4);
    let grain = std::num::NonZeroUsize::MIN;
    let phases = |exec: &mut Exec<'_>| {
        exec.reborrow().fork(7, |i, child| {
            sink.fetch_add(i, Ordering::Relaxed);
            child.fork(2, |_, _| {});
        });
        exec.reborrow().for_ranges(100, grain, |r| {
            sink.fetch_add(r.len(), Ordering::Relaxed);
        });
        exec.reborrow()
            .broadcast(|i| {
                sink.fetch_add(i, Ordering::Relaxed);
            })
            .unwrap();
        exec.reborrow().team(|t| t.run(|_, _| {})).unwrap();
    };
    for _ in 0..50 {
        phases(&mut exec);
    }
    for round in 0..20 {
        assert_eq!(allocs_in(|| phases(&mut exec)), 0, "warm round {round}");
    }
    for round in 0..5 {
        std::thread::sleep(Duration::from_millis(30)); // members park
        assert_eq!(
            allocs_in(|| phases(&mut exec)),
            0,
            "after park, round {round}"
        );
    }
}
