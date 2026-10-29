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

/// A hang is a failure: abort the whole process if the test body stalls.
fn watchdog() -> impl Drop {
    struct Stop(std::sync::mpsc::Sender<()>);
    impl Drop for Stop {
        fn drop(&mut self) {
            let _ = self.0.send(());
        }
    }
    let (tx, rx) = std::sync::mpsc::channel::<()>();
    let (ready_tx, ready_rx) = std::sync::mpsc::channel::<()>();
    std::thread::spawn(move || {
        // Warm the channel's thread-local state and announce readiness before
        // the test starts counting allocations: this thread must not allocate
        // inside a measured window.
        let _ = rx.recv_timeout(Duration::from_millis(1));
        let _ = ready_tx.send(());
        if rx.recv_timeout(Duration::from_secs(45)).is_err() {
            eprintln!("watchdog: test stalled");
            std::process::exit(101);
        }
    });
    ready_rx.recv().unwrap();
    Stop(tx)
}

fn allocs_in(f: impl FnOnce()) -> usize {
    let before = ALLOCS.load(Ordering::SeqCst);
    f();
    ALLOCS.load(Ordering::SeqCst) - before
}

// One test: the counter is process-global.
#[test]
fn team_dispatch_does_not_allocate_after_warm_up() {
    let _w = watchdog();
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

    // Participant pinning on: minting a root token of a persistent team, running
    // a root team scope through it (which is where the pin section places the
    // participants) and dropping the token must not allocate. The leader saves
    // and restores its mask on its stack, a member saves its mask in its own
    // slot, and both use `cpu_set_t` values, never a `Vec`.
    let pool = Pool::builder()
        .threads(4)
        .pin_participants(true)
        .idle_spin(Duration::from_millis(1))
        .acquire_timeout(Duration::from_secs(5))
        .build()
        .unwrap();
    let team = pool.into_team(4).unwrap();
    let token_phases = |team: &threadteam::Team| {
        let mut exec = team.exec(4); // mint
        exec.team(|mut team| {
            team.reborrow().run(|_, _| {});
            team.reborrow().run_with(|p| {
                let _ = p.barrier();
            });
        })
        .expect("the team token opens its scope");
        // pinned members dispatch as usual
        exec.reborrow().fork(2, |_, mut child| {
            child
                .team(|team| team.run(|_, _| {}))
                .expect("carried sub-team");
        });
        drop(exec); // drop
    };
    for _ in 0..50 {
        token_phases(&team); // warm-up
    }
    for round in 0..20 {
        assert_eq!(allocs_in(|| token_phases(&team)), 0, "pinned round {round}");
    }
    drop(team);

    // A zero spin budget: every join and barrier wait that does not finish at
    // once goes to sleep on the pool's parker (`docs/design/waiting.md`), and
    // every member waits for work asleep. Neither the sleep nor the wake-up
    // allocates. The shares are uneven, so that waiters do sleep.
    let pool = Pool::builder()
        .threads(4)
        .spin_polls(0)
        .acquire_timeout(Duration::from_secs(5))
        .build()
        .unwrap();
    let team = pool.into_team(4).unwrap();
    let mut exec = team.exec(4);
    let sleeping_phases = |exec: &mut Exec<'_>| {
        exec.reborrow()
            .team(|mut t| {
                t.reborrow().run(|i, _| {
                    if i == 1 {
                        std::thread::sleep(Duration::from_micros(200));
                    }
                });
                t.reborrow().run_with(|p| {
                    if p.index() == 2 {
                        std::thread::sleep(Duration::from_micros(200));
                    }
                    p.barrier().unwrap();
                    sink.fetch_add(p.index(), Ordering::Relaxed);
                });
            })
            .unwrap();
    };
    for _ in 0..50 {
        sleeping_phases(&mut exec);
    }
    for round in 0..20 {
        assert_eq!(
            allocs_in(|| sleeping_phases(&mut exec)),
            0,
            "sleeping round {round}"
        );
    }
}
