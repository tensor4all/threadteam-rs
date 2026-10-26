//! Persistent reservations: `Pool::into_team`, `Team::exec`, parking.
#![cfg(feature = "parallel")]

use std::collections::HashSet;
use std::num::NonZeroUsize;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Barrier, Mutex};
use std::time::{Duration, Instant};
use threadteam::{Exec, Pool, Refused, WaitOverride};

fn watchdog() -> impl Drop {
    struct Stop(std::sync::mpsc::Sender<()>);
    impl Drop for Stop {
        fn drop(&mut self) {
            let _ = self.0.send(());
        }
    }
    let (tx, rx) = std::sync::mpsc::channel::<()>();
    std::thread::spawn(move || {
        if rx.recv_timeout(Duration::from_secs(120)).is_err() {
            eprintln!("watchdog: test stalled");
            std::process::exit(101);
        }
    });
    Stop(tx)
}

fn pool(n: usize, spin: Duration) -> Pool {
    Pool::builder()
        .threads(n)
        .idle_spin(spin)
        .acquire_timeout(Duration::from_secs(5))
        .build()
        .unwrap()
}

fn wait_until(what: &str, mut cond: impl FnMut() -> bool) {
    let start = Instant::now();
    while !cond() {
        assert!(start.elapsed() < Duration::from_secs(10), "timeout: {what}");
        std::thread::sleep(Duration::from_millis(1));
    }
}

fn one() -> NonZeroUsize {
    NonZeroUsize::MIN
}

#[test]
fn team_tokens_dispatch_fork_on_distinct_reserved_threads() {
    let _w = watchdog();
    let team = pool(5, Duration::from_millis(1)).into_team(5).unwrap();
    assert_eq!(team.size().get(), 5);
    let exec = team.exec(5);
    assert!(exec.can_broadcast());
    let b = Barrier::new(5);
    let threads = Mutex::new(HashSet::new());
    // Five jobs wait for each other: they run on five distinct threads at once.
    exec.fork(5, |_, child| {
        assert_eq!(child.budget().get(), 1);
        threads.lock().unwrap().insert(std::thread::current().id());
        b.wait();
    });
    assert_eq!(threads.into_inner().unwrap().len(), 5);
    assert_eq!(team.pool().resident_members(), 4);
}

#[test]
fn nested_fork_children_carry_their_subteam_and_may_broadcast_or_take_a_team() {
    let _w = watchdog();
    let team = pool(8, Duration::from_millis(1)).into_team(8).unwrap();
    let exec = team.exec(8);
    let b = Barrier::new(8);
    let seen = Mutex::new(HashSet::new());
    exec.fork(2, |c, child| {
        assert_eq!(child.budget().get(), 4);
        assert!(child.can_broadcast(), "a team child owns its members");
        let mut child = child;
        // A broadcast inside the child, concurrently with its sibling.
        child
            .reborrow()
            .broadcast(|_| {
                seen.lock().unwrap().insert(std::thread::current().id());
                b.wait();
            })
            .unwrap();
        // The sub-team is available without acquisition.
        let size = child.team(|t| t.size().get()).unwrap();
        assert_eq!(size, 4, "child {c}");
        // Nested fork keeps going to the carried members.
        child.fork(2, |_, g| assert_eq!(g.budget().get(), 2));
    });
    assert_eq!(seen.into_inner().unwrap().len(), 8);
}

#[test]
fn for_ranges_covers_exactly_once_and_limit_narrows_the_members() {
    let _w = watchdog();
    let team = pool(6, Duration::from_millis(1)).into_team(6).unwrap();
    let mut exec = team.exec(6);
    for len in [0usize, 1, 5, 6, 100, 1001] {
        let covered = AtomicUsize::new(0);
        exec.reborrow().for_ranges(len, one(), |r| {
            covered.fetch_add(r.len(), Ordering::Relaxed);
        });
        assert_eq!(covered.into_inner(), len);
    }
    let threads = Mutex::new(HashSet::new());
    let narrowed = exec.reborrow().limit(2.try_into().unwrap());
    assert_eq!(narrowed.budget().get(), 2);
    narrowed.fork(8, |_, _| {
        threads.lock().unwrap().insert(std::thread::current().id());
        std::thread::sleep(Duration::from_millis(1));
    });
    assert!(threads.into_inner().unwrap().len() <= 2);
    // external hands the width as before.
    let w = exec.external(|w| w.get());
    assert_eq!(w, 6);
}

#[test]
fn members_park_after_the_spin_but_stay_reserved_and_wake_on_dispatch() {
    let _w = watchdog();
    let team = pool(4, Duration::from_millis(2)).into_team(4).unwrap();
    let mut exec = team.exec(4);
    let ids = |exec: &mut Exec<'_>| {
        let ids = Mutex::new(HashSet::new());
        exec.reborrow().fork(4, |_, _| {
            ids.lock().unwrap().insert(std::thread::current().id());
            std::thread::sleep(Duration::from_millis(1));
        });
        ids.into_inner().unwrap()
    };
    let before = ids(&mut exec);
    for _ in 0..5 {
        std::thread::sleep(Duration::from_millis(60)); // far longer than the spin: parked
        assert_eq!(team.pool().resident_members(), 3, "a parked member left");
        assert_eq!(ids(&mut exec), before, "reservation changed across a park");
    }
    // A Rayon job runs on the single free worker, never on a parked member.
    let hits = AtomicUsize::new(0);
    std::thread::scope(|s| {
        s.spawn(|| {
            Exec::new(team.pool(), 2).unwrap().fork(8, |_, _| {
                hits.fetch_add(1, Ordering::Relaxed);
            });
        });
    });
    assert_eq!(hits.load(Ordering::Relaxed), 8);
}

#[test]
fn concurrent_mint_from_two_threads_serializes_the_leaders() {
    let _w = watchdog();
    let team = pool(4, Duration::from_millis(1)).into_team(4).unwrap();
    let inside = AtomicUsize::new(0);
    let max_inside = AtomicUsize::new(0);
    let order = Mutex::new(Vec::new());
    std::thread::scope(|s| {
        for t in 0..3 {
            let (team, inside, max_inside, order) = (&team, &inside, &max_inside, &order);
            s.spawn(move || {
                for _ in 0..5 {
                    let mut exec = team.exec(4);
                    let now = inside.fetch_add(1, Ordering::SeqCst) + 1;
                    max_inside.fetch_max(now, Ordering::SeqCst);
                    exec.reborrow()
                        .fork(4, |_, _| std::thread::sleep(Duration::from_micros(300)));
                    order.lock().unwrap().push(t);
                    inside.fetch_sub(1, Ordering::SeqCst);
                    drop(exec);
                }
            });
        }
    });
    assert_eq!(max_inside.load(Ordering::SeqCst), 1);
    assert_eq!(order.into_inner().unwrap().len(), 15);
}

#[test]
fn try_exec_reports_busy_and_exec_panics_on_reentry() {
    let _w = watchdog();
    let team = pool(3, Duration::from_millis(1)).into_team(3).unwrap();
    let exec = team.exec(3);
    assert_eq!(team.try_exec(3).unwrap_err(), Refused::Busy);
    std::thread::scope(|s| {
        s.spawn(|| assert_eq!(team.try_exec(3).unwrap_err(), Refused::Busy));
    });
    let r = catch_unwind(AssertUnwindSafe(|| {
        let _ = team.exec(3);
    }));
    assert!(r.is_err(), "re-entrant exec must not deadlock");
    drop(exec);
    assert!(team.try_exec(3).is_ok());
}

#[test]
fn dropping_or_unwrapping_the_team_while_parked_returns_the_members() {
    let _w = watchdog();
    // Parked, then into_pool: the members are released and the pool is usable.
    let team = pool(4, Duration::from_millis(1)).into_team(4).unwrap();
    team.exec(4).fork(4, |_, _| {});
    std::thread::sleep(Duration::from_millis(50));
    let pool = team.into_pool();
    wait_until("members returned", || pool.resident_members() == 0);
    // Straight after turning into a team, before any dispatch.
    let pool = pool.into_team(4).unwrap().into_pool();
    wait_until("members returned", || pool.resident_members() == 0);
    // The pool is whole again: a plain token can use every worker.
    let b = Barrier::new(4);
    std::thread::scope(|s| {
        s.spawn(|| {
            Exec::new(&pool, 4).unwrap().fork(4, |_, _| {
                b.wait();
            });
        });
    });
    // Dropping a parked team drops its pool: the workers are joined.
    let team = pool.into_team(4).unwrap();
    std::thread::sleep(Duration::from_millis(20));
    let t = Instant::now();
    drop(team);
    assert!(t.elapsed() < Duration::from_secs(5));
}

#[test]
fn a_panic_in_a_section_with_parked_members_propagates_and_keeps_the_team() {
    let _w = watchdog();
    let team = pool(4, Duration::ZERO).into_team(4).unwrap(); // park immediately
    let mut exec = team.exec(4);
    std::thread::sleep(Duration::from_millis(30));
    let finished = AtomicUsize::new(0);
    let r = catch_unwind(AssertUnwindSafe(|| {
        exec.reborrow().fork(4, |i, _| {
            if i == 2 {
                panic!("job two failed");
            }
            std::thread::sleep(Duration::from_millis(20));
            finished.fetch_add(1, Ordering::SeqCst);
        });
    }));
    let payload = r.expect_err("must propagate");
    assert_eq!(*payload.downcast::<&str>().unwrap(), "job two failed");
    assert_eq!(finished.load(Ordering::SeqCst), 3);
    std::thread::sleep(Duration::from_millis(30)); // park again
                                                   // Barrier cancellation inside a team token's team scope.
    let r = catch_unwind(AssertUnwindSafe(|| {
        exec.reborrow()
            .team(|t| {
                t.run_with(|p| {
                    if p.index() == 1 {
                        panic!("boom");
                    }
                    let _ = p.barrier();
                })
            })
            .unwrap();
    }));
    assert!(r.is_err());
    let n = AtomicUsize::new(0);
    exec.fork(4, |_, _| {
        n.fetch_add(1, Ordering::SeqCst);
    });
    assert_eq!(n.load(Ordering::SeqCst), 4);
    assert_eq!(team.pool().resident_members(), 3);
}

#[test]
fn into_team_with_overrides_the_park_delay_and_zero_width_is_one() {
    let _w = watchdog();
    let team = pool(3, Duration::from_secs(5))
        .into_team_with(0, WaitOverride::new().idle_spin(Duration::ZERO))
        .unwrap();
    assert_eq!(team.size().get(), 1);
    let exec = team.exec(8);
    assert_eq!(exec.budget().get(), 1);
    exec.fork(3, |_, _| {});
    assert_eq!(team.pool().resident_members(), 0);
}

#[test]
fn plain_tokens_are_unchanged_next_to_a_team() {
    let _w = watchdog();
    let team = pool(4, Duration::from_millis(1)).into_team(3).unwrap(); // 2 members, 2 free workers
    let hits = AtomicUsize::new(0);
    std::thread::scope(|s| {
        s.spawn(|| {
            Exec::new(team.pool(), 2).unwrap().fork(8, |_, _| {
                hits.fetch_add(1, Ordering::Relaxed);
            });
        });
    });
    assert_eq!(hits.into_inner(), 8);
    drop(team);
}

#[test]
fn external_on_a_team_token_lends_the_members_to_the_pool_and_takes_them_back() {
    let _w = watchdog();
    let team = pool(4, Duration::from_millis(1)).into_team(4).unwrap(); // 3 members; one worker is free
    std::thread::sleep(Duration::from_millis(20)); // members parked
    let mut exec = team.exec(4);
    let member_threads = {
        let ids = Mutex::new(HashSet::new());
        exec.reborrow().fork(4, |_, _| {
            ids.lock().unwrap().insert(std::thread::current().id());
            std::thread::sleep(Duration::from_millis(1));
        });
        ids.into_inner().unwrap()
    };
    std::thread::sleep(Duration::from_millis(20)); // parked again
                                                   // Four tasks that wait for each other need four threads at once: the one
                                                   // free worker that enters the pool plus the three lent members.
    let threads = Mutex::new(HashSet::new());
    let b = Barrier::new(4);
    let width = exec.reborrow().external(|w| {
        Exec::new(team.pool(), w.get())
            .unwrap()
            .fork(w.get(), |_, _| {
                threads.lock().unwrap().insert(std::thread::current().id());
                b.wait();
            });
        w.get()
    });
    assert_eq!(width, 4, "the width handed over stays the budget");
    let threads = threads.into_inner().unwrap();
    assert_eq!(threads.len(), 4, "w workers actually ran");
    let members: HashSet<_> = member_threads
        .iter()
        .copied()
        .filter(|t| *t != std::thread::current().id())
        .collect();
    assert_eq!(
        members.intersection(&threads).count(),
        3,
        "the lent members served"
    );
    // Members are still reserved and serve the team again.
    assert_eq!(team.pool().resident_members(), 3);
    let after = Mutex::new(HashSet::new());
    exec.fork(4, |_, _| {
        after.lock().unwrap().insert(std::thread::current().id());
        std::thread::sleep(Duration::from_millis(1));
    });
    assert_eq!(after.into_inner().unwrap(), member_threads);
}

#[test]
fn a_wide_team_wakes_all_parked_members_through_the_tree() {
    let _w = watchdog();
    let team = pool(18, Duration::from_millis(1)).into_team(18).unwrap();
    let mut exec = team.exec(18);
    for _ in 0..5 {
        std::thread::sleep(Duration::from_millis(15)); // everybody parks
        let n = AtomicUsize::new(0);
        exec.reborrow().fork(18, |_, _| {
            n.fetch_add(1, Ordering::SeqCst);
        });
        assert_eq!(n.load(Ordering::SeqCst), 18);
        std::thread::sleep(Duration::from_millis(15));
        let n = AtomicUsize::new(0);
        exec.reborrow().fork(3, |_, c| {
            c.for_ranges(12, one(), |r| {
                n.fetch_add(r.len(), Ordering::SeqCst);
            });
        });
        assert_eq!(n.load(Ordering::SeqCst), 36);
    }
}

#[test]
fn a_failed_into_team_gives_the_pool_back() {
    let _w = watchdog();
    // Two of three workers are stuck in their start hook: they cannot take a
    // claim job, like workers busy with a running job.
    let gate = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
    let g = gate.clone();
    let pool = Pool::builder()
        .threads(3)
        .idle_spin(Duration::ZERO)
        .acquire_timeout(Duration::from_millis(100))
        .on_start(move |i| {
            if i != 2 {
                while !g.load(Ordering::SeqCst) {
                    std::thread::sleep(Duration::from_millis(1));
                }
            }
        })
        .build()
        .unwrap();
    let err = pool.into_team(3).expect_err("two workers are unavailable");
    assert_eq!(err.reason(), Refused::Unavailable);
    assert!(err.to_string().contains("team"));
    let pool = err.into_pool();
    assert_eq!(pool.width().get(), 3);
    wait_until("tentative member released", || pool.resident_members() == 0);
    gate.store(true, Ordering::SeqCst);
    let team = pool.into_team(3).expect("all workers free now");
    assert_eq!(team.size().get(), 3);
}

#[test]
fn the_team_owns_the_pool_and_dropping_it_joins_the_workers() {
    let _w = watchdog();
    let team = pool(3, Duration::from_millis(1)).into_team(3).unwrap();
    let mut exec = team.exec(3);
    let n = AtomicUsize::new(0);
    exec.reborrow().fork(3, |_, _| {
        n.fetch_add(1, Ordering::SeqCst);
    });
    assert_eq!(n.load(Ordering::SeqCst), 3);
    drop(exec);
    let t = Instant::now();
    drop(team); // members leave, the pool is dropped and joins its workers
    assert!(t.elapsed() < Duration::from_secs(5));
}

#[test]
fn the_wake_tree_never_loses_a_member_when_parents_start_before_publication_ends() {
    let _w = watchdog();
    let team = pool(18, Duration::from_micros(300)).into_team(18).unwrap();
    let mut exec = team.exec(18);
    for i in 0..400u64 {
        // Sleep around the park delay so that some members spin and some park.
        std::thread::sleep(Duration::from_micros(100 + (i * 37) % 900));
        let n = AtomicUsize::new(0);
        exec.reborrow().fork(18, |_, _| {
            n.fetch_add(1, Ordering::SeqCst);
        });
        assert_eq!(n.load(Ordering::SeqCst), 18, "iteration {i}");
    }
}

/// Threads that burn CPU, so that team threads get preempted in the middle of
/// publication and park/wake windows (the shared CI and developer hosts do the
/// same by accident).
fn burners(n: usize) -> impl Drop {
    use std::sync::atomic::AtomicBool;
    use std::sync::Arc;
    struct Burners(Arc<AtomicBool>, Vec<std::thread::JoinHandle<()>>);
    impl Drop for Burners {
        fn drop(&mut self) {
            self.0.store(true, Ordering::SeqCst);
            for h in self.1.drain(..) {
                let _ = h.join();
            }
        }
    }
    let stop = Arc::new(AtomicBool::new(false));
    let handles = (0..n)
        .map(|i| {
            let stop = stop.clone();
            std::thread::spawn(move || {
                let mut x = i as u64 + 1;
                while !stop.load(Ordering::Relaxed) {
                    for _ in 0..2000 {
                        x = x
                            .wrapping_mul(6364136223846793005)
                            .wrapping_add(1442695040888963407);
                    }
                    std::hint::black_box(x);
                    if x.trailing_zeros() >= 3 {
                        std::thread::yield_now();
                    }
                }
            })
        })
        .collect();
    Burners(stop, handles)
}

#[test]
fn cold_scoped_teams_with_zero_spin_never_lose_a_member() {
    let _w = watchdog();
    let _burn = burners(std::thread::available_parallelism().map_or(2, |n| (n.get() / 4).max(2)));
    let pool = Pool::builder()
        .threads(12)
        .idle_spin(Duration::ZERO)
        .acquire_timeout(Duration::from_secs(5))
        .build()
        .unwrap();
    let mut exec = Exec::new(&pool, 12).unwrap();
    for i in 0..500 {
        while pool.resident_members() != 0 {
            std::thread::yield_now();
        }
        let n = AtomicUsize::new(0);
        exec.team(|t| {
            t.run(|_, _| {
                n.fetch_add(1, Ordering::SeqCst);
            })
        })
        .unwrap();
        assert_eq!(n.load(Ordering::SeqCst), 12, "iteration {i}");
    }
}
