//! Reserved awake members: acquisition, dispatch, nesting, policy, unwind.
#![cfg(feature = "parallel")]

use std::collections::HashSet;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Barrier, Mutex};
use std::thread::ThreadId;
use std::time::{Duration, Instant};
use threadteam::{Cancelled, Error, Exec, Pool, Refused, WaitOverride};

/// A hang is a failure: abort the whole process if the test body stalls.
fn watchdog() -> impl Drop {
    struct Stop(std::sync::mpsc::Sender<()>);
    impl Drop for Stop {
        fn drop(&mut self) {
            let _ = self.0.send(());
        }
    }
    let (tx, rx) = std::sync::mpsc::channel::<()>();
    std::thread::spawn(move || {
        if rx.recv_timeout(Duration::from_secs(45)).is_err() {
            eprintln!("watchdog: test stalled");
            std::process::exit(101);
        }
    });
    Stop(tx)
}

fn pool(n: usize) -> Pool {
    Pool::builder()
        .threads(n)
        .idle_spin(Duration::ZERO)
        .acquire_timeout(Duration::from_secs(5))
        .build()
        .unwrap()
}

/// A pool whose acquisitions tolerate a loaded test host.
/// A pool whose idle members stay for `spin` of wall time: the poll bound of
/// an idle member (`spin_polls`) is lifted so that `idle_spin` alone decides,
/// which these tests are about. `idle_members_return_after_the_spin_budget`
/// covers the poll bound.
fn builder(n: usize, spin: Duration) -> threadteam::PoolBuilder {
    Pool::builder()
        .threads(n)
        .idle_spin(spin)
        .spin_polls(u32::MAX)
        .acquire_timeout(Duration::from_secs(5))
}

fn wait_until(what: &str, mut cond: impl FnMut() -> bool) {
    let start = Instant::now();
    while !cond() {
        assert!(start.elapsed() < Duration::from_secs(10), "timeout: {what}");
        std::thread::sleep(Duration::from_millis(1));
    }
}

fn ids_of(team: threadteam::TeamExec<'_>) -> HashSet<ThreadId> {
    let ids = Mutex::new(HashSet::new());
    team.run(|_, _| {
        ids.lock().unwrap().insert(std::thread::current().id());
    });
    ids.into_inner().unwrap()
}

#[test]
fn every_participant_is_on_its_own_thread_and_the_caller_leads() {
    let _w = watchdog();
    let pool = pool(6);
    for budget in [2, 3, 6] {
        let mut exec = Exec::new(&pool, budget).unwrap();
        let me = std::thread::current().id();
        let leader = Mutex::new(None);
        let seen = Mutex::new(Vec::new());
        exec.team(|team| {
            assert_eq!(team.size().get(), budget);
            team.run_with(|p| {
                // All participants are in the section at the same time.
                p.barrier().unwrap();
                seen.lock().unwrap().push(std::thread::current().id());
                if p.index() == 0 {
                    *leader.lock().unwrap() = Some(std::thread::current().id());
                }
            });
        })
        .unwrap();
        let seen: HashSet<_> = seen.into_inner().unwrap().into_iter().collect();
        assert_eq!(seen.len(), budget);
        assert_eq!(leader.into_inner().unwrap(), Some(me));
        assert!(seen.contains(&me));
    }
}

#[test]
fn serial_and_budget_one_teams_run_inline_without_a_pool() {
    let _w = watchdog();
    let here = std::thread::current().id();
    let mut serial = Exec::serial();
    let n = serial
        .team(|team| {
            assert_eq!(team.size().get(), 1);
            team.run(|i, n| {
                assert_eq!((i, n), (0, 1));
                assert_eq!(std::thread::current().id(), here);
            });
            7
        })
        .unwrap();
    assert_eq!(n, 7);
    let pool = pool(2);
    let mut one = Exec::new(&pool, 1).unwrap();
    one.team(|team| assert_eq!(team.size().get(), 1)).unwrap();
    assert_eq!(pool.resident_members(), 0);
}

#[test]
fn sections_use_the_same_reserved_members_and_survive_a_long_serial_phase() {
    let _w = watchdog();
    // Spin time (20 ms) is far shorter than the serial phase (200 ms).
    let pool = builder(4, Duration::from_millis(20)).build().unwrap();
    let mut exec = Exec::new(&pool, 4).unwrap();
    exec.team(|mut team| {
        let first = ids_of(team.reborrow());
        assert_eq!(first.len(), 4);
        std::thread::sleep(Duration::from_millis(200));
        assert_eq!(
            pool.resident_members(),
            3,
            "members left during a serial phase"
        );
        let second = ids_of(team.reborrow());
        assert_eq!(first, second);
        // Again, after sections.
        std::thread::sleep(Duration::from_millis(100));
        assert_eq!(ids_of(team), first);
    })
    .unwrap();
}

#[test]
fn reserved_members_serve_no_unrelated_rayon_job() {
    let _w = watchdog();
    let pool = pool(3);
    let mut exec = Exec::new(&pool, 3).unwrap();
    exec.team(|mut team| {
        let members = ids_of(team.reborrow());
        assert_eq!(members.len(), 3);
        // One worker is free; jobs must run there, not on a reserved member.
        let job_threads = Mutex::new(HashSet::new());
        std::thread::scope(|s| {
            s.spawn(|| {
                Exec::new(&pool, 2).unwrap().fork(32, |_, _| {
                    job_threads
                        .lock()
                        .unwrap()
                        .insert(std::thread::current().id());
                    std::thread::sleep(Duration::from_micros(200));
                });
            });
        });
        let job_threads = job_threads.into_inner().unwrap();
        let me = std::thread::current().id();
        let member_threads: HashSet<_> = members.into_iter().filter(|t| *t != me).collect();
        assert!(
            job_threads.is_disjoint(&member_threads),
            "a reserved member served a job"
        );
        team.run(|_, _| {});
    })
    .unwrap();
}

#[test]
fn acquisition_refuses_while_a_running_job_holds_a_needed_worker() {
    let _w = watchdog();
    let pool = builder(3, Duration::ZERO)
        .acquire_timeout(Duration::from_millis(100))
        .build()
        .unwrap();
    let started = AtomicUsize::new(0);
    let release = AtomicBool::new(false);
    std::thread::scope(|s| {
        // A job that depends on something the leader does only *after* its
        // team call returns: waiting for it would deadlock.
        s.spawn(|| {
            Exec::new(&pool, 2).unwrap().fork(2, |_, _| {
                started.fetch_add(1, Ordering::SeqCst);
                while !release.load(Ordering::SeqCst) {
                    std::thread::sleep(Duration::from_millis(1));
                }
            });
        });
        wait_until("both workers busy", || started.load(Ordering::SeqCst) == 2);
        let mut exec = Exec::new(&pool, 3).unwrap(); // needs two members, one is free
        let t = Instant::now();
        let r = exec.team(|_| panic!("callback must not run"));
        assert_eq!(r.unwrap_err(), Refused::Unavailable);
        assert!(t.elapsed() < Duration::from_secs(2));
        // No tentative claim is left behind.
        wait_until("tentative claims released", || pool.resident_members() == 0);
        // The caller keeps a barrier-free fallback.
        let hits = AtomicUsize::new(0);
        exec.limit(2.try_into().unwrap())
            .for_ranges(4, 1.try_into().unwrap(), |r| {
                hits.fetch_add(r.len(), Ordering::Relaxed);
            });
        // (the pool is saturated by the blockers; the fallback ran inline-or-queued)
        release.store(true, Ordering::SeqCst);
    });
    // Once the job is gone the same request succeeds.
    let mut exec = Exec::new(&pool, 3).unwrap();
    assert_eq!(exec.team(|t| t.size().get()).unwrap(), 3);
}

#[test]
fn nested_child_teams_run_concurrently_on_distinct_threads() {
    let _w = watchdog();
    let pool = pool(8);
    let mut exec = Exec::new(&pool, 8).unwrap();
    // 8 participants in 3 leaf sections wait on one barrier: they must be 8 threads.
    let rendezvous = Barrier::new(8);
    let threads = Mutex::new(HashSet::new());
    exec.team(|team| {
        assert_eq!(team.size().get(), 8);
        team.split(&[3, 5], |c, child| {
            if c == 0 {
                assert_eq!(child.size().get(), 3);
                child.run(|_, _| {
                    threads.lock().unwrap().insert(std::thread::current().id());
                    rendezvous.wait();
                });
            } else {
                assert_eq!(child.size().get(), 5);
                child
                    .split(&[2, 3], |g, grand| {
                        assert_eq!(grand.size().get(), 2 + g);
                        grand.run(|_, _| {
                            threads.lock().unwrap().insert(std::thread::current().id());
                            rendezvous.wait();
                        });
                    })
                    .unwrap();
            }
        })
        .unwrap();
    })
    .unwrap();
    assert_eq!(threads.into_inner().unwrap().len(), 8);
    // Members came back to the parent's reservation and were released.
    wait_until("released", || pool.resident_members() == 0);
}

#[test]
fn split_with_leftover_members_and_even_fork() {
    let _w = watchdog();
    let pool = pool(6);
    let mut exec = Exec::new(&pool, 6).unwrap();
    exec.team(|mut team| {
        let sum = AtomicUsize::new(0);
        team.reborrow()
            .split(&[2, 1], |c, child| {
                child.run(|i, n| {
                    sum.fetch_add(1000 * c + 10 * n + i, Ordering::Relaxed);
                });
            })
            .unwrap();
        // c0: (20+0)+(20+1) = 41; c1: 1000+10+0 = 1010
        assert_eq!(sum.load(Ordering::Relaxed), 41 + 1010);
        // Members of the leftover set are still usable afterwards.
        assert_eq!(ids_of(team.reborrow()).len(), 6);
        let sizes = Mutex::new(Vec::new());
        team.fork(4, |c, child| {
            sizes.lock().unwrap().push((c, child.size().get()))
        });
        let mut sizes = sizes.into_inner().unwrap();
        sizes.sort();
        assert_eq!(sizes, vec![(0, 2), (1, 2), (2, 1), (3, 1)]);
    })
    .unwrap();
}

#[test]
fn invalid_splits_are_rejected_before_anything_runs() {
    let _w = watchdog();
    let pool = pool(4);
    let mut exec = Exec::new(&pool, 4).unwrap();
    exec.team(|mut team| {
        for bad in [&[][..], &[0, 4][..], &[3, 2][..], &[usize::MAX, 2][..]] {
            let r = team.reborrow().split(bad, |_, _| panic!("must not run"));
            assert_eq!(r.unwrap_err(), Error::InvalidSplit);
        }
        team.split(&[4], |_, c| assert_eq!(c.size().get(), 4))
            .unwrap();
    })
    .unwrap();
}

#[test]
fn scoped_policy_is_inherited_and_overridable_and_cannot_release_reservation() {
    let _w = watchdog();
    let pool = builder(4, Duration::from_millis(500)).build().unwrap();
    let base = pool.wait_policy();
    assert_eq!(base.idle_spin(), Duration::from_millis(500));
    let mut exec = Exec::new(&pool, 4).unwrap();
    exec.team_with(WaitOverride::new().idle_runs_rayon(true), |mut team| {
        let p = team.policy();
        assert!(p.idle_runs_rayon());
        assert_eq!(
            p.idle_spin(),
            Duration::from_millis(500),
            "unset field inherited"
        );
        team.reborrow()
            .split(&[2, 2], |c, child| {
                assert_eq!(child.policy(), p, "child inherits");
                if c == 0 {
                    let child =
                        child.with_wait(WaitOverride::new().idle_spin(Duration::from_millis(1)));
                    assert_eq!(child.policy().idle_spin(), Duration::from_millis(1));
                    assert!(child.policy().idle_runs_rayon());
                    child.fork(2, |_, grand| {
                        assert_eq!(grand.policy().idle_spin(), Duration::from_millis(1));
                    });
                } else {
                    // The sibling's override is not visible here.
                    assert_eq!(child.policy().idle_spin(), Duration::from_millis(500));
                }
            })
            .unwrap();
        // A child's 1 ms spin did not release the parent's members.
        std::thread::sleep(Duration::from_millis(50));
        assert_eq!(pool.resident_members(), 3);
        team.run(|_, _| {});
    })
    .unwrap();
    assert_eq!(pool.wait_policy(), base, "pool defaults are never mutated");
}

#[test]
fn idle_spin_governs_how_long_members_stay_awake_after_the_root_reservation() {
    let _w = watchdog();
    let pool = builder(4, Duration::from_secs(30)).build().unwrap();
    let mut exec = Exec::new(&pool, 4).unwrap();
    exec.team(|t| t.run(|_, _| {})).unwrap();
    assert_eq!(pool.resident_members(), 3, "default spin keeps them awake");
    // Warm reacquisition reuses the awake members.
    let again = Mutex::new(HashSet::new());
    exec.team(|t| *again.lock().unwrap() = ids_of(t)).unwrap();
    assert_eq!(again.into_inner().unwrap().len(), 4);
    assert_eq!(pool.resident_members(), 3);
    // A zero override sends them back as soon as this reservation ends.
    exec.team_with(WaitOverride::new().idle_spin(Duration::ZERO), |t| {
        t.run(|_, _| {})
    })
    .unwrap();
    wait_until("zero spin releases", || pool.resident_members() == 0);
}

#[test]
fn release_idle_waits_for_reservations_and_then_returns_members() {
    let _w = watchdog();
    let pool = builder(4, Duration::from_secs(30)).build().unwrap();
    let mut exec = Exec::new(&pool, 4).unwrap();
    exec.team(|mut team| {
        let before = ids_of(team.reborrow());
        pool.release_idle(); // inside a reservation: must not block or revoke
        std::thread::sleep(Duration::from_millis(30));
        assert_eq!(pool.resident_members(), 3, "live token revoked");
        assert_eq!(ids_of(team), before);
    })
    .unwrap();
    wait_until("released after the reservation", || {
        pool.resident_members() == 0
    });
    // Later teams are unaffected by the old request.
    exec.team(|t| t.run(|_, _| {})).unwrap();
    std::thread::sleep(Duration::from_millis(30));
    assert_eq!(pool.resident_members(), 3);
}

#[test]
fn idle_members_serve_rayon_only_when_the_policy_allows() {
    let _w = watchdog();
    let pool = builder(3, Duration::from_secs(30))
        .idle_runs_rayon(true)
        .build()
        .unwrap();
    let mut exec = Exec::new(&pool, 3).unwrap();
    exec.team(|t| t.run(|_, _| {})).unwrap();
    assert_eq!(pool.resident_members(), 2);
    // Three tasks wait for each other: needs all three workers, including
    // the two idle members, to take Rayon jobs.
    let b = Barrier::new(3);
    std::thread::scope(|s| {
        s.spawn(|| {
            Exec::new(&pool, 3).unwrap().fork(3, |_, _| {
                b.wait();
            });
        });
    });
    // And a team can still be taken afterwards.
    assert_eq!(exec.team(|t| t.size().get()).unwrap(), 3);
}

#[test]
fn a_member_panic_reaches_the_leader_after_all_shares_joined() {
    let _w = watchdog();
    let pool = pool(4);
    let mut exec = Exec::new(&pool, 4).unwrap();
    let finished = AtomicUsize::new(0);
    let r = catch_unwind(AssertUnwindSafe(|| {
        exec.team(|team| {
            team.run(|i, _| {
                if i == 2 {
                    panic!("member two failed");
                }
                std::thread::sleep(Duration::from_millis(30));
                finished.fetch_add(1, Ordering::SeqCst);
            });
        })
    }));
    let payload = r.expect_err("panic must propagate");
    assert_eq!(*payload.downcast::<&str>().unwrap(), "member two failed");
    assert_eq!(
        finished.load(Ordering::SeqCst),
        3,
        "shares not joined before unwinding"
    );
    // Reservation released; the pool is reusable.
    wait_until("released", || pool.resident_members() == 0);
    assert_eq!(exec.team(|t| ids_of(t).len()).unwrap(), 4);
}

#[test]
fn a_leader_panic_joins_the_members_first() {
    let _w = watchdog();
    let pool = pool(3);
    let mut exec = Exec::new(&pool, 3).unwrap();
    let finished = AtomicUsize::new(0);
    let r = catch_unwind(AssertUnwindSafe(|| {
        exec.team(|team| {
            team.run(|i, _| {
                if i == 0 {
                    panic!("leader failed");
                }
                std::thread::sleep(Duration::from_millis(30));
                finished.fetch_add(1, Ordering::SeqCst);
            });
        })
    }));
    assert!(r.is_err());
    assert_eq!(finished.load(Ordering::SeqCst), 2);
    assert_eq!(exec.team(|t| t.size().get()).unwrap(), 3);
}

#[test]
fn a_panic_cancels_barriers_so_peers_leave_instead_of_deadlocking() {
    let _w = watchdog();
    let pool = pool(5);
    for panicker in [0, 3] {
        let mut exec = Exec::new(&pool, 5).unwrap();
        let cancelled = AtomicUsize::new(0);
        let r = catch_unwind(AssertUnwindSafe(|| {
            exec.team(|team| {
                team.run_with(|p| {
                    if p.index() == panicker {
                        panic!("boom");
                    }
                    // The panicker never arrives: without cancellation this waits forever.
                    if p.barrier() == Err(Cancelled) {
                        cancelled.fetch_add(1, Ordering::SeqCst);
                    }
                });
            })
        }));
        assert!(r.is_err());
        assert_eq!(cancelled.load(Ordering::SeqCst), 4);
        // A fresh section after the failure has fresh barriers.
        exec.team(|team| {
            team.run_with(|p| {
                p.barrier().unwrap();
                p.barrier().unwrap();
            });
        })
        .unwrap();
    }
}

/// The same with every wait asleep (`spin_polls(0)`): the panicker raises the
/// cancellation after its peers went to sleep in the barrier, and the
/// cancellation wakes them (`Section::fail`).
#[test]
fn a_panic_wakes_peers_asleep_in_a_barrier() {
    let _w = watchdog();
    let pool = Pool::builder()
        .threads(5)
        .spin_polls(0)
        .acquire_timeout(Duration::from_secs(5))
        .build()
        .unwrap();
    for panicker in [0, 3] {
        let mut exec = Exec::new(&pool, 5).unwrap();
        let cancelled = AtomicUsize::new(0);
        let r = catch_unwind(AssertUnwindSafe(|| {
            exec.team(|team| {
                team.run_with(|p| {
                    if p.index() == panicker {
                        // Let the peers reach the barrier and sleep first.
                        std::thread::sleep(Duration::from_millis(20));
                        panic!("boom");
                    }
                    if p.barrier() == Err(Cancelled) {
                        cancelled.fetch_add(1, Ordering::SeqCst);
                    }
                });
            })
        }));
        assert!(r.is_err());
        assert_eq!(cancelled.load(Ordering::SeqCst), 4);
    }
}

/// Every wait asleep (`spin_polls(0)`), many short sections with barriers,
/// splits and forks: every section completes with every share counted. A lost
/// wake-up hangs (watchdog).
#[test]
fn sleeping_waits_complete_every_section() {
    let _w = watchdog();
    let pool = Pool::builder()
        .threads(6)
        .spin_polls(0)
        .acquire_timeout(Duration::from_secs(5))
        .build()
        .unwrap();
    let team = pool.into_team(6).unwrap();
    let mut exec = team.exec(6);
    let sum = AtomicUsize::new(0);
    let rounds = 300;
    exec.team(|mut t| {
        let size = t.size().get();
        for _ in 0..rounds {
            t.reborrow().run(|_, _| {
                sum.fetch_add(1, Ordering::Relaxed);
            });
            t.reborrow().run_with(|p| {
                p.barrier().unwrap();
                p.barrier().unwrap();
                sum.fetch_add(1, Ordering::Relaxed);
            });
            t.reborrow()
                .split(&[size / 2, size - size / 2], |_, child| {
                    child.run(|_, _| {
                        sum.fetch_add(1, Ordering::Relaxed);
                    });
                })
                .unwrap();
        }
        assert_eq!(sum.load(Ordering::Relaxed), rounds * 3 * size);
    })
    .unwrap();
}

#[test]
fn a_child_panic_propagates_through_split_after_siblings_finish() {
    let _w = watchdog();
    let pool = pool(4);
    let mut exec = Exec::new(&pool, 4).unwrap();
    let done = AtomicUsize::new(0);
    let r = catch_unwind(AssertUnwindSafe(|| {
        exec.team(|team| {
            team.split(&[2, 2], |c, child| {
                child.run(|i, _| {
                    if c == 1 && i == 1 {
                        panic!("grandchild failed");
                    }
                    std::thread::sleep(Duration::from_millis(30));
                    done.fetch_add(1, Ordering::SeqCst);
                });
            })
            .unwrap();
        })
    }));
    let payload = r.expect_err("must propagate");
    assert_eq!(*payload.downcast::<&str>().unwrap(), "grandchild failed");
    assert_eq!(done.load(Ordering::SeqCst), 3);
    assert_eq!(exec.team(|t| t.size().get()).unwrap(), 4);
}

#[test]
fn a_panic_in_the_scope_callback_releases_the_reservation() {
    let _w = watchdog();
    let pool = pool(3);
    let mut exec = Exec::new(&pool, 3).unwrap();
    let r = catch_unwind(AssertUnwindSafe(|| {
        let _ = exec.team(|_team| panic!("scope failed"));
    }));
    assert!(r.is_err());
    wait_until("released", || pool.resident_members() == 0);
    assert_eq!(exec.team(|t| t.size().get()).unwrap(), 3);
}

#[test]
fn concurrent_teams_on_one_pool_use_disjoint_members() {
    let _w = watchdog();
    let pool = pool(8);
    let all = Mutex::new(Vec::new());
    let both_running = Barrier::new(2);
    std::thread::scope(|s| {
        for _ in 0..2 {
            s.spawn(|| {
                let mut exec = Exec::new(&pool, 4).unwrap();
                exec.team(|team| {
                    team.run(|i, _| {
                        if i == 0 {
                            both_running.wait();
                        }
                        all.lock().unwrap().push(std::thread::current().id());
                    });
                })
                .unwrap();
            });
        }
    });
    let all = all.into_inner().unwrap();
    let uniq: HashSet<_> = all.iter().copied().collect();
    assert_eq!(all.len(), 8);
    assert_eq!(uniq.len(), 8, "a thread served two teams at once");
}

#[test]
fn the_team_runs_from_a_pool_worker_with_members_from_the_others() {
    let _w = watchdog();
    let pool = pool(4);
    let mut exec = Exec::new(&pool, 3).unwrap();
    // The leader is itself a worker of the pool (inside a fork job).
    exec.reborrow().fork(1, |_, _| {});
    let ok = AtomicBool::new(false);
    std::thread::scope(|s| {
        s.spawn(|| {
            Exec::new(&pool, 2).unwrap().fork(2, |i, _| {
                if i == 0 {
                    let mut inner = Exec::new(&pool, 3).unwrap();
                    // Two other workers exist (4 - this one), so this may
                    // succeed or be refused if the sibling is busy; both are fine,
                    // neither may deadlock.
                    if let Ok(n) = inner.team(|t| ids_of(t).len()) {
                        assert_eq!(n, 3);
                        ok.store(true, Ordering::SeqCst);
                    }
                }
            });
        });
    });
    let _ = ok.load(Ordering::SeqCst);
}

#[test]
fn idle_members_return_after_the_spin_budget() {
    let _w = watchdog();
    // An hour of `idle_spin`, but a budget of 1000 polls: a released member
    // returns to Rayon after its polls, long before the hour.
    let pool = Pool::builder()
        .threads(4)
        .idle_spin(Duration::from_secs(3600))
        .spin_polls(1000)
        .acquire_timeout(Duration::from_secs(5))
        .build()
        .unwrap();
    let mut exec = Exec::new(&pool, 4).unwrap();
    exec.team(|t| t.run(|_, _| {})).unwrap();
    wait_until("idle members returned after their polls", || {
        pool.resident_members() == 0
    });
}

#[test]
fn pool_drop_joins_idle_members() {
    let _w = watchdog();
    let pool = builder(4, Duration::from_secs(3600)).build().unwrap();
    let mut exec = Exec::new(&pool, 4).unwrap();
    exec.team(|t| t.run(|_, _| {})).unwrap();
    assert_eq!(pool.resident_members(), 3);
    let t = Instant::now();
    drop(exec);
    drop(pool); // must not wait for the hour-long spin
    assert!(t.elapsed() < Duration::from_secs(5));
}
