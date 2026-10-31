//! `Exec::fork_sized`: explicit shares, their validation, leftover slots,
//! concurrency per child and panic propagation.

use std::sync::atomic::{AtomicUsize, Ordering};
use threadteam::{Error, Exec};

/// A zero share is refused before anything runs.
#[test]
fn a_zero_share_is_rejected() {
    let calls = AtomicUsize::new(0);
    let err = Exec::serial()
        .fork_sized(&[1, 0], |_, _| {
            calls.fetch_add(1, Ordering::Relaxed);
        })
        .unwrap_err();
    assert_eq!(err, Error::ZeroBudget);
    assert_eq!(calls.into_inner(), 0);
}

/// Shares summing past the budget are refused, not clamped, and the error
/// names the budget and the sum (saturated when it overflows).
#[test]
fn oversubscribing_shares_are_rejected_with_budget_and_sum() {
    let calls = AtomicUsize::new(0);
    let f = |_: usize, _: Exec<'_>| {
        calls.fetch_add(1, Ordering::Relaxed);
    };
    assert_eq!(
        Exec::serial().fork_sized(&[1, 1], f).unwrap_err(),
        Error::SharesExceedBudget { budget: 1, sum: 2 }
    );
    assert_eq!(
        Exec::serial().fork_sized(&[usize::MAX, 1], f).unwrap_err(),
        Error::SharesExceedBudget {
            budget: 1,
            sum: usize::MAX
        }
    );
    assert_eq!(calls.into_inner(), 0);
}

/// One share and an empty share list both run inline (or not at all).
#[test]
fn single_and_empty_shares_run_inline() {
    let here = std::thread::current().id();
    let calls = AtomicUsize::new(0);
    Exec::serial()
        .fork_sized(&[1], |i, child| {
            assert_eq!((i, child.budget().get()), (0, 1));
            assert_eq!(std::thread::current().id(), here);
            calls.fetch_add(1, Ordering::Relaxed);
        })
        .unwrap();
    Exec::serial()
        .fork_sized(&[], |_, _| {
            calls.fetch_add(1, Ordering::Relaxed);
        })
        .unwrap();
    assert_eq!(calls.into_inner(), 1);
}

#[cfg(feature = "parallel")]
mod parallel {
    use std::collections::HashSet;
    use std::panic::{catch_unwind, AssertUnwindSafe};
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::{Barrier, Mutex};
    use std::time::Duration;
    use threadteam::{Exec, Pool};

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
            if rx.recv_timeout(Duration::from_secs(60)).is_err() {
                eprintln!("watchdog: test stalled");
                std::process::exit(101);
            }
        });
        Stop(tx)
    }

    fn pool(n: usize) -> Pool {
        Pool::builder()
            .threads(n)
            .idle_spin(Duration::from_millis(1))
            .acquire_timeout(Duration::from_secs(5))
            .build()
            .unwrap()
    }

    /// Live-participant counter that records its maximum.
    #[derive(Default)]
    struct Gauge {
        live: AtomicUsize,
        max: AtomicUsize,
    }

    impl Gauge {
        fn enter(&self) {
            let now = self.live.fetch_add(1, Ordering::SeqCst) + 1;
            self.max.fetch_max(now, Ordering::SeqCst);
        }
        fn leave(&self) {
            self.live.fetch_sub(1, Ordering::SeqCst);
        }
    }

    /// On a team token each child's budget is exactly its share and all of
    /// its participants run at once: a barrier over the sum of the shares
    /// opens only when every child's whole share is live, and no child ever
    /// has more participants live than its share. The leftover members run
    /// nothing during the call and are the parent's again after the join.
    #[test]
    fn team_children_get_exactly_their_share_and_run_it_concurrently() {
        let _w = watchdog();
        let team = pool(8).into_team(8).unwrap();
        let mut exec = team.exec(8);
        let shares = [3, 1, 2];
        let sum: usize = shares.iter().sum();
        let all = Barrier::new(sum);
        let gauges: Vec<Gauge> = shares.iter().map(|_| Gauge::default()).collect();
        let budgets: Vec<AtomicUsize> = shares.iter().map(|_| AtomicUsize::new(0)).collect();
        let threads = Mutex::new(HashSet::new());
        exec.reborrow()
            .fork_sized(&shares, |i, child| {
                budgets[i].store(child.budget().get(), Ordering::Relaxed);
                let share = Barrier::new(child.budget().get());
                child
                    .broadcast(|_| {
                        threads.lock().unwrap().insert(std::thread::current().id());
                        gauges[i].enter();
                        share.wait();
                        all.wait();
                        gauges[i].leave();
                    })
                    .expect("a team child owns its members");
            })
            .unwrap();
        for (i, &s) in shares.iter().enumerate() {
            assert_eq!(budgets[i].load(Ordering::Relaxed), s, "budget of child {i}");
            assert_eq!(
                gauges[i].max.load(Ordering::SeqCst),
                s,
                "concurrency of child {i}"
            );
        }
        assert_eq!(threads.into_inner().unwrap().len(), sum);

        // Leftover: the parent has its full budget again, all eight at once.
        assert_eq!(exec.budget().get(), 8);
        let full = Barrier::new(8);
        let gauge = Gauge::default();
        exec.broadcast(|_| {
            gauge.enter();
            full.wait();
            gauge.leave();
        })
        .unwrap();
        assert_eq!(gauge.max.into_inner(), 8);
    }

    /// On a plain pool token each child's budget is exactly its share and
    /// each child can use it: a child forks one job per slot, and those jobs
    /// meet at a barrier of the child's width.
    #[test]
    fn plain_children_get_exactly_their_share() {
        let _w = watchdog();
        let pool = pool(4);
        let mut exec = Exec::new(&pool, 4).unwrap();
        let shares = [2, 1];
        let budgets: Vec<AtomicUsize> = shares.iter().map(|_| AtomicUsize::new(0)).collect();
        exec.reborrow()
            .fork_sized(&shares, |i, child| {
                let width = child.budget().get();
                budgets[i].store(width, Ordering::Relaxed);
                let meet = Barrier::new(width);
                let gauge = Gauge::default();
                child.fork(width, |_, _| {
                    gauge.enter();
                    meet.wait();
                    gauge.leave();
                });
                assert_eq!(gauge.max.into_inner(), width, "concurrency of child {i}");
            })
            .unwrap();
        for (i, &s) in shares.iter().enumerate() {
            assert_eq!(budgets[i].load(Ordering::Relaxed), s);
        }
        // Leftover: the parent has its full budget again.
        assert_eq!(exec.budget().get(), 4);
    }

    /// A single share smaller than the budget runs inline on the caller, and
    /// the team it carries has exactly that many participants.
    #[test]
    fn a_single_team_share_carries_a_subteam_of_its_share() {
        let _w = watchdog();
        let team = pool(8).into_team(8).unwrap();
        let mut exec = team.exec(8);
        let here = std::thread::current().id();
        for share in [1, 2] {
            exec.reborrow()
                .fork_sized(&[share], |_, mut child| {
                    assert_eq!(std::thread::current().id(), here);
                    assert_eq!(child.budget().get(), share);
                    let size = child.team(|t| t.size().get()).unwrap();
                    assert_eq!(size, share);
                })
                .unwrap();
        }
    }

    /// A panic in one child surfaces only after its siblings finished.
    fn panic_after_siblings(exec: Exec<'_>) {
        let siblings = Barrier::new(2);
        let done = AtomicUsize::new(0);
        let caught = catch_unwind(AssertUnwindSafe(|| {
            exec.fork_sized(&[1, 1, 1], |i, _| {
                if i == 1 {
                    panic!("child 1");
                }
                siblings.wait();
                done.fetch_add(1, Ordering::SeqCst);
            })
        }));
        let payload = caught.expect_err("the child's panic propagates");
        assert_eq!(payload.downcast_ref::<&str>(), Some(&"child 1"));
        assert_eq!(done.into_inner(), 2, "the siblings were joined first");
    }

    #[test]
    fn a_panic_in_one_child_propagates_after_the_others_are_joined() {
        let _w = watchdog();
        let plain = pool(3);
        panic_after_siblings(Exec::new(&plain, 3).unwrap());
        let team = pool(3).into_team(3).unwrap();
        panic_after_siblings(team.exec(3));
    }
}
