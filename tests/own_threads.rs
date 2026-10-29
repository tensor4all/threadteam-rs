//! The native workers (`docs/design/own-threads.md`): reuse through the
//! process-wide reserve, cross-pool waits, broadcast order, affinity on lease,
//! and panics.
#![cfg(feature = "parallel")]

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;
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

/// The reserve and the spawn counter are process wide: the tests of this file
/// take turns.
static SERIAL: Mutex<()> = Mutex::new(());

#[test]
fn a_pool_built_after_another_was_dropped_reuses_its_workers() {
    let _w = watchdog();
    let _turn = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    let width = 4;
    // A long acquisition timeout: a worker clears its busy bit just after its
    // last task completed, and on a loaded host that can take a while.
    let build = || {
        Pool::builder()
            .threads(width)
            .acquire_timeout(Duration::from_secs(5))
            .build()
            .unwrap()
    };
    let first = build();
    Exec::new(&first, width).unwrap().fork(width, |_, _| {});
    drop(first);
    for round in 0..5 {
        let before = threadteam::spawned_workers();
        let pool = build();
        let n = AtomicUsize::new(0);
        Exec::new(&pool, width).unwrap().fork(width, |_, _| {
            n.fetch_add(1, Ordering::SeqCst);
        });
        assert_eq!(n.load(Ordering::SeqCst), width);
        // A team on the reused workers, as faer's `with_team` takes one.
        let team = pool.into_team(width).expect("idle reused workers");
        assert_eq!(team.size().get(), width);
        drop(team);
        assert_eq!(
            threadteam::spawned_workers() - before,
            0,
            "round {round}: a pool of the same width spawned threads"
        );
    }
}

#[test]
fn a_pool_with_hooks_gets_dedicated_workers_and_joins_them() {
    let _w = watchdog();
    let _turn = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    let started = std::sync::Arc::new(AtomicUsize::new(0));
    let s = std::sync::Arc::clone(&started);
    let before = threadteam::spawned_workers();
    let pool = Pool::builder()
        .threads(2)
        .thread_name(|i| format!("dedicated-{i}"))
        .on_start(move |_| {
            s.fetch_add(1, Ordering::SeqCst);
        })
        .build()
        .unwrap();
    let names = Mutex::new(Vec::new());
    Exec::new(&pool, 2).unwrap().fork(2, |_, _| {
        names
            .lock()
            .unwrap()
            .push(std::thread::current().name().map(String::from));
    });
    drop(pool);
    assert_eq!(threadteam::spawned_workers() - before, 2);
    assert_eq!(
        started.load(Ordering::SeqCst),
        2,
        "the hook ran on each worker"
    );
    assert!(names
        .into_inner()
        .unwrap()
        .iter()
        .all(|n| n.as_deref().is_some_and(|n| n.starts_with("dedicated-"))));
}

/// A worker of pool A waits for pool B, whose task waits for pool A again.
/// Every waiting worker serves its own pool meanwhile, so the chain completes
/// even when every worker of A is waiting on B.
#[test]
fn a_cross_pool_chain_a_b_a_completes() {
    let _w = watchdog();
    let _turn = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    let a = Pool::builder().threads(2).build().unwrap();
    let b = Pool::builder().threads(2).build().unwrap();
    let inner = AtomicUsize::new(0);
    Exec::new(&a, 2).unwrap().fork(2, |_, _| {
        Exec::new(&b, 2).unwrap().fork(2, |_, _| {
            Exec::new(&a, 2).unwrap().fork(2, |_, _| {
                inner.fetch_add(1, Ordering::SeqCst);
            });
        });
    });
    assert_eq!(inner.load(Ordering::SeqCst), 2 * 2 * 2);
}

/// Two plain threads broadcast on one pool at once: every worker runs both
/// broadcasts, and all workers see them in the same order.
#[test]
fn concurrent_broadcasts_reach_every_worker_in_the_same_order() {
    let _w = watchdog();
    let _turn = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    let width = 3;
    let pool = Pool::builder().threads(width).build().unwrap();
    for _ in 0..20 {
        let seen: Vec<Mutex<Vec<u8>>> = (0..width).map(|_| Mutex::new(Vec::new())).collect();
        std::thread::scope(|s| {
            for tag in [1u8, 2] {
                let (pool, seen) = (&pool, &seen);
                s.spawn(move || {
                    Exec::new(pool, width)
                        .unwrap()
                        .broadcast(|i| seen[i].lock().unwrap().push(tag))
                        .unwrap();
                });
            }
        });
        let orders: Vec<Vec<u8>> = seen.into_iter().map(|m| m.into_inner().unwrap()).collect();
        assert!(orders.iter().all(|o| o.len() == 2), "{orders:?}");
        assert!(orders.windows(2).all(|w| w[0] == w[1]), "{orders:?}");
    }
}

/// A task panics: the others still run, the panic reaches the caller after
/// all of them finished, and the pool keeps working.
#[test]
fn a_task_panic_propagates_after_the_others_and_the_pool_survives() {
    let _w = watchdog();
    let _turn = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    let pool = Pool::builder().threads(3).build().unwrap();
    let done = AtomicUsize::new(0);
    let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        Exec::new(&pool, 3).unwrap().fork(3, |i, _| {
            if i == 1 {
                panic!("boom");
            }
            std::thread::sleep(Duration::from_millis(5));
            done.fetch_add(1, Ordering::SeqCst);
        });
    }));
    assert!(r.is_err());
    assert_eq!(done.load(Ordering::SeqCst), 2);
    let n = AtomicUsize::new(0);
    Exec::new(&pool, 3).unwrap().fork(3, |_, _| {
        n.fetch_add(1, Ordering::SeqCst);
    });
    assert_eq!(n.load(Ordering::SeqCst), 3);
}

/// A reused worker takes the CPU set of the caller that leases it, as a thread
/// that caller spawned would, not the set of whoever spawned it first.
#[cfg(target_os = "linux")]
#[test]
fn a_reused_worker_takes_the_leasing_callers_cpu_set() {
    let _w = watchdog();
    let _turn = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    fn cpus() -> Vec<usize> {
        unsafe {
            let mut set: libc::cpu_set_t = std::mem::zeroed();
            libc::sched_getaffinity(0, std::mem::size_of::<libc::cpu_set_t>(), &mut set);
            (0..libc::CPU_SETSIZE as usize)
                .filter(|&c| libc::CPU_ISSET(c, &set))
                .collect()
        }
    }
    fn set(cpus: &[usize]) -> bool {
        unsafe {
            let mut s: libc::cpu_set_t = std::mem::zeroed();
            libc::CPU_ZERO(&mut s);
            for &c in cpus {
                libc::CPU_SET(c, &mut s);
            }
            libc::sched_setaffinity(0, std::mem::size_of::<libc::cpu_set_t>(), &s) == 0
        }
    }
    let all = cpus();
    if all.len() < 4 {
        eprintln!("skipped: needs 4 allowed CPUs");
        return;
    }
    // Run on a helper thread so the test thread's mask is never changed.
    std::thread::spawn(move || {
        let width = 2;
        assert!(set(&all[..2]));
        drop(Pool::builder().threads(width).build().unwrap());
        assert!(set(&all[2..4]));
        let pool = Pool::builder().threads(width).build().unwrap();
        let masks = Mutex::new(Vec::new());
        Exec::new(&pool, width)
            .unwrap()
            .broadcast(|_| masks.lock().unwrap().push(cpus()))
            .unwrap();
        for m in masks.into_inner().unwrap() {
            assert_eq!(
                m,
                all[2..4].to_vec(),
                "a reused worker kept another caller's CPU set"
            );
        }
    })
    .join()
    .unwrap();
}

/// A dedicated worker keeps the CPU set its `on_start` hook pinned it to: the
/// lease does not reset it to the caller's set.
#[cfg(target_os = "linux")]
#[test]
fn a_dedicated_worker_keeps_the_pin_of_its_start_hook() {
    let _w = watchdog();
    let _turn = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    fn cpus() -> Vec<usize> {
        unsafe {
            let mut set: libc::cpu_set_t = std::mem::zeroed();
            libc::sched_getaffinity(0, std::mem::size_of::<libc::cpu_set_t>(), &mut set);
            (0..libc::CPU_SETSIZE as usize)
                .filter(|&c| libc::CPU_ISSET(c, &set))
                .collect()
        }
    }
    let all = cpus();
    if all.len() < 2 {
        eprintln!("skipped: needs 2 allowed CPUs");
        return;
    }
    let target = all[1];
    let pool = Pool::builder()
        .threads(2)
        .on_start(move |_| unsafe {
            let mut s: libc::cpu_set_t = std::mem::zeroed();
            libc::CPU_ZERO(&mut s);
            libc::CPU_SET(target, &mut s);
            libc::sched_setaffinity(0, std::mem::size_of::<libc::cpu_set_t>(), &s);
        })
        .build()
        .unwrap();
    let masks = Mutex::new(Vec::new());
    Exec::new(&pool, 2)
        .unwrap()
        .broadcast(|_| masks.lock().unwrap().push(cpus()))
        .unwrap();
    for m in masks.into_inner().unwrap() {
        assert_eq!(m, vec![target], "the lease undid the hook's pin");
    }
}

/// Broadcast X's share 0 waits on another pool while X's share 1 waits at a
/// barrier of X's shares; meanwhile broadcast Y, whose shares meet at a barrier
/// too, is published. The waiting share 0 must not start Y's share on top of
/// X's: Y's barrier would wait for worker 1, which waits for X.
#[test]
fn a_waiting_broadcast_share_does_not_start_a_later_broadcast() {
    let _w = watchdog();
    let _turn = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    let a = Pool::builder().threads(2).build().unwrap();
    let b = Pool::builder().threads(2).build().unwrap();
    for _ in 0..20 {
        let x_bar = std::sync::Barrier::new(2);
        let y_bar = std::sync::Barrier::new(2);
        let x_waiting = std::sync::atomic::AtomicBool::new(false);
        std::thread::scope(|s| {
            s.spawn(|| {
                Exec::new(&a, 2)
                    .unwrap()
                    .broadcast(|i| {
                        if i == 0 {
                            x_waiting.store(true, Ordering::SeqCst);
                            Exec::new(&b, 2).unwrap().fork(2, |_, _| {
                                std::thread::sleep(Duration::from_millis(20));
                            });
                        }
                        x_bar.wait();
                    })
                    .unwrap();
            });
            s.spawn(|| {
                while !x_waiting.load(Ordering::SeqCst) {
                    std::thread::yield_now();
                }
                Exec::new(&a, 2)
                    .unwrap()
                    .broadcast(|_| {
                        y_bar.wait();
                    })
                    .unwrap();
            });
        });
    }
}
