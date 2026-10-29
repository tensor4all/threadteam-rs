//! Reuse of the private Rayon pool between pools of the same width (the
//! comparison experiment against own worker threads, tensor4all/faer#107).
#![cfg(feature = "parallel")]

use std::collections::HashSet;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;
use std::thread::ThreadId;
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

/// The cache and the spawn counter are process wide: the tests of this file
/// take turns.
static SERIAL: Mutex<()> = Mutex::new(());

fn worker_ids(pool: &Pool) -> HashSet<ThreadId> {
    let ids = Mutex::new(HashSet::new());
    Exec::new(pool, pool.width().get())
        .unwrap()
        .broadcast(|_| {
            ids.lock().unwrap().insert(std::thread::current().id());
        })
        .unwrap();
    ids.into_inner().unwrap()
}

#[test]
fn a_pool_built_after_another_was_dropped_reuses_its_workers() {
    let _w = watchdog();
    let _turn = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    let width = 4;
    let build = || {
        Pool::builder()
            .threads(width)
            .acquire_timeout(Duration::from_secs(5))
            .build()
            .unwrap()
    };
    let first = build();
    Exec::new(&first, width).unwrap().fork(width, |_, _| {});
    let ids = worker_ids(&first);
    drop(first);
    for round in 0..5 {
        let before = threadteam::spawned_workers();
        let pool = build();
        let n = AtomicUsize::new(0);
        Exec::new(&pool, width).unwrap().fork(width, |_, _| {
            n.fetch_add(1, Ordering::SeqCst);
        });
        assert_eq!(n.load(Ordering::SeqCst), width);
        assert_eq!(worker_ids(&pool), ids, "round {round}: other threads");
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
fn two_live_pools_of_one_width_never_share_workers() {
    let _w = watchdog();
    let _turn = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    drop(Pool::builder().threads(2).build().unwrap());
    let a = Pool::builder().threads(2).build().unwrap();
    let b = Pool::builder().threads(2).build().unwrap();
    let (ia, ib) = (worker_ids(&a), worker_ids(&b));
    assert_eq!(ia.len(), 2);
    assert_eq!(ib.len(), 2);
    assert!(ia.is_disjoint(&ib), "{ia:?} {ib:?}");
}

#[test]
fn a_pool_with_hooks_gets_dedicated_workers_and_joins_them() {
    let _w = watchdog();
    let _turn = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    let started = std::sync::Arc::new(AtomicUsize::new(0));
    for _ in 0..2 {
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
        assert!(names
            .into_inner()
            .unwrap()
            .iter()
            .all(|n| n.as_deref().is_some_and(|n| n.starts_with("dedicated-"))));
    }
    // The drop joined the workers: their hooks ran once each, never again.
    assert_eq!(started.load(Ordering::SeqCst), 4);
}

/// A reused worker takes the CPU set of the caller that builds the pool, as a
/// thread that caller spawned would, not the set of whoever spawned it first.
#[cfg(target_os = "linux")]
#[test]
fn a_reused_worker_takes_the_building_callers_cpu_set() {
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
        let width = 3;
        assert!(set(&all[..2]));
        let first = Pool::builder().threads(width).build().unwrap();
        let ids = worker_ids(&first);
        drop(first);
        assert!(set(&all[2..4]));
        let pool = Pool::builder().threads(width).build().unwrap();
        assert_eq!(worker_ids(&pool), ids, "the workers were not reused");
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
