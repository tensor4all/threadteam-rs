//! Root-scope flags that the placement work carries: whether a root team scope
//! is oversubscribed reaches reborrows, `split`/`fork` children, team tokens and
//! every section of the root.

use crate::awake::{Geometry, Section};
use crate::{Exec, Pool};
use std::time::Duration;

fn own_cpus() -> Vec<usize> {
    // SAFETY: reads the calling thread's mask into a zeroed `cpu_set_t`.
    unsafe {
        let mut set: libc::cpu_set_t = std::mem::zeroed();
        let r = libc::sched_getaffinity(0, std::mem::size_of::<libc::cpu_set_t>(), &mut set);
        assert_eq!(r, 0, "sched_getaffinity failed");
        (0..libc::CPU_SETSIZE as usize)
            .filter(|&cpu| libc::CPU_ISSET(cpu, &set))
            .collect()
    }
}

fn set_own_cpus(cpus: &[usize]) -> bool {
    // SAFETY: builds a `cpu_set_t` and applies it to pid 0, i.e. this thread.
    unsafe {
        let mut set: libc::cpu_set_t = std::mem::zeroed();
        libc::CPU_ZERO(&mut set);
        for &cpu in cpus {
            libc::CPU_SET(cpu, &mut set);
        }
        libc::sched_setaffinity(0, std::mem::size_of::<libc::cpu_set_t>(), &set) == 0
    }
}

fn pool(width: usize) -> Pool {
    Pool::builder()
        .threads(width)
        .idle_spin(Duration::ZERO)
        .acquire_timeout(Duration::from_secs(10))
        .build()
        .unwrap()
}

/// Runs `f` with the calling thread restricted to two CPUs.
fn on_two_cpus(f: impl FnOnce()) {
    let all = own_cpus();
    if all.len() < 3 {
        eprintln!("skipped: need at least 3 allowed CPUs to restrict to 2");
        return;
    }
    if !set_own_cpus(&all[..2]) {
        eprintln!("skipped: cannot restrict the calling thread to two CPUs");
        return;
    }
    f();
    assert!(set_own_cpus(&all), "restoring the calling thread's mask");
}

#[test]
fn a_section_carries_the_oversubscription_flag_it_was_built_with() {
    unsafe fn call(_: *const (), _: &Section, _: usize) {}
    unsafe fn slot_of(_: *const (), pos: usize) -> usize {
        pos
    }
    let geometry = |over| Geometry {
        dispatched: 0,
        size: 1,
        over,
        pin: false,
    };
    let quiet = Section::new(
        call,
        std::ptr::null(),
        slot_of,
        std::ptr::null(),
        geometry(false),
    );
    assert!(!quiet.over);
    let crowded = Section::new(
        call,
        std::ptr::null(),
        slot_of,
        std::ptr::null(),
        geometry(true),
    );
    assert!(crowded.over);
}

#[test]
fn an_oversubscribed_root_keeps_the_flag_through_reborrows_splits_and_fork_children() {
    on_two_cpus(|| {
        let width = 4;
        let pool = pool(width);

        // A scoped acquisition whose participants outnumber the caller's CPUs.
        let mut exec = Exec::new(&pool, width).unwrap();
        exec.team(|mut team| {
            assert!(team.over, "4 participants on 2 CPUs is oversubscribed");
            let over = team.reborrow().over; // reborrows inherit it
            assert!(over);
            team.reborrow()
                .split(&[2, 2], |_, child| {
                    assert!(child.over, "a split child of an oversubscribed root");
                    child.run(|_, _| {});
                })
                .unwrap();
            team.fork(2, |_, child| {
                assert!(child.over, "a fork child of an oversubscribed root");
            });
        })
        .expect("an idle pool");
        drop(exec);

        // A persistent team fixes the flag at its root token and carries it
        // into the carried team and into `fork` children of the token.
        let team = pool.into_team(width).expect("an idle pool");
        let mut exec = team.exec(width);
        assert!(exec.team(|t| t.over).expect("a team token opens its scope"));
        exec.reborrow().fork(2, |_, mut child| {
            assert!(
                child
                    .team(|t| t.over)
                    .expect("a fork child carries the sub-team"),
                "a fork child of an oversubscribed team token"
            );
        });
    });
}

#[test]
fn a_root_scope_that_fits_the_callers_cpus_is_not_oversubscribed() {
    let all = own_cpus();
    let width = all.len().min(3);
    let pool = pool(width);
    let mut exec = Exec::new(&pool, width).unwrap();
    exec.team(|team| {
        assert!(
            !team.over,
            "{width} participants on {} CPUs is not oversubscribed",
            all.len()
        );
    })
    .expect("an idle pool");
}

#[test]
fn a_team_scope_takes_the_oversubscription_from_the_thread_that_enters_it() {
    let all = own_cpus();
    if all.len() < 4 {
        eprintln!("skipped: need at least 4 allowed CPUs for a one-CPU scope mask");
        return;
    }
    let width = 4;
    let pool = pool(width);
    let team = pool.into_team(width).expect("an idle pool");
    // Mint the root token on the full mask: four participants on four CPUs is
    // not oversubscribed at mint.
    let mut exec = team.exec(width);
    // Enter the scope from a thread whose mask holds one CPU (the token could
    // just as well have travelled to another thread): the scope decides from
    // that set, exactly as its pin section does.
    if !set_own_cpus(&all[..1]) {
        eprintln!("skipped: cannot restrict the calling thread to one CPU");
        return;
    }
    let scoped = exec
        .team(|mut team| {
            let root = team.over;
            let child = std::sync::atomic::AtomicBool::new(false);
            team.reborrow().fork(2, |_, child_team| {
                child.store(child_team.over, std::sync::atomic::Ordering::Relaxed);
            });
            (root, child.load(std::sync::atomic::Ordering::Relaxed))
        })
        .expect("a team token opens its scope");
    assert!(set_own_cpus(&all), "restoring the calling thread's mask");
    assert!(
        scoped.0,
        "four participants on a one-CPU scope mask is oversubscribed"
    );
    assert!(scoped.1, "a fork child inherits the root's pressure");
}
