//! A root team scope that has more participants than CPUs in the caller's set
//! parks its members, and its nested sections keep working, with members that
//! sleep and wake between them.
//!
//! These are sleeping and waking smoke tests. The pool's spin budget is lifted
//! to `u32::MAX` polls (minutes), so a member parks within the hang guard only
//! through a short budget: the throttled one the oversubscribed scope chose, or
//! the pressure heuristic of a loaded host (`docs/design/waiting.md`). Which
//! budget a root scope chooses is checked directly in
//! `src/tests/placement.rs`.
#![cfg(all(feature = "trace", feature = "parallel", target_os = "linux"))]

use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Duration, Instant};
use threadteam::trace::{self, Kind};
use threadteam::Pool;

/// Only a hang guard.
const DEADLINE: Duration = Duration::from_secs(10);

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
        if rx.recv_timeout(Duration::from_secs(120)).is_err() {
            eprintln!("watchdog: test stalled");
            std::process::exit(101);
        }
    });
    Stop(tx)
}

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

/// Restores the calling thread's mask on drop.
struct Restrict(Vec<usize>);

impl Drop for Restrict {
    fn drop(&mut self) {
        set_own_cpus(&self.0);
    }
}

/// The members that ran a share, in the order of their first share. Their slot
/// ids are whatever the acquisition claimed, so they are read from the trace.
fn members_that_ran(events: &[trace::Event]) -> Vec<usize> {
    let mut out: Vec<usize> = Vec::new();
    for e in events {
        if e.kind == Kind::SectionEnd && !out.contains(&e.member) {
            out.push(e.member);
        }
    }
    out
}

/// Polls (hang guard only) until every member that ran a share has a
/// `ParkEnter` later than its own last `SectionEnd`.
///
/// No upper bound on how late. With the pool's budget lifted to `u32::MAX`
/// polls, parking within `DEADLINE` comes from a short budget (see the module
/// documentation).
fn wait_until_members_parked(team: &threadteam::Team) {
    let start = Instant::now();
    loop {
        let events = trace::events(team.pool());
        let members = members_that_ran(&events);
        let parked = !members.is_empty()
            && members.iter().all(|&member| {
                let last_end = events
                    .iter()
                    .filter(|e| e.member == member && e.kind == Kind::SectionEnd)
                    .map(|e| e.ns)
                    .max();
                let park = events
                    .iter()
                    .filter(|e| e.member == member && e.kind == Kind::ParkEnter)
                    .map(|e| e.ns)
                    .max();
                matches!((last_end, park), (Some(end), Some(park)) if park > end)
            });
        if parked {
            assert_eq!(
                members.len(),
                team.size().get() - 1,
                "every member ran a share"
            );
            return;
        }
        assert!(
            start.elapsed() < DEADLINE,
            "no member parked within {DEADLINE:?}: the scope kept the pool's lifted spin budget"
        );
        std::thread::sleep(Duration::from_millis(2));
    }
}

#[test]
fn an_oversubscribed_root_scope_parks_its_members_quickly() {
    let _w = watchdog();
    let all = own_cpus();
    if all.len() < 3 {
        eprintln!("skipped: need at least 3 allowed CPUs to restrict to 2");
        return;
    }
    let cpus = vec![all[0], all[1]];
    if !set_own_cpus(&cpus) {
        eprintln!("skipped: cannot restrict the calling thread to {cpus:?}");
        return;
    }
    let _restore = Restrict(all);

    let pool = Pool::builder()
        .threads(4)
        .spin_polls(u32::MAX)
        .acquire_timeout(Duration::from_secs(10))
        .build()
        .unwrap();
    let team = pool.into_team(4).expect("an idle pool");
    let width = team.size().get();
    assert!(width > cpus.len(), "4 participants on 2 CPUs");

    let sum = AtomicUsize::new(0);
    let exec = team.exec(width);
    exec.broadcast(|i| {
        sum.fetch_add(i + 1, Ordering::Relaxed);
    })
    .expect("a team token broadcasts");
    assert_eq!(sum.load(Ordering::Relaxed), (1..=width).sum::<usize>());
    // The root scope is oversubscribed, so its members park without waiting for
    // the 60 s park delay.
    wait_until_members_parked(&team);

    // Nested sections of the same root keep that knowledge: a split, then a
    // section in each child, and the members park again.
    let mut exec = team.exec(width);
    exec.team(|team| {
        team.split(&[2, 2], |_, child| child.run(|_, _| {}))
            .unwrap();
    })
    .expect("a team token opens its scope");
    wait_until_members_parked(&team);
}

#[test]
fn the_scope_that_enters_decides_the_park_delay_not_the_mint() {
    let _w = watchdog();
    let all = own_cpus();
    if all.len() < 2 {
        eprintln!("skipped: need at least 2 allowed CPUs to restrict to 1");
        return;
    }
    // Mint the root token of a persistent team on the full mask: three
    // participants on all CPUs are not oversubscribed, so the members start
    // with the policy's 60 s park delay.
    let pool = Pool::builder()
        .threads(3)
        .spin_polls(u32::MAX)
        .acquire_timeout(Duration::from_secs(10))
        .build()
        .unwrap();
    let team = pool.into_team(3).expect("an idle pool");
    let width = team.size().get();
    let mut exec = team.exec(width);
    // Enter the scope from a one-CPU mask: it is oversubscribed there, so its
    // members must park after the short spin, not after 60 s.
    let cpus = vec![all[0]];
    if !set_own_cpus(&cpus) {
        eprintln!("skipped: cannot restrict the calling thread to {cpus:?}");
        return;
    }
    let _restore = Restrict(all);
    let sum = AtomicUsize::new(0);
    exec.team(|mut team| {
        team.reborrow().run(|i, _| {
            sum.fetch_add(i + 1, Ordering::Relaxed);
        });
    })
    .expect("a team token opens its scope");
    assert_eq!(sum.load(Ordering::Relaxed), (1..=width).sum::<usize>());
    wait_until_members_parked(&team);
}
