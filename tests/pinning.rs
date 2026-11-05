//! Participant placement (Linux only, on by default): every member pins itself
//! onto one CPU of the calling thread's set at a root team scope and restores
//! its own mask when its reservation ends; the leader (the calling thread) is
//! nudged onto the set's first CPU before every section dispatched under that
//! placement, inside the scope or, for a `Team`, through any later token, but
//! its mask is never narrowed; a set with fewer CPUs than participants places
//! nobody and counts the skip.
#![cfg(all(feature = "parallel", target_os = "linux"))]

use std::num::NonZeroUsize;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
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
        if rx.recv_timeout(Duration::from_secs(45)).is_err() {
            eprintln!("watchdog: test stalled");
            std::process::exit(101);
        }
    });
    Stop(tx)
}

/// The set CPUs of the calling thread's affinity mask, ascending.
fn own_cpus() -> Vec<usize> {
    // SAFETY: reads the calling thread's mask into a zeroed `cpu_set_t` of the
    // right size; pid 0 is this thread.
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

/// Restricts the calling thread to `cpus`; restores the previous mask on drop.
struct Restrict(Vec<usize>);

impl Restrict {
    /// `None` when the restriction is refused (the test then skips).
    fn to(cpus: &[usize]) -> Option<Restrict> {
        let before = own_cpus();
        set_own_cpus(cpus).then_some(Restrict(before))
    }
}

impl Drop for Restrict {
    fn drop(&mut self) {
        set_own_cpus(&self.0);
    }
}

fn builder(width: usize, pin: bool) -> threadteam::PoolBuilder {
    Pool::builder()
        .threads(width)
        .pin_participants(pin)
        .idle_spin(Duration::from_millis(1))
        .acquire_timeout(Duration::from_secs(10))
}

/// Worker index -> the CPUs it started with.
type Masks = Arc<Mutex<Vec<(usize, Vec<usize>)>>>;

/// The masks the workers had when they started.
#[derive(Clone, Default)]
struct Origin(Masks);

impl Origin {
    fn record(&self, worker: usize, cpus: Vec<usize>) {
        self.0.lock().unwrap().push((worker, cpus));
    }

    fn of(&self, worker: usize) -> Vec<usize> {
        let table = self.0.lock().unwrap();
        let row = table.iter().find(|(i, _)| *i == worker);
        assert!(row.is_some(), "no mask recorded for worker {worker}");
        row.unwrap().1.clone()
    }

    /// `mask` itself when some worker started with it, otherwise an empty mask
    /// (which then fails the comparison it is used in).
    fn of_any_with(&self, mask: &[usize]) -> Vec<usize> {
        if self.all().iter().any(|m| m == mask) {
            mask.to_vec()
        } else {
            Vec::new()
        }
    }

    fn all(&self) -> Vec<Vec<usize>> {
        self.0
            .lock()
            .unwrap()
            .iter()
            .map(|(_, m)| m.clone())
            .collect()
    }
}

/// A pool whose workers report their initial masks, and the table.
fn origin_pool(width: usize, pin: bool) -> (Pool, Origin) {
    let origin = Origin::default();
    let store = origin.clone();
    let pool = builder(width, pin)
        .on_start(move |worker| store.record(worker, own_cpus()))
        .build()
        .unwrap();
    (pool, origin)
}

/// Collects the mask each participant sees while it runs a share.
#[derive(Default)]
struct Seen {
    masks: Mutex<Vec<(usize, Vec<usize>)>>,
}

impl Seen {
    fn record(&self, index: usize) {
        self.masks.lock().unwrap().push((index, own_cpus()));
    }

    fn take(&self) -> Vec<(usize, Vec<usize>)> {
        let mut v = self.masks.lock().unwrap().clone();
        v.sort_by_key(|&(i, _)| i);
        v
    }
}

#[test]
fn a_root_team_scope_places_each_member_on_its_own_cpu_and_never_moves_the_leader() {
    let _w = watchdog();
    let set = own_cpus();
    if set.len() < 2 {
        eprintln!("skipped: the process has fewer than 2 allowed CPUs");
        return;
    }
    let width = set.len().min(4);
    let (pool, origin) = origin_pool(width, true);
    let before = own_cpus();
    let seen = Seen::default();
    let mut exec = Exec::new(&pool, width).unwrap();
    exec.team(|mut team| {
        assert_eq!(team.size().get(), width);
        team.reborrow().run_with(|p| seen.record(p.index()));
    })
    .expect("an idle pool");

    let masks = seen.take();
    assert_eq!(masks.len(), width, "every participant ran a share");
    // Participant 0 is the leader: it runs its share on the calling thread and
    // its mask is untouched, inside the scope included.
    assert_eq!(
        masks[0].1, before,
        "the leader was moved: {:?} instead of {before:?}",
        masks[0].1
    );
    for (index, mask) in &masks[1..] {
        // Member k sits alone on the k-th CPU of the caller's set.
        assert_eq!(
            mask,
            &vec![set[*index]],
            "participant {index} of {width} on the caller's set {set:?}"
        );
    }
    let distinct: std::collections::HashSet<_> = masks[1..].iter().map(|(_, m)| m[0]).collect();
    assert_eq!(distinct.len(), width - 1, "members are on distinct CPUs");
    // And the calling thread's mask is the same after the scope.
    assert_eq!(own_cpus(), before, "the leader's mask after the scope");

    // A member restores its own mask before it returns its thread to Rayon:
    // every worker of the pool shows its original mask again.
    pool.release_idle();
    Exec::new(&pool, width)
        .unwrap()
        .broadcast(|worker| {
            assert_eq!(
                own_cpus(),
                origin.of(worker),
                "worker {worker} kept a pinned mask after the scope"
            );
        })
        .expect("a root token from a plain thread");
}

#[test]
fn a_persistent_team_pins_every_root_scope_and_never_moves_the_leader() {
    let _w = watchdog();
    let set = own_cpus();
    if set.len() < 2 {
        eprintln!("skipped: the process has fewer than 2 allowed CPUs");
        return;
    }
    let (pool, origin) = origin_pool(set.len().min(4), true);
    let team = pool.into_team(set.len().min(4)).expect("an idle pool");
    let width = team.size().get();
    let seen = Seen::default();
    let before = own_cpus();
    for round in 0..2 {
        let mut exec = team.exec(width);
        // The faer `with_team` path: a root token of the team, used through
        // `team()`, twice on the same persistent reservation.
        exec.team(|mut team| {
            team.reborrow().run_with(|p| seen.record(p.index()));
        })
        .expect("a team token opens its scope");
        drop(exec);
        assert_eq!(
            own_cpus(),
            before,
            "round {round}: the leader's mask after the scope"
        );
    }
    let masks = seen.take();
    assert_eq!(masks.len(), 2 * width);
    let mut rounds = vec![0usize; width];
    for (index, mask) in &masks {
        let round = rounds[*index];
        rounds[*index] += 1;
        if *index == 0 {
            // Participant 0 is the leader, which is never pinned.
            assert_eq!(mask, &before, "round {round}: the leader was moved");
        } else {
            assert_eq!(
                mask,
                &vec![set[*index]],
                "round {round}, participant {index}"
            );
        }
    }
    assert!(
        rounds.iter().all(|&n| n == 2),
        "both rounds ran: {rounds:?}"
    );
    // The members restore what they had before the first scope when the
    // reservation ends.
    let pool = team.into_pool();
    Exec::new(&pool, width)
        .unwrap()
        .broadcast(|worker| {
            assert_eq!(
                own_cpus(),
                origin.of(worker),
                "worker {worker} after the team"
            );
        })
        .expect("a root token from a plain thread");
}

#[test]
fn pinning_is_on_by_default() {
    let pool = Pool::builder().threads(1).build().unwrap();
    assert!(pool.pin_participants(), "on by default");
    assert_eq!(pool.pin_skips(), 0);
}

#[test]
fn the_leader_is_placed_on_its_home_cpu_before_every_section_and_keeps_its_mask() {
    let _w = watchdog();
    let set = own_cpus();
    if set.len() < 3 {
        eprintln!("skipped: need at least 3 allowed CPUs");
        return;
    }
    let width = set.len().min(4);
    let pool = builder(width, true).build().unwrap();
    let team = pool.into_team(width).expect("an idle pool");
    let width = team.size().get();
    let before = own_cpus();
    let moves_before = team.pool().leader_moves().0;
    // Inside one root scope, drag the leader onto a member's CPU (S[1]) between
    // sections, as serial work in a long scope lets the scheduler do, then open
    // a section: the leader must run its share on S[0] with its whole mask.
    // Where the leader runs is the scheduler's to keep, so it is sampled over
    // rounds; the mask is asserted every time.
    let mut on_home = 0;
    let rounds = 20;
    team.exec(width)
        .team(|mut t| {
            for round in 0..rounds {
                assert!(set_own_cpus(&[set[1]]), "moving the leader onto S[1]");
                assert!(set_own_cpus(&before), "restoring the leader's mask");
                let seen = Mutex::new(None);
                t.reborrow().run_with(|p| {
                    if p.index() == 0 {
                        // SAFETY: no arguments; -1 on failure, never a CPU.
                        let cpu = unsafe { libc::sched_getcpu() };
                        *seen.lock().unwrap() = Some((usize::try_from(cpu).ok(), own_cpus()));
                    }
                });
                let (cpu, mask) = seen.into_inner().unwrap().unwrap();
                assert_eq!(
                    mask, before,
                    "round {round}: the leader's mask in its share"
                );
                if cpu == Some(set[0]) {
                    on_home += 1;
                }
            }
        })
        .expect("a team token opens its scope");
    assert_eq!(own_cpus(), before, "the leader's mask after the scope");
    assert!(
        on_home > rounds / 2,
        "the leader ran its share on S[0] in {on_home} of {rounds} sections"
    );
    let moves = team.pool().leader_moves().0 - moves_before;
    assert!(moves > 0, "no placement was counted");
    assert_eq!(team.pool().pin_skips(), 0);
}

/// Drags the calling thread onto `cpu` and gives it `mask` back, as serial
/// work between sections lets the scheduler do.
fn drag(cpu: usize, mask: &[usize]) {
    assert!(set_own_cpus(&[cpu]), "moving the caller onto CPU {cpu}");
    assert!(set_own_cpus(mask), "restoring the caller's mask");
}

/// Where the caller runs a `fork` on `exec`, and the mask it has there: its
/// own share when the token runs one on the caller (a team token), otherwise
/// right after the call (a plain token runs every task on the pool).
fn leader_share(exec: Exec<'_>) -> (Option<usize>, Vec<usize>) {
    let sample = || {
        // SAFETY: no arguments; -1 on failure, never a CPU.
        let cpu = unsafe { libc::sched_getcpu() };
        (usize::try_from(cpu).ok(), own_cpus())
    };
    let here = std::thread::current().id();
    let seen = Mutex::new(None);
    let tasks = exec.budget().get();
    exec.fork(tasks, |_, _| {
        if std::thread::current().id() == here {
            *seen.lock().unwrap() = Some(sample());
        }
    });
    seen.into_inner().unwrap().unwrap_or_else(sample)
}

#[test]
fn a_placed_teams_tokens_home_the_leader_outside_a_root_scope() {
    let _w = watchdog();
    let set = own_cpus();
    if set.len() < 3 {
        eprintln!("skipped: need at least 3 allowed CPUs");
        return;
    }
    let width = set.len().min(4);
    let pool = builder(width, true).build().unwrap();
    // `into_team` places the members in its acknowledgement scope and opens no
    // other: every section below goes through plain team tokens (#33).
    let team = pool.into_team(width).expect("an idle pool");
    let width = team.size().get();
    let before = own_cpus();
    let moves_before = team.pool().leader_moves().0;
    let rounds = 20;
    let mut on_home = 0;
    for round in 0..rounds {
        drag(set[1], &before);
        let (cpu, mask) = leader_share(team.exec(width));
        assert_eq!(
            mask, before,
            "round {round}: the leader's mask in its share"
        );
        if cpu == Some(set[0]) {
            on_home += 1;
        }
    }
    assert_eq!(own_cpus(), before, "the leader's mask afterwards");
    assert!(
        on_home > rounds / 2,
        "the leader ran its share on S[0] in {on_home} of {rounds} sections"
    );
    let moves = team.pool().leader_moves().0 - moves_before;
    assert!(moves > 0, "the moves outside a root scope were not counted");
    assert_eq!(team.pool().pin_skips(), 0);
}

#[test]
fn an_unrelated_token_of_a_placed_teams_pool_never_moves_its_caller() {
    let _w = watchdog();
    let set = own_cpus();
    if set.len() < 3 {
        eprintln!("skipped: need at least 3 allowed CPUs");
        return;
    }
    // A team of two on a pool of four: two workers stay free for other tokens.
    let pool = builder(4, true).build().unwrap();
    let team = pool.into_team(2).expect("an idle pool");
    let moves = team.pool().leader_moves().0;

    // A plain token of the same pool, from a caller whose set holds S[1] and
    // S[2] but not the team's home S[0].
    let cpus = vec![set[1], set[2]];
    let Some(restore) = Restrict::to(&cpus) else {
        eprintln!("skipped: cannot restrict the calling thread to {cpus:?}");
        return;
    };
    for round in 0..10 {
        let (cpu, mask) = leader_share(Exec::new(team.pool(), 2).unwrap());
        assert_eq!(mask, cpus, "round {round}: the caller's mask");
        assert_ne!(
            cpu,
            Some(set[0]),
            "round {round}: the caller was moved home"
        );
        // A team scope of that token with three participants on two CPUs is
        // not placed and has no home: its sections leave the caller alone too.
        let mut exec = Exec::new(team.pool(), 3).unwrap();
        exec.team(|mut t| {
            t.reborrow().run_with(|p| {
                if p.index() == 0 {
                    assert_eq!(own_cpus(), cpus, "round {round}: the scope's leader");
                }
            });
        })
        .expect("idle workers");
    }
    drop(restore);
    assert_eq!(
        team.pool().leader_moves().0,
        moves,
        "a caller of an unrelated token was moved"
    );

    // And the unrelated scopes did not take the team's home away: its own
    // tokens still place their leader.
    let before = own_cpus();
    let mut on_home = 0;
    for _ in 0..10 {
        drag(set[1], &before);
        if leader_share(team.exec(2)).0 == Some(set[0]) {
            on_home += 1;
        }
    }
    assert!(on_home > 5, "the team's leader was home in {on_home} of 10");
    assert!(team.pool().leader_moves().0 > moves);
}

#[test]
fn a_released_or_unpinned_team_no_longer_moves_its_caller() {
    let _w = watchdog();
    let set = own_cpus();
    if set.len() < 3 {
        eprintln!("skipped: need at least 3 allowed CPUs");
        return;
    }
    let width = set.len().min(3);
    let pool = builder(width, true).build().unwrap();
    let team = pool.into_team(width).expect("an idle pool");
    let width = team.size().get();
    // A root scope that cannot be placed (a one-CPU caller) unpins the members
    // and ends the home with them.
    let Some(restore) = Restrict::to(&set[..1]) else {
        eprintln!("skipped: cannot restrict the calling thread to one CPU");
        return;
    };
    team.exec(width)
        .team(|_| {})
        .expect("a team token opens its scope");
    drop(restore);
    assert_eq!(team.pool().pin_skips(), 1);
    let moves = team.pool().leader_moves().0;
    drag(set[2], &set);
    team.exec(width)
        .for_ranges(64, NonZeroUsize::new(16).unwrap(), |_| {});
    assert_eq!(
        team.pool().leader_moves().0,
        moves,
        "an unpinned team moved its caller"
    );

    // Placed again, then released: the pool keeps no home of the team's.
    team.exec(width)
        .team(|_| {})
        .expect("a team token opens its scope");
    let pool = team.into_pool();
    let moves = pool.leader_moves().0;
    for _ in 0..5 {
        drag(set[2], &set);
        let (_, mask) = leader_share(Exec::new(&pool, width).unwrap());
        assert_eq!(mask, set, "the caller's mask after the team");
    }
    assert_eq!(
        pool.leader_moves().0,
        moves,
        "a released team moved its caller"
    );
}

#[test]
fn pinning_off_leaves_every_mask_alone() {
    let _w = watchdog();
    let set = own_cpus();
    let width = set.len().min(3);
    let (pool, origin) = origin_pool(width, false);
    assert!(!pool.pin_participants(), "turned off for this pool");
    let before = own_cpus();
    let seen = Seen::default();
    let mut exec = Exec::new(&pool, width).unwrap();
    exec.team(|mut team| {
        team.reborrow().run_with(|p| seen.record(p.index()));
    })
    .expect("an idle pool");
    let masks = seen.take();
    assert_eq!(masks.len(), width);
    for (index, mask) in &masks {
        // Nothing moves: every thread keeps the mask it started with.
        assert_eq!(mask, &before, "participant {index} was moved");
    }
    assert_eq!(own_cpus(), before);
    assert!(
        origin.all().iter().all(|m| m == &before),
        "a worker started with a different mask than the caller"
    );
}

#[test]
fn a_lent_member_restores_its_mask_before_it_serves_rayon() {
    let _w = watchdog();
    let set = own_cpus();
    if set.len() < 2 {
        eprintln!("skipped: the process has fewer than 2 allowed CPUs");
        return;
    }
    let width = set.len().min(4);
    let (pool, origin) = origin_pool(width, true);
    let team = pool.into_team(width).expect("an idle pool");
    let width = team.size().get();
    let mut exec = team.exec(width);
    // Pin the members for the root scope; a persistent team keeps them pinned
    // until they are released or lent.
    exec.team(|_| {}).expect("a team token opens its scope");
    // `external` lends the members to Rayon: the closure runs on one of them
    // and the jobs it forks run on the others.
    let inside = AtomicUsize::new(0);
    let job_masks = Mutex::new(Vec::new());
    let lent_caller = exec.external(|width| {
        let here = own_cpus();
        Exec::new(team.pool(), width.get())
            .unwrap()
            .fork(width.get(), |_, _| {
                inside.fetch_add(1, Ordering::Relaxed);
                job_masks.lock().unwrap().push(own_cpus());
            });
        here
    });
    assert_eq!(
        lent_caller,
        own_cpus(),
        "the thread that ran the lent call kept a pin"
    );
    assert_eq!(inside.load(Ordering::Relaxed), width);
    for mask in job_masks.into_inner().unwrap() {
        // A member that had not restored would show one CPU, not the set.
        assert!(
            mask.len() > 1,
            "a lent job ran on a pinned thread: {mask:?}"
        );
        assert!(
            origin.all().contains(&mask),
            "a lent job ran with an unexpected mask {mask:?}"
        );
    }
}

#[test]
fn a_team_wider_than_the_callers_set_is_not_pinned_and_the_skip_is_counted() {
    let _w = watchdog();
    let all = own_cpus();
    if all.len() < 3 {
        eprintln!("skipped: need at least 3 allowed CPUs to restrict to 2");
        return;
    }
    let cpus = vec![all[0], all[1]];
    let Some(_restore) = Restrict::to(&cpus) else {
        eprintln!("skipped: cannot restrict the calling thread to {cpus:?}");
        return;
    };
    let width = 4;
    let (pool, origin) = origin_pool(width, true);
    let before = own_cpus();
    assert_eq!(before, cpus);
    let seen = Seen::default();
    let mut exec = Exec::new(&pool, width).unwrap();
    exec.team(|mut team| {
        assert_eq!(team.size().get(), width);
        team.reborrow().run_with(|p| seen.record(p.index()));
    })
    .expect("an idle pool");
    let masks = seen.take();
    assert_eq!(masks.len(), width);
    // Nobody is placed: no member wraps onto a CPU another one has, every
    // participant keeps the mask it had.
    assert_eq!(masks[0].1, before, "the leader was moved");
    for (index, mask) in &masks[1..] {
        assert_eq!(
            mask,
            &origin.of_any_with(mask),
            "participant {index} of {width} was placed on {mask:?}"
        );
        assert!(mask.len() > 1, "participant {index} was pinned to {mask:?}");
    }
    assert_eq!(pool.pin_skips(), 1, "the skip is counted");
    assert_eq!(own_cpus(), before, "the leader's mask after the scope");
}

#[test]
fn a_persistent_team_pinned_on_a_wide_set_unpins_when_a_scope_cannot_be_placed() {
    let _w = watchdog();
    let all = own_cpus();
    if all.len() < 4 {
        eprintln!("skipped: need at least 4 allowed CPUs");
        return;
    }
    let width = 4;
    let (pool, origin) = origin_pool(width, true);
    let team = pool.into_team(width).expect("an idle pool");
    // First scope from the whole set: the members are pinned.
    let pinned = Seen::default();
    team.exec(width)
        .team(|mut t| t.reborrow().run_with(|p| pinned.record(p.index())))
        .expect("a team token opens its scope");
    for (index, mask) in &pinned.take()[1..] {
        assert_eq!(
            mask,
            &vec![all[*index]],
            "participant {index} on the wide set"
        );
    }
    // Second scope from a two-CPU caller: nobody can be placed, so the members
    // pinned by the first scope restore their own masks for this one.
    let cpus = vec![all[0], all[1]];
    let Some(restore) = Restrict::to(&cpus) else {
        eprintln!("skipped: cannot restrict the calling thread to {cpus:?}");
        return;
    };
    let narrow = Seen::default();
    team.exec(width)
        .team(|mut t| t.reborrow().run_with(|p| narrow.record(p.index())))
        .expect("a team token opens its scope");
    drop(restore);
    for (index, mask) in &narrow.take()[1..] {
        assert!(mask.len() > 1, "participant {index} kept a pin: {mask:?}");
        assert!(
            origin.all().contains(mask),
            "participant {index} on {mask:?}"
        );
    }
    assert_eq!(team.pool().pin_skips(), 1);
}

#[test]
fn a_member_leading_a_fork_child_restores_its_mask_before_external() {
    let _w = watchdog();
    let set = own_cpus();
    // The member's child needs a budget of two to enter the pool, so the team
    // must fork into shares of two.
    if set.len() < 4 {
        eprintln!("skipped: a member-led child of width 2 needs 4 allowed CPUs");
        return;
    }
    let width = 4;
    let (pool, origin) = origin_pool(width, true);
    let team = pool.into_team(width).expect("an idle pool");
    let mut exec = team.exec(width);
    // Pin the members for the root scope; a persistent team keeps them pinned.
    exec.team(|_| {}).expect("a team token opens its scope");
    let pre_pin = own_cpus();
    let seen = Mutex::new(Vec::new());
    exec.fork(2, |child, child_token| {
        if child == 1 {
            // Child 1 is led by a member, which the scope pinned to one CPU.
            // `external` runs inline on that thread (it is a worker of the
            // pool), so it must have restored its own mask first.
            let mask = child_token.external(|_width| own_cpus());
            seen.lock().unwrap().push(mask);
        }
    });
    let seen = seen.into_inner().unwrap();
    assert_eq!(seen.len(), 1, "the member-led child ran `external`");
    assert_eq!(
        seen[0], pre_pin,
        "a pinned member kept its mask while serving foreign work"
    );
    assert!(
        origin.all().contains(&pre_pin),
        "the member restored what it had before the first pin"
    );
}

#[test]
fn a_panic_in_a_share_leaves_the_callers_affinity_alone() {
    use std::panic::{catch_unwind, AssertUnwindSafe};
    let _w = watchdog();
    let set = own_cpus();
    if set.len() < 2 {
        eprintln!("skipped: the process has fewer than 2 allowed CPUs");
        return;
    }
    let width = set.len().min(3);

    // A scoped acquisition: the leader is never pinned, so unwinding through
    // the scope cannot leave it on another mask either.
    let (pool, origin) = origin_pool(width, true);
    let before = own_cpus();
    let mut exec = Exec::new(&pool, width).unwrap();
    let panicked = catch_unwind(AssertUnwindSafe(|| {
        let _ = exec.team(|mut team| {
            team.reborrow().run(|index, _| {
                assert!(index < width);
                if index == 1 {
                    panic!("a member panics in its share");
                }
            });
        });
    }));
    assert!(panicked.is_err(), "the panic reached the leader");
    assert_eq!(own_cpus(), before, "the caller after an unwinding scope");
    pool.release_idle();
    Exec::new(&pool, width)
        .unwrap()
        .broadcast(|worker| {
            assert_eq!(
                own_cpus(),
                origin.of(worker),
                "worker {worker} after unwind"
            );
        })
        .expect("a root token from a plain thread");

    // A persistent team's root token: same, and the members restore what they
    // had before the first pin when the reservation ends.
    let (pool, origin) = origin_pool(width, true);
    let team = pool.into_team(width).expect("an idle pool");
    let before = own_cpus();
    let mut exec = team.exec(width);
    let panicked = catch_unwind(AssertUnwindSafe(|| {
        let _ = exec.team(|mut team| {
            team.reborrow().run(|index, _| {
                if index == 1 {
                    panic!("a member panics in its share");
                }
            });
        });
    }));
    assert!(panicked.is_err(), "the panic reached the leader");
    drop(exec);
    assert_eq!(
        own_cpus(),
        before,
        "the caller after an unwinding team scope"
    );
    let pool = team.into_pool();
    Exec::new(&pool, width)
        .unwrap()
        .broadcast(|worker| {
            assert_eq!(
                own_cpus(),
                origin.of(worker),
                "worker {worker} after the team"
            );
        })
        .expect("a root token from a plain thread");
}
