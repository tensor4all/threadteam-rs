//! Participant placement (Linux only, on by default): every member pins itself
//! onto one CPU of the calling thread's set at a root team scope and restores
//! its own mask when its reservation ends; the leader (the calling thread) is
//! pinned to the set's first CPU for the scope, and for each section a `Team`
//! token dispatches without a scope, widened to the set only inside
//! `TeamExec::external`, and has its own mask back when the scope or the
//! section ends; a set with fewer CPUs than participants places nobody and
//! counts the skip.
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
fn a_root_team_scope_places_each_member_on_its_own_cpu_and_the_leader_on_the_first() {
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
    // Participant 0 is the leader: it runs its share on the calling thread,
    // pinned to S[0] for the scope.
    assert_eq!(
        masks[0].1,
        vec![set[0]],
        "the leader's mask in its share: {:?}",
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
fn a_persistent_team_pins_every_root_scope_and_restores_the_leader_after_each() {
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
            // Participant 0 is the leader, pinned to S[0] for each scope.
            assert_eq!(
                mask,
                &vec![set[0]],
                "round {round}: the leader's mask in its share"
            );
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
fn the_leader_is_pinned_to_its_home_cpu_for_the_whole_scope() {
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
    let home = vec![set[0]];
    // Outside the scope the leader has its whole mask.
    assert_eq!(own_cpus(), before);
    team.exec(width)
        .team(|mut t| {
            // From the first instruction of the scope, between sections, in
            // every share and after every section, the mask is {S[0]}: one
            // pin at entry, no per-section check.
            assert_eq!(own_cpus(), home, "at scope entry");
            for round in 0..20 {
                let seen = Mutex::new(None);
                t.reborrow().run_with(|p| {
                    if p.index() == 0 {
                        *seen.lock().unwrap() = Some(own_cpus());
                    }
                });
                assert_eq!(
                    seen.into_inner().unwrap().unwrap(),
                    home,
                    "round {round}: the leader's mask in its share"
                );
                assert_eq!(own_cpus(), home, "round {round}: between sections");
            }
        })
        .expect("a team token opens its scope");
    assert_eq!(own_cpus(), before, "the leader's mask after the scope");
    assert_eq!(team.pool().pin_skips(), 0);
}

#[test]
fn team_external_widens_the_leader_for_the_call_and_restores_the_pin() {
    use std::panic::{catch_unwind, AssertUnwindSafe};
    let _w = watchdog();
    let set = own_cpus();
    if set.len() < 3 {
        eprintln!("skipped: need at least 3 allowed CPUs");
        return;
    }
    let width = set.len().min(3);
    let pool = builder(width, true).build().unwrap();
    let before = own_cpus();
    let team = pool.into_team(width).expect("an idle pool");
    let width = team.size().get();
    let home = vec![set[0]];
    let mut exec = team.exec(width);
    exec.team(|mut t| {
        assert_eq!(own_cpus(), home);
        // Every call widens and restores: nothing remembers "already wide".
        for round in 0..3 {
            let inside = t.external(own_cpus);
            assert_eq!(inside, before, "round {round}: mask inside the call");
            assert_eq!(own_cpus(), home, "round {round}: mask after the call");
        }
        // The restore is not skipped when the call unwinds.
        let r = catch_unwind(AssertUnwindSafe(|| {
            t.external(|| {
                assert_eq!(own_cpus(), before);
                panic!("a vendor call panics");
            })
        }));
        assert!(r.is_err());
        assert_eq!(own_cpus(), home, "mask after an unwinding call");
        // The next section still runs with the leader on S[0].
        let seen = Mutex::new(None);
        t.reborrow().run_with(|p| {
            if p.index() == 0 {
                *seen.lock().unwrap() = Some(own_cpus());
            }
        });
        assert_eq!(seen.into_inner().unwrap().unwrap(), home);
    })
    .expect("a team token opens its scope");
    drop(exec);
    assert_eq!(own_cpus(), before, "the token's end leaves the mask alone");
    drop(team);
    assert_eq!(own_cpus(), before, "the leader's mask after the team");
}

#[test]
fn team_external_is_a_plain_call_where_nothing_was_placed() {
    let _w = watchdog();
    let set = own_cpus();
    let width = set.len().min(3);
    // Pinning off: the leader never changes mask, so neither does the call.
    let pool = builder(width, false).build().unwrap();
    let before = own_cpus();
    Exec::new(&pool, width)
        .unwrap()
        .team(|mut t| {
            assert_eq!(own_cpus(), before);
            assert_eq!(t.external(own_cpus), before);
            assert_eq!(own_cpus(), before);
        })
        .expect("an idle pool");
    // A serial token has no scope to be placed in.
    Exec::serial()
        .team(|mut t| assert_eq!(t.external(own_cpus), before))
        .unwrap();
    assert_eq!(own_cpus(), before);
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

/// The thread that creates a `Team` keeps its own CPU set (#52): every token it
/// mints dispatches with the leader on `{S[0]}` without opening a team scope
/// (`fork_sized`, `for_ranges`, any budget), but only for the section, so
/// `into_team`, the time between tokens and the team's drop all leave the
/// caller's mask as they found it.
#[test]
fn a_team_places_its_leader_for_the_sections_it_dispatches_and_no_longer() {
    let _w = watchdog();
    let set = own_cpus();
    if set.len() < 3 {
        eprintln!("skipped: need at least 3 allowed CPUs");
        return;
    }
    let width = set.len().min(4);
    let pool = builder(width, true).build().unwrap();
    let before = own_cpus();
    let team = pool.into_team(width).expect("an idle pool");
    let width = team.size().get();
    let home = vec![set[0]];
    assert_eq!(own_cpus(), before, "the leader after into_team");
    for round in 0..10 {
        let budget = if round % 2 == 0 { width } else { 2 };
        let mut exec = team.exec(budget);
        for call in 0..3 {
            let (cpu, mask) = leader_share(exec.reborrow());
            assert_eq!(mask, home, "round {round}, call {call}: the leader's share");
            assert_eq!(cpu, Some(set[0]), "round {round}, call {call}: its CPU");
            assert_eq!(own_cpus(), before, "round {round}, call {call}: after it");
        }
        exec.for_ranges(64, NonZeroUsize::new(16).unwrap(), |_| {});
        assert_eq!(own_cpus(), before, "round {round}: between tokens");
    }
    assert_eq!(team.pool().pin_skips(), 0);
    drop(team);
    assert_eq!(own_cpus(), before, "the caller after the team");
}

/// `into_pool` changes no mask either.
#[test]
fn into_pool_gives_the_leader_its_mask_back() {
    let _w = watchdog();
    let set = own_cpus();
    if set.len() < 3 {
        eprintln!("skipped: need at least 3 allowed CPUs");
        return;
    }
    let pool = builder(3, true).build().unwrap();
    let team = pool.into_team(3).expect("an idle pool");
    assert_eq!(own_cpus(), set, "the caller after into_team");
    let _pool = team.into_pool();
    assert_eq!(own_cpus(), set);
}

/// #52: `into_team` gives the calling thread its own CPU set back, also while
/// the team lives and after a section it dispatched. The engine-construction
/// path (`CpuEngine::new`, `pool.into_team(width)`) runs on a caller thread
/// that is not ours, and its mask decides what a vendor thread pool built there
/// gets.
#[test]
fn into_team_leaves_the_callers_cpu_set_as_it_was() {
    let _w = watchdog();
    let set = own_cpus();
    if set.len() < 2 {
        eprintln!("skipped: the process has fewer than 2 allowed CPUs");
        return;
    }
    let width = set.len().min(4);
    let pool = builder(width, true).build().unwrap();
    let before = own_cpus();
    let team = pool.into_team(width).expect("an idle pool");
    let width = team.size().get();
    assert_eq!(own_cpus(), before, "the caller's set after into_team");
    // A section it dispatches narrows the leader for that section only.
    let shares = flat_shares(team.exec(width));
    assert_eq!(
        shares[0].1,
        vec![set[0]],
        "the leader's mask in the section"
    );
    assert_eq!(own_cpus(), before, "the caller's set after the section");
    assert_eq!(team.pool().pin_skips(), 0);
    drop(team);
    assert_eq!(own_cpus(), before, "the caller's set after the team");
}

/// #52: a thread started while a team lives inherits the caller's whole set,
/// not the one CPU the leader holds during a section. A vendor library that
/// builds its thread pool from the engine's thread is this shape.
#[test]
fn a_thread_spawned_while_a_team_lives_is_not_confined_to_one_cpu() {
    let _w = watchdog();
    let set = own_cpus();
    if set.len() < 2 {
        eprintln!("skipped: the process has fewer than 2 allowed CPUs");
        return;
    }
    let width = set.len().min(4);
    let pool = builder(width, true).build().unwrap();
    let before = own_cpus();
    let team = pool.into_team(width).expect("an idle pool");
    let width = team.size().get();
    let spawned = std::thread::scope(|s| s.spawn(own_cpus).join().unwrap());
    assert_eq!(spawned, before, "a thread started after into_team");
    // The same after a section of the team: the leader has its set back.
    flat_shares(team.exec(width));
    let spawned = std::thread::scope(|s| s.spawn(own_cpus).join().unwrap());
    assert_eq!(spawned, before, "a thread started after a section");
}

/// A root scope opened on the creating thread places the members in the
/// caller's `S`, widens the leader in `external` for the call, leaves the
/// leader on `{S[0]}` for the scope and gives it `S` back at the end of it.
#[test]
fn a_scope_on_the_creating_thread_places_the_members_in_the_teams_set() {
    let _w = watchdog();
    let set = own_cpus();
    if set.len() < 3 {
        eprintln!("skipped: need at least 3 allowed CPUs");
        return;
    }
    let width = set.len().min(4);
    let pool = builder(width, true).build().unwrap();
    let before = own_cpus();
    let team = pool.into_team(width).expect("an idle pool");
    let width = team.size().get();
    let home = vec![set[0]];
    let seen = Seen::default();
    for _ in 0..2 {
        team.exec(width)
            .team(|mut t| {
                assert_eq!(own_cpus(), home, "the leader at scope entry");
                t.reborrow().run_with(|p| seen.record(p.index()));
                t.external(|| assert_eq!(own_cpus(), before, "inside external"));
                assert_eq!(own_cpus(), home, "after external");
            })
            .expect("a team token opens its scope");
        assert_eq!(own_cpus(), before, "after the scope");
    }
    for (index, mask) in seen.take() {
        assert_eq!(mask, vec![set[index]], "participant {index}");
    }
    assert_eq!(team.pool().pin_skips(), 0);
}

/// A team is `Send` and `Sync`. A token used on another thread places that
/// thread for its sections, from the set the team recorded, and leaves its mask
/// as it found it; the creating thread is not touched.
#[test]
fn a_token_used_on_another_thread_places_only_that_thread() {
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
    let home = vec![set[0]];
    std::thread::scope(|s| {
        s.spawn(|| {
            let mine = own_cpus();
            let (_, mask) = leader_share(team.exec(width));
            assert_eq!(mask, home, "the other thread leads on the team's S[0]");
            assert_eq!(own_cpus(), mine, "nor moved by the drop");
        });
    });
    assert_eq!(own_cpus(), set, "the creating thread keeps its own mask");
}

/// A team token reborrowed before it opens its scope is still the team: the
/// leader is placed for that scope even though a reborrow carries no leader
/// guard of its own (the shape faer's `with_team` uses).
#[test]
fn a_reborrowed_team_token_places_its_leader_for_the_scope() {
    let _w = watchdog();
    let set = own_cpus();
    if set.len() < 3 {
        eprintln!("skipped: need at least 3 allowed CPUs");
        return;
    }
    let width = set.len().min(3);
    let pool = builder(width, true).build().unwrap();
    let before = own_cpus();
    let team = pool.into_team(width).expect("an idle pool");
    let width = team.size().get();
    let home = vec![set[0]];
    let seen = Mutex::new(None);
    let mut exec = team.exec(width);
    exec.reborrow()
        .team(|mut t| {
            assert_eq!(own_cpus(), home, "the leader at scope entry");
            t.reborrow().run_with(|p| {
                if p.index() == 0 {
                    *seen.lock().unwrap() = Some(own_cpus());
                }
            });
            assert_eq!(own_cpus(), home, "the leader between sections");
        })
        .expect("a reborrowed team token opens its scope");
    assert_eq!(
        seen.into_inner().unwrap(),
        Some(home),
        "the leader's mask in its share"
    );
    assert_eq!(own_cpus(), before, "the caller's mask after the scope");
}

/// A child token that was handed out without a dispatch of its own
/// (`fork_sized` with one share runs inline) still places its leader for the
/// sections it dispatches.
#[test]
fn a_child_reached_without_a_dispatch_still_places_its_leader() {
    let _w = watchdog();
    let set = own_cpus();
    if set.len() < 3 {
        eprintln!("skipped: need at least 3 allowed CPUs");
        return;
    }
    let width = set.len().min(3);
    let pool = builder(width, true).build().unwrap();
    let before = own_cpus();
    let team = pool.into_team(width).expect("an idle pool");
    let width = team.size().get();
    let home = vec![set[0]];
    let seen = Mutex::new(None);
    team.exec(width)
        .fork_sized(&[width], |_, child| {
            assert_eq!(own_cpus(), before, "an inline share pins nothing");
            child.fork(width, |k, _| {
                if k == 0 {
                    *seen.lock().unwrap() = Some(own_cpus());
                }
            });
        })
        .unwrap();
    assert_eq!(
        seen.into_inner().unwrap(),
        Some(home),
        "the leader's mask in the child's section"
    );
    assert_eq!(own_cpus(), before, "the caller's mask afterwards");
}

/// A member that leads a child keeps its own share's CPU: a lent call restores
/// A member that leads a child keeps its own share's CPU: a lent call restores
/// its mask, and the section it dispatches next must not move it onto the team
/// leader's `S[0]`.
#[test]
fn a_member_led_child_keeps_its_own_cpu_after_a_lent_call() {
    let _w = watchdog();
    let set = own_cpus();
    if set.len() < 4 {
        eprintln!("skipped: need at least 4 allowed CPUs");
        return;
    }
    let width = set.len().min(4);
    let pool = builder(width, true).build().unwrap();
    let team = pool.into_team(width).expect("an idle pool");
    let width = team.size().get();
    let mut exec = team.exec(width);
    // Place the members, then fork: child 1 is led by member 1.
    exec.team(|mut t| t.reborrow().run_with(|_| {}))
        .expect("a team token opens its scope");
    let seen = Mutex::new(None);
    exec.fork(2, |child, mut token| {
        if child != 1 {
            return;
        }
        // A lent call: the member restores its own mask for the work it serves.
        token.reborrow().external(|_| ());
        // The section this child dispatches next is led by that member.
        token.reborrow().fork(2, |k, _| {
            if k == 0 {
                *seen.lock().unwrap() = Some(own_cpus());
            }
        });
    });
    let seen = seen
        .into_inner()
        .unwrap()
        .expect("the member-led child ran");
    assert_ne!(
        seen,
        vec![set[0]],
        "the member moved onto the team leader's CPU: {seen:?}"
    );
}

/// A whole-team root scope decides the placement of the team, and a section a
/// token dispatches afterwards follows it: the leader goes home to the CPU the
/// members of the latest placement left free, also when the team was created on
/// a mask too small for it.
#[test]
fn a_whole_team_scope_replaces_the_teams_placement() {
    let _w = watchdog();
    let all = own_cpus();
    if all.len() < 5 {
        eprintln!("skipped: need at least 5 allowed CPUs");
        return;
    }
    // A team created on two CPUs cannot place four participants.
    let Some(_two) = Restrict::to(&all[..2]) else {
        eprintln!("skipped: cannot restrict the calling thread to two CPUs");
        return;
    };
    let pool = builder(4, true).build().unwrap();
    let team = pool.into_team(4).expect("an idle pool");
    let width = team.size().get();
    assert_eq!(
        team.pool().pin_skips(),
        1,
        "the creation scope placed nobody"
    );
    // Widened, the next whole-team scope places the team, and records it.
    let set = all[..4].to_vec();
    let Some(_four) = Restrict::to(&set) else {
        eprintln!("skipped: cannot restrict the calling thread to four CPUs");
        return;
    };
    let seen = Seen::default();
    team.exec(width)
        .team(|mut t| t.reborrow().run_with(|p| seen.record(p.index())))
        .expect("a team token opens its scope");
    for (index, mask) in seen.take() {
        assert_eq!(mask, vec![set[index]], "participant {index} in the scope");
    }
    // A section dispatched without a scope goes home, to a CPU no member has.
    let shares = flat_shares(team.exec(width));
    assert_eq!(shares[0].1, vec![set[0]], "the leader's mask in a section");
    for (k, (_, mask)) in shares.iter().enumerate().skip(1) {
        assert_eq!(mask, &vec![set[k]], "member {k} in a section");
    }
    assert_eq!(team.pool().pin_skips(), 1, "no further skip");
}

/// A scope that cannot be placed clears the placement, so a later section does
/// not send its leader to a CPU a member of an older placement holds.
#[test]
fn a_scope_that_cannot_be_placed_clears_the_placement() {
    let _w = watchdog();
    let set = own_cpus();
    if set.len() < 4 {
        eprintln!("skipped: need at least 4 allowed CPUs");
        return;
    }
    let pool = builder(3, true).build().unwrap();
    let team = pool.into_team(3).expect("an idle pool");
    let width = team.size().get();
    let before = own_cpus();
    let shares = flat_shares(team.exec(width));
    assert_eq!(shares[0].1, vec![set[0]], "the leader's mask in a section");
    // One CPU cannot host three participants: the members are unpinned and the
    // placement is cleared.
    let Some(_one) = Restrict::to(&set[2..3]) else {
        eprintln!("skipped: cannot restrict the calling thread to one CPU");
        return;
    };
    team.exec(width)
        .team(|mut t| {
            t.reborrow().run_with(|p| {
                if p.index() > 0 {
                    assert!(own_cpus().len() > 1, "member {} was placed", p.index());
                }
            })
        })
        .expect("a team token opens its scope");
    drop(_one);
    assert_eq!(team.pool().pin_skips(), 1, "the scope could not be placed");
    // The leader is not sent home to a CPU that may now be a member's.
    let shares = flat_shares(team.exec(width));
    assert_eq!(shares[0].1, before, "the leader stays where its caller is");
}

#[test]
fn an_unrelated_token_of_a_placed_teams_pool_never_pins_its_caller() {
    let _w = watchdog();
    let set = own_cpus();
    if set.len() < 3 {
        eprintln!("skipped: need at least 3 allowed CPUs");
        return;
    }
    // A team of two on a pool of four: two workers stay free for other tokens.
    let pool = builder(4, true).build().unwrap();
    let team = pool.into_team(2).expect("an idle pool");

    // A plain token of the same pool, from a caller whose set holds S[1] and
    // S[2] but not the team's S[0].
    let cpus = vec![set[1], set[2]];
    let Some(restore) = Restrict::to(&cpus) else {
        eprintln!("skipped: cannot restrict the calling thread to {cpus:?}");
        return;
    };
    for round in 0..10 {
        let (_, mask) = leader_share(Exec::new(team.pool(), 2).unwrap());
        assert_eq!(mask, cpus, "round {round}: the caller's mask");
        // A team scope of that token with three participants on two CPUs is
        // not placed: its leader keeps its mask.
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
}

#[test]
fn a_released_or_unpinned_team_leaves_its_caller_alone() {
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
    // A root scope that cannot be placed (entered from another thread with a
    // one-CPU mask) unpins the members.
    let entered = std::thread::scope(|s| {
        s.spawn(|| {
            if !set_own_cpus(&set[..1]) {
                eprintln!("skipped: cannot restrict a thread to one CPU");
                return false;
            }
            team.exec(width)
                .team(|_| assert_eq!(own_cpus(), vec![set[0]]))
                .expect("a team token opens its scope");
            true
        })
        .join()
        .unwrap()
    });
    if !entered {
        return;
    }
    assert_eq!(team.pool().pin_skips(), 1);
    // The members are unpinned and the placement is cleared, so nothing sends
    // the leader anywhere either.
    let shares = flat_shares(team.exec(width));
    assert_eq!(shares[0].1, set, "the leader's mask");
    for (k, (_, mask)) in shares.iter().enumerate().skip(1) {
        assert!(mask.len() > 1, "member {k} kept a pin: {mask:?}");
    }

    // Placed again, then released: the pool keeps nothing of the team's.
    team.exec(width)
        .team(|_| {})
        .expect("a team token opens its scope");
    assert_eq!(own_cpus(), set, "the team pins nobody between sections");
    let pool = team.into_pool();
    assert_eq!(own_cpus(), set, "into_pool changed the mask");
    let (_, mask) = leader_share(Exec::new(&pool, width).unwrap());
    assert_eq!(mask, set, "the caller's mask after the team");
}

/// One `fork_sized` section of one-slot shares on `exec`: per share, the CPU it
/// ran on and its mask. Share 0 runs on the caller, share `k` on member `k`.
fn flat_shares(exec: Exec<'_>) -> Vec<(Option<usize>, Vec<usize>)> {
    let width = exec.budget().get();
    let seen = Mutex::new(vec![(None, Vec::new()); width]);
    exec.fork_sized(&vec![1; width], |k, _| {
        // SAFETY: no arguments; -1 on failure, never a CPU.
        let cpu = unsafe { libc::sched_getcpu() };
        seen.lock().unwrap()[k] = (usize::try_from(cpu).ok(), own_cpus());
    })
    .expect("one-slot shares sum to the budget");
    seen.into_inner().unwrap()
}

/// After a narrower root scope of a placed team of three (members on `S[1]`,
/// `S[2]`): the members are still where the full placement put them, and a
/// whole-team token opens its section with the leader on `S[0]`.
fn assert_full_placement_kept(team: &threadteam::Team, set: &[usize]) {
    for round in 0..10 {
        let shares = flat_shares(team.exec(3));
        assert_eq!(
            shares[0].1,
            vec![set[0]],
            "round {round}: the leader's mask"
        );
        for (k, (_, mask)) in shares.iter().enumerate().skip(1) {
            assert_eq!(
                mask,
                &vec![set[k]],
                "round {round}: member {k} lost its pin"
            );
        }
    }
}

#[test]
fn a_narrower_root_scope_that_cannot_be_placed_changes_nothing() {
    let _w = watchdog();
    let set = own_cpus();
    if set.len() < 3 {
        eprintln!("skipped: need at least 3 allowed CPUs");
        return;
    }
    let pool = builder(3, true).build().unwrap();
    let team = pool.into_team(3).expect("an idle pool");
    // A root scope of two from a one-CPU caller. Were it to unpin, it could
    // reach member 1 only, while member 2 stayed pinned on S[2].
    let Some(restore) = Restrict::to(&set[..1]) else {
        eprintln!("skipped: cannot restrict the calling thread to one CPU");
        return;
    };
    team.exec(2)
        .team(|_| {})
        .expect("a team token opens its scope");
    drop(restore);
    assert_eq!(
        team.pool().pin_skips(),
        0,
        "a narrower scope places nothing"
    );
    assert_full_placement_kept(&team, &set);
}

#[test]
fn a_narrower_root_scope_pins_no_leader_and_moves_no_member() {
    let _w = watchdog();
    let set = own_cpus();
    if set.len() < 4 {
        eprintln!("skipped: need at least 4 allowed CPUs");
        return;
    }
    let pool = builder(3, true).build().unwrap();
    // Members on S[1] and S[2].
    let team = pool.into_team(3).expect("an idle pool");
    // A root scope of two from [S[2], S[3]]. Were it to place, it would move
    // member 1 onto S[3] and pin the leader to S[2], which member 2 holds.
    let cpus = vec![set[2], set[3]];
    let Some(restore) = Restrict::to(&cpus) else {
        eprintln!("skipped: cannot restrict the calling thread to {cpus:?}");
        return;
    };
    team.exec(2)
        .team(|_| {
            assert_eq!(own_cpus(), cpus, "a narrower scope pins no leader");
        })
        .expect("a team token opens its scope");
    drop(restore);
    assert_full_placement_kept(&team, &set);
}

/// The path of #33 and #45: a team from `into_team` and nothing but plain team
/// tokens afterwards, at the machine's width (up to 96). Every section opens
/// with its leader placed (see
/// `a_team_places_its_leader_for_the_sections_it_dispatches_and_no_longer`), so
/// every participant is alone on its own CPU, the leader on `S[0]`.
#[test]
fn the_reported_wide_fork_sized_path_keeps_every_participant_alone_on_its_cpu() {
    use std::time::Instant;
    let _w = watchdog();
    let set = own_cpus();
    if set.len() < 3 {
        eprintln!("skipped: need at least 3 allowed CPUs");
        return;
    }
    let width = set.len().min(96);
    let pool = builder(width, true).build().unwrap();
    let team = pool.into_team(width).expect("an idle pool");
    let width = team.size().get();
    let before = own_cpus();
    let calls = 40;
    let mut latest_us = Vec::with_capacity(calls);
    for call in 0..calls {
        let t0 = Instant::now();
        if call % 2 == 0 {
            // Flat: one section of `width` one-slot shares.
            let shares = flat_shares(team.exec(width));
            latest_us.push(t0.elapsed().as_micros());
            assert_eq!(shares[0].1, vec![set[0]], "call {call}: the leader's mask");
            assert_eq!(shares[0].0, Some(set[0]), "call {call}: the leader's CPU");
            for (k, (cpu, mask)) in shares.iter().enumerate().skip(1) {
                assert_eq!(mask, &vec![set[k]], "call {call}: member {k}");
                assert_eq!(*cpu, Some(set[k]), "call {call}: member {k}'s CPU");
            }
        } else {
            // Tree: nested `fork_sized(&[b / 2, b - b / 2])`, as in the report.
            fn tree(exec: Exec<'_>) {
                let b = exec.budget().get();
                if b > 1 {
                    let left = b / 2;
                    exec.fork_sized(&[left, b - left], |_, child| tree(child))
                        .expect("shares sum to the budget");
                }
            }
            tree(team.exec(width));
            latest_us.push(t0.elapsed().as_micros());
        }
    }
    latest_us.sort_unstable();
    eprintln!(
        "width {width}: call time median {} us, max {} us",
        latest_us[calls / 2],
        latest_us[calls - 1]
    );
    assert_eq!(own_cpus(), before, "the caller's mask afterwards");
    assert_eq!(team.pool().pin_skips(), 0);
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
    let before = own_cpus();
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
        lent_caller, before,
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

/// A scope whose caller cannot be home for the team's recorded set places
/// nobody: the members pinned earlier restore their own masks instead of
/// keeping a placement whose leader CPU this caller is not allowed on.
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
    // Second scope from another thread that is allowed two CPUs, neither of
    // them the team's `S[0]`: the leader cannot be home there and two CPUs
    // cannot host four participants, so nobody is placed and the members pinned
    // by the first scope restore their own masks for this one.
    let cpus = vec![all[1], all[2]];
    let narrow = Seen::default();
    let entered = std::thread::scope(|s| {
        s.spawn(|| {
            if !set_own_cpus(&cpus) {
                eprintln!("skipped: cannot restrict a thread to {cpus:?}");
                return false;
            }
            team.exec(width)
                .team(|mut t| t.reborrow().run_with(|p| narrow.record(p.index())))
                .expect("a team token opens its scope");
            true
        })
        .join()
        .unwrap()
    });
    if !entered {
        return;
    }
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
    // The caller's mask before the team pins it to `S[0]`.
    let pre_pin = own_cpus();
    let team = pool.into_team(width).expect("an idle pool");
    let mut exec = team.exec(width);
    // Pin the members for the root scope; a persistent team keeps them pinned.
    exec.team(|_| {}).expect("a team token opens its scope");
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

#[test]
fn a_team_as_wide_as_the_mask_gives_every_participant_its_own_cpu() {
    let _w = watchdog();
    let all = own_cpus();
    if all.len() < 4 {
        eprintln!("skipped: the process has fewer than 4 allowed CPUs");
        return;
    }
    // The regression this covers (#33): a team whose width is the whole mask
    // must still leave the leader a CPU no member takes. Member `k` takes
    // `S[k]`, so width == count(S) leaves `S[0]` for the leader by construction;
    // anything that wraps a member onto `S[0]` puts a spinning member under the
    // leader's serial work, which one reporter measures as a whole timeslice per
    // section on a 96-CPU host.
    let width = all.len().min(8);
    let set = all[..width].to_vec();
    let Some(_restrict) = Restrict::to(&set) else {
        eprintln!("skipped: the restriction was refused");
        return;
    };
    let (pool, origin) = origin_pool(width, true);
    let seen = Seen::default();
    let team = pool.into_team(width).expect("an idle pool");
    assert_eq!(team.size().get(), width, "a team as wide as the mask");
    assert_eq!(
        own_cpus(),
        set,
        "into_team leaves the creating thread's set alone"
    );
    {
        let mut exec = team.exec(width);
        let seen = &seen;
        exec.team(|mut t| t.reborrow().run_with(|p| seen.record(p.index())))
            .expect("a team token opens a root scope");
    }
    let masks = seen.take();
    assert_eq!(masks.len(), width, "every participant ran a share");
    assert_eq!(
        masks[0].1,
        vec![set[0]],
        "the leader's mask in its share: {:?}",
        masks[0].1
    );
    for (index, mask) in &masks[1..] {
        assert_eq!(
            mask,
            &vec![set[*index]],
            "participant {index} of {width} on the caller's set {set:?}"
        );
    }
    let distinct: std::collections::HashSet<_> = masks.iter().map(|(_, m)| m[0]).collect();
    assert_eq!(
        distinct.len(),
        width,
        "no two participants share a CPU: {masks:?}"
    );
    let pool = team.into_pool();
    assert_eq!(own_cpus(), set, "the leader has S back when the team ends");
    // And the members are back on the masks they started with.
    pool.release_idle();
    Exec::new(&pool, width)
        .unwrap()
        .broadcast(|worker| {
            assert_eq!(
                own_cpus(),
                origin.of(worker),
                "worker {worker} kept a pinned mask after the team"
            );
        })
        .expect("a root token from a plain thread");
}
