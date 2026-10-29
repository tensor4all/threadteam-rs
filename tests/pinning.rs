//! Opt-in member placement (Linux only): every member pins itself onto one CPU
//! of the calling thread's set for a root team scope and restores its own mask
//! afterwards, while the leader (the calling thread) is never moved.
#![cfg(all(feature = "parallel", target_os = "linux"))]

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
fn pinning_off_is_the_default_and_leaves_every_mask_alone() {
    let _w = watchdog();
    let set = own_cpus();
    let width = set.len().min(3);
    let (pool, origin) = origin_pool(width, false);
    assert!(!pool.pin_participants(), "off by default");
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
fn a_team_wider_than_the_callers_set_wraps_around_it() {
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
    let pool = builder(width, true).build().unwrap();
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
    // The leader keeps the whole set; member k wraps onto S[k % 2].
    assert_eq!(masks[0].1, before, "the leader was moved");
    for (index, mask) in &masks[1..] {
        assert_eq!(
            mask,
            &vec![cpus[index % cpus.len()]],
            "participant {index} of {width} on the caller's set {cpus:?}"
        );
    }
    assert_eq!(own_cpus(), before, "the leader's mask after the scope");
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
