//! Root-scope flags that the placement work carries: whether a root team scope
//! is oversubscribed reaches reborrows, `split`/`fork` children, team tokens and
//! every section of the root.

use crate::awake::{Geometry, Section};
use crate::{Exec, PinStrategy, Pool};
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
        None,
    );
    assert!(!quiet.over);
    let crowded = Section::new(
        call,
        std::ptr::null(),
        slot_of,
        std::ptr::null(),
        geometry(true),
        None,
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
    // not oversubscribed there.
    let mut exec = team.exec(width);
    // Enter the scope from another thread whose mask holds one CPU: the scope
    // decides from that set, exactly as its pin section does.
    let scoped = std::thread::scope(|s| {
        s.spawn(|| {
            if !set_own_cpus(&all[..1]) {
                eprintln!("skipped: cannot restrict the calling thread to one CPU");
                return None;
            }
            exec.team(|mut team| {
                let root = team.over;
                let child = std::sync::atomic::AtomicBool::new(false);
                team.reborrow().fork(2, |_, child_team| {
                    child.store(child_team.over, std::sync::atomic::Ordering::Relaxed);
                });
                (root, child.load(std::sync::atomic::Ordering::Relaxed))
            })
            .ok()
        })
        .join()
        .unwrap()
    });
    drop(exec);
    let Some(scoped) = scoped else { return };
    assert!(
        scoped.0,
        "four participants on a one-CPU scope mask is oversubscribed"
    );
    assert!(scoped.1, "a fork child inherits the root's pressure");
}

/// The budget a root scope chooses at entry: the throttled one when its
/// participants outnumber the caller's CPUs, the pool's otherwise; and a
/// persistent team's token does the same at every root-scope entry.
#[test]
fn a_root_scope_chooses_the_throttled_spin_budget_when_oversubscribed() {
    use crate::park::THROTTLED_POLLS;
    let budget = 50_000;
    let pool = |width| {
        Pool::builder()
            .threads(width)
            .spin_polls(budget)
            .acquire_timeout(Duration::from_secs(10))
            .build()
            .unwrap()
    };
    on_two_cpus(|| {
        let crowded = pool(4);
        let mut exec = Exec::new(&crowded, 4).unwrap();
        exec.team(|team| {
            let awake = team.awake.expect("a pool");
            assert!(!team.ids.is_empty());
            for &id in team.ids {
                assert_eq!(awake.slot_spin_budget(id), THROTTLED_POLLS);
            }
        })
        .expect("an idle pool");
        drop(exec);
        let team = crowded.into_team(4).expect("an idle pool");
        let mut exec = team.exec(4);
        exec.team(|t| {
            for &id in t.ids {
                assert_eq!(t.awake.unwrap().slot_spin_budget(id), THROTTLED_POLLS);
            }
        })
        .expect("a team token opens its scope");

        let fits = pool(2);
        let mut exec = Exec::new(&fits, 2).unwrap();
        exec.team(|team| {
            for &id in team.ids {
                assert_eq!(team.awake.unwrap().slot_spin_budget(id), budget);
            }
        })
        .expect("an idle pool");
    });
}

/// A pool of `width` workers whose waits spin for `budget` polls.
fn spinning_pool(width: usize, budget: u32) -> Pool {
    Pool::builder()
        .threads(width)
        .spin_polls(budget)
        .acquire_timeout(Duration::from_secs(10))
        .build()
        .unwrap()
}

/// The spin budget of every member slot of `pool` that `team` reserved.
fn member_budgets(team: &crate::Team) -> Vec<u32> {
    let awake = &team.pool().awake;
    team.reservation_ids()
        .iter()
        .map(|&id| awake.slot_spin_budget(id))
        .collect()
}

/// threadteam-rs#33. A `Team`'s token is a team of `|S|` CPUs even though its
/// leader is placed only for a section: two participants on a two-CPU set are
/// not oversubscribed, so the members keep the pool's spin budget instead of
/// `THROTTLED_POLLS`.
#[test]
fn a_team_token_minted_by_its_leader_keeps_the_pool_spin_budget() {
    use crate::park::THROTTLED_POLLS;
    let budget = 50_000;
    on_two_cpus(|| {
        let set = own_cpus();
        let team = spinning_pool(2, budget).into_team(2).expect("an idle pool");
        assert_eq!(own_cpus(), set, "into_team leaves the caller's mask");
        assert_eq!(crate::pin::available(), 2, "the leader leads |S| CPUs");
        let exec = team.exec(2);
        let budgets = member_budgets(&team);
        assert!(!budgets.is_empty());
        assert!(
            budgets.iter().all(|&b| b == budget && b != THROTTLED_POLLS),
            "2 participants on a 2-CPU S, token minted by the team's leader: {budgets:?}"
        );
        drop(exec);
    });
}

/// The same token, reached through `fork_sized` without opening a scope: the
/// child carries the pressure decided at mint, so it is not oversubscribed and
/// its members keep the pool's spin budget.
#[test]
fn a_fork_sized_child_of_a_team_token_is_not_oversubscribed() {
    let budget = 50_000;
    on_two_cpus(|| {
        let set = own_cpus();
        let team = spinning_pool(2, budget).into_team(2).expect("an idle pool");
        assert_eq!(own_cpus(), set, "into_team leaves the caller's mask");
        let seen = std::sync::Mutex::new(None);
        team.exec(2)
            .fork_sized(&[2], |_, mut child| {
                let got = child
                    .team(|t| {
                        let awake = t.awake.expect("a carried sub-team");
                        let budgets: Vec<u32> =
                            t.ids.iter().map(|&id| awake.slot_spin_budget(id)).collect();
                        (t.over, budgets)
                    })
                    .expect("a fork child carries the sub-team");
                *seen.lock().unwrap() = Some(got);
            })
            .unwrap();
        let (over, budgets) = seen.into_inner().unwrap().expect("the child ran");
        assert!(!over, "a fork_sized child of a team token");
        assert!(!budgets.is_empty(), "the child carries the member");
        assert!(budgets.iter().all(|&b| b == budget), "{budgets:?}");
    });
}

/// The fix keeps pacing where it belongs: a `Team` of four participants
/// created on a two-CPU thread is genuinely oversubscribed. Its leader is not
/// placed (`into_team` records no set), so its tokens, and a
/// `fork_sized` child reached without a scope, keep the throttled spin budget.
#[test]
fn a_team_created_on_too_few_cpus_is_still_paced() {
    use crate::park::THROTTLED_POLLS;
    let budget = 50_000;
    on_two_cpus(|| {
        let set = own_cpus();
        let team = spinning_pool(4, budget).into_team(4).expect("an idle pool");
        assert_eq!(own_cpus(), set, "an oversubscribed team pins no leader");
        assert_eq!(crate::pin::available(), 2);
        let exec = team.exec(4);
        let budgets = member_budgets(&team);
        assert!(!budgets.is_empty());
        assert!(
            budgets.iter().all(|&b| b == THROTTLED_POLLS),
            "4 participants on a 2-CPU set: {budgets:?}"
        );
        let over = std::sync::atomic::AtomicBool::new(false);
        exec.fork_sized(&[4], |_, mut child| {
            let o = child
                .team(|t| t.over)
                .expect("a fork child carries the sub-team");
            over.store(o, std::sync::atomic::Ordering::Relaxed);
        })
        .unwrap();
        assert!(
            over.into_inner(),
            "a fork_sized child of an oversubscribed team"
        );
    });
}

/// Puts the host's topology back when the test ends, also on unwind.
struct HostTopology;

impl Drop for HostTopology {
    fn drop(&mut self) {
        crate::pin::set_test_topology(None);
    }
}

/// The mask each participant of a two-wide root team scope ran its share in,
/// participant 0 (the leader) first.
fn participant_masks(pool: &Pool) -> Vec<Vec<usize>> {
    // `run_with` joins every participant before it returns, so a plain
    // `Mutex` holding the rows is enough for the closure to borrow.
    let seen = std::sync::Mutex::new(Vec::new());
    let mut exec = Exec::new(pool, 2).expect("a two-wide token");
    exec.team(|mut team| {
        team.reborrow().run_with(|p| {
            seen.lock().unwrap().push((p.index(), own_cpus()));
        });
    })
    .expect("an idle pool");
    let mut rows = seen.into_inner().unwrap();
    rows.sort_by_key(|&(index, _)| index);
    assert_eq!(rows.len(), 2, "both participants ran a share");
    rows.into_iter().map(|(_, mask)| mask).collect()
}

/// `exec::pin_scope` hands the pool's own strategy to `pin::Placement::of`. On
/// one NUMA node `Spread` and `Pack` place identically, so the node map is a
/// synthetic two-node machine over this host's own CPUs: `Spread` then takes the
/// second participant across to the first CPU of the second node, `Pack` keeps
/// it on the second CPU of the first.
///
/// A `pin_scope` that passes a hard-coded strategy (or the wrong field) moves
/// one of the two members onto the other's CPU and fails here. That the builder's
/// default is `Spread` is asserted in `tests/pinning.rs`.
#[test]
fn a_root_scope_places_under_the_pools_pin_strategy() {
    let set = own_cpus();
    if set.len() < 4 {
        eprintln!(
            "skipped: needs 4 allowed CPUs to build two synthetic nodes, this process has {}",
            set.len()
        );
        return;
    }
    let nodes = vec![vec![set[0], set[1]], vec![set[2], set[3]]];
    let topology: &'static crate::pin::Topology =
        Box::leak(Box::new(crate::pin::Topology::from_nodes(&nodes)));
    crate::pin::set_test_topology(Some(topology));
    let _host = HostTopology;

    let member_cpu = |strategy: PinStrategy| {
        let pool = Pool::builder()
            .threads(2)
            .pin_strategy(strategy)
            .idle_spin(Duration::ZERO)
            .acquire_timeout(Duration::from_secs(10))
            .build()
            .unwrap();
        assert_eq!(pool.pin_strategy(), strategy);
        let masks = participant_masks(&pool);
        // Participant 0, the leader, is on `S[0]` under either strategy, so the
        // member's CPU is the one the strategy decides.
        assert_eq!(masks[0], vec![set[0]], "the leader's CPU for {strategy:?}");
        assert_eq!(pool.pin_skips(), 0, "nothing skipped for {strategy:?}");
        masks[1].clone()
    };

    assert_eq!(
        member_cpu(PinStrategy::Spread),
        vec![set[2]],
        "Spread puts the second participant on node 2's first CPU"
    );
    assert_eq!(
        member_cpu(PinStrategy::Pack),
        vec![set[1]],
        "Pack puts it on node 1's second CPU"
    );
    assert_eq!(
        own_cpus(),
        set,
        "the calling thread's mask after both scopes"
    );
}

/// threadteam-rs#62: the memo is refreshed by the pin section of a whole-team
/// scope, not by a mint and not by a narrower scope. On a thread another library
/// restricts while nothing is placed, the mints before the next whole-team scope
/// keep the count the thread had, and that scope is paced by it; its pin section
/// then reads the mask anyway, so every mint after it is paced by the restricted
/// set.
#[test]
fn a_mask_changed_by_another_library_is_seen_by_the_scope_that_follows() {
    use crate::park::THROTTLED_POLLS;
    let all = own_cpus();
    if all.len() < 4 {
        eprintln!("skipped: need at least 4 allowed CPUs for a 4-participant team");
        return;
    }
    let budget = 50_000;
    let team = spinning_pool(4, budget).into_team(4).expect("an idle pool");
    // One mint on the wide mask caches the count of that mask. No scope, so
    // nothing moves the thread and the memo keeps it.
    drop(team.exec(4));
    // Restrict the calling thread behind the crate: one CPU for four
    // participants.
    if !set_own_cpus(&all[..1]) {
        eprintln!("skipped: cannot restrict the calling thread to one CPU");
        return;
    }
    // A narrower root scope runs no pin section, so it reads nothing. With no
    // placement recorded (this team never opened a whole-team scope) it pins
    // nothing either, so the memo keeps the old count however many of them run.
    // Once a placement exists the narrower scope pins its leader at the home CPU
    // and gives it its mask back, and those two mask changes clear the memo (see
    // the last case of this test).
    for _ in 0..3 {
        team.exec(2)
            .team(|_| {})
            .expect("a narrower team token opens its scope");
    }
    let exec = team.exec(4);
    let stale = member_budgets(&team);
    assert!(
        stale.iter().all(|&b| b == budget),
        "neither a mint nor a narrower scope reads the mask: {stale:?}"
    );
    drop(exec);
    // The whole-team scope reads the mask in its pin section, but it is still
    // paced by the count the thread had: the budgets are set at the mint.
    let inside = std::sync::Mutex::new(None);
    team.exec(4)
        .team(|t| {
            let awake = t.awake.expect("a pool");
            let budgets: Vec<u32> = t.ids.iter().map(|&id| awake.slot_spin_budget(id)).collect();
            *inside.lock().unwrap() = Some((t.over, budgets));
        })
        .expect("a team token opens its scope");
    let (over, budgets) = inside.into_inner().unwrap().expect("the scope ran");
    assert!(!over, "the scope itself keeps the stale decision");
    assert!(
        budgets.iter().all(|&b| b == budget),
        "and the stale budget: {budgets:?}"
    );
    // Every mint after it is paced by the restricted set.
    let exec = team.exec(4);
    let throttled = member_budgets(&team);
    assert!(
        throttled.iter().all(|&b| b == THROTTLED_POLLS),
        "the pin section refreshed the memo: {throttled:?}"
    );
    drop(exec);
    // A narrower scope does refresh the memo once the team has a placement: the
    // narrower token pins its leader at the home CPU and gives it its mask back,
    // and both are mask changes of ours. Four participants on the two CPUs that
    // hold the home are oversubscribed, so a refreshed count shows as the
    // throttled budget.
    assert!(set_own_cpus(&all[..4]), "widening the calling thread");
    team.exec(4)
        .team(|_| {})
        .expect("a whole-team scope records the placement");
    drop(team.exec(4));
    assert!(set_own_cpus(&all[..2]), "two CPUs that hold the home");
    team.exec(2)
        .team(|_| {})
        .expect("a narrower scope pins its leader home");
    let exec = team.exec(4);
    let refreshed = member_budgets(&team);
    assert!(
        refreshed.iter().all(|&b| b == THROTTLED_POLLS),
        "the home pin of the narrower scope cleared the memo: {refreshed:?}"
    );
    drop(exec);
    assert!(set_own_cpus(&all), "restoring the calling thread mask");
}

/// threadteam-rs#62: with participant pinning off no scope of this crate touches
/// the calling thread mask, so the memo keeps whatever count the thread had, even
/// across whole-team scopes.
#[test]
fn a_pool_with_pinning_off_never_refreshes_the_count() {
    let all = own_cpus();
    if all.len() < 4 {
        eprintln!("skipped: need at least 4 allowed CPUs for a 4-participant team");
        return;
    }
    let budget = 50_000;
    let pool = Pool::builder()
        .threads(4)
        .pin_participants(false)
        .spin_polls(budget)
        .acquire_timeout(Duration::from_secs(10))
        .build()
        .unwrap();
    let team = pool.into_team(4).expect("an idle pool");
    // One mint on the wide mask caches the count of that mask.
    drop(team.exec(4));
    if !set_own_cpus(&all[..1]) {
        eprintln!("skipped: cannot restrict the calling thread to one CPU");
        return;
    }
    // No scope runs a pin section, so nothing reads the mask: the count the thread
    // had stays, and so does the budget it chose.
    for _ in 0..3 {
        team.exec(4)
            .team(|_| {})
            .expect("a team token opens its scope");
        let exec = team.exec(4);
        let kept = member_budgets(&team);
        assert!(
            kept.iter().all(|&b| b == budget),
            "pinning off runs no pin section: {kept:?}"
        );
        drop(exec);
    }
    assert!(set_own_cpus(&all), "restoring the calling thread mask");
}
