//! Ledger: team participants count as live tasks, never more than the budget.
#![cfg(all(feature = "parallel", feature = "ledger"))]

use std::time::Duration;
use threadteam::{ledger, Exec, Pool};

fn pool(n: usize) -> Pool {
    Pool::builder()
        .threads(n)
        .idle_spin(Duration::ZERO)
        .acquire_timeout(Duration::from_secs(5))
        .build()
        .unwrap()
}

fn nap() {
    std::thread::sleep(Duration::from_millis(2));
}

#[test]
fn team_sections_and_nested_children_never_exceed_the_root_budget() {
    let pool = pool(8);
    for budget in [2, 4, 8] {
        let mut exec = Exec::new(&pool, budget).unwrap();
        let root = exec.ledger_root().unwrap();
        exec.team(|mut team| {
            team.reborrow().run(|_, _| nap());
            team.reborrow().run_with(|p| {
                p.barrier().unwrap();
                nap();
            });
            team.fork(2, |_, child| {
                child.fork(2, |_, grand| {
                    grand.run(|_, _| nap());
                });
            });
        })
        .unwrap();
        assert!(ledger::max_live(root) <= budget, "budget {budget}");
        assert!(ledger::max_live(root) >= 1);
        assert_eq!(ledger::live(root), 0);
    }
}

#[test]
fn a_team_inside_a_fork_child_stays_within_the_budget() {
    let pool = pool(8);
    let exec = Exec::new(&pool, 8).unwrap();
    let root = exec.ledger_root().unwrap();
    exec.fork(2, |_, mut child| {
        // Each child holds 4 slots: it leads a team of 4 or falls back.
        let _ = child.team(|team| team.run(|_, _| nap()));
    });
    assert!(ledger::max_live(root) <= 8);
    assert_eq!(ledger::live(root), 0);
}
