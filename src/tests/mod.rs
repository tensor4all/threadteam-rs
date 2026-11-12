#[cfg(all(feature = "parallel", target_os = "linux"))]
mod placement;
mod split;
mod waiting;

/// threadteam-rs#68: without a multi-node term, making the children of a team
/// token reads no node slot (`fork`, `fork` with more jobs than tasks,
/// `TeamExec::split` and `TeamExec::fork`); only a pool with a term pays for
/// the leader's node.
#[cfg(feature = "parallel")]
#[test]
fn default_child_construction_reads_no_node_slot() {
    use std::time::Duration;
    let pool = crate::Pool::builder()
        .threads(4)
        .idle_spin(Duration::ZERO)
        .acquire_timeout(Duration::from_secs(10))
        .build()
        .unwrap();
    assert!(!pool.awake.costs.team_row().has_multi_node());
    let team = pool.into_team(4).expect("an idle pool");
    let reads = || {
        team.pool()
            .awake
            .node_reads
            .load(core::sync::atomic::Ordering::Relaxed)
    };
    let before = reads();
    team.exec(4).fork(2, |_, _| {});
    team.exec(4).fork(8, |_, _| {});
    team.exec(4)
        .team(|mut t| {
            t.reborrow().split(&[2, 2], |_, _| {}).unwrap();
            t.fork(2, |_, _| {});
        })
        .expect("carried members");
    assert_eq!(reads(), before);
}
