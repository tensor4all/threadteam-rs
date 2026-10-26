//! The idle clock of a reserved member (feature `trace`): the time since the
//! member's own last share (or since it started), per member. A member parks
//! once that exceeds the park delay and never earlier; sections that arrive
//! sooner keep it spinning. Timings are generous; the test takes well under a
//! second.
#![cfg(all(feature = "trace", feature = "parallel"))]

use std::time::Duration;
use threadteam::trace::{self, Kind};
use threadteam::Pool;

const SPIN: Duration = Duration::from_millis(40);

fn team(width: usize) -> threadteam::Team {
    Pool::builder()
        .threads(width)
        .idle_spin(SPIN)
        .acquire_timeout(Duration::from_secs(10))
        .build()
        .unwrap()
        .into_team(width)
        .unwrap()
}

fn kinds(team: &threadteam::Team, kind: Kind) -> Vec<(usize, u64)> {
    trace::events(team.pool())
        .into_iter()
        .filter(|e| e.kind == kind && e.member != usize::MAX)
        .map(|e| (e.member, e.ns))
        .collect()
}

#[test]
fn members_park_after_the_park_delay_since_their_last_share_and_not_before() {
    let team = team(3);
    let mut exec = team.exec(3);
    if team.size().get() < 3 {
        return;
    }
    // Sections 5 ms apart for 150 ms: nobody parks (park delay 40 ms).
    for _ in 0..30 {
        exec.reborrow().broadcast(|_| {}).unwrap();
        std::thread::sleep(Duration::from_millis(5));
    }
    assert!(
        kinds(&team, Kind::ParkEnter).is_empty(),
        "a member parked although sections arrived every 5 ms"
    );
    // Then no section: every member parks once, not before the delay.
    let last_end = kinds(&team, Kind::SectionEnd)
        .iter()
        .map(|&(_, ns)| ns)
        .max()
        .unwrap();
    std::thread::sleep(Duration::from_millis(300));
    let parks = kinds(&team, Kind::ParkEnter);
    assert_eq!(
        parks.len(),
        2,
        "each of the two members parks once: {parks:?}"
    );
    for (m, ns) in parks {
        let after = ns.saturating_sub(last_end);
        let own_end = kinds(&team, Kind::SectionEnd)
            .iter()
            .filter(|&&(mm, _)| mm == m)
            .map(|&(_, n)| n)
            .max()
            .unwrap();
        assert!(
            ns - own_end >= SPIN.as_nanos() as u64,
            "member {m} parked {} us after its own last share",
            (ns - own_end) / 1000
        );
        assert!(after < 250_000_000, "member {m} parked too late");
    }
    // A section wakes them; the clock restarts from that share.
    exec.reborrow().broadcast(|_| {}).unwrap();
    assert_eq!(kinds(&team, Kind::ParkExit).len(), 2);
}
