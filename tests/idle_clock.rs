//! The idle clock of a reserved member (feature `trace`): the time since the
//! member's own last share (or since it started, or since it woke from a park).
//!
//! Everything here is judged by the **order of a member's own trace events**,
//! never by a wall-clock bound on how fast the leader gets around to the next
//! section: on a small or loaded host one `sleep` can wake tens of milliseconds
//! late, so "sections every 5 ms" is not something a test may assume. Member
//! slot ids are read from the trace as well, because the acquisition may claim
//! any free slots.
//!
//! * Never before the delay: for every park of member `m` at time `t`,
//!   `t - max(m's last share end, m's last wake-up, m's start) >= SPIN`. This
//!   holds for every park in the whole test.
//! * After the delay: once no section arrives, every member parks; polled until
//!   it happens, with no upper bound on how late.
//! * A following section wakes every member that was parked: the number of
//!   `ParkExit`s rises by the number of members parked at that moment.
#![cfg(all(feature = "trace", feature = "parallel"))]

use std::time::{Duration, Instant};
use threadteam::trace::{self, Kind};
use threadteam::Pool;

const SPIN: Duration = Duration::from_millis(40);

/// Only a hang guard: no assertion below is a bound on how long the members or
/// the leader take.
const DEADLINE: Duration = Duration::from_secs(10);

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

/// The member events of the pool, in time order.
fn member_events(team: &threadteam::Team) -> Vec<trace::Event> {
    trace::events(team.pool())
        .into_iter()
        .filter(|e| e.member != usize::MAX)
        .collect()
}

fn count(team: &threadteam::Team, kind: Kind) -> usize {
    member_events(team)
        .iter()
        .filter(|e| e.kind == kind)
        .count()
}

/// The members that ran a share, in the order of their first share.
fn members_that_ran(events: &[trace::Event]) -> Vec<usize> {
    let mut out: Vec<usize> = Vec::new();
    for e in events {
        if e.kind == Kind::SectionEnd && !out.contains(&e.member) {
            out.push(e.member);
        }
    }
    out
}

/// Members asleep now: their latest park event is a `ParkEnter`.
fn parked_now(team: &threadteam::Team) -> usize {
    let mut parked = std::collections::HashSet::new();
    for e in member_events(team) {
        match e.kind {
            Kind::ParkEnter => {
                parked.insert(e.member);
            }
            Kind::ParkExit => {
                parked.remove(&e.member);
            }
            _ => {}
        }
    }
    parked.len()
}

/// Waits (hang guard only) until `cond` holds, polling the trace.
fn wait_until(what: &str, mut cond: impl FnMut() -> bool) {
    let start = Instant::now();
    while !cond() {
        assert!(start.elapsed() < DEADLINE, "timeout waiting for {what}");
        std::thread::sleep(Duration::from_millis(2));
    }
}

/// Did every member that ran a share park after its own last share end?
fn every_member_parked_after_its_last_share(events: &[trace::Event]) -> bool {
    let members = members_that_ran(events);
    !members.is_empty()
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
        })
}

/// Asserts the "never before the delay" property over every park in the trace.
fn assert_no_park_before_the_delay(team: &threadteam::Team) {
    let events = member_events(team);
    let parks: Vec<_> = events
        .iter()
        .filter(|e| e.kind == Kind::ParkEnter)
        .collect();
    assert!(!parks.is_empty(), "no member parked at all");
    for park in parks {
        // The park delay runs from the later of the member's own last share
        // end, its own last wake-up and its start.
        let anchor = events
            .iter()
            .filter(|e| e.member == park.member && e.ns <= park.ns)
            .filter(|e| matches!(e.kind, Kind::SectionEnd | Kind::ParkExit | Kind::Enter))
            .map(|e| e.ns)
            .max();
        let Some(anchor) = anchor else {
            panic!("member {} parked without an anchor event", park.member);
        };
        assert!(
            park.ns - anchor >= SPIN.as_nanos() as u64,
            "member {} parked {} us after its own last share, wake-up or start",
            park.member,
            (park.ns - anchor) / 1000
        );
    }
}

#[test]
fn members_park_after_the_park_delay_since_their_own_last_share_or_wake() {
    let team = team(3);
    let mut exec = team.exec(3);
    if team.size().get() < 3 {
        return;
    }
    // Sections a few milliseconds apart for about 150 ms. This does *not*
    // assert that nobody parks: on a loaded host the leader's sleeps can wake
    // late enough for a member to idle through its delay, and that is allowed.
    for _ in 0..30 {
        exec.reborrow().broadcast(|_| {}).unwrap();
        std::thread::sleep(Duration::from_millis(5));
    }
    // Then no section at all: every member parks, however long it takes.
    wait_until("every member parked after its own last share", || {
        every_member_parked_after_its_last_share(&member_events(&team))
    });
    let members = members_that_ran(&member_events(&team));
    assert_eq!(
        members.len(),
        team.size().get() - 1,
        "every member ran a share: {members:?}"
    );
    // The "not before" property holds for every park of the whole test, the
    // ones inside the phase of regular sections included.
    assert_no_park_before_the_delay(&team);
    // A following section wakes every member that is parked now: the wake-ups
    // are exactly the parked members' `ParkExit`s.
    let parked = parked_now(&team);
    assert_eq!(
        parked,
        team.size().get() - 1,
        "every member is parked after the last section"
    );
    let exits = count(&team, Kind::ParkExit);
    exec.reborrow().broadcast(|_| {}).unwrap();
    wait_until("every parked member woke", || {
        count(&team, Kind::ParkExit) >= exits + parked
    });
    // The clock restarts from the share they just ran: a later park is again at
    // least SPIN after it.
    assert_no_park_before_the_delay(&team);
}
