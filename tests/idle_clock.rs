//! The idle episodes of a reserved member (feature `trace`): after its own last
//! share (or its start, or a wake-up) a member spins for the pool's spin budget
//! of polls, yields a few times and parks (`docs/design/waiting.md`). The budget
//! is a count, so no wall-clock lower bound on a park exists; the gating of the
//! count itself is unit tested (`src/tests/waiting.rs`). Here the budget is zero
//! and the test checks the episode's order and the wake-up.
//!
//! Everything here is judged by the **order of a member's own trace events**,
//! never by a wall-clock bound on how fast the leader gets around to the next
//! section: on a small or loaded host one `sleep` can wake tens of milliseconds
//! late, so "sections every 5 ms" is not something a test may assume. Member
//! slot ids are read from the trace as well, because the acquisition may claim
//! any free slots.
//!
//! * A park follows an anchor: every `ParkEnter` of member `m` comes after one of
//!   `m`'s own `SectionEnd`, `ParkExit` or `Enter` events, with no other
//!   `ParkEnter` of `m` in between (one park per idle episode).
//! * Once no section arrives, every member parks; polled until it happens, with
//!   no upper bound on how late.
//! * A following section wakes every member that was parked: the number of
//!   `ParkExit`s rises by the number of members parked at that moment.
#![cfg(all(feature = "trace", feature = "parallel"))]

use std::time::{Duration, Instant};
use threadteam::trace::{self, Kind};
use threadteam::Pool;

/// Only a hang guard: no assertion below is a bound on how long the members or
/// the leader take.
const DEADLINE: Duration = Duration::from_secs(10);

fn team(width: usize) -> threadteam::Team {
    Pool::builder()
        .threads(width)
        .spin_polls(0)
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

/// Asserts that every park follows its own anchor (a share end, a wake-up or the
/// start) with no second park in the same idle episode.
fn assert_one_park_per_episode(team: &threadteam::Team) {
    let events = member_events(team);
    let parks: Vec<_> = events
        .iter()
        .filter(|e| e.kind == Kind::ParkEnter)
        .collect();
    assert!(!parks.is_empty(), "no member parked at all");
    for park in parks {
        let before: Vec<_> = events
            .iter()
            .filter(|e| e.member == park.member && e.ns <= park.ns)
            .filter(|e| {
                matches!(
                    e.kind,
                    Kind::SectionEnd | Kind::ParkExit | Kind::Enter | Kind::ParkEnter
                )
            })
            .collect();
        // The last event before this park (itself excluded) is an anchor.
        let prev = before
            .iter()
            .rev()
            .find(|e| !std::ptr::eq(**e, park))
            .unwrap_or_else(|| panic!("member {} parked without an anchor event", park.member));
        assert!(
            prev.kind != Kind::ParkEnter,
            "member {} parked twice in one idle episode",
            park.member
        );
    }
}

#[test]
fn members_park_once_per_idle_episode_and_wake_on_the_next_section() {
    let team = team(3);
    let mut exec = team.exec(3);
    if team.size().get() < 3 {
        return;
    }
    // Sections a few milliseconds apart: with a zero budget the members park
    // between them, and every section wakes them.
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
    // One park per idle episode, over the whole test.
    assert_one_park_per_episode(&team);
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
    // The episode restarts from the share they just ran.
    wait_until("every member parked again", || {
        parked_now(&team) == team.size().get() - 1
    });
    assert_one_park_per_episode(&team);
}
