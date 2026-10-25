//! Forced interleavings of the park/wake handshake (feature `test-hooks`).
//!
//! A hook pauses the member (or the leader) at a chosen point of the handshake,
//! the test dispatches a section meanwhile, then lets the paused thread go. The
//! real protocol must complete the section; a deliberately broken protocol
//! (`Mutation`) must stall in exactly this interleaving, which shows the test
//! can see the bug. Each scenario takes well under a second.
#![cfg(all(feature = "test-hooks", feature = "parallel"))]

use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::mpsc::{channel, Receiver, Sender};
use std::sync::{Arc, Mutex};
use std::time::Duration;
use threadteam::test_hooks::{self, Mutation, Point};
use threadteam::{Pool, Team};

/// The hooks are process-global: one scenario at a time.
static SERIAL: Mutex<()> = Mutex::new(());

/// A hang is a failure: abort the whole process if the test body stalls.
fn watchdog() -> impl Drop {
    struct Stop(Sender<()>);
    impl Drop for Stop {
        fn drop(&mut self) {
            let _ = self.0.send(());
        }
    }
    let (tx, rx) = channel::<()>();
    std::thread::spawn(move || {
        if rx.recv_timeout(Duration::from_secs(45)).is_err() {
            eprintln!("watchdog: test stalled");
            std::process::exit(101);
        }
    });
    Stop(tx)
}

/// Pauses the first thread that hits it once armed, until released.
struct Pauser {
    armed: AtomicBool,
    reached: Mutex<Sender<()>>,
    go: Mutex<Receiver<()>>,
}

struct Handle {
    pauser: Arc<Pauser>,
    reached: Receiver<()>,
    go: Sender<()>,
}

impl Handle {
    fn new() -> Handle {
        let (reached_tx, reached) = channel();
        let (go, go_rx) = channel();
        Handle {
            pauser: Arc::new(Pauser {
                armed: AtomicBool::new(false),
                reached: Mutex::new(reached_tx),
                go: Mutex::new(go_rx),
            }),
            reached,
            go,
        }
    }

    fn arm(&self) {
        self.pauser.armed.store(true, Ordering::SeqCst);
    }

    fn wait_reached(&self) {
        self.reached
            .recv_timeout(Duration::from_secs(5))
            .expect("the paused thread reached its hook point");
    }

    fn release(&self) {
        self.go.send(()).unwrap();
    }
}

impl Pauser {
    fn hit(&self) {
        if self.armed.swap(false, Ordering::SeqCst) {
            let _ = self.reached.lock().unwrap().send(());
            let _ = self.go.lock().unwrap().recv();
        }
    }
}

/// Waits until the member has reached its first wait so that arming a pause cannot
/// catch the member before the scenario starts.
static PARKS: AtomicUsize = AtomicUsize::new(0);

fn wait_first_park() {
    let start = std::time::Instant::now();
    while PARKS.load(Ordering::SeqCst) == 0 {
        assert!(
            start.elapsed() < Duration::from_secs(5),
            "the member never parked"
        );
        std::thread::sleep(Duration::from_millis(1));
    }
    // Let it enter the wait itself.
    std::thread::sleep(Duration::from_millis(20));
}

/// Clears hook and mutation when a scenario ends, also on panic.
struct Reset;

impl Drop for Reset {
    fn drop(&mut self) {
        test_hooks::set_hook(None);
        test_hooks::set_mutation(Mutation::None);
    }
}

/// Two members' worth of pool is not needed: one member, parked at once
/// (`idle_spin` zero), is the whole scene.
fn team() -> Team {
    Pool::builder()
        .threads(2)
        .idle_spin(Duration::ZERO)
        .acquire_timeout(Duration::from_secs(5))
        .build()
        .unwrap()
        .into_team(2)
        .expect("an idle pool")
}

fn run_section(team: &Team, hits: &AtomicUsize) {
    team.exec(2).fork(2, |_, _| {
        hits.fetch_add(1, Ordering::SeqCst);
    });
}

fn install(raised: &Handle, checked: &Handle, wake: &Handle) {
    let (a, b, c) = (
        raised.pauser.clone(),
        checked.pauser.clone(),
        wake.pauser.clone(),
    );
    test_hooks::set_hook(Some(Arc::new(move |point| match point {
        Point::ParkRaised => a.hit(),
        Point::ParkChecked => {
            PARKS.fetch_add(1, Ordering::SeqCst);
            b.hit()
        }
        Point::WakeBegin => c.hit(),
    })));
}

/// The member is paused at `point` just before it would sleep; a section is
/// dispatched by another thread; the member is released after a pause. Reports
/// whether the section finished (a stalled protocol is recovered with a correct
/// wake-up so that the scenario can end).
fn member_paused_then_dispatch(point: Point, mutation: Mutation) -> bool {
    let _serial = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    let _reset = Reset;
    PARKS.store(0, Ordering::SeqCst);
    let (raised, checked, wake) = (Handle::new(), Handle::new(), Handle::new());
    install(&raised, &checked, &wake);
    let team = team();
    wait_first_park();
    let paused = if point == Point::ParkRaised {
        &raised
    } else {
        &checked
    };
    paused.arm();
    let hits = AtomicUsize::new(0);
    // Section A wakes the member; afterwards it parks again and meets the pause.
    run_section(&team, &hits);
    paused.wait_reached();
    test_hooks::set_mutation(mutation);

    let (done_tx, done_rx) = channel();
    let (team_ref, hits_ref) = (&team, &hits);
    let finished = std::thread::scope(|s| {
        s.spawn(move || {
            run_section(team_ref, hits_ref); // section B
            let _ = done_tx.send(());
        });
        // B has published (or is blocked on the lock) while the member is paused.
        std::thread::sleep(Duration::from_millis(50));
        paused.release();
        let finished = done_rx.recv_timeout(Duration::from_millis(600)).is_ok();
        if !finished {
            test_hooks::set_mutation(Mutation::None);
            test_hooks::wake_parked(team_ref.pool());
        }
        finished
    });
    if finished {
        assert_eq!(hits.load(Ordering::SeqCst), 4, "both sections ran fully");
    }
    finished
}

#[test]
fn a_section_published_while_the_member_is_about_to_park_is_not_missed() {
    let _w = watchdog();
    assert!(member_paused_then_dispatch(
        Point::ParkRaised,
        Mutation::None
    ));
}

#[test]
fn a_section_published_between_the_members_check_and_its_wait_is_not_missed() {
    let _w = watchdog();
    assert!(member_paused_then_dispatch(
        Point::ParkChecked,
        Mutation::None
    ));
}

#[test]
fn the_first_interleaving_stalls_when_the_member_skips_its_recheck() {
    let _w = watchdog();
    assert!(
        !member_paused_then_dispatch(Point::ParkRaised, Mutation::SkipRecheck),
        "the scenario must be able to see the lost wake-up"
    );
}

#[test]
fn the_second_interleaving_stalls_when_the_leader_skips_the_lock_handoff() {
    let _w = watchdog();
    assert!(
        !member_paused_then_dispatch(Point::ParkChecked, Mutation::NoLockHandoff),
        "the scenario must be able to see the lost wake-up"
    );
}

#[test]
fn a_leader_paused_between_bumping_and_reading_parked_does_not_lose_the_member() {
    let _w = watchdog();
    let _serial = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    let _reset = Reset;
    PARKS.store(0, Ordering::SeqCst);
    let (raised, checked, wake) = (Handle::new(), Handle::new(), Handle::new());
    install(&raised, &checked, &wake);
    let team = team();
    wait_first_park();
    raised.arm();
    let hits = AtomicUsize::new(0);
    run_section(&team, &hits); // A: the member wakes, re-parks, meets the pause
    raised.wait_reached();
    // The leader of B bumps the generation, then pauses before reading `parked`.
    wake.arm();
    let (done_tx, done_rx) = channel();
    let (team_ref, hits_ref) = (&team, &hits);
    std::thread::scope(|s| {
        s.spawn(move || {
            run_section(team_ref, hits_ref);
            let _ = done_tx.send(());
        });
        wake.wait_reached();
        // The member now re-checks: it must see the new generation.
        raised.release();
        std::thread::sleep(Duration::from_millis(50));
        wake.release();
        assert!(done_rx.recv_timeout(Duration::from_secs(2)).is_ok());
    });
    assert_eq!(hits.load(Ordering::SeqCst), 4);
}
