//! Model check of the park/wake handshake of reserved members.
//!
//! Run with `RUSTFLAGS="--cfg loom" cargo test --release --test loom_park`.
//! The model is the real `Parker` (compiled against `loom` primitives): one
//! leader and two members that decided to park, one change of the state they
//! wait on, and the wake-up. Loom explores the interleavings; a lost wake-up
//! shows up as a deadlock of the leader's join.
#![cfg(loom)]

use loom::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use loom::sync::Arc;
use loom::thread;
use threadteam::Parker;

const IDLE: usize = 1;
const RESERVED: usize = 3;

fn model(f: impl Fn() + Sync + Send + 'static) {
    let mut b = loom::model::Builder::new();
    b.preemption_bound = Some(3);
    b.check(f);
}

/// A member that found no work: waits while the generation is unchanged and
/// the reservation holds.
fn member(
    gen: &Arc<AtomicUsize>,
    state: &Arc<AtomicUsize>,
    parker: &Arc<Parker>,
) -> thread::JoinHandle<()> {
    let (gen, state, parker) = (gen.clone(), state.clone(), parker.clone());
    thread::spawn(move || {
        parker.park_while(|| {
            gen.load(Ordering::SeqCst) == 0 && state.load(Ordering::SeqCst) == RESERVED
        });
    })
}

#[test]
fn a_published_section_wakes_every_parking_member() {
    model(|| {
        let gen = Arc::new(AtomicUsize::new(0));
        let state = Arc::new(AtomicUsize::new(RESERVED));
        let parker = Arc::new(Parker::new());
        let ms = [member(&gen, &state, &parker), member(&gen, &state, &parker)];
        // Leader: publish, then wake.
        gen.fetch_add(1, Ordering::SeqCst);
        parker.wake();
        for m in ms {
            m.join().unwrap();
        }
    });
}

#[test]
fn releasing_the_reservation_wakes_every_parking_member() {
    model(|| {
        let gen = Arc::new(AtomicUsize::new(0));
        let state = Arc::new(AtomicUsize::new(RESERVED));
        let parker = Arc::new(Parker::new());
        let ms = [member(&gen, &state, &parker), member(&gen, &state, &parker)];
        state.store(IDLE, Ordering::SeqCst);
        parker.wake();
        for m in ms {
            m.join().unwrap();
        }
    });
}

#[test]
fn a_member_that_parks_after_the_first_wake_is_woken_by_the_second() {
    model(|| {
        let gen = Arc::new(AtomicUsize::new(0));
        let state = Arc::new(AtomicUsize::new(RESERVED));
        let parker = Arc::new(Parker::new());
        let m = {
            let (gen, state, parker) = (gen.clone(), state.clone(), parker.clone());
            thread::spawn(move || {
                // Two rounds: wait for generation 1, then for generation 2.
                for round in 1..=2 {
                    parker.park_while(|| {
                        gen.load(Ordering::SeqCst) < round
                            && state.load(Ordering::SeqCst) == RESERVED
                    });
                }
            })
        };
        gen.fetch_add(1, Ordering::SeqCst);
        parker.wake();
        gen.fetch_add(1, Ordering::SeqCst);
        parker.wake();
        m.join().unwrap();
    });
}

/// Per-slot parkers: a leader wakes only the slot it published to; the member
/// of another slot stays asleep until its own state changes, and is then woken
/// by its own parker (no lost wake-up on either slot).
#[test]
fn a_wake_of_one_slot_leaves_the_other_asleep_until_its_own_change() {
    model(|| {
        let gens = [Arc::new(AtomicUsize::new(0)), Arc::new(AtomicUsize::new(0))];
        let state = Arc::new(AtomicUsize::new(RESERVED));
        let parkers = [Arc::new(Parker::new()), Arc::new(Parker::new())];
        let ms: Vec<_> = (0..2)
            .map(|i| {
                let (gen, state, parker) = (gens[i].clone(), state.clone(), parkers[i].clone());
                thread::spawn(move || {
                    parker.park_while(|| {
                        gen.load(Ordering::SeqCst) == 0 && state.load(Ordering::SeqCst) == RESERVED
                    });
                    // Woken by its own slot: the generation moved or the
                    // reservation ended.
                    assert!(
                        gen.load(Ordering::SeqCst) != 0 || state.load(Ordering::SeqCst) != RESERVED
                    );
                })
            })
            .collect();
        // Section addressed to slot 0 only.
        gens[0].fetch_add(1, Ordering::SeqCst);
        parkers[0].wake();
        // Release: every slot changed, every parker is woken.
        state.store(IDLE, Ordering::SeqCst);
        for p in &parkers {
            p.wake();
        }
        for m in ms {
            m.join().unwrap();
        }
    });
}

// ---------------------------------------------------------------------------
// The staged waits of a section (`docs/design/waiting.md`): the exact
// `Completion` and `Barrier` code, with a zero spin budget so every waiter goes
// straight to sleep on the shared parker.

use threadteam::{Barrier, Completion, Waiter};

/// The leader sleeps in `join` until the last of two members finishes; the
/// member that brings the count to zero wakes it. A lost wake-up deadlocks.
#[test]
fn the_last_member_wakes_a_leader_asleep_in_join() {
    model(|| {
        let done = Arc::new(Completion::new(2));
        let sync = Arc::new(Parker::new());
        let ms: Vec<_> = (0..2)
            .map(|_| {
                let (done, sync) = (done.clone(), sync.clone());
                thread::spawn(move || unsafe {
                    Completion::finish(Arc::as_ptr(&done), Some(&sync))
                })
            })
            .collect();
        done.wait(Some(&sync), &mut Waiter::new(0, None));
        assert!(done.is_done());
        for m in ms {
            m.join().unwrap();
        }
    });
}

/// Two participants pass two consecutive barrier rounds, sleeping on the
/// shared parker: the reset of `arrived` and the epoch bump of one round never
/// lose the other's arrival or wake-up in the next.
#[test]
fn consecutive_barrier_rounds_wake_every_sleeper() {
    model(|| {
        let barrier = Arc::new(Barrier::new(2));
        let cancelled = Arc::new(AtomicBool::new(false));
        let sync = Arc::new(Parker::new());
        let peer = {
            let (barrier, cancelled, sync) = (barrier.clone(), cancelled.clone(), sync.clone());
            thread::spawn(move || {
                for _ in 0..2 {
                    assert!(barrier.arrive(&cancelled, Some(&sync), &mut Waiter::new(0, None)));
                }
            })
        };
        for _ in 0..2 {
            assert!(barrier.arrive(&cancelled, Some(&sync), &mut Waiter::new(0, None)));
        }
        peer.join().unwrap();
    });
}

/// A participant asleep in a barrier that can never complete (the third
/// participant failed) is woken by the cancellation and leaves with `false`.
#[test]
fn a_cancellation_wakes_a_participant_asleep_in_a_barrier() {
    model(|| {
        let barrier = Arc::new(Barrier::new(3));
        let cancelled = Arc::new(AtomicBool::new(false));
        let sync = Arc::new(Parker::new());
        let waiter = {
            let (barrier, cancelled, sync) = (barrier.clone(), cancelled.clone(), sync.clone());
            thread::spawn(move || {
                barrier.arrive(&cancelled, Some(&sync), &mut Waiter::new(0, None))
            })
        };
        // `Section::fail`: raise the flag (SeqCst), then wake.
        cancelled.store(true, Ordering::SeqCst);
        sync.wake();
        assert!(!waiter.join().unwrap());
    });
}
