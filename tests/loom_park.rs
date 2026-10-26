//! Model check of the park/wake handshake of reserved members.
//!
//! Run with `RUSTFLAGS="--cfg loom" cargo test --release --test loom_park`.
//! The model is the real `Parker` (compiled against `loom` primitives): one
//! leader and two members that decided to park, one change of the state they
//! wait on, and the wake-up. Loom explores the interleavings; a lost wake-up
//! shows up as a deadlock of the leader's join.
#![cfg(loom)]

use loom::sync::atomic::{AtomicUsize, Ordering};
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
