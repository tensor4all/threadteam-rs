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
        done.wait(Some(&sync), &mut Waiter::new(0));
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
                    assert!(barrier.arrive(&cancelled, Some(&sync), &mut Waiter::new(0)));
                }
            })
        };
        for _ in 0..2 {
            assert!(barrier.arrive(&cancelled, Some(&sync), &mut Waiter::new(0)));
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
            thread::spawn(move || barrier.arrive(&cancelled, Some(&sync), &mut Waiter::new(0)))
        };
        // `Section::fail`: raise the flag (SeqCst), then wake.
        cancelled.store(true, Ordering::SeqCst);
        sync.wake();
        assert!(!waiter.join().unwrap());
    });
}

// ---------------------------------------------------------------------------
// The native scheduler (`docs/design/own-threads.md`): the slot state word,
// the queue of stack batches, the broadcast list, and an idle worker's sleep
// against a publication. The exact code, under `loom`.

use threadteam::{Batch, Broadcast, BroadcastList, Queue, SlotState};

/// A team reservation and a worker taking work race for a free slot: exactly
/// one wins, and a release or a detach while the worker is busy neither frees
/// it early nor is erased when it finishes.
#[test]
fn a_free_slot_is_reserved_or_taken_never_both() {
    model(|| {
        let s = Arc::new(SlotState::new());
        let leader = {
            let s = s.clone();
            thread::spawn(move || s.try_reserve())
        };
        let took = s.try_take();
        let reserved = leader.join().unwrap();
        assert!(took != reserved, "exactly one wins");
        if took {
            // Busy: a detach arrives, then the work ends.
            s.detach();
            s.end_busy();
            assert_eq!(s.load(), 8, "DETACH survives the end of the work");
        } else {
            s.release();
            assert_eq!(s.load(), 0);
        }
    });
}

/// A lent member that is running a task when its reservation is released
/// stays busy (not claimable) until the task ends.
#[test]
fn a_release_does_not_free_a_busy_lent_member() {
    model(|| {
        let s = Arc::new(SlotState::new());
        assert!(s.try_reserve());
        s.set_lent(true);
        assert!(s.try_take());
        let releaser = {
            let s = s.clone();
            thread::spawn(move || s.release())
        };
        let claimer = {
            let s = s.clone();
            thread::spawn(move || s.try_reserve())
        };
        releaser.join().unwrap();
        assert!(!claimer.join().unwrap(), "a busy worker is not reservable");
        s.end_busy();
        assert!(s.try_reserve(), "free again once the task ended");
    });
}

unsafe fn count_task(data: *const (), _i: usize) {
    (*data.cast::<AtomicUsize>()).fetch_add(1, Ordering::SeqCst);
}

/// An owner publishes a batch of two tasks; two workers claim and run them
/// while the owner claims its own and then sleeps until the last completion:
/// every task runs exactly once and the owner wakes (no lost wake-up).
#[test]
fn a_batch_runs_every_task_once_and_wakes_its_sleeping_owner() {
    model(|| {
        let q = Arc::new(Queue::new());
        let ran = Arc::new(AtomicUsize::new(0));
        let waker = std::sync::Arc::new(Parker::new());
        // SAFETY: `ran` and `waker` outlive the batch, which the owner waits
        // for below.
        let batch =
            Arc::new(unsafe { Batch::new(count_task, Arc::as_ptr(&ran).cast(), 2, &waker) });
        q.publish(&batch);
        let ws: Vec<_> = (0..2)
            .map(|_| {
                let q = q.clone();
                thread::spawn(move || {
                    if let Some((b, i)) = q.claim() {
                        unsafe { Batch::run_claimed(b, i) };
                    }
                })
            })
            .collect();
        while let Some(i) = q.claim_own(&batch) {
            unsafe { Batch::run_claimed(Arc::as_ptr(&batch), i) };
        }
        batch.done.wait(Some(&waker), &mut Waiter::new(0));
        assert_eq!(ran.load(Ordering::SeqCst), 2);
        for w in ws {
            w.join().unwrap();
        }
    });
}

unsafe fn share_task(data: *const (), i: usize) {
    (*data.cast::<[AtomicUsize; 2]>())[i].fetch_add(1, Ordering::SeqCst);
}

/// A broadcast to two workers: each runs its own share exactly once, in the
/// order of its cursor, and the caller wakes before it unlinks the broadcast.
#[test]
fn a_broadcast_runs_each_share_once_and_wakes_its_caller() {
    model(|| {
        let list = Arc::new(BroadcastList::new());
        let shares = Arc::new([AtomicUsize::new(0), AtomicUsize::new(0)]);
        let waker = std::sync::Arc::new(Parker::new());
        // SAFETY: as above.
        let b =
            Arc::new(unsafe { Broadcast::new(share_task, Arc::as_ptr(&shares).cast(), 2, &waker) });
        list.publish(&b);
        let ws: Vec<_> = (0..2)
            .map(|i| {
                let list = list.clone();
                thread::spawn(move || {
                    let mut seq = 0;
                    while let Some(b) = list.next_for(i, &mut seq) {
                        unsafe { Broadcast::run_share(b, i) };
                    }
                })
            })
            .collect();
        b.done.wait(Some(&waker), &mut Waiter::new(0));
        list.unlink(&b);
        assert_eq!(shares[0].load(Ordering::SeqCst), 1);
        assert_eq!(shares[1].load(Ordering::SeqCst), 1);
        for w in ws {
            w.join().unwrap();
        }
    });
}

/// An idle worker going to sleep on its parker against a publication: the
/// publisher links the batch and wakes; the worker either sees the new work
/// before it sleeps or is woken (no lost wake-up), then claims it.
#[test]
fn an_idle_worker_is_never_left_asleep_by_a_publication() {
    model(|| {
        let q = Arc::new(Queue::new());
        let park = Arc::new(Parker::new());
        let ran = Arc::new(AtomicUsize::new(0));
        let waker = std::sync::Arc::new(Parker::new());
        let worker = {
            let (q, park) = (q.clone(), park.clone());
            thread::spawn(move || {
                let g = q.gen();
                park.park_while(|| q.gen() == g && !q.has_work());
                if let Some((b, i)) = q.claim() {
                    unsafe { Batch::run_claimed(b, i) };
                }
            })
        };
        // SAFETY: as above.
        let batch = unsafe { Batch::new(count_task, Arc::as_ptr(&ran).cast(), 1, &waker) };
        q.publish(&batch);
        park.wake();
        worker.join().unwrap();
        // The worker claimed it, or it is still linked for the owner.
        while let Some(i) = q.claim_own(&batch) {
            unsafe { Batch::run_claimed(&batch, i) };
        }
        batch.done.wait(Some(&waker), &mut Waiter::new(0));
        assert_eq!(ran.load(Ordering::SeqCst), 1);
    });
}
