//! The owner checks of `park::Latch` (debug builds only, where they exist).

use crate::park::Latch;

/// A latch is waited on by the thread that created it. The latch is set first,
/// so without the check the wait would return at once instead of panicking.
#[test]
#[should_panic(expected = "a latch is waited on by the thread that created it")]
fn a_latch_waited_on_by_another_thread_panics() {
    let latch = Latch::new();
    // SAFETY: `latch` outlives the call, and the calling thread created it.
    unsafe { Latch::set(&latch, std::thread::current()) };
    let waited = std::thread::scope(|s| s.spawn(|| latch.wait()).join());
    std::panic::resume_unwind(waited.unwrap_err());
}

/// A latch is owned by the thread that creates it.
#[test]
#[should_panic(expected = "a latch is owned by the thread that creates it")]
fn a_latch_owned_by_another_thread_panics() {
    let other = std::thread::spawn(|| {});
    let stranger = other.thread().clone();
    other.join().unwrap();
    let _ = Latch::owned_by(&stranger);
}

/// A latch wakes the thread that created it.
#[test]
#[should_panic(expected = "a latch wakes the thread that created it")]
fn a_latch_set_for_another_thread_panics() {
    let latch = Latch::new();
    let other = std::thread::spawn(|| {});
    let stranger = other.thread().clone();
    other.join().unwrap();
    // SAFETY: `latch` outlives the call; the waiter is deliberately wrong.
    unsafe { Latch::set(&latch, stranger) };
}
