//! The staged wait of `crate::park::Waiter` (`docs/design/waiting.md`).
//!
//! Nothing here depends on how the scheduler runs this thread: the pressure
//! deadlines are set explicitly, and the only real delay (a sleep) can only make
//! a measured gap longer.

use crate::park::{Pressure, Step, Waiter, CONTENDED_HOLD_NS, THROTTLED_POLLS};
use std::time::Duration;

/// Steps until the waiter says sleep; the number of `Poll` steps before it.
fn polls_until_sleep(w: &mut Waiter<'_>) -> u32 {
    let mut n = 0;
    while w.step() == Step::Poll {
        n += 1;
        assert!(n < 10_000_000, "never sleeps");
    }
    n
}

#[test]
fn a_wait_spins_its_budget_then_yields_sixteen_times_then_sleeps() {
    for budget in [0u32, 1, 63, 64, 1000] {
        let mut w = Waiter::new(budget, None);
        assert_eq!(polls_until_sleep(&mut w), budget + 16, "budget {budget}");
        // It stays asleep-ready: every further step says sleep.
        assert_eq!(w.step(), Step::Sleep);
    }
}

#[test]
fn a_pool_under_pressure_starts_its_waits_throttled() {
    // Throttle on the first observation, with a deadline that cannot pass
    // during the test.
    let p = Pressure::tuned(THROTTLED_POLLS, CONTENDED_HOLD_NS, 1, 0);
    p.note(u64::MAX - CONTENDED_HOLD_NS);
    let mut throttled = Waiter::new(100_000, Some(&p));
    assert_eq!(polls_until_sleep(&mut throttled), THROTTLED_POLLS + 16);
    // A small budget is never raised to the throttled one.
    let mut small = Waiter::new(70, Some(&p));
    assert_eq!(polls_until_sleep(&mut small), 70 + 16);
}

#[test]
fn a_descheduled_spinner_ends_its_spin_and_records_the_pressure() {
    let p = Pressure::new();
    assert_eq!(p.events(), 0);
    let mut w = Waiter::new(1_000_000, Some(&p));
    // The first clock read is at poll 64; a gap longer than 200 µs before the
    // second one (poll 128) stands for a descheduled waiter.
    for _ in 0..64 {
        assert_eq!(w.step(), Step::Poll);
    }
    std::thread::sleep(Duration::from_millis(2));
    for _ in 64..128 {
        assert_eq!(w.step(), Step::Poll);
    }
    // The spin is over: what remains is the sixteen yields.
    assert_eq!(polls_until_sleep(&mut w), 16);
    assert_eq!(p.events(), 1, "the pool records the observation");
}

#[test]
fn a_burst_throttles_the_pool_and_isolated_preemptions_do_not() {
    // Four observations within the hold window throttle the pool (the
    // default threshold).
    let p = Pressure::tuned(THROTTLED_POLLS, CONTENDED_HOLD_NS, 4, 0);
    let t = 1_000_000_000;
    for k in 0..3 {
        p.note(t + k * 1_000_000);
        assert!(!p.active(t + k * 1_000_000), "observation {k} alone");
    }
    p.note(t + 3_000_000);
    assert!(p.active(t + 3_000_000), "the fourth within the window");
    assert!(
        !p.active(t + 3_000_000 + CONTENDED_HOLD_NS),
        "the hold ends"
    );
    // Observations spread wider than the window never add up.
    let q = Pressure::tuned(THROTTLED_POLLS, CONTENDED_HOLD_NS, 4, 0);
    for k in 0..10 {
        let at = t + k * (CONTENDED_HOLD_NS + 1);
        q.note(at);
        assert!(!q.active(at), "isolated observation {k}");
    }
}

#[test]
fn with_a_per_thread_hold_a_preemption_throttles_the_thread_that_saw_it() {
    use core::sync::atomic::AtomicU64;
    // The per-thread hold is a benchmark knob (off by default); the pool
    // threshold is off here so that only the thread's own deadline acts.
    let p = Pressure::tuned(THROTTLED_POLLS, CONTENDED_HOLD_NS, 0, CONTENDED_HOLD_NS);
    let own = AtomicU64::new(0);
    // A member's wait sees itself descheduled (a gap before its second clock
    // read): its spin ends, and its slot's deadline is set.
    let mut w = Waiter::of_thread(1_000_000, Some(&p), Some(&own));
    for _ in 0..64 {
        assert_eq!(w.step(), Step::Poll);
    }
    std::thread::sleep(Duration::from_millis(2));
    for _ in 64..128 {
        assert_eq!(w.step(), Step::Poll);
    }
    assert_eq!(polls_until_sleep(&mut w), 16);
    assert!(own.load(core::sync::atomic::Ordering::Relaxed) > 0);
    // The pool is not throttled: another thread's wait spins its full budget.
    let other_own = AtomicU64::new(0);
    let mut other = Waiter::of_thread(100_000, Some(&p), Some(&other_own));
    assert_eq!(polls_until_sleep(&mut other), 100_000 + 16);
    // The same thread's next wait is throttled while its deadline runs (set
    // here to one that cannot pass during the test).
    own.store(u64::MAX, core::sync::atomic::Ordering::Relaxed);
    let mut again = Waiter::of_thread(100_000, Some(&p), Some(&own));
    assert_eq!(polls_until_sleep(&mut again), THROTTLED_POLLS + 16);
}

#[test]
fn by_default_a_preemption_sets_no_per_thread_deadline() {
    use core::sync::atomic::{AtomicU64, Ordering};
    let p = Pressure::new();
    let own = AtomicU64::new(0);
    let mut w = Waiter::of_thread(1_000_000, Some(&p), Some(&own));
    for _ in 0..64 {
        w.step();
    }
    std::thread::sleep(Duration::from_millis(2));
    for _ in 64..128 {
        w.step();
    }
    assert_eq!(polls_until_sleep(&mut w), 16, "its own spin ends");
    assert_eq!(
        own.load(Ordering::Relaxed),
        0,
        "no lasting per-thread deadline"
    );
}
