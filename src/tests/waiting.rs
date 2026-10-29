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
    let p = Pressure::tuned(THROTTLED_POLLS, CONTENDED_HOLD_NS, 1);
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
fn one_preemption_throttles_only_its_waiter_and_a_burst_throttles_the_pool() {
    let p = Pressure::new(); // four observations within 100 ms throttle the pool
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
    let q = Pressure::new();
    for k in 0..10 {
        let at = t + k * (CONTENDED_HOLD_NS + 1);
        q.note(at);
        assert!(!q.active(at), "isolated observation {k}");
    }
}
