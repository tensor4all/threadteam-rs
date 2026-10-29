//! The staged wait of `crate::park::Waiter` (`docs/design/waiting.md`).

use crate::park::{Step, Waiter};

/// Steps until the waiter says sleep; the number of `Poll` steps before it.
fn polls_until_sleep(w: &mut Waiter) -> u32 {
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
        let mut w = Waiter::new(budget);
        assert_eq!(polls_until_sleep(&mut w), budget + 16, "budget {budget}");
        assert!(w.slept());
        // It stays asleep-ready: every further step says sleep.
        assert_eq!(w.step(), Step::Sleep);
    }
}

#[test]
fn a_descheduled_spinner_keeps_its_budget() {
    // Nothing reads a clock: a waiter that was off its CPU for a while spins
    // exactly as many polls as one that was not (libgomp's waits).
    let mut w = Waiter::new(1_000);
    for _ in 0..64 {
        assert_eq!(w.step(), Step::Poll);
    }
    std::thread::sleep(std::time::Duration::from_millis(2));
    assert_eq!(polls_until_sleep(&mut w), 1_000 - 64 + 16);
}
