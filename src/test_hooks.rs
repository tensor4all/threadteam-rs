//! Test-only hooks into the park/wake handshake (feature `test-hooks`).
//!
//! A hook pauses a thread at a chosen point of the handshake so that a test
//! can force an interleaving; a mutation breaks one rule of the protocol to
//! show that the forced interleaving does catch the breakage. Global state:
//! tests that use it must run one at a time (they live in their own test
//! binary and take a lock). Not part of the supported API.

pub use crate::park::{Mutation, Point};
use crate::pool::Pool;
use std::sync::atomic::{AtomicU8, Ordering};
use std::sync::{Arc, PoisonError, RwLock};

type Hook = Arc<dyn Fn(Point) + Send + Sync>;

static HOOK: RwLock<Option<Hook>> = RwLock::new(None);
static MUTATION: AtomicU8 = AtomicU8::new(0);

/// Installs (or clears) the hook called at every [`Point`], on the thread that
/// reaches it.
pub fn set_hook(hook: Option<Hook>) {
    *HOOK.write().unwrap_or_else(PoisonError::into_inner) = hook;
}

/// Selects a protocol breakage (or none).
pub fn set_mutation(m: Mutation) {
    MUTATION.store(
        match m {
            Mutation::None => 0,
            Mutation::SkipRecheck => 1,
            Mutation::NoLockHandoff => 2,
        },
        Ordering::SeqCst,
    );
}

pub(crate) fn mutation() -> Mutation {
    match MUTATION.load(Ordering::SeqCst) {
        1 => Mutation::SkipRecheck,
        2 => Mutation::NoLockHandoff,
        _ => Mutation::None,
    }
}

pub(crate) fn fire(point: Point) {
    let hook = HOOK.read().unwrap_or_else(PoisonError::into_inner).clone();
    if let Some(hook) = hook {
        hook(point);
    }
}

/// A correct wake-up of every parked member of `pool`, to recover a test that
/// deliberately stalled the protocol.
pub fn wake_parked(pool: &Pool) {
    pool.awake.wake_parked();
}
