//! Feature-gated trace of what each reserved member does over time (feature
//! `trace`).
//!
//! Every state transition of a member loop is stamped with the time since the
//! pool was built and stored in a preallocated ring (one per member slot, one
//! extra for leaders), so recording allocates nothing and costs two atomic
//! stores. With the feature off none of this exists. Not part of the supported
//! API.
//!
//! A member's timeline reads: `Enter` (loop started), then repeatedly
//! `SectionBegin` / `SectionEnd` around a share, `ParkEnter` / `ParkExit`
//! around a sleep on the park condition variable, `LentBegin` / `LentEnd`
//! while the thread serves Rayon, and `Leave`. The time from `Enter` or
//! `SectionEnd` to the next `ParkEnter` is spent spinning; a leader records
//! `Publish` when it bumps the generations of a section.

use crate::pool::Pool;
use core::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::time::Instant;

/// Capacity of each ring; older events are overwritten.
const CAPACITY: usize = 1 << 14;

/// What happened.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    /// The member loop started on this worker.
    Enter = 1,
    /// A share started.
    SectionBegin,
    /// A share finished.
    SectionEnd,
    /// The member is about to sleep on the park condition variable.
    ParkEnter,
    /// The member woke up.
    ParkExit,
    /// The thread was lent to Rayon.
    LentBegin,
    /// The thread was taken back.
    LentEnd,
    /// The loop returned the thread to Rayon.
    Leave,
    /// A leader published a section (recorded in the leader ring).
    Publish,
}

impl Kind {
    fn from_u64(v: u64) -> Option<Kind> {
        Some(match v {
            1 => Kind::Enter,
            2 => Kind::SectionBegin,
            3 => Kind::SectionEnd,
            4 => Kind::ParkEnter,
            5 => Kind::ParkExit,
            6 => Kind::LentBegin,
            7 => Kind::LentEnd,
            8 => Kind::Leave,
            9 => Kind::Publish,
            _ => return None,
        })
    }
}

/// One recorded transition.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Event {
    /// Member slot index, or `usize::MAX` for the leader ring.
    pub member: usize,
    /// Nanoseconds since the pool was built.
    pub ns: u64,
    /// The transition.
    pub kind: Kind,
}

struct Ring {
    head: AtomicUsize,
    buf: Box<[AtomicU64]>,
}

impl Ring {
    fn new() -> Self {
        Ring {
            head: AtomicUsize::new(0),
            buf: (0..CAPACITY).map(|_| AtomicU64::new(0)).collect(),
        }
    }
}

pub(crate) struct Trace {
    base: Instant,
    rings: Box<[Ring]>,
}

impl Trace {
    pub(crate) fn new(width: usize) -> Self {
        Trace {
            base: Instant::now(),
            rings: (0..=width).map(|_| Ring::new()).collect(),
        }
    }

    /// Records `kind` for member `id` (`None`: the leader ring).
    pub(crate) fn record(&self, id: Option<usize>, kind: Kind) {
        let ring = &self.rings[id.unwrap_or(self.rings.len() - 1)];
        let ns = u64::try_from(self.base.elapsed().as_nanos()).unwrap_or(u64::MAX >> 4);
        let i = ring.head.fetch_add(1, Ordering::Relaxed) % CAPACITY;
        ring.buf[i].store((ns << 4) | kind as u64, Ordering::Relaxed);
    }

    fn events(&self) -> Vec<Event> {
        let mut out = Vec::new();
        for (r, ring) in self.rings.iter().enumerate() {
            let head = ring.head.load(Ordering::Relaxed);
            let member = if r + 1 == self.rings.len() {
                usize::MAX
            } else {
                r
            };
            let (from, to) = (head.saturating_sub(CAPACITY), head);
            for k in from..to {
                let v = ring.buf[k % CAPACITY].load(Ordering::Relaxed);
                if let Some(kind) = Kind::from_u64(v & 0xF) {
                    out.push(Event {
                        member,
                        ns: v >> 4,
                        kind,
                    });
                }
            }
        }
        out.sort_by_key(|e| e.ns);
        out
    }

    fn clear(&self) {
        for ring in self.rings.iter() {
            ring.head.store(0, Ordering::Relaxed);
        }
    }

    pub(crate) fn now_ns(&self) -> u64 {
        u64::try_from(self.base.elapsed().as_nanos()).unwrap_or(u64::MAX >> 4)
    }
}

/// All recorded events of `pool`, ordered by time. Call while no section runs.
pub fn events(pool: &Pool) -> Vec<Event> {
    pool.awake.trace.events()
}

/// Forgets the recorded events of `pool`.
pub fn clear(pool: &Pool) {
    pool.awake.trace.clear();
}

/// Nanoseconds since `pool` was built, on the clock of the events.
pub fn now_ns(pool: &Pool) -> u64 {
    pool.awake.trace.now_ns()
}
