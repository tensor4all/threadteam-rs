//! Primitives of the park/wake handshake and of the submission latch: `std`,
//! or `loom` under `--cfg loom` so that the exact handshake code can be
//! model-checked.

#[cfg(loom)]
pub(crate) use loom::sync::{
    atomic::{AtomicBool, AtomicUsize, Ordering},
    Condvar, Mutex,
};
#[cfg(loom)]
pub(crate) use loom::thread;
#[cfg(not(loom))]
pub(crate) use std::sync::{
    atomic::{AtomicBool, AtomicUsize, Ordering},
    Condvar, Mutex,
};
#[cfg(not(loom))]
pub(crate) use std::thread;
