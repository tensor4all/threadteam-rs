//! Primitives of the park/wake handshake: `std`, or `loom` under `--cfg loom`
//! so that the exact handshake code can be model-checked.

#[cfg(loom)]
pub(crate) use loom::sync::{
    atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering},
    Condvar, Mutex,
};
#[cfg(not(loom))]
pub(crate) use std::sync::{
    atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering},
    Condvar, Mutex,
};
