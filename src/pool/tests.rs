use super::*;
use std::sync::{mpsc, Arc};
use std::time::Duration;

/// The last reference is dropped on a worker of the pool itself: the drop
/// must not wait for its own thread.
#[test]
fn drop_from_a_worker_of_the_pool_does_not_hang() {
    let pool = Arc::new(Pool::builder().threads(2).build().unwrap());
    let last = Arc::clone(&pool);
    let (gate_tx, gate_rx) = mpsc::channel::<()>();
    let (done_tx, done_rx) = mpsc::channel::<()>();
    pool.inner().spawn(move || {
        gate_rx.recv().unwrap();
        drop(last); // the last reference: runs `Pool::drop` on this worker
        done_tx.send(()).unwrap();
    });
    drop(pool);
    gate_tx.send(()).unwrap();
    done_rx
        .recv_timeout(Duration::from_secs(10))
        .expect("drop on a worker hung");
}

/// The cache never holds more threads than its cap, whatever the number
/// of pools dropped.
#[test]
fn the_cache_keeps_at_most_its_cap_of_threads() {
    let cap = std::thread::available_parallelism().map_or(1, NonZeroUsize::get);
    let pools: Vec<Pool> = (0..cap + 2)
        .map(|_| Pool::builder().threads(1).build().unwrap())
        .collect();
    drop(pools);
    // Other tests of this binary build wider pools concurrently, which
    // can only raise the cap.
    let cached = cache::cached_threads();
    let cap = cap.max(cache::widest());
    assert!(cached <= cap, "{cached} cached threads, cap {cap}");
}
