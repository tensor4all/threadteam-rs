//! Where does the latency of the first dispatch after a park go?
//!
//! For each width and QoS: time from the leader's call to every participant
//! starting its share, for members that were spinning (short pause) and for
//! members that were parked (long pause), against a plain `Exec::fork` after the
//! same pause (Rayon workers asleep). Prints the median total, the median of
//! the slowest member, and how often some member started more than 1 ms late.

use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};
use threadteam::{Exec, Pool, Qos};

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

#[cfg(target_vendor = "apple")]
fn qos_of_current() -> String {
    let mut class = libc::qos_class_t::QOS_CLASS_UNSPECIFIED;
    let mut rel = 0;
    unsafe { libc::pthread_get_qos_class_np(libc::pthread_self(), &mut class, &mut rel) };
    format!("{class:?}")
}
#[cfg(not(target_vendor = "apple"))]
fn qos_of_current() -> String {
    "n/a".into()
}

struct Stats {
    total: f64,
    slowest: f64,
    late: f64,
    max: f64,
}

fn probe(w: usize, qos: Option<Qos>, pause: Duration, team: bool) -> Stats {
    let mut b = Pool::builder()
        .threads(w)
        .idle_spin(Duration::from_millis(2))
        .acquire_timeout(Duration::from_secs(5));
    if let Some(q) = qos {
        b = b.qos(q);
    }
    let pool = b.build().unwrap();
    let base = Instant::now();
    let starts: Vec<AtomicU64> = (0..w).map(|_| AtomicU64::new(0)).collect();
    // The plain variant gets its own pool: a pool whose members are reserved
    // would leave plain tokens a single free worker.
    let (tm, pool) = if team {
        (Some(pool.into_team(w).unwrap()), None)
    } else {
        (None, Some(pool))
    };
    let mut texec = tm.as_ref().map(|t| t.exec(w));
    let mut plain = pool.as_ref().map(|p| Exec::new(p, w).unwrap());
    let (mut totals, mut slowest, mut late, mut max) = (vec![], vec![], 0usize, 0f64);
    let trials = 100;
    for _ in 0..trials + 10 {
        std::thread::sleep(pause);
        for s in &starts {
            s.store(0, Ordering::Relaxed);
        }
        let t0 = base.elapsed().as_nanos() as u64;
        let mark = |i: usize| starts[i].store(base.elapsed().as_nanos() as u64, Ordering::Release);
        if team {
            texec.as_mut().unwrap().reborrow().broadcast(mark).unwrap();
        } else {
            plain.as_mut().unwrap().reborrow().fork(w, |i, _| mark(i));
        }
        let t1 = base.elapsed().as_nanos() as u64;
        let lat: Vec<f64> = starts
            .iter()
            .map(|s| s.load(Ordering::Acquire).saturating_sub(t0) as f64)
            .collect();
        totals.push((t1 - t0) as f64);
        let m = lat.iter().cloned().fold(0.0, f64::max);
        slowest.push(m);
        max = max.max(m);
        if m > 1e6 {
            late += 1;
        }
    }
    totals.drain(..10);
    slowest.drain(..10);
    drop(texec);
    Stats {
        total: median(totals),
        slowest: median(slowest),
        late: late as f64 / (trials + 10) as f64,
        max,
    }
}

/// Raw `thread::park` / `unpark` on plain threads, no pool involved: how long a
/// serial unpark takes and how long a parked thread needs to run again.
fn raw_park(n: usize) {
    use std::sync::atomic::AtomicBool;
    use std::sync::Arc;
    let base = Instant::now();
    let go = Arc::new(AtomicBool::new(false));
    let woke: Arc<Vec<AtomicU64>> = Arc::new((0..n).map(|_| AtomicU64::new(0)).collect());
    let stop = Arc::new(AtomicBool::new(false));
    let handles: Vec<_> = (0..n)
        .map(|i| {
            let (go, woke, stop) = (go.clone(), woke.clone(), stop.clone());
            std::thread::spawn(move || {
                while !stop.load(Ordering::Acquire) {
                    while !go.load(Ordering::Acquire) {
                        if stop.load(Ordering::Acquire) {
                            return;
                        }
                        std::thread::park();
                    }
                    woke[i].store(base.elapsed().as_nanos() as u64, Ordering::Release);
                    while go.load(Ordering::Acquire) {
                        std::thread::yield_now();
                    }
                }
            })
        })
        .collect();
    let threads: Vec<_> = handles.iter().map(|h| h.thread().clone()).collect();
    let (mut call, mut first, mut last) = (vec![], vec![], vec![]);
    for _ in 0..110 {
        std::thread::sleep(Duration::from_millis(12));
        for w in woke.iter() {
            w.store(0, Ordering::Relaxed);
        }
        let t0 = base.elapsed().as_nanos() as u64;
        go.store(true, Ordering::SeqCst);
        for t in &threads {
            t.unpark();
        }
        let t1 = base.elapsed().as_nanos() as u64;
        while woke.iter().any(|w| w.load(Ordering::Acquire) == 0) {
            std::hint::spin_loop();
        }
        let lat: Vec<u64> = woke
            .iter()
            .map(|w| w.load(Ordering::Acquire) - t0)
            .collect();
        call.push((t1 - t0) as f64);
        first.push(*lat.iter().min().unwrap() as f64);
        last.push(*lat.iter().max().unwrap() as f64);
        go.store(false, Ordering::SeqCst);
        std::thread::sleep(Duration::from_millis(1));
    }
    stop.store(true, Ordering::SeqCst);
    for t in &threads {
        t.unpark();
    }
    for h in handles {
        h.join().unwrap();
    }
    for v in [&mut call, &mut first, &mut last] {
        v.drain(..10);
    }
    println!(
        "{n:>5} {:>14.1} {:>14.1} {:>14.1}",
        median(call) / 1e3,
        median(first) / 1e3,
        median(last) / 1e3
    );
}

fn main() {
    let mut l = [0f64; 3];
    unsafe { libc::getloadavg(l.as_mut_ptr(), 3) };
    println!(
        "load average at start {l:?}; caller thread QoS: {}",
        qos_of_current()
    );
    println!();
    println!("raw park/unpark on plain threads (serial unpark), microseconds");
    println!(
        "{:>5} {:>14} {:>14} {:>14}",
        "n", "unpark calls", "first runs", "last runs"
    );
    for n in [1usize, 2, 6, 12, 17] {
        raw_park(n);
    }
    println!();
    println!("microseconds; total = call to return, slowest = call to the last share starting");
    println!(
        "{:>5} {:<22} {:<8} {:>9} {:>9} {:>8} {:>10}",
        "width", "variant", "state", "total", "slowest", "late>1ms", "max"
    );
    for w in [2usize, 6, 12, 18] {
        for (label, qos) in [
            ("team qos=builder-default", None),
            ("team qos=Default", Some(Qos::Default)),
            ("team qos=UserInteractive", Some(Qos::UserInteractive)),
        ] {
            for (state, pause) in [
                ("spinning", Duration::from_micros(100)),
                ("parked", Duration::from_millis(12)),
            ] {
                let s = probe(w, qos, pause, true);
                println!(
                    "{w:>5} {label:<22} {state:<8} {:>9.1} {:>9.1} {:>7.0}% {:>10.1}",
                    s.total / 1e3,
                    s.slowest / 1e3,
                    s.late * 100.0,
                    s.max / 1e3
                );
            }
        }
        {
            let (state, pause) = ("idle", Duration::from_millis(12));
            let s = probe(w, None, pause, false);
            println!(
                "{w:>5} {:<22} {state:<8} {:>9.1} {:>9.1} {:>7.0}% {:>10.1}",
                "plain Exec::fork",
                s.total / 1e3,
                s.slowest / 1e3,
                s.late * 100.0,
                s.max / 1e3
            );
        }
    }
    unsafe { libc::getloadavg(l.as_mut_ptr(), 3) };
    println!("\nload average at end {l:?}");
}
