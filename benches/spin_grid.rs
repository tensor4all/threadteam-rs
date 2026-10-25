//! Dispatch latency and idle CPU against the park delay (`idle_spin`) and the
//! gap between sections, for a team of 6 participants.
//!
//! For each (spin, gap) cell a fresh team is built, then the leader repeats
//! "wait `gap`, dispatch a trivial section to all participants". Reported per
//! cell:
//!
//! * `lat`: median time from the leader's call to the *last* participant
//!   starting its share (first-dispatch latency after the gap);
//! * `cores`: CPU time of the process other than the leader thread, divided by
//!   wall time, over the whole cell: the cost of members that spin or wake
//!   (the members' idle CPU in units of cores; 5 = all five members spinning).
//!
//! The leader waits by sleeping when `gap >= 1 ms` and by busy-waiting below
//! (a sleep cannot be shorter than that); the busy-wait is the leader's own
//! thread and is excluded from `cores`.
//!
//! `cargo bench --bench spin_grid -- [cell_seconds]`

use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};
use threadteam::{raise_current_thread_qos, Pool, Qos};

fn cpu_ns(clock: libc::clockid_t) -> u64 {
    let mut ts = libc::timespec {
        tv_sec: 0,
        tv_nsec: 0,
    };
    unsafe { libc::clock_gettime(clock, &mut ts) };
    ts.tv_sec as u64 * 1_000_000_000 + ts.tv_nsec as u64
}

fn wait(gap: Duration) {
    if gap >= Duration::from_millis(1) {
        std::thread::sleep(gap);
    } else {
        let t = Instant::now();
        while t.elapsed() < gap {
            std::hint::spin_loop();
        }
    }
}

fn median(v: &mut [f64]) -> f64 {
    v.sort_by(f64::total_cmp);
    v[v.len() / 2]
}

fn cell(spin: Duration, gap: Duration, seconds: f64) -> (f64, f64, f64) {
    let pool = Pool::builder()
        .threads(6)
        .qos(Qos::UserInitiated)
        .idle_spin(spin)
        .acquire_timeout(Duration::from_secs(30))
        .build()
        .unwrap();
    let team = pool.into_team(6).unwrap();
    let mut exec = team.exec(6);
    let base = Instant::now();
    let starts: Vec<AtomicU64> = (0..6).map(|_| AtomicU64::new(0)).collect();
    let one = |exec: &mut threadteam::Exec<'_>| -> f64 {
        for s in &starts {
            s.store(0, Ordering::Relaxed);
        }
        let t0 = base.elapsed().as_nanos() as u64;
        exec.reborrow()
            .broadcast(|i| starts[i].store(base.elapsed().as_nanos() as u64, Ordering::Release))
            .unwrap();
        starts
            .iter()
            .map(|s| s.load(Ordering::Acquire).saturating_sub(t0))
            .max()
            .unwrap() as f64
    };
    for _ in 0..5 {
        wait(gap);
        one(&mut exec);
    }
    let n = ((seconds / (gap.as_secs_f64() + 20e-6)) as usize).clamp(20, 100_000);
    let (p0, l0, w0) = (
        cpu_ns(libc::CLOCK_PROCESS_CPUTIME_ID),
        cpu_ns(libc::CLOCK_THREAD_CPUTIME_ID),
        Instant::now(),
    );
    let mut lat = Vec::with_capacity(n);
    for _ in 0..n {
        wait(gap);
        lat.push(one(&mut exec));
    }
    let wall = w0.elapsed().as_nanos() as f64;
    let others = (cpu_ns(libc::CLOCK_PROCESS_CPUTIME_ID) - p0) as f64
        - (cpu_ns(libc::CLOCK_THREAD_CPUTIME_ID) - l0) as f64;
    let mut sorted = lat.clone();
    let med = median(&mut sorted);
    let p90 = sorted[sorted.len() * 9 / 10];
    (med * 1e-3, p90 * 1e-3, others / wall)
}

fn main() {
    let seconds: f64 = std::env::args()
        .skip(1)
        .find_map(|a| a.parse().ok())
        .unwrap_or(1.0);
    raise_current_thread_qos(Qos::UserInitiated);
    let spins = [
        Duration::ZERO,
        Duration::from_micros(50),
        Duration::from_micros(200),
        Duration::from_millis(1),
        Duration::from_millis(10),
        Duration::from_millis(200),
    ];
    let gaps = [
        Duration::from_micros(10),
        Duration::from_micros(100),
        Duration::from_millis(1),
        Duration::from_millis(10),
        Duration::from_millis(100),
    ];
    println!("team of 6, QoS UserInitiated, cell time about {seconds} s");
    println!("each cell: median / p90 dispatch latency in us, members' CPU in cores");
    print!("{:>10}", "spin\\gap");
    for g in gaps {
        print!("{:>26}", format!("{g:?}"));
    }
    println!();
    for s in spins {
        print!("{:>10}", format!("{s:?}"));
        for g in gaps {
            let (med, p90, cores) = cell(s, g, seconds);
            print!("{:>26}", format!("{med:6.1} /{p90:6.1} us {cores:4.2}c"));
        }
        println!();
    }
}
