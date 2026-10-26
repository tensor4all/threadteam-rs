//! What does each reserved member do over time while the leader works
//! sequentially? Requires the feature `trace`.
//!
//! A team of 6 participants is held for the whole run (as the stage-2 harness
//! of faer-benchmark#24 does). The leader runs phases of sequential work with
//! and without team sections; for each phase the trace gives, per member, the
//! time spent spinning (reserved, polling), parked, running a share and lent,
//! and the process CPU time over wall time is printed next to it.
//!
//! `cargo bench --features trace --bench idle_trace -- [spin_ms]`

use std::hint::black_box;
use std::time::{Duration, Instant};
use threadteam::trace::{self, Event, Kind};
use threadteam::{raise_current_thread_qos, Pool, Qos};

fn process_cpu_ns() -> u64 {
    let mut ts = libc::timespec {
        tv_sec: 0,
        tv_nsec: 0,
    };
    unsafe { libc::clock_gettime(libc::CLOCK_PROCESS_CPUTIME_ID, &mut ts) };
    ts.tv_sec as u64 * 1_000_000_000 + ts.tv_nsec as u64
}

/// Sequential work of about `d` (the leader computes, no section).
fn busy(d: Duration) {
    let t = Instant::now();
    let mut x = 1.0f64;
    while t.elapsed() < d {
        for _ in 0..200 {
            x = black_box(x * 1.0000001 + 1e-9);
        }
    }
    black_box(x);
}

#[derive(Default, Clone, Copy)]
struct Occupancy {
    spin: u64,
    parked: u64,
    section: u64,
    lent: u64,
    absent: u64,
}

/// Occupancy of member `m` over `[t0, t1]` from its events.
fn occupancy(ev: &[Event], m: usize, t0: u64, t1: u64) -> Occupancy {
    #[derive(Clone, Copy, PartialEq)]
    enum S {
        Absent,
        Spin,
        Section,
        Parked,
        Lent,
    }
    let mut o = Occupancy::default();
    let (mut state, mut at) = (S::Absent, 0u64);
    let mut add = |s: S, a: u64, b: u64| {
        let (a, b) = (a.max(t0), b.min(t1));
        if b > a {
            let d = b - a;
            match s {
                S::Absent => o.absent += d,
                S::Spin => o.spin += d,
                S::Section => o.section += d,
                S::Parked => o.parked += d,
                S::Lent => o.lent += d,
            }
        }
    };
    for e in ev.iter().filter(|e| e.member == m) {
        add(state, at, e.ns);
        at = e.ns;
        state = match e.kind {
            Kind::Enter | Kind::SectionEnd | Kind::ParkExit | Kind::LentEnd => S::Spin,
            Kind::SectionBegin => S::Section,
            Kind::ParkEnter => S::Parked,
            Kind::LentBegin => S::Lent,
            Kind::Leave => S::Absent,
            Kind::Publish => state,
        };
    }
    add(state, at, t1);
    o
}

struct Phase<'a> {
    name: &'a str,
    pool: &'a Pool,
}

impl Phase<'_> {
    fn run(&self, f: impl FnOnce()) {
        let (c0, w0, t0) = (process_cpu_ns(), Instant::now(), trace::now_ns(self.pool));
        f();
        let (c1, w1, t1) = (process_cpu_ns(), Instant::now(), trace::now_ns(self.pool));
        let ev = trace::events(self.pool);
        let wall = (w1 - w0).as_nanos() as f64;
        print!(
            "{:<34} wall {:7.1} ms  cpu/wall {:4.2}  |",
            self.name,
            wall * 1e-6,
            (c1 - c0) as f64 / wall
        );
        for m in 0..self.pool.width().get() {
            let o = occupancy(&ev, m, t0, t1);
            let pct = |x: u64| 100.0 * x as f64 / (t1 - t0).max(1) as f64;
            if o.absent as f64 >= 0.99 * (t1 - t0) as f64 {
                continue;
            }
            print!(
                " m{m}: spin {:3.0}% park {:3.0}% run {:3.0}%{}",
                pct(o.spin),
                pct(o.parked),
                pct(o.section),
                if o.lent > 0 { " lent" } else { "" }
            );
        }
        println!();
    }
}

fn main() {
    let spin_ms: u64 = std::env::args()
        .skip(1)
        .find_map(|a| a.parse().ok())
        .unwrap_or(200);
    raise_current_thread_qos(Qos::UserInitiated);
    let pool = Pool::builder()
        .threads(6)
        .qos(Qos::UserInitiated)
        .idle_spin(Duration::from_millis(spin_ms))
        .acquire_timeout(Duration::from_secs(30))
        .build()
        .unwrap();
    let team = pool.into_team(6).unwrap();
    let mut exec = team.exec(6);
    let pool = team.pool();
    println!("team of 6, idle_spin {spin_ms} ms, QoS UserInitiated; members m1..m5 (the pool has 6 workers, one is unused)");
    let ph = |name| Phase { name, pool };
    // Sections are 20 us shares, like a small GEMM tile.
    let sections = |exec: &mut threadteam::Exec<'_>, n: usize| {
        for _ in 0..n {
            exec.reborrow()
                .broadcast(|_| busy(Duration::from_micros(20)))
                .unwrap();
        }
    };
    ph("A seq 120 ms (no section)").run(|| busy(Duration::from_millis(120)));
    ph("B seq 800 ms (no section)").run(|| busy(Duration::from_millis(800)));
    ph("C 2000 sections back to back").run(|| sections(&mut exec, 2000));
    ph("D seq 150 ms after sections").run(|| busy(Duration::from_millis(150)));
    ph("E seq 800 ms after D").run(|| busy(Duration::from_millis(800)));
    ph("F 1 section, then seq 150 ms").run(|| {
        sections(&mut exec, 1);
        busy(Duration::from_millis(150));
    });
    ph("G 10 sections, 100 ms gaps").run(|| {
        for _ in 0..10 {
            sections(&mut exec, 1);
            busy(Duration::from_millis(100));
        }
    });
    ph("H seq 800 ms (no section)").run(|| busy(Duration::from_millis(800)));
}
