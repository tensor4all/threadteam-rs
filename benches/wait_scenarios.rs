//! How team threads wait, measured the way callers use them (`docs/design/waiting.md`).
//!
//! `cargo bench --bench wait_scenarios -- [options]` prints one table row per cell and, with
//! `--csv <file>`, appends the same rows to a CSV file (`docs/design/waiting-data/`). Every cell
//! repeats a *call* and reports the median, the 90th percentile and the maximum per call.
//!
//! A section is `width` participants each busy-waiting `work` microseconds on the clock, so the
//! ideal time of a call is `sections * work`; the `overhead` column is the median minus that.
//!
//! Scenarios (`--scenario`, default `all`):
//! * `sections`: a call is 200 sections of 5, 20 or 50 µs, on a team held across calls;
//! * `gaps`: a call is 10 sections of 5 µs, the caller sleeps 0 / 100 µs / 1 ms / 10 ms between
//!   calls; reports the first section of each call (wake latency + 5 µs) and the whole call;
//! * `acquire`: a call is 50 sections of 20 µs, with the team held and a token minted per call
//!   (`held_team`, as tenferro's `CpuEngine`), with the pool held and the members acquired per
//!   call (`held_pool`, `Exec::new(..).team(..)`), and with a new pool per call (`fresh_pool`, as
//!   faer's `with_team`; `total` times the build, the acquisition, the call and the drop);
//! * `nested`: a call is 100 splits of the team into pairs, each pair running one 20 µs
//!   section: many child leaders join at once (they share one sleeping place in the pool).
//!
//! Every scenario runs at the widths given by `--widths` (default: all CPUs of the affinity set,
//! and three quarters of them), on threadteam and, as a reference, on a Rayon `scope` of the same
//! width (`--impl threadteam,rayon`). The libgomp reference is `benches/omp/wait_omp.c` (README,
//! "Wait benchmarks").
//!
//! Foreign load (`--load K:P`): `K` child processes of this binary, each busy `P` % of every 10 ms
//! period, unpinned, for the whole run. They stand for another user's processes on a shared host.
//!
//! Options: `--scenario sections|gaps|acquire|nested|all`, `--widths 64,48`, `--impl threadteam,rayon`,
//! `--load K:P`, `--calls N` (default 200; `gaps` and `acquire` use N/2), `--csv <file>`,
//! `--label <text>` (the version or commit, written to the CSV); tuning of the waits (this
//! version only, `docs/design/waiting.md`): `--spin-polls N`, `--throttled-polls N`,
//! `--pressure-hold-ms N` (0 throttles only the waiter that saw a preemption),
//! `--pressure-events N` (preemptions within the hold that throttle the pool). Under each
//! threadteam row a `waits per call` line counts join and barrier sleeps, member parks and the
//! preemptions waiters saw.

use std::io::Write;
use std::time::{Duration, Instant};
use threadteam::{Pool, Team, WaitCounters};

fn busy(us: u64) {
    let end = Instant::now() + Duration::from_micros(us);
    while Instant::now() < end {
        std::hint::spin_loop();
    }
}

fn cpus() -> usize {
    #[cfg(target_os = "linux")]
    unsafe {
        let mut set: libc::cpu_set_t = std::mem::zeroed();
        if libc::sched_getaffinity(0, std::mem::size_of::<libc::cpu_set_t>(), &mut set) == 0 {
            let n = libc::CPU_COUNT(&set) as usize;
            if n > 0 {
                return n;
            }
        }
    }
    std::thread::available_parallelism().map_or(1, |n| n.get())
}

fn load_average() -> f64 {
    let mut l = [0f64; 3];
    unsafe { libc::getloadavg(l.as_mut_ptr(), 3) };
    l[0]
}

fn host() -> String {
    let mut buf = [0u8; 256];
    unsafe { libc::gethostname(buf.as_mut_ptr().cast(), buf.len()) };
    let end = buf.iter().position(|&b| b == 0).unwrap_or(buf.len());
    String::from_utf8_lossy(&buf[..end]).into_owned()
}

/// Child processes that burn a share of a CPU each until dropped.
struct Load(Vec<std::process::Child>);

impl Load {
    fn start(spec: Option<(usize, u64)>) -> Self {
        let Some((k, pct)) = spec else {
            return Load(Vec::new());
        };
        let exe = std::env::current_exe().expect("own path");
        Load(
            (0..k)
                .map(|_| {
                    std::process::Command::new(&exe)
                        .args(["--load-child", &pct.to_string()])
                        .spawn()
                        .expect("a load child")
                })
                .collect(),
        )
    }
}

impl Drop for Load {
    fn drop(&mut self) {
        for c in &mut self.0 {
            let _ = c.kill();
            let _ = c.wait();
        }
    }
}

fn load_child(pct: u64) -> ! {
    let period = Duration::from_millis(10);
    let on = period * pct.min(100) as u32 / 100;
    loop {
        let t = Instant::now();
        while t.elapsed() < on {
            std::hint::spin_loop();
        }
        if let Some(rest) = period.checked_sub(t.elapsed()) {
            std::thread::sleep(rest);
        }
    }
}

struct Stats {
    median: f64,
    p90: f64,
    max: f64,
}

fn stats(mut v: Vec<f64>) -> Stats {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    Stats {
        median: v[v.len() / 2],
        p90: v[(v.len() * 9 / 10).min(v.len() - 1)],
        max: v[v.len() - 1],
    }
}

/// Pool settings from the command line (`--spin-polls`, `--throttled-polls`,
/// `--pressure-hold-ms`); `None` keeps the library default.
type Tuning = (Option<u32>, Option<(u32, Duration, u32)>);
static TUNING: std::sync::OnceLock<Tuning> = std::sync::OnceLock::new();

fn pool(width: usize) -> Pool {
    let mut b = Pool::builder()
        .threads(width)
        .acquire_timeout(Duration::from_secs(30));
    if let Some((spin, throttle)) = TUNING.get() {
        if let Some(n) = spin {
            b = b.spin_polls(*n);
        }
        if let Some((polls, hold, events)) = throttle {
            b = b.wait_tuning(*polls, *hold, *events);
        }
    }
    b.build().expect("a pool")
}

/// Counters of the waits between two snapshots.
fn waits_between(a: WaitCounters, b: WaitCounters) -> WaitCounters {
    WaitCounters {
        join_sleeps: b.join_sleeps - a.join_sleeps,
        barrier_sleeps: b.barrier_sleeps - a.barrier_sleeps,
        member_parks: b.member_parks - a.member_parks,
        pressure_events: b.pressure_events - a.pressure_events,
    }
}

fn waits_sum(a: WaitCounters, b: WaitCounters) -> WaitCounters {
    WaitCounters {
        join_sleeps: a.join_sleeps + b.join_sleeps,
        barrier_sleeps: a.barrier_sleeps + b.barrier_sleeps,
        member_parks: a.member_parks + b.member_parks,
        pressure_events: a.pressure_events + b.pressure_events,
    }
}

fn snap(t: &Option<Team>) -> WaitCounters {
    t.as_ref()
        .map(|t| t.pool().wait_counters())
        .unwrap_or_default()
}

/// Prints what the waits of the cell just reported did (stdout only).
fn waits_line(c: WaitCounters, calls: usize) {
    let per = |n: u64| n as f64 / calls.max(1) as f64;
    println!(
        "          waits per call: join sleeps {:.1}, barrier sleeps {:.1}, member parks {:.1}, preemptions seen {:.2}",
        per(c.join_sleeps),
        per(c.barrier_sleeps),
        per(c.member_parks),
        per(c.pressure_events)
    );
}

fn team(width: usize) -> Team {
    pool(width).into_team(width).expect("an idle pool")
}

/// One call of `sections` sections of `work` µs on a team; the time of the first section and of
/// the whole call, in µs.
fn call_team(team: &Team, width: usize, sections: usize, work: u64) -> (f64, f64) {
    let mut exec = team.exec(width);
    exec.team(|mut t| {
        let t0 = Instant::now();
        t.reborrow().run(|_, _| busy(work));
        let first = t0.elapsed().as_secs_f64() * 1e6;
        for _ in 1..sections {
            t.reborrow().run(|_, _| busy(work));
        }
        (first, t0.elapsed().as_secs_f64() * 1e6)
    })
    .expect("a team token")
}

/// One call of `splits` splits of the team into pairs, each pair running one section of
/// `work` µs; the time of the call in µs.
fn call_nested(team: &Team, width: usize, splits: usize, work: u64) -> f64 {
    let mut exec = team.exec(width);
    exec.team(|mut t| {
        let pairs = vec![2usize; t.size().get() / 2];
        let t0 = Instant::now();
        for _ in 0..splits {
            t.reborrow()
                .split(&pairs, |_, child| child.run(|_, _| busy(work)))
                .expect("a split into pairs");
        }
        t0.elapsed().as_secs_f64() * 1e6
    })
    .expect("a team token")
}

/// One call that acquires its members from a held pool (`Exec::new(..).team(..)`); the time of
/// the acquisition and the call, in µs.
fn call_pool(pool: &Pool, width: usize, sections: usize, work: u64) -> f64 {
    let t0 = Instant::now();
    threadteam::Exec::new(pool, width)
        .expect("a token")
        .team(|mut t| {
            for _ in 0..sections {
                t.reborrow().run(|_, _| busy(work));
            }
        })
        .expect("members for the scope");
    t0.elapsed().as_secs_f64() * 1e6
}

/// The same call on a Rayon pool: one `scope` per section, `width - 1` spawned shares and one on
/// the caller.
fn call_rayon(pool: &rayon::ThreadPool, width: usize, sections: usize, work: u64) -> (f64, f64) {
    pool.install(|| {
        let t0 = Instant::now();
        let mut first = 0.0;
        for s in 0..sections {
            rayon::scope(|sc| {
                for _ in 1..width {
                    sc.spawn(|_| busy(work));
                }
                busy(work);
            });
            if s == 0 {
                first = t0.elapsed().as_secs_f64() * 1e6;
            }
        }
        (first, t0.elapsed().as_secs_f64() * 1e6)
    })
}

struct Row<'a> {
    scenario: &'a str,
    imp: &'a str,
    width: usize,
    param: String,
    metric: &'a str,
    s: Stats,
    ideal_us: f64,
}

struct Out {
    csv: Option<std::fs::File>,
    prefix: String,
}

impl Out {
    fn row(&mut self, r: Row<'_>) {
        println!(
            "{:<9} {:<10} {:>5} {:<22} {:<7} {:>11.1} {:>11.1} {:>11.1} {:>10.1}",
            r.scenario,
            r.imp,
            r.width,
            r.param,
            r.metric,
            r.s.median,
            r.s.p90,
            r.s.max,
            r.s.median - r.ideal_us
        );
        if let Some(f) = &mut self.csv {
            writeln!(
                f,
                "{},{},{},{},{},{},{:.2},{:.2},{:.2},{:.2}",
                self.prefix,
                r.scenario,
                r.imp,
                r.width,
                r.param,
                r.metric,
                r.s.median,
                r.s.p90,
                r.s.max,
                r.s.median - r.ideal_us
            )
            .expect("csv write");
        }
    }
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let value = |name: &str| {
        args.iter()
            .position(|a| a == name)
            .and_then(|i| args.get(i + 1).cloned())
    };
    if let Some(pct) = value("--load-child") {
        load_child(pct.parse().expect("--load-child <percent>"));
    }
    let scenario = value("--scenario").unwrap_or_else(|| "all".into());
    let n = cpus();
    let widths: Vec<usize> = value("--widths")
        .map(|v| v.split(',').map(|w| w.parse().expect("--widths")).collect())
        .unwrap_or_else(|| {
            let mut w = vec![n, (n * 3 / 4).max(1)];
            w.dedup();
            w
        });
    let imps: Vec<String> = value("--impl")
        .unwrap_or_else(|| "threadteam,rayon".into())
        .split(',')
        .map(String::from)
        .collect();
    let load_spec = value("--load").map(|v| {
        let (k, p) = v.split_once(':').expect("--load K:P");
        (k.parse().expect("--load K"), p.parse().expect("--load P"))
    });
    let calls: usize = value("--calls").map_or(200, |v| v.parse().expect("--calls"));
    let label = value("--label").unwrap_or_default();
    let spin = value("--spin-polls").map(|v| v.parse().expect("--spin-polls"));
    let throttle = match (
        value("--throttled-polls"),
        value("--pressure-hold-ms"),
        value("--pressure-events"),
    ) {
        (None, None, None) => None,
        (t, h, e) => Some((
            t.map_or(100, |v| v.parse().expect("--throttled-polls")),
            Duration::from_millis(h.map_or(100, |v| v.parse().expect("--pressure-hold-ms"))),
            e.map_or(4, |v| v.parse().expect("--pressure-events")),
        )),
    };
    TUNING.get_or_init(|| (spin, throttle));
    let load_text = load_spec.map_or("none".into(), |(k, p)| format!("{k}x{p}%"));

    let csv = value("--csv").map(|path| {
        let fresh = !std::path::Path::new(&path).exists();
        let mut f = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&path)
            .expect("csv file");
        if fresh {
            writeln!(
                f,
                "host,label,cpus,load,scenario,impl,width,param,metric,median_us,p90_us,max_us,overhead_us"
            )
            .expect("csv header");
        }
        f
    });
    let mut out = Out {
        csv,
        prefix: format!("{},{label},{n},{load_text}", host()),
    };

    let _load = Load::start(load_spec);
    if load_spec.is_some() {
        std::thread::sleep(Duration::from_millis(200));
    }
    println!(
        "host {} cpus {n} load average {:.2} foreign load {load_text} label {label}",
        host(),
        load_average()
    );
    println!(
        "{:<9} {:<10} {:>5} {:<22} {:<7} {:>11} {:>11} {:>11} {:>10}",
        "scenario", "impl", "width", "param", "metric", "median us", "p90 us", "max us", "overhead"
    );

    let want = |s: &str| scenario == "all" || scenario == s;
    for &w in &widths {
        let rayon_pool = imps.iter().any(|i| i == "rayon").then(|| {
            rayon::ThreadPoolBuilder::new()
                .num_threads(w)
                .build()
                .expect("a rayon pool")
        });
        let mut held = imps.iter().any(|i| i == "threadteam").then(|| team(w));

        if want("sections") {
            for work in [5u64, 20, 50] {
                let sections = 200;
                let param = format!("work={work}us,sections={sections}");
                let ideal = (sections as u64 * work) as f64;
                if let Some(t) = &held {
                    let c0 = snap(&held);
                    let v = (0..calls)
                        .map(|_| call_team(t, w, sections, work).1)
                        .collect();
                    out.row(Row {
                        scenario: "sections",
                        imp: "threadteam",
                        width: w,
                        param: param.clone(),
                        metric: "call",
                        s: stats(v),
                        ideal_us: ideal,
                    });
                    waits_line(waits_between(c0, snap(&held)), calls);
                }
                if let Some(p) = &rayon_pool {
                    let v = (0..calls)
                        .map(|_| call_rayon(p, w, sections, work).1)
                        .collect();
                    out.row(Row {
                        scenario: "sections",
                        imp: "rayon",
                        width: w,
                        param,
                        metric: "call",
                        s: stats(v),
                        ideal_us: ideal,
                    });
                }
            }
        }

        if want("gaps") {
            for gap_us in [0u64, 100, 1_000, 10_000] {
                let (sections, work) = (10, 5u64);
                let param = format!("gap={gap_us}us");
                let gap = Duration::from_micros(gap_us);
                let run = |f: &mut dyn FnMut() -> (f64, f64)| {
                    let mut first = Vec::new();
                    let mut whole = Vec::new();
                    for _ in 0..calls / 2 {
                        if !gap.is_zero() {
                            std::thread::sleep(gap);
                        }
                        let (a, b) = f();
                        first.push(a);
                        whole.push(b);
                    }
                    (stats(first), stats(whole))
                };
                if let Some(t) = &held {
                    let c0 = snap(&held);
                    let (a, b) = run(&mut || call_team(t, w, sections, work));
                    let c = waits_between(c0, snap(&held));
                    out.row(Row {
                        scenario: "gaps",
                        imp: "threadteam",
                        width: w,
                        param: param.clone(),
                        metric: "first",
                        s: a,
                        ideal_us: work as f64,
                    });
                    out.row(Row {
                        scenario: "gaps",
                        imp: "threadteam",
                        width: w,
                        param: param.clone(),
                        metric: "call",
                        s: b,
                        ideal_us: (sections as u64 * work) as f64,
                    });
                    waits_line(c, calls / 2);
                }
                if let Some(p) = &rayon_pool {
                    let (a, b) = run(&mut || call_rayon(p, w, sections, work));
                    out.row(Row {
                        scenario: "gaps",
                        imp: "rayon",
                        width: w,
                        param: param.clone(),
                        metric: "first",
                        s: a,
                        ideal_us: work as f64,
                    });
                    out.row(Row {
                        scenario: "gaps",
                        imp: "rayon",
                        width: w,
                        param,
                        metric: "call",
                        s: b,
                        ideal_us: (sections as u64 * work) as f64,
                    });
                }
            }
        }

        if want("nested") && w >= 4 {
            if let Some(t) = &held {
                let (splits, work) = (100, 20u64);
                let c0 = snap(&held);
                let v = (0..calls / 2)
                    .map(|_| call_nested(t, w, splits, work))
                    .collect();
                out.row(Row {
                    scenario: "nested",
                    imp: "threadteam",
                    width: w,
                    param: "pairs,splits=100,work=20us".into(),
                    metric: "call",
                    s: stats(v),
                    ideal_us: (splits as u64 * work) as f64,
                });
                waits_line(waits_between(c0, snap(&held)), calls / 2);
            }
        }

        if want("acquire") && imps.iter().any(|i| i == "threadteam") {
            // No other team of this process stays reserved while these arms run.
            drop(held.take());
            let (sections, work) = (50, 20u64);
            let ideal = (sections as u64 * work) as f64;
            let mut row = |param: &str, metric: &'static str, v: Vec<f64>| {
                out.row(Row {
                    scenario: "acquire",
                    imp: "threadteam",
                    width: w,
                    param: param.into(),
                    metric,
                    s: stats(v),
                    ideal_us: ideal,
                })
            };
            {
                let t = team(w);
                let c0 = t.pool().wait_counters();
                let v = (0..calls / 2)
                    .map(|_| call_team(&t, w, sections, work).1)
                    .collect();
                row("held_team", "call", v);
                waits_line(waits_between(c0, t.pool().wait_counters()), calls / 2);
            }
            {
                let p = pool(w);
                let c0 = p.wait_counters();
                let v = (0..calls / 2)
                    .map(|_| call_pool(&p, w, sections, work))
                    .collect();
                row("held_pool", "call", v);
                waits_line(waits_between(c0, p.wait_counters()), calls / 2);
            }
            let (mut call, mut total) = (Vec::new(), Vec::new());
            let mut c = WaitCounters::default();
            for _ in 0..calls / 2 {
                let t0 = Instant::now();
                let t = team(w);
                let c0 = t.pool().wait_counters();
                call.push(call_team(&t, w, sections, work).1);
                c = waits_sum(c, waits_between(c0, t.pool().wait_counters()));
                drop(t);
                total.push(t0.elapsed().as_secs_f64() * 1e6);
            }
            row("fresh_pool", "call", call);
            row("fresh_pool", "total", total);
            waits_line(c, calls / 2);
        }
    }
}
