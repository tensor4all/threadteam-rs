//! Cost of splitting a copy whose source the calling thread has just written (issue #27).
//!
//! The width model charges a parallel section `D(k) + est / k`, where `est` is the serial time
//! measured on data resident in the caller's caches. When the caller has just produced the data,
//! the members of a team fork do not find it in their own caches: they pull it from the leader's.
//! This bench measures that transfer, so a per-host transfer row can be calibrated.
//!
//! Each sample:
//!
//! 1. the jobs pre-touch their own destinations (not timed);
//! 2. the leader writes the whole source in one pass;
//! 3. one `team.exec(k).fork(k, ..)` copies with `copy_from_slice`; job 0 runs on the leader, jobs
//!    `1..k` on reserved members. Timed: the fork..join wall and each job's busy time around its
//!    copy.
//!
//! Modes:
//!
//! * `D` (disjoint): job `i` copies its own `1/k` of the source, so the members move
//!   `bytes * (k - 1) / k` out of the leader's caches;
//! * `S` (same): every job copies the same first quarter, so the members share one transfer.
//!
//! States: `hot` as above; `cold` streams an `--evict-mib` buffer (default 1024 MiB, larger than
//! the last-level cache) after writing the source, so the copy reads it from memory instead.
//! `k = 1` is the serial reference (the leader copies inline).
//!
//! # Run
//!
//! `cargo bench --bench resident_split -- [options] > out.csv`
//!
//! Options: `--sizes B1,B2,..` (bytes; default 16 KiB to 256 MiB in powers of 4, plus 392 KiB),
//! `--widths 1,2,4,..` (default 1, 2, 4, 8, 16, 32, 48, 96; widths above the team are skipped),
//! `--evict-mib N` (default 1024), `--hot-bytes N` (hot samples per point are `N / size`, clamped
//! to 21..=401; default 1 GiB), `--cold-reps N` (default 15).
//!
//! The team takes `available_parallelism()` threads; restrict placement from outside, for example
//! `numactl --cpunodebind=0 --membind=0` for one die. On Linux the CPU of every job of the widest
//! fork is printed on stderr, to check the placement.
//!
//! # Output
//!
//! One CSV row per (state, bytes, mode, k), times in ns, medians and quartiles over `reps` samples
//! after one warm-up sample: `wall_*` the fork..join; `sum_*` the summed busy of all jobs;
//! `leader_med` job 0; `members_sum_med`, `members_max_med` jobs `1..k`; `job_max_med`,
//! `job_med_med` the slowest and the median job per sample.
//!
//! # Reading it
//!
//! * The extra CPU time of the split in mode `D` is `sum_med(k) - sum_med(1)`; in mode `S`, a
//!   member's busy against the leader's own copy of the quarter.
//! * The members pull concurrently from one source, so the wall pays one drain of the moved
//!   bytes, not one per member: for `D`, `members_max_med` stays about flat over `k` while
//!   `members_sum_med` grows like `k - 1`.
//! * Transfer row: on one die, `hot`, mode `D`, `k >= 4`, take the rate
//!   `bytes * (k - 1) / k / members_max_med` (bytes per ns = GB/s). It is about constant over
//!   `k`; the section then pays about `D(k) + moved / rate` in wall. Compare with `cold` to see
//!   whether the source being resident still helps the members.

use std::cell::UnsafeCell;
use std::hint::black_box;
use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
use std::time::Instant;

use threadteam::{Pool, Team};

struct Args {
    sizes: Vec<usize>,
    widths: Vec<usize>,
    evict_mib: usize,
    hot_bytes: usize,
    cold_reps: usize,
}

fn parse_args() -> Args {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let value = |name: &str| {
        args.iter()
            .position(|a| a == name)
            .and_then(|i| args.get(i + 1).cloned())
    };
    let list = |s: String| -> Vec<usize> {
        s.split(',')
            .map(|x| x.parse().expect("comma-separated numbers"))
            .collect()
    };
    let num =
        |name: &str, default: usize| value(name).map_or(default, |v| v.parse().expect("a number"));
    let mut sizes = value("--sizes").map(list).unwrap_or_else(|| {
        // 16 KiB .. 256 MiB in powers of 4, plus 392 KiB (196 blocks of 16x16 f64).
        (0..8)
            .map(|i| 16usize << (10 + 2 * i))
            .chain([196 * 256 * 8])
            .collect()
    });
    sizes.sort_unstable();
    Args {
        sizes,
        widths: value("--widths")
            .map(list)
            .unwrap_or_else(|| vec![1, 2, 4, 8, 16, 32, 48, 96]),
        evict_mib: num("--evict-mib", 1024),
        hot_bytes: num("--hot-bytes", 1 << 30),
        cold_reps: num("--cold-reps", 15),
    }
}

/// One destination per job; job `i` is the only one touching slot `i`.
struct Slots(Vec<UnsafeCell<Vec<f64>>>);
// SAFETY: within one fork, job `i` accesses only `self.0[i]`, and fork indices are distinct, so no
// slot is aliased mutably across threads.
unsafe impl Sync for Slots {}

impl Slots {
    /// # Safety
    /// The caller is job `i` of the running fork, or no fork is running (see `Sync` above).
    #[allow(clippy::mut_from_ref)]
    unsafe fn job(&self, i: usize) -> &mut Vec<f64> {
        unsafe { &mut *self.0[i].get() }
    }
}

#[derive(Clone, Copy, PartialEq)]
enum Mode {
    D,
    S,
}

fn range(mode: Mode, n: usize, k: usize, i: usize) -> std::ops::Range<usize> {
    match mode {
        Mode::D => i * n / k..(i + 1) * n / k,
        Mode::S => 0..n / 4,
    }
}

#[cfg(target_os = "linux")]
fn cpu() -> i64 {
    // SAFETY: sched_getcpu takes no arguments and only reads the calling thread's state.
    unsafe { libc::sched_getcpu() as i64 }
}
#[cfg(not(target_os = "linux"))]
fn cpu() -> i64 {
    -1
}

fn quantile(v: &mut [u64], p: f64) -> u64 {
    v.sort_unstable();
    v[((v.len() - 1) as f64 * p).round() as usize]
}

fn main() {
    let a = parse_args();
    let threads = std::thread::available_parallelism().map_or(1, |n| n.get());
    let team: Team = Pool::builder()
        .threads(threads)
        .build()
        .expect("pool")
        .into_team(threads)
        .unwrap_or_else(|_| panic!("team reservation refused"));
    let size = team.size().get();
    let widths: Vec<usize> = a.widths.iter().copied().filter(|&k| k <= size).collect();
    eprintln!("# available_parallelism={threads} team_size={size} widths={widths:?}");

    // Warm the team: every member runs a few sections.
    for _ in 0..200 {
        team.exec(size).fork(size, |_, _| {
            black_box(0);
        });
    }
    let cpus: Vec<AtomicU64> = (0..size).map(|_| AtomicU64::new(0)).collect();
    team.exec(size)
        .fork(size, |i, _| cpus[i].store(cpu() as u64, Relaxed));
    let cpus: Vec<i64> = cpus.iter().map(|c| c.load(Relaxed) as i64).collect();
    eprintln!("# job cpus (job 0 = leader): {cpus:?}");

    let mut evict = vec![0u8; a.evict_mib << 20];
    println!(
        "state,bytes,mode,k,reps,wall_med,wall_q1,wall_q3,sum_med,sum_q1,sum_q3,leader_med,\
         members_sum_med,members_max_med,job_max_med,job_med_med"
    );
    for &bytes in &a.sizes {
        let n = bytes / 8;
        let mut src = vec![0f64; n];
        for state in ["hot", "cold"] {
            let reps = if state == "hot" {
                (a.hot_bytes / bytes).clamp(21, 401)
            } else {
                a.cold_reps
            };
            for mode in [Mode::D, Mode::S] {
                for &k in &widths {
                    let slots = Slots(
                        (0..k)
                            .map(|i| UnsafeCell::new(vec![0f64; range(mode, n, k, i).len()]))
                            .collect(),
                    );
                    let busy: Vec<AtomicU64> = (0..k).map(|_| AtomicU64::new(0)).collect();
                    let mut rec: [Vec<u64>; 7] = Default::default();
                    let mut seed = 0f64;
                    // One untimed warm-up sample, then `reps` timed ones.
                    for rep in 0..=reps {
                        seed += 1.0;
                        let touch = || {
                            team.exec(k).fork(k, |i, _| {
                                // SAFETY: this is job `i`.
                                unsafe { slots.job(i) }.fill(seed);
                            })
                        };
                        if state == "hot" {
                            touch();
                        }
                        for (j, x) in src.iter_mut().enumerate() {
                            *x = j as f64 + seed;
                        }
                        if state == "cold" {
                            evict.fill(seed as u8);
                            black_box(&mut evict);
                            touch();
                        }
                        let src = &src;
                        let t0 = Instant::now();
                        team.exec(k).fork(k, |i, _| {
                            let r = range(mode, n, k, i);
                            // SAFETY: this is job `i`.
                            let d = unsafe { slots.job(i) };
                            let t = Instant::now();
                            d.copy_from_slice(&src[r]);
                            black_box(d);
                            busy[i].store(t.elapsed().as_nanos() as u64, Relaxed);
                        });
                        let wall = t0.elapsed().as_nanos() as u64;
                        if rep == 0 {
                            continue;
                        }
                        let mut b: Vec<u64> = busy.iter().map(|x| x.load(Relaxed)).collect();
                        let members = &b[1..];
                        rec[0].push(wall);
                        rec[1].push(b.iter().sum());
                        rec[2].push(b[0]);
                        rec[3].push(members.iter().sum());
                        rec[4].push(members.iter().copied().max().unwrap_or(0));
                        rec[5].push(b.iter().copied().max().unwrap_or(0));
                        rec[6].push(quantile(&mut b, 0.5));
                    }
                    // Check one destination so the copy cannot be elided.
                    // SAFETY: no fork is running.
                    let last = unsafe { slots.job(k - 1) };
                    let r = range(mode, n, k, k - 1);
                    assert!(last.is_empty() || last[0] == r.start as f64 + seed);
                    let m = if mode == Mode::D { "D" } else { "S" };
                    let [w, s, l, ms, mm, jm, jmed] = &mut rec;
                    println!(
                        "{state},{bytes},{m},{k},{reps},{},{},{},{},{},{},{},{},{},{},{}",
                        quantile(w, 0.5),
                        quantile(w, 0.25),
                        quantile(w, 0.75),
                        quantile(s, 0.5),
                        quantile(s, 0.25),
                        quantile(s, 0.75),
                        quantile(l, 0.5),
                        quantile(ms, 0.5),
                        quantile(mm, 0.5),
                        quantile(jm, 0.5),
                        quantile(jmed, 0.5)
                    );
                }
            }
        }
    }
}
