//! Dispatch latency of an empty `TeamExec::run` against `Exec::fork`, and the
//! CPU time idle members burn while spinning.
//!
//! `cargo bench --bench team_dispatch` (prints a table; no statistics beyond
//! the median of repeated batches). Report the host load printed first.

use std::hint::black_box;
use std::time::{Duration, Instant};
use threadteam::{Exec, Pool, WaitOverride};

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

/// Median over `batches` of the mean nanoseconds per call of `f`.
fn per_call_ns(batches: usize, iters: usize, mut f: impl FnMut()) -> f64 {
    for _ in 0..iters / 10 + 10 {
        f();
    }
    median(
        (0..batches)
            .map(|_| {
                let t = Instant::now();
                for _ in 0..iters {
                    f();
                }
                t.elapsed().as_nanos() as f64 / iters as f64
            })
            .collect(),
    )
}

fn cpu_time() -> Duration {
    let mut ru: libc::rusage = unsafe { std::mem::zeroed() };
    unsafe { libc::getrusage(libc::RUSAGE_SELF, &mut ru) };
    let tv = |t: libc::timeval| Duration::new(t.tv_sec as u64, t.tv_usec as u32 * 1000);
    tv(ru.ru_utime) + tv(ru.ru_stime)
}

fn load() -> [f64; 3] {
    let mut l = [0f64; 3];
    unsafe { libc::getloadavg(l.as_mut_ptr(), 3) };
    l
}

fn main() {
    let cores = std::thread::available_parallelism().map_or(1, |n| n.get());
    println!("host: {cores} logical cores, load average {:?}", load());
    println!();
    println!("dispatch latency, empty section, median ns per call");
    println!(
        "{:>6} {:>14} {:>14} {:>14}",
        "width", "team.run", "Exec::fork", "ratio f/t"
    );
    for w in [2usize, 6, 12, 18] {
        let pool = Pool::builder()
            .threads(w)
            .acquire_timeout(Duration::from_secs(5))
            .build()
            .unwrap();
        let mut exec = Exec::new(&pool, w).unwrap();
        if pool.width().get() < w {
            println!("{w:>6} (pool narrower than {w}: skipped)");
            continue;
        }
        let fork = per_call_ns(7, 5_000, || {
            exec.reborrow().fork(w, |_, _| {
                black_box(());
            });
        });
        let team = exec
            .team(|mut team| {
                per_call_ns(7, 100_000, || {
                    team.reborrow().run(|_, _| {
                        black_box(());
                    });
                })
            })
            .expect("team");
        println!("{w:>6} {team:>14.0} {fork:>14.0} {:>14.1}", fork / team);
    }

    println!();
    println!("acquisition (team scope with one empty section), median ns per scope");
    println!(
        "{:>6} {:>14} {:>14}",
        "width", "cold (spin 0)", "warm (spin 200ms)"
    );
    for w in [2usize, 6, 12, 18] {
        let build = |spin| {
            Pool::builder()
                .threads(w)
                .idle_spin(spin)
                .acquire_timeout(Duration::from_secs(5))
                .build()
                .unwrap()
        };
        let cold_pool = build(Duration::ZERO);
        let mut e = Exec::new(&cold_pool, w).unwrap();
        let cold = per_call_ns(5, 100, || {
            // let the members leave so that the next scope starts them afresh
            while cold_pool.resident_members() != 0 {
                std::hint::spin_loop();
            }
            e.team(|t| t.run(|_, _| {})).unwrap();
        });
        let warm_pool = build(Duration::from_millis(200));
        let mut e = Exec::new(&warm_pool, w).unwrap();
        let warm = per_call_ns(5, 2_000, || {
            e.team(|t| t.run(|_, _| {})).unwrap();
        });
        println!("{w:>6} {cold:>14.0} {warm:>14.0}");
    }

    println!();
    println!("CPU time while idle after a team of 6 ended (600 ms wall window, whole process)");
    println!("{:>12} {:>14}", "idle_spin", "cpu ms");
    for spin in [Duration::ZERO, Duration::from_millis(200)] {
        let pool = Pool::builder()
            .threads(6)
            .idle_spin(spin)
            .acquire_timeout(Duration::from_secs(5))
            .build()
            .unwrap();
        let mut exec = Exec::new(&pool, 6).unwrap();
        exec.team_with(WaitOverride::new(), |t| t.run(|_, _| {}))
            .unwrap();
        let (c0, t0) = (cpu_time(), Instant::now());
        while t0.elapsed() < Duration::from_millis(600) {
            std::thread::sleep(Duration::from_millis(5));
        }
        let cpu = cpu_time() - c0;
        println!(
            "{:>12} {:>14.1}",
            format!("{spin:?}"),
            cpu.as_secs_f64() * 1e3
        );
    }
    println!("\nhost load average at end {:?}", load());
}
