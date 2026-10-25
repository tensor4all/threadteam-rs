//! Break-even sweep for faer GEMM and LU: serial vs plain token (outside the pool, one install per
//! call; inside the pool, loop run in `external`) vs team token, 6 participants.
//!
//! `thr-faer FLOOR(off|default) ROUNDS` prints CSV: family,shape,mode,round,ns
//!
//! `off` sets the `gemm` crate's threading threshold and the LU `par_threshold` to zero so the
//! sweep shows where the parallel route itself starts to pay; `default` keeps faer's own floors.
use faer::dyn_stack::{MemBuffer, MemStack};
use faer::linalg::lu::partial_pivoting::factor as lu;
use faer::linalg::matmul::matmul;
use faer::{c64, Accum, Auto, Mat, Par};
use std::sync::{Arc, Mutex};
use std::num::NonZeroUsize;
use std::time::{Duration, Instant};
use threadteam::{raise_current_thread_qos, Exec, Pool, Qos};

const W: usize = 6;

fn rnd(m: usize, n: usize, seed: u64) -> Mat<f64> {
    let mut x = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    Mat::from_fn(m, n, |_, _| {
        x = x.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((x >> 11) as f64 / (1u64 << 53) as f64) - 0.5
    })
}

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(f64::total_cmp);
    v[v.len() / 2]
}

/// Median over 11 batches (each at least 4 ms) of the mean ns per call.
fn time_calls(mut call: impl FnMut()) -> f64 {
    let mut n = 1usize;
    loop {
        let t = Instant::now();
        for _ in 0..n {
            call();
        }
        if t.elapsed() >= Duration::from_millis(4) || n >= 1 << 22 {
            break;
        }
        n *= 2;
    }
    median(
        (0..11)
            .map(|_| {
                let t = Instant::now();
                for _ in 0..n {
                    call();
                }
                t.elapsed().as_nanos() as f64 / n as f64
            })
            .collect(),
    )
}

/// A workload: `run(par)` performs one call; `restore()` resets its inputs (timed and subtracted).
struct Work {
    family: &'static str,
    shape: String,
    run: Box<dyn FnMut(Par<'_>) + Send>,
    restore: Box<dyn FnMut() + Send>,
}

fn gemm_f64(family: &'static str, m: usize, n: usize, k: usize) -> Work {
    let a = rnd(m, k, 1);
    let b = rnd(k, n, 2);
    let mut c = Mat::<f64>::zeros(m, n);
    Work {
        family,
        shape: format!("{m}x{n}x{k}"),
        run: Box::new(move |par| matmul(c.as_mut(), Accum::Replace, a.as_ref(), b.as_ref(), 1.0, par)),
        restore: Box::new(|| {}),
    }
}

fn gemm_c64(n: usize) -> Work {
    let f = |s| {
        let (r, i) = (rnd(n, n, s), rnd(n, n, s + 50));
        Mat::<c64>::from_fn(n, n, |x, y| c64::new(r[(x, y)], i[(x, y)]))
    };
    let (a, b) = (f(1), f(2));
    let mut c = Mat::<c64>::zeros(n, n);
    Work {
        family: "gemm_c64",
        shape: format!("{n}x{n}x{n}"),
        run: Box::new(move |par| matmul(c.as_mut(), Accum::Replace, a.as_ref(), b.as_ref(), c64::new(1.0, 0.0), par)),
        restore: Box::new(|| {}),
    }
}

fn lu_f64(n: usize, off: bool) -> Work {
    lu_cfg(n, if off { 0 } else { 128 * 128 }, None)
}

/// LU with an explicit `par_threshold` and, when given, the `gemm` threading floor set before each call.
fn lu_cfg(n: usize, par_threshold: usize, floor: Option<usize>) -> Work {
    let src = rnd(n, n, 7);
    let w = Arc::new(Mutex::new(src.clone()));
    let w2 = w.clone();
    let mut perm = vec![0usize; n];
    let mut perm_inv = vec![0usize; n];
    let mut p = <lu::PartialPivLuParams as Auto<f64>>::auto();
    p.par_threshold = par_threshold;
    let nz = NonZeroUsize::new(W).unwrap();
    let r1 = lu::lu_in_place_scratch::<usize, f64>(n, n, faer::ParShape::Team(nz), p.into());
    let r2 = lu::lu_in_place_scratch::<usize, f64>(n, n, faer::ParShape::Rayon(nz), p.into());
    let bytes = r1.unaligned_bytes_required().max(r2.unaligned_bytes_required());
    let mut mem = MemBuffer::new(faer::dyn_stack::StackReq::new::<u8>(bytes + 256));
    Work {
        family: Box::leak(match floor {
            None => "lu_f64".to_string(),
            Some(f) => format!("lu_pt{}_gf{}", par_threshold.min(9_999_999), f),
        }.into_boxed_str()),
        shape: format!("{n}"),
        run: Box::new(move |par| {
            if let Some(f) = floor {
                gemm::set_threading_threshold(f);
            }
            let mut g = w.lock().unwrap();
            lu::lu_in_place(g.as_mut(), &mut perm, &mut perm_inv, par, MemStack::new(&mut mem), p.into());
        }),
        restore: Box::new(move || {
            let mut g = w2.lock().unwrap();
            g.as_mut().copy_from(src.as_ref());
        }),
    }
}

fn lu_sweep() -> Vec<Work> {
    let mut v = Vec::new();
    for (pt, gf) in [(16384, 589824), (0, 0), (16384, 0), (65536, 0), (262144, 0), (65536, 589824)] {
        for n in [128, 192, 256, 384, 512, 768] {
            v.push(lu_cfg(n, pt, Some(gf)));
        }
    }
    v
}

fn workloads(off: bool) -> Vec<Work> {
    let mut v = Vec::new();
    for n in [16, 24, 32, 40, 48, 56, 64, 80, 96, 112, 128, 160, 192, 256] {
        v.push(gemm_f64("gemm_sq", n, n, n));
    }
    for m in [64, 128, 256, 512, 1024, 2048, 4096, 8192] {
        v.push(gemm_f64("gemm_tall64", m, 64, 64));
    }
    for l in [32, 48, 64, 96, 128, 192, 256, 512, 1024] {
        v.push(gemm_f64("gemm_k16", l, l, 16));
    }
    for n in [16, 24, 32, 48, 64, 96, 128] {
        v.push(gemm_c64(n));
    }
    for n in [32, 48, 64, 96, 128, 192, 256, 384, 512, 768] {
        v.push(lu_f64(n, off));
    }
    v
}

#[derive(Clone, Copy, PartialEq)]
enum Mode {
    Seq,
    PlainOut,
    PlainIn,
    Team,
}

fn measure(mode: Mode, w: &mut Work) -> f64 {
    let restore = {
        let r = &mut w.restore;
        time_calls(|| r())
    };
    let one = NonZeroUsize::MIN;
    let _ = one;
    let ns = match mode {
        Mode::Seq => time_calls(|| {
            (w.restore)();
            (w.run)(Par::Seq)
        }),
        Mode::PlainOut => {
            let pool = POOL.with(|p| p.borrow().as_ref().map(|p| p as *const Pool).unwrap());
            // SAFETY: the pool outlives the phase that calls this
            let pool = unsafe { &*pool };
            let mut tok = Exec::new(pool, W).unwrap();
            time_calls(|| {
                (w.restore)();
                tok.reborrow().external(|n| (w.run)(Par::rayon(n.get())))
            })
        }
        Mode::PlainIn => {
            let pool = POOL.with(|p| p.borrow().as_ref().map(|p| p as *const Pool).unwrap());
            let pool = unsafe { &*pool };
            let tok = Exec::new(pool, W).unwrap();
            tok.external(|n| {
                time_calls(|| {
                    (w.restore)();
                    (w.run)(Par::rayon(n.get()))
                })
            })
        }
        Mode::Team => {
            let team = TEAM.with(|t| t.borrow().as_ref().map(|t| t as *const threadteam::Team).unwrap());
            let team = unsafe { &*team };
            let mut tok = team.exec(W);
            tok.team(|mut t| {
                time_calls(|| {
                    (w.restore)();
                    (w.run)(Par::Team(t.reborrow()))
                })
            })
            .unwrap()
        }
    };
    (ns - restore).max(0.0)
}

thread_local! {
    static POOL: std::cell::RefCell<Option<Pool>> = const { std::cell::RefCell::new(None) };
    static TEAM: std::cell::RefCell<Option<threadteam::Team>> = const { std::cell::RefCell::new(None) };
}

fn build() -> Pool {
    Pool::builder().threads(W).qos(Qos::UserInitiated).acquire_timeout(Duration::from_secs(30)).build().unwrap()
}

/// Count the team sections one call opens: `sections` subcommand, untimed.
fn sections() {
    println!("work,config,run_sections,split_or_fork_regions,gemm_tiled,gemm_below_floor,gemm_one_tile");
    let team = build().into_team(W).expect("team");
    let mut tok = team.exec(W);
    tok.team(|mut t| {
        let mut one = |name: String, cfg: &str, floor: usize, mut w: Work| {
            gemm::set_threading_threshold(floor);
            (w.restore)();
            (w.run)(Par::Team(t.reborrow()));
            (w.restore)();
            faer::utils::team::stats::enable(true);
            let _ = faer::utils::team::stats::take();
            (w.run)(Par::Team(t.reborrow()));
            faer::utils::team::stats::enable(false);
            let s = faer::utils::team::stats::take();
            let sum = |v: &Vec<usize>| v.iter().skip(2).sum::<usize>();
            println!("{name},{cfg},{},{},{},{},{}", sum(&s.run), sum(&s.split), s.gemm_tiled, s.gemm_below_floor, s.gemm_one_tile);
        };
        for n in [64, 96, 128, 256] {
            one(format!("gemm_sq {n}"), "gf0", 0, gemm_f64("gemm_sq", n, n, n));
        }
        for n in [64, 128, 192, 256, 384, 512, 768] {
            one(format!("lu {n}"), "pt0_gf0", 0, lu_cfg(n, 0, Some(0)));
            one(format!("lu {n}"), "default", 589824, lu_cfg(n, 16384, Some(589824)));
        }
    })
    .unwrap();
}

fn main() {
    if std::env::args().nth(1).as_deref() == Some("sections") {
        raise_current_thread_qos(Qos::UserInitiated);
        sections();
        return;
    }
    let mut args = std::env::args().skip(1);
    let first = args.next();
    let sweep = first.as_deref() == Some("lusweep");
    let off = first.as_deref() != Some("default");
    let rounds: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(3);
    if off {
        gemm::set_threading_threshold(0);
    }
    let _ = sweep;
    raise_current_thread_qos(Qos::UserInitiated);
    println!("family,shape,mode,round,ns");
    let mut works = if sweep { lu_sweep() } else { workloads(off) };
    for round in 0..rounds {
        for (mode, name) in [(Mode::Seq, "seq"), (Mode::PlainOut, "plain_out"), (Mode::PlainIn, "plain_in"), (Mode::Team, "team")] {
            if sweep && mode == Mode::PlainOut {
                continue;
            }
            match mode {
                Mode::PlainOut | Mode::PlainIn => POOL.with(|p| *p.borrow_mut() = Some(build())),
                Mode::Team => TEAM.with(|t| *t.borrow_mut() = Some(build().into_team(W).expect("team"))),
                Mode::Seq => {}
            }
            for w in works.iter_mut() {
                let ns = measure(mode, w);
                println!("{},{},{},{},{:.1}", w.family, w.shape, name, round, ns);
            }
            POOL.with(|p| *p.borrow_mut() = None);
            TEAM.with(|t| *t.borrow_mut() = None);
            std::thread::sleep(Duration::from_millis(300));
        }
    }
}
