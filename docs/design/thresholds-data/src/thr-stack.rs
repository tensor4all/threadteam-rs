//! Break-even sweep for strided elementwise copy/map/reduction and a tcontract contraction:
//! serial vs plain token (outside the pool; inside the pool) vs team token, 6 participants.
//!
//! `thr-stack ROUNDS` prints CSV: family,shape,mode,round,ns
//!
//! Built against a strided-rs checkout whose `MINTHREADLENGTH` is lowered to 16 and tcontract with
//! its `WidthPolicy` set to zero for the `tc_off` family (the width rule then picks the full budget),
//! so the sweep shows where the parallel route itself starts to pay. `tc_default` keeps the
//! default policy (the width rule decides).
use std::sync::Mutex;
use std::time::{Duration, Instant};
use strided_basic::{copy_into, map_into, sum};
use strided_view::{StridedView, StridedViewMut};
use tcontract::api::{CSpec, DType, Labels, LayoutSpec, Op, OperandSpec, Problem};
use tcontract::{Plan, PlanConfig};
use tcontract_exec::{ArenaProvider, BatchPolicy, WidthPolicy, WorkModel};
use threadteam::{raise_current_thread_qos, Exec, Pool, Qos, Team};

const W: usize = 6;

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(f64::total_cmp);
    v[v.len() / 2]
}

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

struct Work {
    family: &'static str,
    shape: String,
    run: Box<dyn FnMut(Exec<'_>) + Send>,
}

fn strided(kind: &'static str, n: usize) -> Work {
    let src: Vec<f64> = (0..n).map(|i| (i % 97) as f64 * 0.01).collect();
    let mut dst = vec![0f64; n];
    let run: Box<dyn FnMut(Exec<'_>) + Send> = match kind {
        "copy" => Box::new(move |exec| {
            let s: StridedView<f64> = StridedView::new(&src, &[n], &[1], 0).unwrap();
            let mut d = StridedViewMut::new(&mut dst, &[n], &[1], 0).unwrap();
            copy_into(&mut d, &s, exec).unwrap();
        }),
        "map" => Box::new(move |exec| {
            let s: StridedView<f64> = StridedView::new(&src, &[n], &[1], 0).unwrap();
            let mut d = StridedViewMut::new(&mut dst, &[n], &[1], 0).unwrap();
            map_into(&mut d, &s, |x| x * 2.0 + 1.0, exec).unwrap();
        }),
        "exp" => Box::new(move |exec| {
            let s: StridedView<f64> = StridedView::new(&src, &[n], &[1], 0).unwrap();
            let mut d = StridedViewMut::new(&mut dst, &[n], &[1], 0).unwrap();
            map_into(&mut d, &s, |x| x.exp(), exec).unwrap();
        }),
        "sum" => Box::new(move |exec| {
            let s: StridedView<f64> = StridedView::new(&src, &[n], &[1], 0).unwrap();
            std::hint::black_box(sum(&s, exec).unwrap());
        }),
        _ => unreachable!(),
    };
    Work { family: kind, shape: format!("{n}"), run }
}

fn col_major(dims: &[usize]) -> OperandSpec {
    let mut strides = Vec::new();
    let mut acc = 1isize;
    for &d in dims {
        strides.push(acc);
        acc *= d as isize;
    }
    OperandSpec::new(LayoutSpec::new(dims, &strides, 0).unwrap()).with_op(Op::Identity)
}

fn tcontract_matmul(family: &'static str, n: usize, off: bool, arena: &'static ArenaProvider) -> Work {
    let (ia, ib, id): (Vec<i64>, Vec<i64>, Vec<i64>) = (vec![105, 107], vec![107, 106], vec![105, 106]);
    let dims = [n, n];
    let problem = Problem::from_labels(
        DType::F64,
        col_major(&dims),
        col_major(&dims),
        CSpec::Absent,
        col_major(&dims),
        &Labels::new(&ia, &ib, &id),
    )
    .unwrap();
    let plan = Plan::<f64>::new(&problem, &PlanConfig::default()).unwrap();
    let a: Vec<f64> = (0..n * n).map(|i| (i % 13) as f64 * 0.1).collect();
    let b: Vec<f64> = (0..n * n).map(|i| (i % 7) as f64 * 0.1).collect();
    let mut d = vec![0f64; n * n];
    let st = [1isize, n as isize];
    let policy = if off {
        BatchPolicy::default().with_model(WorkModel::default().with_width(WidthPolicy {
            serial_below_ns: 0.0,
            entry_base_ns: 0.0,
            entry_per_thread_ns: 0.0,
        }))
    } else {
        BatchPolicy::default()
    };
    Work {
        family,
        shape: format!("{n}"),
        run: Box::new(move |tok| {
            let mut exec = tcontract_exec::Exec::from_token(tok, Some(arena), policy);
            let va = StridedView::new(&a, &dims, &st, 0).unwrap();
            let vb = StridedView::new(&b, &dims, &st, 0).unwrap();
            let mut vd = StridedViewMut::new(&mut d, &dims, &st, 0).unwrap();
            plan.execute_into(&mut exec, 1.0, &va, &vb, &mut vd).unwrap();
        }),
    }
}

#[derive(Clone, Copy, PartialEq)]
enum Mode {
    Seq,
    PlainOut,
    PlainIn,
    Team,
}

static POOL: Mutex<Option<Pool>> = Mutex::new(None);
static TEAM: Mutex<Option<Team>> = Mutex::new(None);

fn measure(mode: Mode, w: &mut Work) -> f64 {
    match mode {
        Mode::Seq => time_calls(|| (w.run)(Exec::serial())),
        Mode::PlainOut => {
            let g = POOL.lock().unwrap();
            let pool = g.as_ref().unwrap();
            let mut tok = Exec::new(pool, W).unwrap();
            time_calls(|| (w.run)(tok.reborrow()))
        }
        Mode::PlainIn => {
            let g = POOL.lock().unwrap();
            let pool = g.as_ref().unwrap();
            let tok = Exec::new(pool, W).unwrap();
            tok.external(|_| {
                let mut t = Exec::new(pool, W).unwrap();
                time_calls(|| (w.run)(t.reborrow()))
            })
        }
        Mode::Team => {
            let g = TEAM.lock().unwrap();
            let team = g.as_ref().unwrap();
            let mut tok = team.exec(W);
            time_calls(|| (w.run)(tok.reborrow()))
        }
    }
}

fn build() -> Pool {
    Pool::builder().threads(W).qos(Qos::UserInitiated).acquire_timeout(Duration::from_secs(30)).build().unwrap()
}

fn main() {
    let rounds: usize = std::env::args().nth(1).and_then(|s| s.parse().ok()).unwrap_or(3);
    raise_current_thread_qos(Qos::UserInitiated);
    let arena: &'static ArenaProvider = Box::leak(Box::new(ArenaProvider::new()));
    let mut works = Vec::new();
    for kind in ["copy", "map", "exp", "sum"] {
        for e in 8..=21 {
            works.push(strided(kind, 1usize << e));
        }
    }
    for n in [8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384] {
        works.push(tcontract_matmul("tc_off", n, true, arena));
    }
    for n in [8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384] {
        works.push(tcontract_matmul("tc_default", n, false, arena));
    }
    println!("family,shape,mode,round,ns");
    for round in 0..rounds {
        for (mode, name) in [(Mode::Seq, "seq"), (Mode::PlainOut, "plain_out"), (Mode::PlainIn, "plain_in"), (Mode::Team, "team")] {
            match mode {
                Mode::PlainOut | Mode::PlainIn => *POOL.lock().unwrap() = Some(build()),
                Mode::Team => *TEAM.lock().unwrap() = Some(build().into_team(W).expect("team")),
                Mode::Seq => {}
            }
            for w in works.iter_mut() {
                let ns = measure(mode, w);
                println!("{},{},{},{},{:.1}", w.family, w.shape, name, round, ns);
            }
            *POOL.lock().unwrap() = None;
            *TEAM.lock().unwrap() = None;
            std::thread::sleep(Duration::from_millis(300));
        }
    }
}
