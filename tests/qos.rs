//! Worker and leader quality of service on macOS.
#![cfg(all(target_vendor = "apple", feature = "parallel"))]

use std::collections::HashSet;
use std::sync::Mutex;
use std::time::Duration;
use threadteam::{raise_current_thread_qos, Exec, Pool, Qos};

use libc::qos_class_t as C;

fn current() -> C {
    let mut class = C::QOS_CLASS_UNSPECIFIED;
    let mut rel = 0;
    assert_eq!(
        unsafe { libc::pthread_get_qos_class_np(libc::pthread_self(), &mut class, &mut rel) },
        0
    );
    class
}

fn set(class: C) {
    assert_eq!(unsafe { libc::pthread_set_qos_class_self_np(class, 0) }, 0);
}

fn worker_classes(pool: &Pool) -> HashSet<u32> {
    let seen = Mutex::new(HashSet::new());
    std::thread::scope(|s| {
        s.spawn(|| {
            Exec::new(pool, pool.width().get())
                .unwrap()
                .fork(64, |_, _| {
                    seen.lock().unwrap().insert(current() as u32);
                    std::thread::sleep(Duration::from_micros(200));
                });
        });
    });
    seen.into_inner().unwrap()
}

#[test]
fn workers_report_the_configured_class_and_default_is_user_initiated() {
    let build = |q: Option<Qos>| {
        let mut b = Pool::builder().threads(3);
        if let Some(q) = q {
            b = b.qos(q);
        }
        b.build().unwrap()
    };
    let want = |c: C| HashSet::from([c as u32]);
    assert_eq!(
        worker_classes(&build(None)),
        want(C::QOS_CLASS_USER_INITIATED)
    );
    assert_eq!(
        worker_classes(&build(Some(Qos::Default))),
        want(C::QOS_CLASS_DEFAULT)
    );
    assert_eq!(
        worker_classes(&build(Some(Qos::Utility))),
        want(C::QOS_CLASS_UTILITY)
    );
    assert_eq!(
        worker_classes(&build(Some(Qos::UserInteractive))),
        want(C::QOS_CLASS_USER_INTERACTIVE)
    );
}

#[test]
fn raising_never_lowers_the_callers_class() {
    // Threads of their own: the class of the test thread must not leak.
    let run = |start: C, raise: Qos| {
        std::thread::spawn(move || {
            set(start);
            raise_current_thread_qos(raise);
            current()
        })
        .join()
        .unwrap()
    };
    assert_eq!(
        run(C::QOS_CLASS_UTILITY, Qos::UserInitiated) as u32,
        C::QOS_CLASS_USER_INITIATED as u32
    );
    assert_eq!(
        run(C::QOS_CLASS_DEFAULT, Qos::UserInitiated) as u32,
        C::QOS_CLASS_USER_INITIATED as u32
    );
    assert_eq!(
        run(C::QOS_CLASS_DEFAULT, Qos::UserInteractive) as u32,
        C::QOS_CLASS_USER_INTERACTIVE as u32
    );
    // Already higher or equal: unchanged.
    assert_eq!(
        run(C::QOS_CLASS_USER_INTERACTIVE, Qos::UserInitiated) as u32,
        C::QOS_CLASS_USER_INTERACTIVE as u32
    );
    assert_eq!(
        run(C::QOS_CLASS_USER_INITIATED, Qos::UserInitiated) as u32,
        C::QOS_CLASS_USER_INITIATED as u32
    );
    assert_eq!(
        run(C::QOS_CLASS_DEFAULT, Qos::Utility) as u32,
        C::QOS_CLASS_DEFAULT as u32
    );
}
