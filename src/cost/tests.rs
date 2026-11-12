use super::*;

fn nz(n: usize) -> NonZeroUsize {
    NonZeroUsize::new(n).unwrap()
}
fn us(n: u64) -> Duration {
    Duration::from_micros(n)
}

#[test]
fn anchors_are_returned_exactly() {
    let c = DispatchCosts::for_class(HostClass::Aarch64Apple);
    assert_eq!(c.cost(DispatchKind::Team, nz(2)), us_f(3.0));
    assert_eq!(c.cost(DispatchKind::Team, nz(4)), us_f(2.3));
    assert_eq!(c.cost(DispatchKind::Team, nz(6)), us_f(1.4));
    assert_eq!(c.cost(DispatchKind::Outside, nz(6)), us_f(19.0));
}

#[test]
fn interpolation_and_extrapolation() {
    let c = WidthCosts::new([us(3), us(2), us(1)]);
    assert_eq!(c.at(nz(1)), us(3));
    assert_eq!(c.at(nz(3)), Duration::from_nanos(2500));
    assert_eq!(c.at(nz(5)), Duration::from_nanos(1500));
    // Above the last anchor the cost grows with the width.
    assert_eq!(c.at(nz(8)), Duration::from_nanos(1333));
    assert_eq!(c.at(nz(12)), us(2));
    // Non-monotone anchors interpolate without underflow.
    let n = WidthCosts::new([us(1), us(5), us(0)]);
    assert_eq!(n.at(nz(5)), Duration::from_nanos(2500));
    assert_eq!(n.at(nz(12)), Duration::ZERO);
}

#[test]
fn a_flat_row_extrapolates_in_proportion_to_the_width() {
    let c = WidthCosts::flat(us(1));
    assert_eq!(c.at(nz(6)), us(1));
    assert_eq!(c.at(nz(12)), us(2));
    assert_eq!(c.at(nz(24)), us(4));
    assert_eq!(c.at(nz(48)), us(8));
    assert_eq!(c.at(nz(64)), Duration::from_nanos(10_666));
    assert_eq!(c.min(), us(1));
}

/// Above the last anchor `w_m` the cost never falls and is at least
/// `cost(w_m) * w / w_m`.
#[test]
fn above_the_last_anchor_is_non_decreasing_and_at_least_linear() {
    let rows: [[Duration; 3]; 5] = [
        [us_f(0.6), us_f(0.7), us_f(0.9)],
        [us_f(3.0), us_f(3.6), us_f(3.5)],
        [us_f(9.7), us_f(10.1), us_f(8.7)],
        [us(3), us(2), us(1)],
        [us(1), us(5), us(0)],
    ];
    for row in rows {
        let c = WidthCosts::new(row);
        let cm = u128::from(to_ns(c.at(nz(6))));
        let mut prev = c.at(nz(6));
        for w in 7..=64u64 {
            let v = c.at(nz(w as usize));
            assert!(v >= prev, "{row:?}: width {w} fell");
            assert!(
                u128::from(to_ns(v)) >= cm * u128::from(w) / 6,
                "{row:?}: width {w} below the linear bound"
            );
            prev = v;
        }
    }
}

#[test]
fn anchors_beyond_six_interpolate_and_extrapolate() {
    let c = WidthCosts::from_anchors(&[
        (2, Duration::from_nanos(8000)),
        (4, Duration::from_nanos(4000)),
        (8, Duration::from_nanos(2000)),
        (16, Duration::from_nanos(1000)),
        (32, Duration::from_nanos(500)),
        (64, Duration::from_nanos(500)),
    ])
    .unwrap();
    let ns = |w: usize| c.at(nz(w)).as_nanos() as u64;
    assert_eq!(ns(2), 8000);
    assert_eq!(ns(3), 6000); // between 2 and 4
    assert_eq!(ns(6), 3000); // between 4 and 8, past the old last anchor
    assert_eq!(ns(12), 1500); // between 8 and 16
    assert_eq!(ns(24), 750); // between 16 and 32
    assert_eq!(ns(48), 500); // between 32 and 64
    assert_eq!(ns(64), 500); // the last anchor
    assert_eq!(ns(96), 750); // above the last anchor
    assert_eq!(ns(128), 1000);
    assert_eq!(c.min(), Duration::from_nanos(500));
}

#[test]
fn from_anchors_rejects_invalid_rows() {
    assert!(WidthCosts::from_anchors(&[]).is_none());
    assert!(WidthCosts::from_anchors(&[(0, us(1))]).is_none());
    assert!(WidthCosts::from_anchors(&[(1, us(1)), (2, us(1))]).is_none());
    assert!(WidthCosts::from_anchors(&[(2, us(1)), (2, us(1))]).is_none());
    assert!(WidthCosts::from_anchors(&[(4, us(1)), (2, us(1))]).is_none());
    let too_wide = [(usize::from(u16::MAX) + 1, us(1))];
    assert!(WidthCosts::from_anchors(&too_wide).is_none());
    assert!(WidthCosts::from_anchors(&[(usize::from(u16::MAX), us(1))]).is_some());
    // MAX_ANCHORS anchors fit, one more is rejected.
    let mut many = [(0usize, Duration::ZERO); WidthCosts::MAX_ANCHORS + 1];
    for (k, a) in many.iter_mut().enumerate() {
        *a = (k + 2, us(1));
    }
    assert!(WidthCosts::from_anchors(&many[..WidthCosts::MAX_ANCHORS]).is_some());
    assert!(WidthCosts::from_anchors(&many).is_none());
}

/// `from_anchors` is usable in a `const` item and the unused capacity does
/// not take part in equality.
#[test]
fn from_anchors_is_const_and_capacity_is_invisible() {
    const ROW: WidthCosts = match WidthCosts::from_anchors(&[
        (2, Duration::from_micros(3)),
        (4, Duration::from_micros(2)),
        (8, Duration::from_micros(1)),
    ]) {
        Some(c) => c,
        None => panic!("valid anchors"),
    };
    assert_eq!(WidthCosts::MAX_ANCHORS, 12);
    assert_eq!(ROW.at(nz(6)), Duration::from_nanos(1500));
    assert_eq!(ROW.at(nz(16)), us(2));
    assert_eq!(ROW.min(), us(1));

    let three = WidthCosts::from_anchors(&[(2, us(3)), (4, us(2)), (6, us(1))]).unwrap();
    assert_eq!(three, WidthCosts::new([us(3), us(2), us(1)]));
    let four = WidthCosts::from_anchors(&[(2, us(3)), (4, us(2)), (6, us(1)), (8, us(1))]).unwrap();
    assert_ne!(three, four);
}

#[test]
fn a_duration_max_row_never_pays() {
    // `Duration::MAX` saturates to `u64::MAX` nanoseconds in a row.
    let cap = Duration::from_nanos(u64::MAX);
    let c = WidthCosts::flat(Duration::MAX);
    assert_eq!(c.at(nz(6)), cap);
    assert_eq!(c.at(nz(64)), cap); // the extrapolation saturates
    assert_eq!(c.min(), cap);
    assert!(!pays_at(cap, 1, c.at(nz(64))));
    assert!(!pays_at(cap, 2, c.at(nz(64))));
    let mixed = WidthCosts::from_anchors(&[(2, Duration::ZERO), (4, Duration::MAX)]).unwrap();
    assert_eq!(mixed.at(nz(64)), cap);
    assert_eq!(mixed.min(), Duration::ZERO);
}

/// A wide region selects a useful interior width above six, not always six
/// or the budget.
#[test]
fn a_wide_region_selects_an_interior_width_above_six() {
    let row = WidthCosts::flat(us(1));
    let d = |k: NonZeroUsize| row.at(k);
    let budget = nz(64);
    // obj(k) = D(k) + 24 us / k with D(k) = 1 us * k / 6 above 6: obj(12) =
    // 2 + 2 = 4 us against obj(6) = 1 + 4 = 5 us, obj(11) = 1.833 + 2.18 and
    // obj(13) = 2.167 + 1.85.
    assert_eq!(width_for_by(us(24), 1, budget, d).get(), 12);
    // A large enough estimate pushes the optimum to the budget.
    assert_eq!(width_for_by(us(1_000_000), 1, budget, d).get(), 64);
    // A region that does not pay at width 6 does not pay at 64 either.
    let below = Duration::from_nanos(3 * 1000 - 1);
    assert!(!pays_at(below, 1, row.at(nz(6))));
    assert!(!pays_at(below, 1, row.at(nz(64))));
    assert_eq!(width_for_by(below, 1, budget, d).get(), 1);
}

/// The model as it was before the anchors were stored: i128 interpolation on
/// every call, flat above width 6.
fn old_at(ns: [u64; 3], w: u64) -> u64 {
    if w <= 2 {
        ns[0]
    } else if w >= 6 {
        ns[2]
    } else {
        let i = usize::from(w > 4);
        let (w0, w1) = ([2u64, 4, 6][i], [2u64, 4, 6][i + 1]);
        let (c0, c1) = (i128::from(ns[i]), i128::from(ns[i + 1]));
        (c0 + (c1 - c0) * i128::from(w - w0) / i128::from(w1 - w0)).max(0) as u64
    }
}

#[test]
fn rows_keep_the_old_values_inside_the_anchors() {
    let tables = [
        [3000, 2300, 1400],
        [2800, 2200, 1700],
        [14000, 18000, 19000],
        [9500, 10100, 8800],
        [3100, 3600, 3500],
        [1000, 5000, 0],
        [0, 0, 0],
        [7, 3, 11],
    ];
    for ns in tables {
        let c = WidthCosts::new(ns.map(Duration::from_nanos));
        for w in 1..=6u64 {
            assert_eq!(
                c.at(nz(w as usize)),
                Duration::from_nanos(old_at(ns, w)),
                "{ns:?} width {w}"
            );
        }
    }
    // The minimum over the anchors is the minimum over all widths: neither
    // interpolation nor the extrapolation can undershoot its endpoint.
    for d in [
        DispatchCosts::for_class(HostClass::Aarch64Apple),
        DispatchCosts::for_class(HostClass::X86_64Avx512),
        DispatchCosts::for_class(HostClass::Generic),
    ] {
        for kind in [
            DispatchKind::Team,
            DispatchKind::InPool,
            DispatchKind::Outside,
        ] {
            let m = (1..=64).map(|w| d.cost(kind, nz(w))).min().unwrap();
            assert_eq!(d.min_cost(kind), m);
        }
    }
}

/// The measured rows at widths 1 to 6 as the pre-anchor model gave them, so
/// this change moves no decision inside the anchors.
#[test]
fn measured_rows_are_pinned_at_widths_one_to_six() {
    let avx512 = DispatchCosts::for_class(HostClass::X86_64Avx512);
    let apple = DispatchCosts::for_class(HostClass::Aarch64Apple);
    let generic = DispatchCosts::for_class(HostClass::Generic);
    for (costs, kind, expected) in [
        (
            avx512,
            DispatchKind::Team,
            [600u64, 600, 650, 700, 800, 900],
        ),
        (
            avx512,
            DispatchKind::InPool,
            [3000, 3000, 3300, 3600, 3550, 3500],
        ),
        (
            avx512,
            DispatchKind::Outside,
            [9700, 9700, 9900, 10100, 9400, 8700],
        ),
        (
            apple,
            DispatchKind::Team,
            [3000, 3000, 2650, 2300, 1850, 1400],
        ),
        (
            apple,
            DispatchKind::InPool,
            [6000, 6000, 8500, 11000, 8250, 5500],
        ),
        (
            apple,
            DispatchKind::Outside,
            [14000, 14000, 16000, 18000, 18500, 19000],
        ),
        (generic, DispatchKind::Team, [3000; 6]),
        (generic, DispatchKind::InPool, [15000; 6]),
        (generic, DispatchKind::Outside, [40000; 6]),
    ] {
        for (k, want) in expected.into_iter().enumerate() {
            assert_eq!(
                costs.cost(kind, nz(k + 1)).as_nanos() as u64,
                want,
                "{kind:?} width {}",
                k + 1
            );
        }
    }
    // The measured width-8 team anchor replaces the extrapolation there;
    // width 7 interpolates between the width-6 and width-8 team anchors.
    assert_eq!(avx512.cost(DispatchKind::Team, nz(8)), us_f(0.98));
    assert_eq!(
        avx512.cost(DispatchKind::Team, nz(7)),
        Duration::from_nanos(940)
    );
    assert_eq!(
        avx512.cost(DispatchKind::Team, nz(64)),
        Duration::from_nanos(7_840)
    );
    // The width-8 values in pool and outside come from the pre-fix run and
    // are not anchors: the extrapolation applies to them.
    assert_eq!(
        avx512.cost(DispatchKind::InPool, nz(8)),
        Duration::from_nanos(4666)
    );
    assert_eq!(
        avx512.cost(DispatchKind::Outside, nz(8)),
        Duration::from_nanos(11_600)
    );
}

#[test]
fn min_is_right_for_a_rising_table_and_a_falling_one() {
    let rising = WidthCosts::new([us(2), us(5), us(9)]);
    assert_eq!(rising.min(), us(2));
    let falling = WidthCosts::new([us(9), us(5), us(2)]);
    assert_eq!(falling.min(), us(2));
    let dip = WidthCosts::new([us(9), us(1), us(9)]);
    assert_eq!(dip.min(), us(1));
    assert_eq!(WidthCosts::flat(us(4)).min(), us(4));
}

#[test]
fn pays_matches_the_float_rule() {
    assert_eq!(PAYS, PAYS_INT as f64);
    for cost in [1u64, 700, 1700, 2800, 19_000] {
        for sections in [1u32, 2, 5] {
            let edge = 3 * u64::from(sections) * cost;
            for est in [edge - 1, edge, edge + 1] {
                let f = est as f64 >= PAYS * f64::from(sections) * cost as f64;
                assert_eq!(
                    pays_at(
                        Duration::from_nanos(est),
                        sections,
                        Duration::from_nanos(cost)
                    ),
                    f
                );
            }
        }
    }
}

#[test]
fn pays_is_the_multiple_of_the_cost() {
    let d = us(2);
    assert!(pays_at(us(6), 1, d));
    assert!(!pays_at(Duration::from_nanos(5999), 1, d));
    assert!(pays_at(us(12), 2, d));
    assert!(!pays_at(us(11), 2, d));
    assert!(!pays_at(us(1_000_000), 1, Duration::MAX));
    assert!(pays_at(Duration::ZERO, 0, d));
}

#[test]
fn width_for_minimises_the_model() {
    let budget = nz(6);
    let d = |k: NonZeroUsize| us(2) * k.get() as u32;
    // 2 us * k + est / k: for est = 200 us the optimum is k = 10 clamped to
    // the budget; for est = 50 us it is k = 5 (obj 20) against k = 4 (20.5)
    // and k = 6 (20.3).
    assert_eq!(width_for_by(us(200), 1, budget, d).get(), 6);
    assert_eq!(width_for_by(us(50), 1, budget, d).get(), 5);
    // Below the multiple at every width: serial.
    assert_eq!(width_for_by(us(5), 1, budget, d).get(), 1);
    // More sections move the optimum down.
    assert!(width_for_by(us(200), 4, budget, d).get() < 6);
    // A serial cost never pays.
    assert_eq!(
        width_for_by(us(1_000_000), 1, budget, |_| Duration::MAX).get(),
        1
    );
    assert_eq!(width_for_by(us(1_000_000), 1, nz(1), d).get(), 1);
}

#[test]
fn wide_values_take_the_128_bit_path_with_the_same_result() {
    // Values whose products overflow 64 bits use the 128-bit fallback; it
    // must give what the 128-bit formula gives.
    let big = Duration::from_nanos(u64::MAX / 3);
    let huge = Duration::from_nanos(u64::MAX / 2);
    let c = WidthCosts::from_anchors(&[(2, big), (64, huge)]).unwrap();
    let (c0, c1) = (i128::from(u64::MAX / 3), i128::from(u64::MAX / 2));
    for w in [3usize, 17, 33, 63] {
        let want = c0 + (c1 - c0) * (w as i128 - 2) / 62;
        assert_eq!(u128::from(c.ns_at(nz(w))), want as u128, "width {w}");
    }
    // Above the last anchor the product saturates instead of wrapping.
    assert_eq!(c.ns_at(nz(200)), u64::MAX);
    let want = u128::from(u64::MAX / 2) * 100 / 64;
    assert_eq!(u128::from(c.ns_at(nz(100))), want.min(u128::from(u64::MAX)));
}

#[test]
fn falling_interpolation_truncates_toward_zero() {
    // 4 + (0 - 4) * 1 / 3 = 4 - 1.33..: truncation toward zero gives 3, floor
    // rounding would give 2. Pins the old model's integer semantics.
    let ns = Duration::from_nanos;
    let c = WidthCosts::from_anchors(&[(2, ns(4)), (5, ns(0))]).unwrap();
    assert_eq!(c.at(nz(3)), ns(3));
    assert_eq!(c.at(nz(4)), ns(2));
}

#[test]
fn a_transfer_charge_is_one_drain_at_the_row_rate() {
    let rate = TransferRate::per_kib(41);
    // 1 MiB at 41 ns/KiB is 41 * 1024 ns.
    assert_eq!(rate.time(1 << 20), Duration::from_nanos(41 * 1024));
    assert_eq!(rate.ns_of_bytes(0), 0);
    // Saturation, not wrap.
    assert_eq!(
        TransferRate::per_kib(u64::MAX).time(u64::MAX),
        Duration::from_nanos(u64::MAX)
    );
    // The measured rows: the M5 is charged less per KiB than a die of the
    // measured Xeon, and an unmeasured class takes the slowest of them.
    assert!(TransferRate::for_class(HostClass::Aarch64Apple).ns_per_kib() < 41);
    assert_eq!(
        TransferRate::for_class(HostClass::Generic),
        TransferRate::per_kib(41)
    );
}

#[test]
fn zero_resident_bytes_is_exactly_the_width_model() {
    let rate = TransferRate::per_kib(41);
    let costs = DispatchCosts::for_class(HostClass::Aarch64Apple);
    for est_us in [1u64, 7, 40, 1_000, 20_000] {
        let est = us(est_us);
        for sections in [1u32, 4] {
            let split = SplitCost {
                estimate: est,
                sections,
                resident_bytes: 0,
            };
            assert_eq!(
                width_for_split_by(split, nz(6), |k| costs.cost(DispatchKind::Team, k), rate),
                width_for_by(est, sections, nz(6), |k| costs.cost(DispatchKind::Team, k)),
                "estimate {est_us} us, {sections} sections"
            );
        }
    }
}

#[test]
fn a_resident_source_makes_a_region_stay_serial() {
    // A 2 ms region over 64 MiB it has just written: the drain is 2.6 ms at
    // the default row, more than the region costs on one thread.
    let rate = TransferRate::per_kib(41);
    let est = Duration::from_millis(2);
    let cold = width_for_by(est, 1, nz(32), |_| Duration::ZERO);
    let hot = width_for_split_by(
        SplitCost {
            estimate: est,
            sections: 1,
            resident_bytes: 64 << 20,
        },
        nz(32),
        |_| Duration::ZERO,
        rate,
    );
    assert_eq!(cold.get(), 32);
    assert_eq!(hot.get(), 1);
}

#[test]
fn the_measured_m5_widths_at_1_mib() {
    // 1 MiB hot copy on the M5 Max (`benches/resident_split.rs`,
    // 2026-11-03): 15.4 us serial, 5.96 us at k = 4, 6.29 us at k = 8. The
    // model keeps the region parallel and lands on its widest width.
    let costs = DispatchCosts::for_class(HostClass::Aarch64Apple);
    let split = SplitCost {
        estimate: Duration::from_nanos(15_375),
        sections: 1,
        resident_bytes: 1 << 20,
    };
    let w = width_for_split_by(
        split,
        nz(6),
        |k| costs.cost(DispatchKind::Team, k),
        TransferRate::for_class(HostClass::Aarch64Apple),
    );
    assert_eq!(w.get(), 6);
    // The transfer is 6 us of the 15.4 us region at this width, and it does
    // not change the choice the plain model makes: the term refines the
    // width where the drain is comparable to the work, it does not veto a
    // section that plainly pays.
    assert_eq!(
        w,
        width_for_by(split.estimate, 1, nz(6), |k| costs
            .cost(DispatchKind::Team, k))
    );
}

#[test]
fn the_measured_die_split_at_392_kib_stays_serial() {
    // threadteam-rs#27: on one die of the measured Xeon, 392 KiB splits
    // never beat the 9.2 us serial copy at k = 2 (11.6 us at best).
    let costs = DispatchCosts::for_class(HostClass::X86_64Avx512);
    let split = SplitCost {
        estimate: Duration::from_nanos(9_200),
        sections: 1,
        resident_bytes: 392 << 10,
    };
    let w = width_for_split_by(
        split,
        nz(24),
        |k| costs.cost(DispatchKind::Team, k),
        TransferRate::for_class(HostClass::X86_64Avx512),
    );
    assert_eq!(w.get(), 1);
}

/// threadteam-rs#68: the rows this crate builds carry no multi-node term, so
/// every node count is priced exactly as `at`, the single-node row.
#[test]
fn the_default_rows_price_every_node_count_as_one_node() {
    let classes = [
        HostClass::X86_64Avx512,
        HostClass::X86_64Avx2,
        HostClass::X86_64Scalar,
        HostClass::Aarch64Apple,
        HostClass::Aarch64,
        HostClass::Generic,
    ];
    for class in classes {
        let row = *DispatchCosts::for_class(class).team_row();
        assert!(!row.has_multi_node(), "{class:?}");
        for w in 2..=128 {
            for n in 1..=4 {
                assert_eq!(
                    row.at_nodes(nz(w), nz(n)),
                    row.at(nz(w)),
                    "{class:?} w={w} n={n}"
                );
            }
        }
    }
    // Today's x86_64-avx512 team row, exactly, on one node and on four.
    let row = *DispatchCosts::for_class(HostClass::X86_64Avx512).team_row();
    for (w, ns) in [
        (2, 600),
        (4, 700),
        (6, 900),
        (8, 980),
        (32, 3920),
        (96, 11_760),
    ] {
        for n in [1, 4] {
            assert_eq!(
                row.at_nodes(nz(w), nz(n)),
                Duration::from_nanos(ns),
                "w={w} n={n}"
            );
        }
    }
    assert_eq!(
        WidthCosts::flat(us(1)).with_multi_node(Duration::ZERO, Duration::ZERO),
        WidthCosts::flat(us(1))
    );
}

/// threadteam-rs#68 shape on the 4-node Xeon 9242 fixture: the single-node
/// anchors 8/16/24 = 3.30/4.63/5.90 us and the term fitted from 4-node minus
/// 1-node at the same widths (1.31 us + 221 ns per participant). Two and four
/// nodes take the same step; one node is the row.
#[test]
fn a_multi_node_term_is_one_step_with_its_own_slope() {
    let ns = Duration::from_nanos;
    let row = WidthCosts::from_anchors(&[(8, ns(3300)), (16, ns(4630)), (24, ns(5900))])
        .unwrap()
        .with_multi_node(ns(1310), ns(221));
    assert!(row.has_multi_node());
    for w in [2, 8, 16, 24, 32, 48, 96] {
        assert_eq!(row.at_nodes(nz(w), nz(1)), row.at(nz(w)), "w={w}");
        let term = ns(1310 + 221 * w as u64);
        assert_eq!(row.at_nodes(nz(w), nz(2)), row.at(nz(w)) + term, "w={w}");
        assert_eq!(
            row.at_nodes(nz(w), nz(4)),
            row.at_nodes(nz(w), nz(2)),
            "w={w}"
        );
    }
    assert_eq!(row.at_nodes(nz(8), nz(2)), ns(6378));
    assert_eq!(row.at_nodes(nz(24), nz(4)), ns(12_514));
    // Above the last anchor the row extrapolates as before, the term on top.
    assert_eq!(row.at_nodes(nz(48), nz(2)), ns(11_800 + 1310 + 221 * 48));
    // The cheap rejection stays a lower bound.
    assert_eq!(row.min(), ns(3300));
    // Saturates instead of wrapping.
    let huge = WidthCosts::flat(us(1)).with_multi_node(Duration::MAX, Duration::MAX);
    assert_eq!(huge.at_nodes(nz(64), nz(2)), Duration::from_nanos(u64::MAX));
}
