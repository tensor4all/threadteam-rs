//! `PinStrategy` is a closed enum, and this file is what makes that true.
//!
//! An integration test compiles as its own crate, so the `match` below is an
//! *external* one. If `#[non_exhaustive]` is ever put back on the enum, this
//! file stops compiling with `E0004` ("non-exhaustive patterns: `_` not
//! covered") instead of the regression passing silently — a unit test inside
//! the crate cannot detect it, because Rust lets the defining crate match
//! exhaustively regardless of the attribute.
//!
//! Cost of the rule: adding a variant is a minor bump. That is the trade we
//! chose (`CHANGELOG.md`, "The enum is closed").

use threadteam::PinStrategy;

/// No `_ =>` arm, on purpose. Do not add one.
fn rank(strategy: PinStrategy) -> u8 {
    match strategy {
        PinStrategy::Spread => 0,
        PinStrategy::Pack => 1,
    }
}

#[test]
fn every_strategy_is_matched_without_a_wildcard() {
    assert_eq!(rank(PinStrategy::Spread), 0);
    assert_eq!(rank(PinStrategy::Pack), 1);
}
