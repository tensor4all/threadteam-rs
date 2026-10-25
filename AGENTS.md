# AGENTS.md

Before acting, read the latest shared tensor4all agent rules from the
[`tensor4all-agent-rules`](https://github.com/tensor4all/tensor4all-agent-rules)
repository. Start from:

- `https://github.com/tensor4all/tensor4all-agent-rules/blob/main/rules/index.md`

If internet access is unavailable, use the sibling checkout
`../tensor4all-agent-rules/rules/index.md`. Load only the rule files relevant
to the task (common, Rust, performance, docs). `rules/common/provenance.md`
applies whenever code is written while referencing third-party code.

## Project overview

`threadteam` is a single-crate repository: a scoped CPU thread-team placement
token (`Pool`, non-`Clone` `Exec` with `fork` / `for_ranges` / `external` /
`broadcast`). It is **the only crate in the tensor4all stack that starts
tasks**. strided-rs, tcontract-rs, cpueinsum-rs, tlinalg-rs and tenferro-rs
take an `Exec` and split it; none of them depends on Rayon directly. The code is
original; there is no third-party provenance beyond the licenses.

Design record: `docs/design/placement.md`. Durable invariants:
`REPOSITORY_RULES.md`.

## Invariants

- **Budget conservation**: tasks running at once under one root token never
  exceed its budget at any nesting depth; splits conserve slots.
- **No implicit global pool**: serial, budget-one and one-task entries run
  inline; the process-global Rayon pool is never started.
- **No Rayon type is public**: `scripts/check-no-public-rayon.sh` must pass.
- **Allocation-free steady state**: `tests/alloc_free.rs` must pass; do not add
  allocation to the entry paths.
- `Exec` stays neither `Clone` nor `Copy` (compile-fail doctests).

## Pre-push checklist

```bash
cargo fmt --all -- --check
cargo clippy --all-targets --all-features -- -D warnings
cargo clippy --all-targets --no-default-features -- -D warnings
cargo test
cargo test --all-features
cargo test --no-default-features
cargo deny check bans
scripts/check-no-public-rayon.sh      # needs cargo-public-api and a nightly toolchain
cargo doc --no-deps
```
