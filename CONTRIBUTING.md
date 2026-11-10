<!-- unified contribution policy: ../../docs/contributing/rust-repositories.md in tensor4all/tensor4all-meta -->
# Contributing to threadteam-rs

The rules that apply to every Rust repository in `tensor4all` are in one place:
**[tensor4all-meta · docs/contributing/rust-repositories.md](https://github.com/tensor4all/tensor4all-meta/blob/main/docs/contributing/rust-repositories.md)**.
Read that first — it covers the issue-first workflow, what a pull request body has to contain, red-first tests, the
gates every repository runs, the review requirement, and the version policy.

This file only records what is specific to `threadteam-rs`:

## Local facts

- **MSRV**: none is declared — no `rust-version` in `Cargo.toml` and no `rust-toolchain.toml`. CI pins `stable`, and
  `nightly` only in the `placement` lane that runs `cargo-public-api`, so there is no MSRV claim for CI to hold and
  no `msrv` job.
- **Feature combinations CI builds**: `cargo test` (default, which is `parallel`), `cargo test --all-features` and
  `cargo test --no-default-features`, each on Linux and macOS; `clippy --all-targets` with `--all-features` and with
  `--no-default-features`; and a `loom` lane running `RUSTFLAGS="--cfg loom" cargo test --release --test loom_park`.
  The `ledger`, `test-hooks` and `trace` features are test support and are enabled only by the test lanes.
- **Platforms claimed**: `ubuntu-latest` and `macos-latest`. The crate has Linux-only (`sched_getaffinity`) and
  Apple-only (`pthread_set_qos_class_self_np`) code; no other target is built.
- **Local gate**: `cargo test --all-features`.

## Anything here that departs from the shared rules

One does: the `msrv` gate is not run, because no `rust-version` is declared for it to hold CI to. Declaring one and
adding the job is the way to close that gap; this file records the state and does not exempt the repository from the
rest of the gates.

This repository adds a `loom` model of the park/wake handshake and a `deny`/`cargo-public-api` placement check (no
`rayon` type in the public API).
