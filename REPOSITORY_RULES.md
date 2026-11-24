# Repository rules

1. Only this crate depends on `rayon` among tensor4all crates; `deny.toml`
   bans `rayon` for everything else and bans `faer`, `t4a-faer` and `spindle`
   outright. `Exec::external` hands a plain width to such libraries.
2. No type from `rayon` appears in the public API (no constructor from, no
   accessor returning, a scheduler pool).
3. `Exec` is not `Clone` or `Copy`; every parallel entry consumes it; splits
   conserve slots (every share is positive and the shares sum to at most the
   parent's budget; `fork`'s shares also differ by at most one and sum to
   exactly the budget).
4. No ambient thread-local state (the placement record `pin::PLACED` is the one
   thread-local, a non-allocating `Cell`) and no implicit global pool; lazy
   inline execution for serial, budget-one and single-task cases.
5. The steady-state entries `fork`, `fork_sized`, `fork_sized_slots`,
   `for_ranges` and `external` do not allocate, with one exception: such an
   entry made from a worker of another Rayon pool allocates the one job that
   `ThreadPool::spawn` boxes when it submits the work, and nothing else.
   `tests/alloc_free.rs` pins both, the exception at one per entry.
   `broadcast` is outside this rule: Rayon's `ThreadPool::broadcast` collects
   one job per worker into a `Vec` on every call.
6. API changes keep the contracts in `README.md` and `docs/design/placement.md`
   in sync.
