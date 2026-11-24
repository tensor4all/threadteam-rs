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
5. A steady-state entry allocates nothing of its own: whatever it allocates is
   allocated by Rayon inside the call that hands its work to the pool. (In the
   current code those allocations are the job `ThreadPool::spawn` boxes for a
   worker of another pool and the job list `ThreadPool::broadcast` collects,
   besides the blocks of Rayon's job queue.) Acquiring a reservation is not an
   entry. `tests/alloc_free.rs` and `tests/team_alloc_free.rs` check it, after
   warm-up, for the entries they call.
6. API changes keep the contracts in `README.md` and `docs/design/placement.md`
   in sync.
