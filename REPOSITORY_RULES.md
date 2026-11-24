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
5. Steady-state entries add no allocation of their own. The only allocation
   an entry makes per call is the one Rayon makes for the call that hands the
   work to the pool, if the entry makes one: the job `ThreadPool::spawn` boxes
   when a worker of another pool submits work, or the job list
   `ThreadPool::broadcast` collects. An allocation amortized over many calls
   (a block of Rayon's job queue) is not per call. Acquiring a reservation
   (`Exec::team`, `Pool::into_team`) is not an entry. `tests/alloc_free.rs`
   measures the plain-token entries other than `broadcast` from a plain
   thread, from a worker of the pool and, one job per entry, from a worker of
   another pool; `tests/team_alloc_free.rs` measures dispatch inside a
   reservation.
6. API changes keep the contracts in `README.md` and `docs/design/placement.md`
   in sync.
