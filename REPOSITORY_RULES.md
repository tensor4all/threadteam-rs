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
4. No thread-local state and no implicit global pool; lazy inline execution
   for serial, budget-one and single-task cases.
5. Steady-state entries do not allocate.
6. API changes keep the contracts in `README.md` and `docs/design/placement.md`
   in sync.
