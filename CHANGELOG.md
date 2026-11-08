# Changelog

All notable changes to `threadteam` are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the crate follows
[Semantic Versioning](https://semver.org/spec/v2.0.0.html). Entries start at
0.5.0; earlier releases are not recorded here.

## [Unreleased]

### Fixed

- **`Pool::into_team` no longer pins the calling thread** (#52). The leader of a
  `Team` token is placed on `S[0]` for the one section it dispatches, and has
  its own mask back when the section ends; nothing is pinned for the team's
  life. 0.5.0's `pin::TeamPin` left the caller of `into_team` on `{S[0]}` until
  the `Team` was dropped, so a vendor thread pool (MKL, OpenBLAS) or a plain
  thread started from that thread while the engine lived inherited one CPU: an
  engine's construction thread is not ours. Measured on evo2 (32 CPUs, width
  8): the caller's `Cpus_allowed_list` is `0-7` before `into_team`, `0` after it
  and in everything started from there on 0.5.1, and `0-7` throughout now. The
  leader is still on `S[0]` in 40 of 40 calls (0.5.1: 0 of 40 with the caller
  off `S[0]`, 5.9 ms per call against 10 us). The pin costs one
  `sched_getaffinity` and two `sched_setaffinity` calls per outermost section
  of a team token that opens no scope, and nothing inside a team scope.
  `pin::TeamPin` is replaced by one record on the `Team` (`pin::Home`, the CPU
  no member takes), which pins nothing: a whole-team root scope writes it and
  the sections read it. Other threads are no longer second class: a token used
  on another thread places that thread for its sections too, inside its own
  mask.

## [0.5.1] - 2026-11-07

A `Team` created on 0.5.0 paced its own members by its leader's one-CPU mask.
0.5.0 pins the thread that creates a `Team` to `{S[0]}` for the team's life,
and `Team::exec` then decided oversubscription against that mask. So every team
of two or more participants, whose mask `S` had a CPU for each participant and
was therefore not oversubscribed, ran with its members' spin budget cut to
`THROTTLED_POLLS`. The same flag was carried into `fork_sized`, `for_ranges` and
`broadcast` reached without opening a team scope (#33). 0.5.1 decides this by
the team's width instead: `Team::token` no longer consults the post-pin mask on
the leader the team placed, and `pin::available` answers `|S|` on a leader this
crate placed (#49). A `Team` whose `S` has fewer CPUs than participants is
still paced. No public API changes, so this is a patch release.

## [0.5.0] - 2026-11-06

The leader of a team is pinned, where 0.4.0 only moved it. Inside a placed team
scope the leader's affinity mask is `{S[0]}`, the one CPU no member takes, for
as long as the scope or the `Team` lives; it no longer keeps all of `S`.

### Breaking

The minor bump is for the surface below.

- **`Pool::leader_moves()` is removed** (#45). It returned the count and total
  time of the per-section leader moves, and the moves it counted are gone. It
  was `#[doc(hidden)]`, so it never appeared in rustdoc or in
  `cargo public-api`; a caller who used it can grep for `leader_moves`.
- **The per-section leader nudge is removed** (#45). Internally this is
  `Awake::place_leader`, `pin::nudge_to`, and the `leader_home`,
  `leader_moves` and `leader_move_ns` counters of `Awake`. All of them were
  `pub(crate)`; the only public trace was `Pool::leader_moves()`.
- **The `Team`-lived leader home is removed** (#45): the `home` field of `Team`
  and of `TeamAuth`, and the `home` argument of the internal `split_raw` and
  `spread_raw`. There has never been a public `Team::home`, so consumer code
  has nothing to fix; grep `home` only if you mirror this crate's internals.
- **The leader's mask is one CPU while a team lives.** A vendor library called
  by the leader directly now sees `{S[0]}` and builds a one-thread team.
  `TeamExec::external` is the only call that widens it: the mask becomes `S`
  for the call and `{S[0]}` again after it, two `sched_setaffinity` calls
  (about 2.3 µs on broccoli) per call. `Exec::external` is unchanged.
- **`Pool::into_team` pins the calling thread** (#46). The thread that calls it
  stays on `{S[0]}` until the `Team` is dropped or turned into its pool with
  `Team::into_pool`, also between tokens and on unwind: `pin::TeamPin`, a field
  of the `Team`, holds the pin and gives `S` back. A `Team` driven by
  `fork_sized`, `for_ranges` or `broadcast` without a team scope is therefore
  placed for its whole life; in 0.4.0 the leader got `S` back at the end of the
  acknowledgement scope and the team ran in the #33 slow mode. A token minted
  on another thread has an unplaced leader, and a `Team` dropped on another
  thread restores nothing there.

### Added

- `Exec::fork_sized_slots(&shares, &mut slots, f)` (#37): `fork_sized` where
  child `i` also gets an exclusive `&mut slots[i]`, without a lock, allocating
  nothing when warm. A slot count that differs from the share count is the new
  `Error::SlotsMismatch`.
- `TeamExec::external(f)` (#45): the widening call above.

### Unchanged

- A team of one (a serial token, budget one) still runs inline and touches no
  pool.
- A mask with fewer CPUs than participants still pins nobody, and
  `Pool::pin_skips()` counts the scope; nothing wraps.
- `PoolBuilder::pin_participants(false)` still pins nothing.

### Notes

- #43 (a leader home kept on the `Team` and a nudge onto it before every
  section) **landed but was never measured**: the reporter's re-run is on the
  commit before it, and the first nodes of the same model did not reproduce #33
  on 0.4.0 either, so no run has measured #43 either way. #45 removed it in
  favour of the scope- and team-held pin.
- The runs, medians and probed fields for the scope- and team-held pin are in
  [`docs/design/placement.md`](docs/design/placement.md).
- Documentation only: #40 and #41 state what the placement and waiting code
  does.
