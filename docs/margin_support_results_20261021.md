# Finite margin support control results

The frozen standard-library batch completed and was accepted online on
2026-10-21. The finite tilt rule produced the predicted positive given-source
bound and retained its negative counterexample. This is an implemented,
exactly checked mechanism control, not trained-model or runtime-effectiveness
evidence. G1–G6 remain OPEN.

- Execution: `144f5053fc19713b685a491e45860ca60cb1e58c`.
- [Protocol](margin_support_control_20261021.md) and
  [freeze](margin_support_freeze_20261021.json), SHA-256
  `02df94104ab97f5549736ec673e8361c2d0925f5674155fb2ae8d8d9dd4d8bd4`.
- [Lossless terminal, worker and saved-evidence audit](margin_support_records_20261021.json).
- Original directory: `/data1/Kane/MOE/results/margin_support_20261021_r1`.
- One admission and one attempt; no repair, rerun, extra source or parameter search.

## Mechanism result and negative control

| Fixed given source | Exact checked aggregate | Interpretation |
| --- | ---: | --- |
| Nonconstant shared/private example, zero tilts | -89/256 | This certificate is nonpositive |
| Same source and 128 cells, fixed median tilts | 19/2048 | Positive bound on the declared continuous source |
| Same source with reversed pair | 19/2048 | Orientation transformation preserves the result |
| Same property via two classes and nonzero constant | 19/2048 | Exact projection and constant handling agree |
| Diamond | -1/16 | Expected unresolved bound, retained without refinement |
| Identically zero margin and half gate | 1 | Constant-gate case |
| Nonzero inequality / equality dual controls | 0 / 0 | Valid nonpositive lower bounds, not safety acceptance |

The main positive is above the predeclared finite-cell guarantee 3/512 and
below the paper lower bound 3/256. The change from zero to median tilt is the
intended controlled factor. It uses two versus four objective slots and ZERO
native support solves; literal candidate multipliers and box residuals suffice
on these sources. Therefore this is not an equal-solver-call performance win.

The additional retained diamond-slice support checked to 1/16 using nonzero
constraint multipliers. It exercises inherited constraints, not just box support.
The diamond full-source nonpositive result is not evidence that its true output
is unsafe; it is a known limitation of this fixed two-anchor surrogate.

All 13 frozen groups passed. Missing support/cells, false sigmoid premises,
changed scope/constraints/private ownership, forged bounds, wrong policy, bit/shape
limits and injected deadline/Work stops were rejected as specified.
The latter are finite fault controls, not a 300-second soak or a new test of
the OS supervisor's correctness.

## Acceptance and complete bounded cost

The original owned-session supervisor returned `COMPLETED`, child exit0,
`LEADER_REAPED_NO_LIVE_SESSION`, no execution/cleanup error. Online reception
recomputed all eight full source certificates and the single slice support.
The outer execution command also exited0. FD count was4 before and4 after.
No native numerical package, optimizer, checkpoint, training data or GPU was used.

| Recorded cost | Value |
| --- | ---: |
| Controller entry through terminal-file export, final stdout receipt | 0.9900553803890944 s |
| Worker entry through pre-export timestamp | 0.45898719877004623 s |
| Owned-session execution | 0.6585949342697859 s |
| Owned cleanup | 0.04790113493800163 s |
| Worker instrumented Work | 76,165 |
| Online reception additional Work | 25,666 |
| Combined online Work / frozen cap | 101,831 / 8,000,000 |
| Peak sampled session plus caller RSS | 43,634,688 bytes |
| Original worker member | 56,301 bytes |
| Original terminal member | 3,769 bytes |
| Compact lossless archive | 61,884 bytes |

Phase times are nested observations, not disjoint values to sum. The observed
export duration does not cover the interpreter pre-entry or final stdout/exit
tail; `full_cold_start_seconds` remains null. The hard alarm stayed armed and
the command exit was separately observed. The small synthetic runtime is not
an estimate for a trained request or a speedup over H2/N1/M3.

Per-process 8GiB AS, 2GiB sampled aggregate RSS, 128MiB artifacts and all exact
bit limits remained unchanged. Original runtime members total60,070 bytes.
The artifact contract additionally reserves publication copies and32KiB logs;
the offline archive is separately retained, far below the same total allowance.
No cache or historical artifact was removed.

## Saved evidence audit and trusted boundary

The saved-only audit exited0 and returned `PASS_SAVED_EVIDENCE`, preserving the
original online PASS. Its separate post-run cost was25,666 Work and
0.15697501227259636 seconds before archive publication; it is not charged as
another online proof or used to repair a missing receipt.

Original terminal SHA-256:
`29098167a0900e541e74cc2e1ef4044aa19fb5bb0ab63e504eb2dff552c0e1ac`.
Original worker SHA-256:
`6777a90fa7033227fbbf93ed2e6124e174c3d3632f71f714fff0950a01f567a1`.
Archive SHA-256:
`d3e35c3cde57dbaa2cc6e715a5e42fe02a2d4d54c6fe5646f432f74612c24424`.

The guarantee is `CHECKED_LOWER_BOUND_ON_GIVEN_CONTINUOUS_SOURCE`. Source parsing,
projection, finite policy and the analytic sigmoid rule form a shared exact
kernel; producer and checker separately compute residual and cell-aggregation
arithmetic. The frozen external source/policy identities, rather than hashes
chosen only by the candidate, bind reception.

This does not check network-to-HZ conversion, input inclusion or actual
tie-legal route reachability. The main example has one fixed unordered pair;
reversing its internal rank is not a route-set-changing witness. The production
SAFE gate, complete request coverage and old source gaps are unchanged.
Two additional read-only AI reviews checked the archived arithmetic and execution
fields; they are not human technical review.

## Next decision

Continue only toward a bounded integration with the existing HybridZ support
candidate interface, with the original joint source and independent residual
checking kept intact. First specify that mapping and the candidate/cost contract;
do not repeat these successful sources as a substitute for it.

Any native execution still requires its own frozen scope and the unresolved
observation-contract decision; this successful standard-library batch does not
restore old native attempt authority. No automatic model, GPU, GenBaB or RSS
repair follows. Strong slicing and lifted/product-aware competitors may reproduce
the gain, so a full-request comparison, not this control count, must establish
practical research value.
