# True monolithic weighted top-2 F0 baseline

Status: implementation, correctness tests, smoke, full 20-row execution, and
independent audit complete.

## Fair common semantics

Selected-softmax top-2 contains a sigmoid gate and cannot be represented
exactly by a finite MILP without an additional nonlinear approximation. The
baseline therefore uses exactly the same guarded router-margin range and
property-directed F0 McCormick relaxation as the staged Route A verifier. It
does not encode `exp`, division, or a symbolic sigmoid graph.

For one safety property, each exact feasible unordered top-2 pair supplies one
guarded shared-input two-expert F0 HZ. The monolithic baseline places every
such branch into one MILP with one pair selector. It uses bounded
homogenization:

- continuous branch factors obey `-s <= x <= s`;
- binary branch factors obey `0 <= z <= s`;
- every branch row `l <= A x <= u` becomes
  `l s <= A x <= u s`;
- exactly one pair selector is active.

This is an exact representation of the union of the supplied F0 relaxations
and introduces no new arbitrary big-M. One solver call per property allocates
all feasible pair/expert branches simultaneously. Route A instead solves the
same guarded F0 semantics one pair at a time.

## Verdict discipline

Only a solver-certified, outward-rounded positive lower bound can establish
SAFE. A negative relaxation incumbent remains UNKNOWN. A recovered candidate
is called UNSAFE only after the full concrete selected-softmax model changes
prediction inside the registered input box. Every tie-inclusive feasible pair
is present in the disjunction.

## Frozen evaluation

`configs/monolithic_f0_baseline_r1.json` fixes the same 20 sample/radius
identities and 900-second per-row deadline as Experiment 1D. Natural property
order is fixed before execution. After shared pair propagation, each remaining
property receives an equal share of the remaining deadline. The pre-existing
Route A result is hash-pinned for common-cohort status comparison; its timing
was not interleaved, so runtime comparisons are descriptive rather than a
paired microbenchmark.

The old `monolithic_hz_status` field remains a route-unguarded decomposed
reference and is not renamed or used as this baseline.

## Smoke outcome

Rank 110 has three feasible pair branches. Its single formulation contains
9,660 variables, 148 binaries, and 19,798 constraints per property. The
120-second smoke completed five of nine properties before the external hard
deadline; every completed property returned a solver-limit UNKNOWN and the
row returned TIMEOUT. No relaxation result was promoted to UNSAFE or SAFE.
Independent audit reported zero issues. This passes the construction and
fail-closed smoke gate; it is not reported as a solved-rate result.

## Full common-cohort outcome

The registered 20-row run completed in 11,007.57 seconds and passed an
independent audit with zero issues. The monolithic formulation returned 6
SAFE, 2 full-forward-validated UNSAFE, 1 UNKNOWN, and 11 TIMEOUT rows: 8/20
rows solved. The frozen Route A reference returned 10 SAFE, 2 validated
UNSAFE, 7 UNKNOWN, and 1 TIMEOUT: 12/20 rows solved.

The paired table contains five Route-A-only solved rows and one
monolithic-only solved row; the exact two-sided paired binomial p-value is
0.21875. Accordingly, the supported conclusion is descriptive: Route A solves
four more rows on this small frozen cohort, while monolithic uniquely proves
one SAFE row. Neither result set contains the other, and the experiment does
not establish blanket solver dominance. Runtime is also descriptive because
the Route A reference predates this run and was not interleaved.

The tracked audit manifest is
`results/monolithic_f0_baseline_20260902_r1_audit.json`; raw solver artifacts
remain under `data/moe/results/monolithic_f0_baseline_r1` and are hash-pinned.
