# True monolithic weighted top-2 F0 baseline

Status: implementation and correctness tests complete; frozen smoke and
20-row execution pending.

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
