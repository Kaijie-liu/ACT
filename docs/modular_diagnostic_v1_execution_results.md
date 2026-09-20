# Frozen modular diagnostics: execution and archive

Date: 2026-09-20. Execution HEAD:
`ab4a70728241c9d34c6a813b909e50064274d20c` (clean and pushed at launch).

## Outcome

The four registered diagnostics ran once. **All four returned LIMIT; none
produced a complete reconstructed vector, an original-LP checked feasible
point, or a checked feasible upper bound.** The batch completed its ledger and
audit successfully; this is not mathematical success of the four LP queries.

| Frozen job | Primes tried / merged | CRT modulus bits | Stop phase / operation | Provided-LP request seconds |
| --- | ---: | ---: | --- | ---: |
| input220_p0 | 81 / 80 | 2400 | finite_field / map | 21.672129 |
| input222_p1 | 77 / 77 | 2310 | prime_schedule / trial_division | 21.179189 |
| input230_p2 | 50 / 49 | 1470 | finite_field / elimination | 26.315105 |
| input232_p0 | 50 / 49 | 1470 | finite_field / map | 26.293886 |

Every stop is the shared **20,000,000 operation cap**, with observed counter
20,000,001. These are configured arithmetic operations, not CPU instructions.
None reached the 4096-bit cap, request deadline, or 128-prime maximum. Every
merged round records `RECONSTRUCTION_INCOMPLETE`. There were no bad-denominator
or singular primes and no complete-vector exact residual check. In particular,
zero residual rejections does not establish validity of partial candidates.

All four basis structures and assembled exact-system hashes agree with the
previous primitive diagnostics. The recorded finite-field products never exceed
60 bits. This run avoids the previously observed 4097--4116-bit raw row-product
failure, but replaces that stopping condition with operation-budget exhaustion;
**it adds no checked feasible endpoint**. Partial reconstruction numerator and
denominator sizes are not the size of a known exact solution.

## Frozen scope and execution

The unchanged [freeze](modular_diagnostic_v1_freeze.json) and
[selection review](modular_diagnostic_v1_selection_review.json) bind the four
original LPs, source identities, runtime, and protocol. Execution used:

```text
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
/data1/Kane/miniconda3/envs/act-py312/bin/python \
-m modular_diagnostic.run launch --execute-frozen
```

The batch used its existing lock/nice policy and one native fidelity call and
one basis per request. No retries, alternate bases, larger budgets, fallback,
new samples, or changes to frozen source/results were introduced. Internal
prime rounds are the already registered algorithm, not additional LP queries.
The 218/298/300-second cutoffs and all numerical acceptance gates are unchanged.

Raw artifacts remain in
`data/moe/results/modular_diagnostic_real_20260920_v1`; raw source matrices and
external/checkpoint data are not added to Git. Older primitive results remain
unchanged. Literal NOT_EXECUTED status in the historical freeze/readiness
record describes its creation time; this new execution archive records launch.

## Complete cost accounting

| Cost scope | Seconds |
| --- | ---: |
| Four provided-LP requests through terminal publication | 95.460309 |
| Post-terminal request audits | 45.543369 |
| Batch wall time, including those requests and audits | 141.006109 |
| Preflight, outside batch clock | 3.926427 |
| Final-summary audit/serialization, outside batch clock | 45.409580 |
| Separate archive construction | 47.829198 |
| Fresh independent archive/result review | 48.022604 |

The archive additionally records resource wait, attempt clocks and residual
overhead. Constructor durations (8.958422, 9.321034, 9.210599, 9.624302 seconds)
are nested inside request time; do not add them again. Repeated finite-field,
CRT, and reconstruction windows are **summed**, not represented by the last
window. Their instrumented phase durations include journal work and are not
isolated kernel benchmarks.

No package or original-LP check was reached: per-request costs for these stages
are **null**, not observed zero. Completed LIMIT journals are complete records
of an unsuccessful construction, not complete proof packages. Upstream network
propagation, ranges and F0 generation were supplied historically and not rerun;
these costs do not represent end-to-end MoE verification. Comparison with the
earlier primitive run is a descriptive same-system comparison, not a matched
timing experiment or speedup claim.

## Independent review and controls

- [Execution archive](modular_diagnostic_v1_execution_results.json): 1980 raw
  artifact hashes, complete four-row outcomes and cost accounting.
- [Fresh review](modular_diagnostic_v1_execution_review.json): PASS, zero issues;
  re-collects the frozen archive, checks identities, outcomes, cyclic journals,
  cost completeness, absent candidate/check artifacts and prior-system matches.
- [Reviewer](../scripts/review_modular_diagnostic_results.py) and
  [controls](../scripts/test_review_modular_diagnostic_results.py).
- [Control receipt](modular_execution_review_controls_attempt001.json): 4/4 PASS;
  validates all saved rows, repeated-phase cost aggregation and rejection of
  eight mutated outcome/cost/progress records. Tests do not edit raw records or
  initiate native, reconstruction or bound queries.

Archive and fresh review performed zero additional solve/reconstruction calls.
They establish record identity and consistency, not an independent proof of
every finite-field elimination. The original-LP checker has no new candidate
to check in this run.

## Interpretation and closure

The finite-field approach controlled the recorded large intermediate products
on these identical systems but did not reconstruct a complete point within the
shared operation budget. This does **not** establish LP infeasibility, network
unsafety, a solution requiring more than 4096 bits, or that more primes/time
would close the proof. It also does not identify a unique cause of incomplete
rational reconstruction from partial-candidate bit maxima.

This four-query frozen stage is closed and retained in full. The next bounded
research step, if commissioned, should use saved journals to distinguish
repeated finite-field work from reconstruction limitations before changing an
algorithm. No unchanged rerun, cap extension or production-policy relaxation
is part of this archive.
