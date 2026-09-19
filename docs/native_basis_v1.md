# Native basis mapping controls: HiGHS 1.14.0

## Completed scope

New `native_basis/` captures an actual small HiGHS solve, preserves the submitted
model and native record, then separately maps basis statuses into the unchanged
`exact_basis/` original-coordinate manifest. Rational reconstruction and full
original-LP checking remain separate, mandatory steps.

[Attempt001](native_basis_controls_attempt001.json): **54/54 controls PASS**,
including7 new tests and47 prior regressions. There were6 native analytic
captures in the new tests, not zero native calls. There were **zero real-network
LP calls or reconstructions**. Old modules/freezes and the four sealed real
diagnostic outcomes remain unchanged. No dependency installation occurred.

This closes the small native-adapter prerequisite. It does NOT deliver a
production-integrated, outer-supervised or full-costed real-request pipeline.
No speedup, large-LP compatibility or real certificate is claimed.

## Submission and capture identity

The installed `act-py312` highspy reports1.14.0; this adapter rejects other
versions until separately tested. Recorded identity includes version and native
binary SHA256. Fixed options are simplex, presolve off, simplex scaling strategy0,
threads1, parallel off, random seed0, output flag false. Actual option readback
must match; one native call receives at most10 seconds of the caller's remaining
absolute deadline. Old production or diagnostic solver options are not changed.

The rational LP is submitted as its explicitly recorded float64 conversion,
with original x columns and E-then-A row order. Original rational coefficients
are NOT overwritten. The submitted objective, offset, column bounds, row bounds
and sparse coefficients are read back before and after solving. A mismatch
rejects mapping. Exact reconstruction subsequently uses the original rational
coefficients, not the rounded submitted matrix. Thus native rounding may change
the useful basis proposal but cannot replace original-LP proof obligations.

`input.json` and `submission.json` precede solving; `capture.json` precedes
mapping/reconstruction/checking. Native point, row/column statuses, valid flags,
basic-variable identifiers, model/API status and objective are untrusted
metadata. No native optimal status is accepted as an exact proof. File creation
is no-overwrite, one-shot. Limits remain those of the analytic basis interface.

## Explicit status mapping

For original A_i x<=b_i, our slack is s_i=b_i-A_i x, never the raw solver row
variable with an assumed sign. The basis is rebuilt using original coefficients
and +1 slack columns. The tested mapping is:

| Native item / status | Original-coordinate interpretation |
| --- | --- |
| Structural column kBasic | Basic x column |
| Structural column kLower / kUpper | Exact original lower / upper anchor |
| A-row kBasic | Basic positive-slack column for that original A row |
| A-row kUpper | Slack anchored at zero |
| E-row kLower / kUpper | Equality row fixed at original h; no extra E slack |
| Basic/free E-row, free/unknown structural status, unsupported A status | UNSUPPORTED_MAPPING |

This is a deliberately restricted mapping. Finite x bounds are part of the LP
contract. No arbitrary interpretation of kZero/kNonbasic, transformed row IDs,
presolved models or scale vectors is attempted. Both raw basis IDs and statuses
are saved, but the conversion uses named statuses plus original row/column
identity, not inferred signs of raw basic-variable integers.

An external capture hash, LP/statement hashes, complete readback and options
bind the mapping. Its successful result is only `MAPPED_HINT_ONLY`, never SAFE
or feasible. A later `CANDIDATE_ONLY` exact reconstruction still needs the
unchanged isolated full checker to validate every original constraint.

## Actual analytic observations

For x+y=1, 3x<=1, x,y in[0,1], minimizing **-x-1**, HiGHS gives both structural
columns basic and the A-row kUpper. The adapter reconstructs (1/3,2/3); a moved
`python -I -S` checker proves **U=-4/3** without solver/model imports. This
objective deliberately differs from the earlier manually selected +x-1 example
so the native optimizer chooses the active row. Neither case is a real request.

With +x-1, the native solution is (0,1): x is at its lower bound, y and the
inactive A-row are basic. The original positive slack reconstructs as1. Fixed
columns and independently permuted original structural columns/A rows also
produce fully checked feasible points.

A redundant-equality control, x+y=1 and2x+2y=2, produces basic equality-row
variables. The current original-basis schema has no corresponding E-slack
coordinate. The adapter preserves the capture and returns
`UNSUPPORTED_MAPPING`, not LP infeasibility and not a dropped equality. This
is an actual compatibility limitation observed in native execution, not merely
a synthetic malformed-input test. Supporting it needs an explicit extension
and separate controls; do not silently remove redundant rows.

Adversarial tests reject changed capture hashes, presolve/scaling settings,
readback matrices, dimensions and unsupported statuses. Expired/oversized
inputs make no native call. Reusing an output directory is rejected. Prior
tests retain exact arithmetic, degeneracy, fill-in, identity and full-check
rejection coverage. No feasibility tolerance is weakened.

## Literature/API boundary and next step

The official [HiGHS Python examples](https://ergo-code.github.io/HiGHS/dev/interfaces/python/example-py/)
document model submission, `run()`, solution/basis retrieval and option access.
They do not replace this adapter's task-specific original-coordinate checks.
This work is an evidence interface using established LP tools, not a claim of
new exact-linear-programming theory.

Next, integrate the **supported** capture→mapping→exact construction→packaging→
isolated full-check path into a new owned outer supervisor. Include deadline
inheritance, termination during native/exact work, mapping failure, partial
artifacts, all terminal denominators and complete cost accounting in analytic
controls. A single caller budget must cover imports/conversions/capture and
checking; these control-call timings are not that end-to-end comparison.

Keep unsupported equality-row mappings explicit. Their eventual support and
large sparse factorization require distinct design work; neither may be hidden
inside a runtime integration or a cap increase. No real experiment is frozen
or launched by this release. Network→HZ, guard, route exclusion and F0 lowering
remain trusted upstream components, and LP feasible points are not network
counterexamples.
