# Primitive integer rows: separate exact-arithmetic development

## Status and scope

Under the 2026-09-20 instruction to study bit-growth control separately, this
stage implements `primitive_basis/`, tests it on analytic systems and synthetic
sparse scale, and checks candidate points against the unchanged original-LP
checker. It does not modify V2, rerun any of the four real LPs, raise the 4096-bit
cap, add dependencies or change production acceptance. **No real-LP efficacy
result is claimed.** This is a candidate arithmetic method, not a default switch.

Current receipt: `primitive_basis_controls_attempt002.json`, **62/62 pass**
(16 new tests, 46 regressions). There are 24 exact rational differential systems,
a same-cap benefit control, adverse controls, original-LP feasibility/rejection
controls, mutation/deadline/resource checks, and a structured 4096-row system.
The fresh review checks 114 retained artifacts, independently evaluates 26
successful systems' original equations, preserves five unresolved controls,
and reruns five moved packages with `python -I -S`. Four points are exactly
feasible and one is correctly rejected. Review PASS, zero issues.

Attempt001's 61 tests passed. Its first review failed because script-only
execution metadata was compared directly to an in-process reference that lacks
those fields. The failure and partial moved files are preserved in
`primitive_basis_review_attempt001_failure.json`. The reviewer now independently
requires isolated/no-site/no-solver execution and compares all mathematical
fields exactly; an added mutation test covers both checks. No bound tolerance
or mathematical result was changed to obtain the passing review.

## Algorithm: row content removal, not a Bareiss implementation

The component assembles the same original-coordinate square basis as the frozen
sparse constructor. It retains every original equality and inequality via
`E x + e = h` with required `e=0`, and `A x + s=b` with required `s>=0`.
Anchors and source/statement/hint identities are checked as before. The original
LP stored in the final bundle is not scaled or rewritten.

Only the internal linear-equation solver changes:

1. For each rational equation, compute a positive common denominator and form
   integer coefficients and RHS exactly.
2. Divide the **whole row including RHS** by their positive gcd, and fix its
   sign deterministically. Store this primitive integer row.
3. Use the same minimum-column-degree / shortest-row / index pivot rule as the
   frozen sparse solver. Do not normalize the pivot row to unit diagonal.
4. If pivot coefficient is `a != 0` and an affected row has coefficient `b`, set
   `g=gcd(|a|,|b|)` and replace that row by `(a/g) row - (b/g) pivot`. Remove its
   content again. Preserve sparsity and all exact nonzeros; no pruning.
5. Perform exact rational back-substitution only after elimination finishes.

This is a gcd-normalized integer row algorithm, **not** Bareiss's previous-pivot
exact-division algorithm, modular reconstruction or an optimal pivot heuristic.
No novelty or universal complexity improvement is claimed for these operations.

### Why the equation transformations are equivalent

Multiplication by a nonzero common denominator and division by a nonzero content
preserve a row's solution set. The pivot row is retained. Since `a/g != 0`, the
affected original row can be recovered as a linear combination of the retained
pivot and the replacement row. Thus each completed transformation is reversible
over the rationals. Exact back-substitution solves the resulting triangular
system when the chosen square basis is nonsingular. A singular chosen basis is
unresolved, not a proof that the original LP is infeasible.

Row scaling does not change exact zero patterns or row lengths. Accordingly,
absent a resource failure, the declared structural pivot rule is unchanged; this
study does not mix in a new pivot heuristic. A bounded pivot prefix and rolling
hash support later audit. They are diagnostic records, not a full elimination
proof. Most importantly, the constructor still returns only `CANDIDATE_ONLY`:
the separate checker must validate all original equalities, inequalities, boxes
and the objective before any feasible upper bound is accepted.

## Bit and resource contract

The component inherits the frozen structural limits: 16,384 variables/rows,
one million input entries, two million active+pivot entries, one million fill
insertions, 200,000 heap entries, twenty million counted events and one attempt.
It accepts an absolute deadline at most 300 seconds ahead, without resetting it
between assembly and elimination. This is a cooperative **component** deadline,
not yet the full outer supervisor or a complete request-budget experiment.

The 4096-bit cap remains. Every rational input/reduced fractional intermediate,
LCM, scaled integer coefficient, integer cross-product and subtraction is
checked. An over-cap integer is rejected **before** a later subtraction or gcd
could make it smaller. This is deliberately more conservative than guarding
only a reduced row; it can lose cases the rational solver handles. Python must
materialize an arithmetic result to inspect its bit length, just as the old
Fraction path did; no claim that transient allocation is literally capped at
4096 bits is made. Structural entry counts are not process RSS limits either.

The first limit records phase, operation, row/column where available, observed
bit size, fixed cap and value hash. Maxima include the rejected value. Assembly,
row clearing, elimination and back-substitution can therefore be distinguished.
After a limit, the proposal contains no point/bundle; partial counters remain.
Some unit controls temporarily use *smaller* caps to exercise failures; these
are marked in their saved policy and are not new real-experiment settings.

## What the controls establish—and do not establish

| Control | Frozen rational method | New method | Interpretation |
| --- | --- | --- | --- |
| Large common row scale, unique solution `(1,1)` | 4096-bit LIMIT | Candidate, original LP checks feasible | Avoidable intermediate growth exists in this constructed case |
| Coprime denominator LCM, unique zero solution | Completes | Row-clear LIMIT at 5036 bits | This representation can be worse |
| Triangular system with a 4201-bit exact answer | Not needed for the analytic conclusion | Back-substitution LIMIT at 4201 bits | Cannot fit an intrinsically oversized solution into the cap |
| Cross-products exceed a test cap before cancellation | Not a comparative endpoint | LIMIT, no candidate | Gcd reduction is not used to bypass the arithmetic guard |
| Inexact basis with nonzero equality residual | Original LP must reject | Candidate rejected, upper bound null | Solving the augmented system is not enough for LP feasibility |

For the benefit control, let `p=2^3000+1`, `K=2^2000`:

```
p x + y = p + 1
K x + 2K y = 3K
```

Old eager rational normalization forms the unreduced-row RHS
`K(2p-1)/p`, whose reduced numerator has 5002 bits, even though the unique answer
is `(1,1)`. The new method removes `K` as row content before elimination. Its
largest observed integer is 3002 bits, and the original LP with bounds `[-2,2]`
and objective `-x` independently checks U=-1. This is **not** a network UNSAFE
witness and does not predict success on the four real LPs.

For the adverse control, one row contains `1/2^2500` and `1/3^1600`. Their LCM
has 5036 bits, so this implementation stops before elimination. The old rational
method solves that system as zero without crossing its cap. Hence automatically
replacing the old algorithm, or assuming integer arithmetic is always smaller,
is unsupported.

The 4096-row lower-bidiagonal synthetic system solves exactly as `1/3` in every
coordinate, with zero fill and 4096 pivots. Component time in attempt002 was
about 0.052 s. Its coefficients require at most four integer bits; it is a
structured scale control, not a timing forecast for dense or ill-scaled real
bases. Test timings exclude a production request's upstream pipeline and
standalone proof-generation/checking costs.

## Next bounded step

This completes the separate arithmetic study, **not** integration or a real
rerun. The next engineering task is a separately versioned optional component
under the unified outer supervisor: preserve original start time, partial
metadata, exception/kill behavior and complete assembly/clear/eliminate/package/
check costs. Admission can record denominator-clearing obstacles; it must not
silently drop entries or enlarge caps. Neither a retry of the original basis
nor a switch to another algorithm is free—any future portfolio needs an explicit
budget and stopping rule.

Only after integration controls should a new real diagnostic protocol be
frozen. The four V2 limits remain unchanged. Small original coefficient bit
sizes alone do not bound an assembled row's LCM, intermediate minors or final
solution size. Whether this method improves any real obligation is still open.
