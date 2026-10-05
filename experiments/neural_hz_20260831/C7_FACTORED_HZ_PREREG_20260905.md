# C7 V1: exact shared affine-factor HZ for the same residual CNN suffix

Previous C6 actual-state result is bound by SHA-256
ba3771d55d09447b32a9bbe6cf4cee00ad784c21bbf7f52120d49456f59d5803.
Its 14-term uncached support-composition upper bound is 2,208,268,400; four
branches fail to certify the old per-branch ceiling. This is the evidence
for changing representation, not permission to increase any budget.

## One structural rule and exact set identity

Factor common linear suffixes by ORIGINAL operator identity:
sum_i W*x_i = W*(sum_i x_i). Sources stay keyed by identity, not payload
equality; all sources' original predicates and continuous/binary coordinates
remain. Reconstruct a shared affine DAG containing source, sum and linear
nodes. Propagate exact structural supports, then retain the coordinates needed
by selected outputs through that DAG. This is graph dependency analysis,
not bound refinement, phase search, solver feedback or a family/iid rule.

For each required node coordinate y, allocate a fresh continuous eta and an
exact power-of-two scale s with |y| <= s on the existing factor boxes. Add the
defining equality eta = y/s. A conservative exponent bound is derived from
the number and maximum binary exponent of nonzero summands; no LP or external
bounds are required. Operator coefficients are only scaled by powers of two,
not multiplied into long composed coefficient matrices. Every nonzero scale
operation must be finite and reversible exactly in float64; otherwise reject.
No binary factor is pivoted, merged, relaxed or removed. Source constraints,
including zero-valued sources, are conjoined through the unchanged native
same-frame join. Existing continuous/binary coordinate prefixes are unchanged.

The added equalities form a triangular definition graph with a unique eta
assignment for each old feasible latent assignment. The chosen boxes contain
that assignment. Conversely, projecting any lifted feasible assignment onto
the old coordinates satisfies every old predicate and yields exactly the
original affine-program output. Witness reconstruction is old-prefix
projection; the forward extension is uniquely defined. The result is still
a genuine nonconvex SparseHZono, even when its value map reads only fresh
continuous factors: the binary phases remain in the coupled predicates.

Semantics are exact REAL arithmetic on the frozen float64 coefficients of
the uncomposed affine expression and its already separate full bias. Rational
elimination must reproduce those coefficients exactly on focused tests.
This is not a claim of byte identity to a differently associated, rounded
materialized matrix. No live integration or sound CERT/ADV promotion follows
from the representation theorem alone; native integration, bounds, solver
numerics, concrete witnesses and full retention remain independently gated.
The physical comparator remains phase_selective_expanded_v1, not C6 or an
unrelated larger network. More auxiliaries alone are not simplification.

## Fixed offline target and ceilings

Use the same sealed actual ADD75 snapshot and native Dense77 append as C6:
d08086844eacebbd69c2ddc3c4ffc77ebf1aabc90539e95547744da929273fed.
All 200 output coordinates and all 14 terms/9 sources are included. No
production dispatch, fresh live propagation, nonlinear or solver step here.

Before encoding, count every required source coefficient, operator edge,
sum edge, new defining row and output entry on the COMPLETE factored graph.
Bound encoding arithmetic by 8 operations per retained coefficient plus
8 per new row; include charged support visits in the 256M whole construction
work ceiling, with each source-to-output branch <=200M. Count all old
predicates and the lifted HZ against 64M stored entries. Keep 1 GiB measured
construction, CPU numeric threads 1, 16 GiB worker memory, 240-second wall.
Nonfinite/unrepresentable/unsupported/overbudget states reject before any
publication. No fallback or larger-cap retry of a failed source version.

First prove source/shape/frame binding, rational identity in both directions,
binary/predicate retention, exact scale reversibility, common suffix/source
sharing, zero/partial outputs, non-dyadic coefficients, grouped/masked Conv,
duplicate terms, mutation rejection, and budget failure on focused tests.
Then perform ONE exclusive offline complete construction if preflight passes.
After construction, independently audit EVERY defining equality against
original source coefficients and the original unfused operator row oracle,
including coordinate mapping, exact reverse scaling, redundant boxes and the
full original term multiset. Stream at most 256M original row entries, without
retaining an expanded operator. Measure this oracle separately from candidate
construction. Its completion is required; focused tests do not substitute for
the full actual coefficient audit.
Automatically retain tests, source freeze, provenance, graph/work counts,
result or failure, construction metrics and exit hashes in
results/c7_factored_hz_20260905_v1/. Save a new lifted checkpoint only there.

The offline numeric-root ledger must include the ENTIRE loaded native state,
new HZ, reconstruction graph/slots/scales/supports, original sources/operators,
and retained proof state, not only the final generator map. For the same
expanded comparator, preregister at most the TWO largest distinct reachable
Conv leaves with distinct contents as well as identities, first occurrence
on ties, as a reference lower-bound witness.
Their aggregate expanded entries must stay <=64M and their jointly retained
numeric payload <=1 GiB. This is the unchanged subset/owner-union proof,
not a new comparator or a complete reference-size claim. Reference building
is separately measured after candidate construction; it must not contaminate
the candidate construction peak. If strict complete bytes AND entries decrease
cannot be proved, stop this version. Arbitrary Python heap remains separately
measured, not silently included in the exact numeric metric.

The C5 prefix and all its existing quarter-work proofs remain unchanged.
C7 cannot advance live until its own complete work and numerical/physical
qualification is audited; an upper bound alone is not a measured speed gain.
Formal 1870/2413, independent E0 61/400, default-off status, 13-family and
400-row single-path retention, witness and four-concurrent gates are unchanged.
