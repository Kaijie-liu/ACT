# Large sparse basis construction and equality-residual mapping

## Delivered scope

New `sparse_basis/` is a separate, opt-in candidate-construction interface.
It does not edit `exact_basis/`, `native_basis/`, `basis_supervised/`, their
limits, or any frozen real diagnostic result. Starting HEAD was
`9ef9e63698fe8aa4a87681acb9e7ff50c4c6fb91`. The user explicitly authorized this
separate large-sparse/mapping development after the compatibility review.

[Attempt004](sparse_basis_controls_attempt004.json): **95/95 controls PASS**,
16 new controls plus79 existing compatibility/supervision/component regressions.
The new tests contain **4 actual native analytic captures**, including a
4,096-variable/8,192-equality system; prior suites have additional analytic
calls. **Zero real-network LP calls or reconstructions** occurred. No model,
training recipe, original LP or mathematical acceptance threshold changed.

[Fresh-process review](sparse_basis_v1_review.json):36 stored analytic artifacts
match hashes;4 native mappings reproduce;5 standalone original-LP checks pass
their expected positive/negative feasibility outcomes. No new native solve or
candidate reconstruction was used in that review. This is not network proof.

Preserved attempts:001 had4 native-interface errors before the solves because
an already parsed `Fraction` was incorrectly passed back through the checker’s
serialized-number parser. Only the new conversion wrapper was fixed; the
frozen checker was not changed. The large synthetic construction already passed
in that attempt. Attempt002 passed91/91;003 added arithmetic-growth, permutation,
deadline and heap controls,95/95;004 retains the full native mapping/proposal/
check-reference artifacts for fresh replay,95/95. No failed attempt was erased.

## New algorithm, not just a larger gate

The previous sparse elimination scanned previously accumulated pivots for each
incoming row and was explicitly analytic-small. The new engine assembles only
the selected original-coordinate basis and maintains a **column→active-row
incidence index**. A heap selects the minimum active column degree; among its
incident rows it selects the shortest row, breaking ties by original index.
This is a declared heuristic, not global Markowitz minimization or an optimal
fill ordering.

A singleton-column pivot requires no other row update. A general pivot updates
only incident rows, maintaining the incidence sets exactly. Stale heap entries
are discarded; periodic heap rebuilds bound their growth. Reverse substitution
uses the recorded pivot order. The assembler’s row list is consumed rather than
retained as a duplicate matrix. There is no dense m×m array, floating solve of
the basis, near-zero pruning, omitted coefficient, or alternate-basis search.

The engine records active/pivot storage, fill insertions, heap size/rebuilds,
row updates, candidate pivot-row inspections, singleton pivots, arithmetic/work
events and time. `peak_live_nnz` counts active plus stored pivot coefficients;
it is **not** process RSS and does not include every input/serialization buffer
or transient Python object. Byte-level resource accounting remains separate.

All coefficient arithmetic is exact rational arithmetic. Serialized floats are
interpreted as their exact binary rationals. Input decoding, elimination growth,
fill and deadlines are guarded. The new contract is:

| Resource | New component cap |
| --- | ---: |
| Original x variables |16,384|
| Original E+A rows |16,384|
| Input CSR stored entries |1,000,000|
| Active plus pivot entries |2,000,000|
| Fill insertions |1,000,000|
| Pivot heap entries |200,000|
| Rational numerator/denominator bits |4,096|
| Accounted arithmetic/work events |20,000,000|
| Native call |one, at most10 seconds|

These are limits, not completion guarantees. `LIMIT`, `TIMEOUT` or
`UNRESOLVED_SINGULAR_BASIS` emits no candidate bundle and proves no LP
infeasibility. A singular selected basis may coexist with a feasible LP.
Component calls require a finite caller deadline no more than300 seconds ahead;
they do not themselves provide an owned outer watchdog.

## Original-coordinate mapping including basic equality rows

For the supplied LP

```
E x = h,  A x <= b,  l <= x <= u,
```

use named augmented coordinates

```
E x + e = h,    e = 0,
A x + s = b,    s >= 0.
```

`e` is `E_residual`; `s` is `A_slack`. Both use **+1** columns, defined from
the original equations, not guessed signs of native row-variable integers.
The native point and basis statuses remain untrusted hints. Mapping rules:

| Native item | New coordinate/action |
| --- | --- |
| Structural kBasic |basic x|
| Structural kLower/kUpper |exact original bound anchor|
| E-row kBasic |basic E_residual, subsequently required to be zero|
| E-row kLower/kUpper |E_residual anchored zero|
| A-row kBasic |basic A_slack|
| A-row kUpper |A_slack anchored zero|
| Other/free/unknown status |UNSUPPORTED_MAPPING|

Every original row has its identity/hash in the manifest; all original and
residual coordinates have a complete, disjoint basic/anchored partition. A
basic equality residual **does not waive that equality**. The constructor may
produce nonzero e or negative s; its output is still only `CANDIDATE_ONLY`.
All original constraints must be checked by the unchanged standalone checker.

### Correctness boundary

An original feasible x uniquely extends to `(x,e=0,s=b-Ax)`, satisfying the
augmented rows and residual bounds. Conversely, a point satisfying those rows
and residual bounds projects to an original feasible x. This is an exact
reformulation of the feasible set, not deletion of redundant constraints.

For any proposed basis/anchors, exact elimination only constructs a potential
augmented solution. It does not establish residual bounds or x bounds. The
independent original-LP checker verifies the projection x against **every**
original E row, A row and box bound and recomputes its objective exactly.
Thus a valid upper bound does not depend on trusting the native basis, pivot
heuristic or candidate construction. A nonzero basic E residual fails the
original equality check; the implementation never labels it feasible merely
because the augmented linear equations were solved.

The checker does not certify native optimality or completeness of the generator.
A checked feasible nonpositive LP point is not a full-model counterexample.
Network→HZ, guards, route exclusions and property/F0 lowering remain trusted
upstream components; this interface starts with a given serialized LP.

## Native capture at larger dimensions

`sparse_basis/native.py` is a new capture path, not a monkey-patch of the old
size limits. It retains highspy1.14.0, simplex, presolveOFF, scaling0, threads1,
parallelOFF, seed0 and actual option readback. No dependency was installed.

Readback traverses the native sparse rowwise/columnwise representation once and
reconstructs canonical original rows in Python. It does not issue a repeated
native row extraction for each row. Row-local canonical sorting remains charged;
the single traversal is not a worst-case linear bound for arbitrary sorting.
Submitted/readback values must match the
explicit original-to-float conversion; these floating values never replace the
original rational LP used by the constructor/checker.

`input.json` and `submission.json` precede optimization; `raw_native.json`
preserves statuses, point and objective immediately after native return, before
post-solve readback or mapping can fail. `capture.json` adds native version,
binary hash, input/statement hashes, policy, options and complete snapshots.
No-overwrite creation and one-call semantics remain. A future outer supervisor
must preserve partial records if capture cannot complete.

## What the controls establish

1. The earlier two-variable redundant-E native case now maps both basic E rows
   explicitly and passes all original constraints, without deleting either row.
2. A native4,096-variable system with8,192 equality rows has4,096 basic
   E-residual columns. Exact reconstruction checks all8,192 rows and obtains
   U=−4096/3. The selected system used8,192 singleton-column pivots and no fill.
3. A preconstructed triangular system with9,500 variables,6,500 rows and
   **356,015 stored entries** uses6,500 singleton-column pivots, zero other-row
   updates, zero fill insertions, and1,179,547 accounted events. Its moved
   `python -I -S` package checks U=−6500/3 without solver/model imports.
4. Two distinct exact RHS values,1/3 and1/3+2^−60, collapse to the same submitted
   binary64 value. Native capture supplies a basis, but reconstruction retains
   the original RHS difference. Its E residual includes2^−60 and the full
   checker returns `NOT_EXACTLY_FEASIBLE`, upper=null. Native success cannot
   erase this discrepancy.

The large controls are deliberately structured synthetic systems, **not real
network LPs or predictions of their runtime**. Their singleton-rich graphs are
easy for this ordering. Equal size/nnz does not imply similar fill or bit growth.
Recorded construction/check times are analytic component observations, not a
matched speed comparison or a300-second full-request production measurement.

Other controls compare exact results to the old elimination on small coupled
systems; exercise nonzero fill and rejection at its cap; test bit growth during
elimination, heap and original deadlines, singularity, fixed bounds, exact binary
coefficients, permuted structural/basis columns, mixed row slacks, native option
and identity drift, unknown statuses, missing/duplicate rows/coordinates and
nonzero basic equality residuals. Full checking never skips unselected rows.

## Next integration boundary

The new component’s declared dimensions cover the sizes from the read-only
real-LP inventory. This removes the old analytic-size restriction **for this
new component**, not for the unchanged `basis_supervised` production path.
No actual real-LP basis, rank, fill, bit-growth or feasible point has been
measured. The old compatibility report remains correct for the old interface.

Before freezing real diagnostics, bind the new schema/policy and raw-capture
layout to a **new** single-original-budget supervisor/terminal audit and test
native/exact/check cutoffs and partial evidence at the larger interface. Do not
silently swap it into the frozen old runner. Then make a separate compatibility
and selection decision; do not rerun or repair the sealed four diagnostic
records in place. This release supplies no new real diagnostic launch/freeze.
