# Fresh lower-bound candidates for the complete source-checked request

This finite follow-up targets all9 NEW output LPs of
`docs/full_source_v1_results.md`. It moves from a checked full-expert enclosure
and LP construction to independently checked output lower bounds. It does not
reuse the older positive duals or alter any existing matrix, range or result.

## Frozen request and method

Same convolutional model/input98, represented2/255 box, label0, sole legal
pair[1,2]. Parent manifest:
`ade6c1b34db6fa4923c9c8f13e6a547771aa0aa723fa32027b77c93a93bb295d`.
New joint:
`93b15af1a0ff299aa60a6be0525513947ad0640f51749471e596a8709db0e3f3`.
Common LP base:
`96a1f576413532c0c3dd3914506ff0cef6e623a265dd92ba1b312842c160831a`.

All competitors1..9 run in ascending order, at most one native call each.
SciPy1.16.3 `linprog(method="highs")`, NumPy2.3.5, **SciPy's bundled
HiGHS1.8.0**, one thread, native time limit16s per call. This is not the
separately installed highspy1.14.0. The explicit threads option is forwarded by
SciPy with its normal warning. No precision, solver, gate[0,1], difference,
McCormick, ReLU bound, input, property, sample or optimizer-option search.

Numeric matrix conversion is shared within this fresh worker. No saved source
cache, old solve state, basis, dual or check verdict is supplied. Materialized
exact CSR LP identities remain bound independently, even if tiny coefficients
are rounded or dropped by the numerical proposal path.

## What is checked

The untrusted worker saves successful solver multipliers as exact binary64
numbers. Inequality multipliers are projected to nonpositive values. Solver
objectives and approximate primal points are recorded as diagnostics only;
neither is an accepting bound or a validated counterexample. If the solver
does not complete, no multiplier candidate is accepted from that call.

The independent stdlib checker uses the existing unmodified rational
`sparse_lp_certificate.evaluate`, without requiring any claimed lower bound.
For `min c*x+d`, `Ax<=b`, `Ex=h`, finite `l<=x<=u`, and signed multipliers
`y<=0`, unrestricted `z`, it recomputes

`r = c-A^T*y-E^T*z`,
`L = d+b^T*y+h^T*z + sum_j min(r_j*l_j,r_j*u_j)`.

For every feasible point, `y*Ax>=y*b`, so the expression is a valid lower
bound; no approximate stationarity or solver success/objective is trusted.
The exact residual-box contribution, residual L1 and nonzero-coordinate count
are reported for each available candidate. Positive means strictly L>0; zero
does not pass. No production optimal-status or numerical-policy gate changes.
The residual-box term includes legitimate finite-bound dual contributions; it
is not by itself a measure of numerical error. Compare the checked bound with
the separately labeled floating solver objective before interpreting the gap.

Before any output aggregation, the moved checker freshly verifies the entire
parent input/affine/ReLU/pool/flatten/Linear/factor chain, exact route coverage
and all9 LP constructions. Every candidate is then bound to the request, new
source, common base, competitor and full materialized LP identity.

Only9/9 positive checked bounds may return
`CHECKED_POSITIVE_DECLARED_REAL_MOE`. Missing candidates return
`UNKNOWN_MISSING_BOUND_EVIDENCE`; complete but nonpositive evidence returns
`UNKNOWN_NONPOSITIVE_BOUNDS`. None of the negative cases establishes UNSAFE,
an intrinsic LP gap or a network counterexample. Never combine the old proof's
positive bounds with this larger source or silently omit a difficult property.

Even a positive aggregate is for the **declared real graph and pinned represented
box**. Captured graph-to-intended-program correspondence and checker execution
remain assumptions; native floating execution is not proved. The fixed model
is67.06% clean accuracy and this request is single-route, not the outstanding
high-accuracy/cross-family route-changing target.

## Complete budget and lifecycle

One300s stored-source budget: prepare at most15s, proposal batch at most180s,
seal at most10s, independent complete checking within the actual remainder,
and2s reserved for publication. Phase caps never add to a larger total.
Copying/validating the old **new-source** bundle, float conversion, all native
calls, candidate serialization, namespace/code packaging, complete source
checking, exact residual checks and terminal publication are charged. Original
full-source generation is excluded: this is not production end-to-end timing.

Each candidate is written to a fresh partial file, fsynced, and atomically
published without overwrite. A killed producer leaves partial artifacts in
the raw directory. The sealer only consumes published candidates; missing
properties remain explicit roster entries. A proposal timeout may be followed
by checks of available evidence within the same total budget. An execution
exception cannot become a successful terminal. No late publication may
establish a positive result. Only this task's owned child process group is killed.

Control tests include exact residual comparison with the original checker,
negative/zero bounds, malformed signs/identities, all-necessary-property coverage,
partial candidate publication, exception/deadline handling, and a moved tiny
full-source positive example. Wrong request/property/LP, duplicate or missing
outcomes and bad dual signs reject even after transport hashes are recomputed.
An explicit absent candidate returns UNKNOWN rather than a false complete proof.
Existing source and portability regressions remain enabled.

After preparation commit/push, execute once:

```sh
python -m full_bounds.run
python -m full_bounds.review data/moe/results/full_bounds_conv98_20260920_v1
```

A separate relocated mathematical recheck and independent cost/identity audit
are timed outside the execution budget. Archive either outcome, including all
missing or nonpositive rows. No automatic retries, scope expansion, new samples,
model training, source reconstruction or renewed arithmetic/backend search.

## Decision after the run

If positive, report precisely the new source-to-output proof contract and keep
native floating/high-accuracy/route-changing claims open. If nonpositive, use
saved solver objectives and exact residual corrections only to distinguish
observed numerical-candidate weakness from unresolved optimization/relaxation
causes. A floating primal estimate is not exact feasibility evidence. If no
candidate survives, report that finite execution limit without asserting the
model is unsafe or the LP cannot prove it. Any subsequent improvement needs a
separate bounded hypothesis based on these complete-request records.
