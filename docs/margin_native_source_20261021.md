# Native HybridZ source bridge for margin support

This default-off implementation connects the existing shared-expert HZ and CPU
support-candidate code to exact margin-property views. It is the first part of
A1, **not an executed integration result**. No trained source, numerical search,
GPU or control batch has run in this stage. Production verification is unchanged.

Entry was clean `feat/moe-route-verification` at
`e541f9602ca4ce8a4d63e2e67fa21c3e1f081d3b`. The
[current goal](ALGORITHM_RESEARCH_GOAL.md) supplies standing implementation and
bounded-control authority; no additional approval is needed to finish A1 within it.

## Implemented source and objective contract

[Native adapter](../act/back_end/moe/margin_hz_source.py) calls the existing
`shared_input_pair_hz` composition after validating the original CSR buffers.
It checks that every router constraint is retained in the guarded entry by
explicit row maps, then appends only the two router output rows. It does not
concatenate another copy of the router constraint system or change the expert P.

The supplied joint output is `[E_a[0:K], E_b[0:K], r_a, r_b]`. The
[standard-library receiver](../scoped_source/check_margin_hz_source.py) separately
reconstructs the entry/expert composition, shared/private column maps, retained
router rows and all constraints. It binds request, model, materialized-input,
domain and guard identities, canonical pair direction and the complete property.
These identities are caller declarations; checking their consistency is not a
proof of their connection to the original network or data.

All K property coefficients and the constant are retained. From original stored
binary64 coefficients, interpreted exactly as rationals, the property view forms

```
b = q^T E_b + c
d = q^T(E_a - E_b)
m = r_a - r_b
objective = b + t*d + nu*m
```

No floating dot product precedes this projection. Continuous and binary blocks
remain separate until the final coefficient vector; the router binary shift is
the **joint continuous width**, not the router's or shared-prefix width. A source
with K=10 retains all22 output rows, including classes8 and9. Missing, changed or
extra property coefficients cannot be silently dropped.

Only the optional `MARGIN_NATIVE_SOURCE_V1` profile expands parser dimensions:
32,768 factors,65,536 constraints and16,384 source output rows, using existing
store shape constants. Input/entry sources may have3,072 outputs even though the
two ten-class experts plus router have22. Legacy `None` profile remains
128 factors/256 rows/128 outputs; the old margin prototype remains untouched.
This is versioned source expressibility, not admission of any full model shape.
The128MiB serialization contract,4,096 stored bits and8,192 intermediate bits
remain. Large captures can still fail time, bytes or address-space constraints.

## Candidate and exact reception

The adapter reuses `_candidate_columns` with its existing128-update CPU strategy,
single-thread requirement, and caller deadline. At most eight explicitly supplied
goals are permitted. Existing exact-planner `Fraction` anchors/tilts are accepted
through a canonical stored-bit check; they are not rounded into new objectives.

Generated and zero dual candidates are both evaluated with a separately written,
metered producer residual calculation. The receiver then recomputes, from the
original HZ rather than the proposed LP,

```
L = constant + rhs_inequality^T y + rhs_equality^T z - sum(abs(residual))
y <= 0
```

Every inherited constraint is inspected, including rows whose multiplier is zero.
The binary factors are explicitly relaxed from `{-1,+1}` to `[-1,1]`.
The caller supplies the full expected objective roster; absent candidates and
wrong identities cannot become accepted partial supports. Producer arithmetic,
capture, projection and exact reception use the same Work/deadline object.
The existing floating search is still untrusted; its native arithmetic is charged
by wall time, not misreported as fully enumerated Work operations.

Receipt grade is `CHECKED_SUPPORT_ON_GIVEN_HZ_CONTINUOUS_RELAXATION`, with
`scalar_aggregate_checked=false` and `complete_request=false`. The interface
does not yet connect the sparse view to finite-anchor selection, complete scalar
cells or the verifier's complete route/property aggregation. It does not close
input inclusion, network conversion, route reachability or G3.

## Resource policy status

[DevelopmentWork](../scoped_source/development_work.py) is a new opt-in account:
it inherits the original deadline and exact-bit checks, accumulates every defined
visit, and omits only the8M rejection gate. Continued worker counts cannot reset
a nonempty account. Legacy `Work` is unchanged.

This is not a hard supervisor. The approved per-process8GiB AS, telemetry-only
RSS/ru_maxrss, same300-second request budget and complete online reception still
need minimal versioned wiring in the existing supervisor before native controls.
No old native attempt is restored. Full capture and repeated source checking may
be expensive; this stage has no timing result or speedup claim.

## Static review and next executable gate

AST parsing of all five affected Python files and `git diff --check` passed.
Two read-only AI reviews identified and prompted fixes before execution:
the initially retained128-output limit would reject3,072-row input sources;
Fraction goals did not match the old parser types; and old producer `_evaluate`
did not meter its exact arithmetic or intermediate bits. Original CSR validation
now also precedes native composition. These are static fixes, not passing runtime
controls or human review.

The next step is to finish the minimal development supervisor integration and
freeze **one** A1 integration batch, at most300 seconds, containing only new
interface/resource cases. Required cases are:

- Complete ten-class and full-height input views, nonzero constants and a
  dense signed q, compared with direct rational coordinate sums.
- Shared/A-private/B-private continuous and binary columns; unchanged P;
  router retained-row maps and the joint binary offset.
- Existing Fraction-plan goals, canonical direction, missing objectives,
  forged lower bounds, changed property/source/scope and zero-dual row coverage.
- Malformed CSR rejected before native composition; immutable capture and
  live-source alias mutation at candidate boundaries.
- New-versus-legacy capacity and Work behavior, unchanged stored/intermediate
  bit limits, one continued deadline/account, and expired/partial reception.
- A finite actual CPU candidate call, all costs retained; missing RSS telemetry
  must not imply exit, while timeout, exception and cleanup errors still reject.
- Saved-source/property/candidate reception using only standard-library code
  from a new location, without native packages, checkpoint or historical paths.

The roster, counts, sources, code hashes and termination rules are **not yet
frozen**, so this list is not permission to launch an ad-hoc pretrial. Do not
rerun the old13 toy groups or use these interface controls as A2 effectiveness.
Real-development ledger remains0 stages/0 calls; the next stage needs no repeated
PI permission within the already adopted contract. G1–G6 remain OPEN.
