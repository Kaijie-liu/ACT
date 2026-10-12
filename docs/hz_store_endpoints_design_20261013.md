# Checked-store endpoint consumption: finite freeze

Frozen before numerical controls, following `e53b6cad9023e45a7c9d18a697971a448b0acc6d`.
This is a new component protocol, not permission for real-model, sealed-input,
physical GPU, convolutional or production-supervisor execution.

## Algorithm and acceptance

One immutable request-owned HZ store supplies common constraints, eight or fewer
unconditional private expert suffixes, and every unordered pair guard. No joint
HZ, repeated source snapshot dictionaries, or dense LP is constructed. Sparse
native common/template matrices are prepared once per request; pair guards are
prepared once per pair. At most eight objective columns are live per tile.

The proposal algorithm remains CPU float64, one thread, 128 projected subgradient
updates with step `0.125/sqrt(i+1)`, initial zero multipliers and best-iterate
retention. Equality multipliers are free, inequality multipliers nonpositive.
CPU exact arithmetic checks original rows and the zero fallback. Shared-factor
residuals are added before absolute values. Global registry order differs from
the old continuous/binary concatenation; do not promise bit-identical floating
iterates or speedup. Exact same-candidate differentials must agree.

An externally anchored property contract fixes store, source/request, class
count, label, margin, selected-softmax top-2 and ANY_LEGAL_TOPK semantics.
Properties are independently derived as every `y_label-y_k-margin`, not a
producer-selected subset. Every pair is retained, including ties/empty guards.
For source-checked objects the smaller-index expert's gate interval is derived
from the exact box of its router-score difference: tie `[1/2,1/2]`, nonpositive
`[0,1/2]`, nonnegative `[1/2,1]`, otherwise `[0,1]`. Given-HZ controls use `[0,1]`
with an explicit supplied-gate premise. Distinct endpoints cannot be dropped.
Each obligation takes the minimum checked endpoint bound; all obligations must
exceed the unchanged `1/10000000` threshold. Missing evidence is UNKNOWN;
duplicate, foreign or malformed evidence is rejected. No negative bound is UNSAFE.

The checker is standard-library-only and does not invoke proposal/source
producers. Source mode independently checks the declaration and store; its
guarantee concerns declared real operators, not captured checkpoint identity or
native floating execution. Given-HZ mode never upgrades to a source proof.

## Fixed controls and resource envelope

Five normal calls per attempt, in this order, each with one 300-second deadline:

1. `source8_tile8`: x in `[-1,1]^8`, router `r_i=x_i`, eight experts with
   `h_i=ReLU(x_i)`, outputs `y_0=2+h_i`, `y_k=(k/32)h_i`, k=1..9.
   Label 0, margin 0: 28 pairs, 252 obligations, 504 endpoints, 84 tiles.
2. `source8_tile1`: identical source/property contract, one column per tile.
   Query/obligation identity is invariant to tiling. Retain any optimizer
   differences; same saved candidates must have identical exact checks.
3. `negative3_tile8`: x in `[-1,1]^3`, three constant-zero router scores;
   expert i uses ReLU(x_i), outputs `[2+h_i, h_i/32, 3+h_i]`.
   Label 0, margin 0: three tie-legal pairs, six duties, six unique endpoints.
   This must remain UNKNOWN; no complete positive is permitted.
4. `mixed_tile8`: the already frozen `fixture('mixed')` source recipe from the
   source-row stage, label 0, margin 0, all pairs/properties. No expected positive
   count is selected; bounds and complete/partial status are retained.
5. `given_large_tile8`: the existing fixed 25,648-factor given-HZ recipe,
   eight experts, ten classes, label 0, margin 0, supplied `[0,1]` gates;
   all 504 endpoints. This is not checked network provenance. It independently
   tests combined factor/query consumption, not trained source density.

The old per-operation store limit of 8M visits does not compose to 504 queries.
This **new endpoint composition** explicitly permits at most 96M cumulative
visits, at most 8 simultaneous targets, 504 total targets and one deadline.
No counter resets per pair/tile. Underlying dimensions, rational bit caps and
acceptance threshold are unchanged. The component limit is not a claim that a
real complete request fits 300 seconds. At most 2 generation attempts; preserve
the first failure and snapshot before any repair. Combined archive cap 128 MiB,
1 MiB reserved for terminal records. One CPU thread, no GPU/native solver.
Stop on timeout/resource refusal; do not shrink a fixture or extend the deadline.

Record launch-to-receipt and internal source creation, parsing/checking, native
preparation, proposal, exact selection, serialization and offline checking
separately. Overlapping substage times are labelled, not summed twice. Test
harness deadlines and offline checks do not establish production supervision.

Controls cover partial/duplicate/foreign pair/property/endpoints, changed label,
margin/source/store/gate orientation, inward gate claims, bad dual signs,
wrong bound, tile tail, deadline/work-limit refusal, caller alias pollution,
and tile-1/tile-8 query equality. Small materialized coefficient/dual differentials
check the support formula independently. A relocated `python -B -I -S` checker
must need no checkpoint, dataset, repository, numerical library or solver.

## Decision

Continue only if complete original-row endpoint coverage and source/property
binding pass independently. Failure remains visible; fix a specific interface
defect only within the two-attempt cap. Stop this control study after the finite
decision. Production hard-budget intake and the same-algorithm block GPU kernel
remain separate next gates. No real competitive or high-accuracy gain follows
from these synthetic controls; all goal gates remain OPEN.
