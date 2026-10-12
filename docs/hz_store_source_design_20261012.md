# Frozen source-row component on the request-owned HybridZ store

Frozen before numerical execution, 2026-10-12, parent
`32459552ee6022e89570ca9b10772e843119f3cb`.
This is the next component of the [request-store design](hz_request_store_design_20261012.md),
not admission for real models, CUDA, new endpoint queries or sealed inputs.

## Mechanism and guarantee

Consume checked immutable view rows directly. Reconstruct each exact operator
reference transiently; compare its binary64 outer enclosure with the target.
The parent is the preceding checked binary64 view, never the exact reference
whose new constraints may require rounding compensation. Inherited constraints
remain identical objects. Do not expand views through legacy `unpack`, `parse`,
`snapshot` or small-source checkers. A single bounded live SparseHZono workspace
is allowed; explicit global-to-continuous/binary maps must cover every factor.

Input checks require diagonal nonnegative radii and exact rational containment.
Affine checks derive the nominal old-factor polynomial from target binary64
coefficients and independently compute the exact L1 residual against the declared
operator. The producer obtains that nominal from actual `sparse_hz_linear`.
Each nonzero exact residual introduces a distinct continuous error factor.
ReLU checks derive box ranges, active/inactive/unstable cases and the compressed
exact graph, with negative/positive continuous factors and a private sign binary
for each unstable output. Inequalities use ACT's blocked ordering (all negative,
then all positive), including the unchanged inherited prefix.

New exact output/equality rows are enclosed by rounding coefficients and adding
distinct continuous compensation; inequalities weaken RHS by the exact coefficient
residual. Allocate all operator factors first, then all output compensation,
then all equality compensation across the whole layer, not separately per page.
Common restoration keeps router constraints/factors but restores input outputs.
Every tie-legal unordered pair receives all selected/outside score inequalities;
these guards are conditional, not globally redundant assertions. No conditional
fact is promoted into an unconditional expert template.

The standard-library checker verifies operator identity, every original and new
coefficient, inherited edges, complete row inventory, factor kind/scope/owner and
compensation. Actual SparseHZono matching independently traverses all nine fields
and explicit kind maps. This checks declared finite operators, not correspondence
of a checkpoint to a declaration, native floating execution or complete MoE output.

## Fixed controls and coefficients

1. **Wide affine:** 3,072 input coordinates, center zero, radius
   `w=1+2^-52`; diagonal operator `w`, bias zero. Every nominal diagonal product
   is `1+2^-51`, requiring exact residual `2^-104`. All coordinates are consumed;
   this is not a wide-factor/two-output parser test. Three router outputs select
   coordinates 0, 1 and 3071; three two-output expert projections and all three
   guards complete the structural store, without endpoint proposals.
2. **Mixed graph:** 70 inputs. Default center zero/radius one; indices 0 and 65
   have center `2^-54`, radius one; indices 1..5 respectively have (2,1), (-2,1),
   (0,0), (1,1), (-1,1). Two identity-affine/ReLU layers exercise inherited
   inequalities, multiple unstable rows, existing binaries and compensation on
   both sides of a 64-row page boundary. A final three-output projection uses
   indices 0, 1 and 65. Three experts apply the same two-layer pattern from the
   restored input, followed by two-output projections; their private factors
   remain disjoint. No query outcome is used to choose these coefficients.
3. **Input enclosure:** three intervals `[1/3,2/3]`, `[0,1]`, `[1,1]`;
   nominal centers/radii outward cover exact endpoints. Negative inward tail
   mutation and incorrect coordinate mapping must refuse.
4. Page-size 1/64 differential for the mixed mathematical object; complete row
   coefficients, factor identities, scopes and ordering must agree.
5. Independent analytic checks for all wide affine rows; legacy small operator
   equations are a reference, not a whole-source parsing dependency. Direct CSR
   mutation, bad factor maps, missing last row, wrong source/operator binding,
   absent compensation, wrong ReLU ordering, guard sign, alias pollution,
   partial checking and deadline exhaustion must refuse. A guard tie control
   retains every unordered pair even when score differences vanish.

These are four saved normal objects (wide, mixed at two page sizes, input).
Any extra small object used for ties/mutations is auxiliary and fully counted.
No accepting result may be a hash-only test of a producer-created object.

Pre-execution coefficient clarification: every expert's final two-output
projection selects coordinates 0 and 1 with unit weights and zero biases;
the wide experts have no preceding hidden layers. Mixed experts use identity
affine → ReLU → identity affine → ReLU before that projection. The input
control uses three constant-one router outputs (all three pairs remain), and
the same expert projections. Every input center is the nearest binary64 exact
midpoint; its radius is the least binary64 not below the larger exact endpoint
distance from that center. ReLU ranges are the exact coefficient box ranges.
Actual first-layer compensation at mixed rows 0 and 65 must be checked, not just
an aggregate nonzero count. These clarifications precede all numerical controls.

## Resources, costs and decision

Existing Store V1 capacities remain unchanged. Operator declarations are bounded
to 4 MiB, 4,096 rows/columns and 262,144 parameter nonzeros per step; standard
Work limits and exact bit checks apply. Use CPU, one thread, existing act-py312,
no native solver or device initialization. Each complete control process has a
300-second outer timeout; this is a component guard, not a production supervisor.
No warmups or timing repetitions. Record source creation, native nominal work,
serialization, source checking, actual CSR matching and complete observed cost.
Do not describe offline rechecking as generation cost or performance gain.

At most two complete attempts: preserve a failed first attempt and allow one
documented correctness/receipt repair with unchanged mathematical fixtures and
limits. Unexpected resource failure stops the stage. All attempts, mutations,
relocated checks and code copies together have a 64 MiB regular-file payload cap
with 1 MiB reserved for terminal records; projected admission must include all
planned copies. Refuse rather than delete scientific evidence or enlarge limits.

If controls pass, stop this source component and implement complete endpoint
tiling against these views. If not, record the concrete failure without a real
rerun. Full source capture/tensor streaming, convolution/pooling, all 252 duties,
portable full-budget supervision and block GPU remain separate open work.
All G1–G6 gates stay OPEN.
