# Checked declared-source connection to HybridZ endpoints

Frozen before implementation and execution on 2026-10-01, starting at
`367ef5beb7403cd967cfbe975a396063e64c200c`. This is an additive CPU control,
not permission to rerun real/full-size sources, use CUDA or reopen sealed inputs.
The [protocol](../configs/hz_source_connection_20261001.json) fixes the four
existing H2 declarations without changing their coefficients. No old positive
bound or LP matrix is reused. All pairs and all non-label margins remain duties;
the historical partial-reuse fixture name does not enable free reuse here.

## Construction and independent acceptance

The source parser binds the stored binary64 parameters, exact clipped input box,
eval graph and selected-softmax/ANY_LEGAL_TOPK semantics. The checked input HZ
must contain that box. Affine propagation calls `sparse_hz_linear`; its floating
nominal coefficients are only proposals. Exact rational residual polynomials
determine additional independent continuous error factors, and a separate
checker verifies both their radius and placement. Thus floating affine error is
enclosed, not ignored or repaired by a fixed epsilon.

ReLU calls the actual `sparse_hz_apply_relu_exact` kernel with rational box
ranges, represented exactly in binary64. The independent checker verifies the
result against the existing exact ReLU graph theorem, translating only the
documented blocked inequality order to its interleaved order. Every inherited
row, new slot, range, private name and output expression is checked. A conversion
to live SparseHZono is refused if any rational coefficient is not exactly
representable; full snapshot equality is required. This capacity restriction
is explicit, not a claim of generic exact floating propagation.

For each unordered pair, retain the complete checked router factor space and
constraints, restore the original input output expression in that space, and
append, for selected a and outside o, the exact rows
`(Gc_o-Gc_a) xi + (Gb_o-Gb_a) zeta <= c_a-c_o`.
These are conditional restrictions, not globally redundant guards. Every real
execution whose pair is tie-legal has an enclosing router witness satisfying
these non-strict rows; that same witness carries the original input into both
expert propagations. Private expert factors have distinct names. No route
infeasibility claim drops a pair.

For canonical a<b, the weight is sigmoid(r_a-r_b). A checked global router-box
margin proves [0,1/2], [1/2,1], [1/2,1/2] or, absent sign information, [0,1].
This deliberately loose sign rule is frozen. The source checker independently
binds the input/guard/expert terminal states and gate to the actual endpoint
adapter's snapshots, and reconstructs every class margin with offset -margin.
The existing independent endpoint checker then checks fresh batch dual evidence
for every gate endpoint. Positivity uses the unchanged strict 1e-7 threshold.

For a fixed factor assignment, weighted output is affine in the scalar weight;
its minimum on the covering gate interval occurs at an endpoint. Requiring both
endpoints over the checked pair enclosure therefore suffices for that pair.
Together with all tie-legal pairs, this proves the declared real-network output
property when all bounds are accepted. Nonpositive or absent evidence is UNKNOWN,
not a network counterexample. Relaxing binary factors remains an outer bound.

## Scope and stop rule

This connects actual sparse affine/ReLU and shared/private pair kernels to a new
checked source trace. It does not retrospectively prove the legacy analyzer or
make the new path a production default. In particular the prior conditional
support subclass has different range/lowering assumptions. Program/declaration
correspondence, the exact checker implementation and deployed floating-point
execution remain separate trust questions. This stage does not produce a
portable distribution or a source-path hard-budget supervisor.

Record construction, proposals, checking and serialization costs under one
cooperative deadline; imports and archive/test orchestration are separately
identified, not hidden in a speed claim. Test source binding, inward boxes,
deleted affine compensation, ReLU rows/ranges, missing layers, reversed guards,
aliased private factors, wrong weight orientation, endpoint source mismatch,
missing pairs/properties, partial/stale proofs, nonrepresentable coefficients
and expiration. Do not tune the four cases to obtain positives. Complete the
controls and independently recheck saved packages; retain failed attempts.
