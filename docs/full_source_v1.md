# Full expert source chain and NEW output LP obligations V1

This stage extends the checked input98 first-Conv/ReLU source prefix through
both complete experts. It addresses a named complete-MoE proof gap: the
previous positive output proof relied on unchecked full-expert lowering.
It does not optimize another old LP, tune a range, change the production
acceptance policy, or splice an old positive dual into new matrices.

## Frozen scope and acceptance

One unchanged convolutional request: input98, model state
`44572f06b657883fe169406d3cc260878d1cf41e762cbacdb15fe006da6487f5`,
the represented box of radius2/255, label0, sole legal pair[1,2]. All five
excluded pairs must be freshly checked from the original router parameters.
The existing prefix manifest is
`01e7a3b62ae49c1ed89eb6ddc3aaaf23ce8f1ca202c062be5483b04b896a2579`.

Each expert must continue through coordinate lifting, Conv2d, ReLU, AvgPool2d,
Flatten, Linear, ReLU, Linear. The original parameter inventory, operator
options, shapes, layer order and shared/private factor identities are bound.
Missing a layer, property or route disposition rejects the composed result.

Success is `CHECKED_FULL_EXPERTS_AND_NEW_LP_CONSTRUCTIONS`: complete declared
real expert containment plus all9 new weighted classification LP constructions.
It is **not** SAFE, a checked positive lower bound, or a deployed floating-point
execution proof. There are0 optimizer calls,0 old LP certificates and0 reused
membership facts. Positive lower-bound generation is a separate next task.

## Sparse exact affine lift

Naively propagating all input/error generators through every affine operator
can create long expanded rows. Instead use an exact lifted graph. If a source
row combination is `a = c + g*xi + h*beta`, compute all coefficients as exact
rationals and `r = sum(abs(g)) + sum(abs(h))`. For r>0 introduce a fresh
continuous factor eta in[-1,1], output `c+r*eta`, and retain the equality
`g*xi+h*beta-r*eta=0`. For r=0 output the constant c with no new factor.

For every old feasible assignment, eta=(g*xi+h*beta)/r lies in[-1,1], so
the extension exactly equals the affine output. Conversely the equality fixes
the output. No rounding compensation is needed for these rational operations.
All prior factors and constraints survive. An initial identity-coordinate lift
after the checked prefix keeps later convolution rows sparse; it changes the
representation, not the represented output relation.

The independent checker recomputes coefficients, radius, new factor names and
every equality. It does not call the producer. Conv indexing is checked using
the existing independent CHW parser; pooling is exact nonoverlap averaging,
flatten preserves CHW coordinate order, and Linear is checked against bound
parameter bytes. The existing exact ReLU graph check is reused unchanged.

This is not an unchecked interval reset: the box is used to choose a finite
factor scale, while the exact defining equality retains its input relation.
Generator-box ReLU ranges can nevertheless be loose. Later LP relaxation of
the binaries may therefore lose precision; no positive result is promised.

## Storage and complete-output construction

Layer deltas store only new factors/constraints and new outputs. Their source
and target hashes bind a lossless append-only reconstruction; the restored
state still undergoes the full mathematical step check. Flatten changes only
the checked shape. A final explicit map joins both experts on the checked
common input, with all private factors disjoint.

For each competitor k!=0 the objective is
`E_b,0-E_b,k + lambda*((E_a,0-E_a,k)-(E_b,0-E_b,k))`.
The newly checked joint state supplies both terms in one factor vector.
Use the universal real selected-softmax bound lambda in[0,1] (no historical
sigmoid/range certificate). The difference bounds are recomputed exactly from
the NEW state's coefficients. Four rational McCormick planes and finite corner
bounds enclose the product. Binary +/-1 factors are explicitly relaxed to[-1,1].

One common LP constraint file and nine property blocks avoid storing the source
matrices nine times. `full_source.obligations.materialize` can produce each
ordinary CSR LP without a solver. Independent construction checking validates
the common constraints, complete competitor inventory, shared source identity,
objective, difference bounds, all four planes and variable bounds. A synthetic
differential also compares materialized LPs with the existing rational builder
and its independent construction checker.

All nine lower-bound-certificate slots must be null in this construction-only
stage. Hash agreement or successful construction does not establish positivity.

## Controls and resource protocol

Seven new controls cover exact affine witness extensions, constant and binary
cases, source/delta binding, pool/flatten semantics, LP differential, moved
full-network composition with7 semantic mutations, and supervisor deadline,
exception and partial-output handling. Existing prefix/router/portable and
lifecycle regressions remain enabled. A no-op coefficient mutation in the first
test draft was corrected to an actually different coefficient; no real run
or source threshold was changed.

Resource admission precedes execution. One CPU process, existing act-py312,
no GPU, no dependency changes; original other-user jobs are untouched.
Single300s budget for copying the prefix, one checkpoint parameter capture,
new remaining-layer propagation, delta/LP serialization, fresh full checking,
and terminal publication. Both child stages share the actual remaining budget,
with2s reserved for publication; no retry or budget increase after the result.
No whole-network forward or native LP solve occurs. Old prefix generation is
excluded and explicitly disclosed: this is not production end-to-end timing.

Architecture-only worst-case factor count is at most34,974 continuous and5,125
binary factors after all layers, before the two McCormick variables. This is a
size bound, not a performance prediction. Actual size, nnz, RSS and per-layer
costs are recorded. A separate relocated recheck is timed outside the execution
budget. Partial evidence is retained, never upgraded after a deadline.

New directory `data/moe/results/full_source_conv98_20260920_v1`.
After preparation commit/push: `python -m full_source.run`, then
`python -m full_source.review <new directory>`. Freeze and archive either outcome.

## Trusted boundary and next decision

The checker proves containment and LP construction for the **declared real
graphs and pinned represented input box**. Mapping the captured topology to the
intended program, checker/interpreter correctness, and native floating execution
remain distinct questions. No claim about the exact requested epsilon ball or
preprocessing is introduced. No training, new sample, sealed search or claim
upgrade is authorized by successful source construction alone.

If complete construction checks, only newly bound lower-bound evidence on these
matrices can close the output obligations. Old positive9-property certificates
remain valid only under their original upstream assumptions. If this run reaches
its limit, preserve the partial trace and identify the unfinished source step;
do not call the model unsafe or increase the budget automatically.
