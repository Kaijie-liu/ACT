# A phase preserving activation difference factor

This investigation obtains a compact ideal formulation for a projected pair of ReLU differences, retaining both original phase bits. It works when the input difference crosses zero; no ordered-gate assumption or exact three-gate identity is needed. It strictly strengthens even separate single-gate ideal hulls combined with the exact phase-free difference hull on an ordinary example. This is a useful mathematical component, not yet a new Neural-HZ domain, an established novelty claim, or a benchmark gain.

Two tempting alternatives were checked and rejected as the main definition: replacing each output by an anchored residual is a coordinate change, and replacing a ReLU by a shared min/overlap generally retains its original nonlinear cost. Their precise boundaries are recorded below. The positive result instead keeps quantitative coupling between differences and original phases.

## Exact projected relation

Let two original gates on the same source be

```text
r = ReLU(f),     p = ReLU(h),
delta = f-h,     d = r-p,
```

with their original bits `beta` and `eta`. A bit of zero requires nonpositive preactivation and zero output; a bit of one requires nonnegative preactivation and output equal to the preactivation. All choices at a zero preactivation remain legal independently. Assume certified bounds `L=-a<0<U=b` on `delta`, with `a,b>0`.

Define `K` as the projection of this two-gate relation onto `(delta,d,beta,eta)` when the common baseline `h` is unrestricted. This definition intentionally drops `h`, `p`, and the original source predicates. Its binary slices are:

| beta eta | Difference relation |
| --- | --- |
| 0 0 | `d=0`, `L<=delta<=U` |
| 1 1 | `d=delta`, `L<=delta<=U` |
| 1 0 | `0<=d<=delta<=U` |
| 0 1 | `L<=delta<=d<=0` |

These are mathematical slices for the proof, not a runtime phase-enumeration algorithm. For example, in the mixed positive slice choose `f=d` and `h=d-delta`; for the negative slice choose `h=-d` and `f=delta-d`. In same-phase slices a sufficiently positive or negative common baseline realizes every permitted `delta`. At `delta=d=0`, choosing `f=h=0` realizes all four bit assignments.

## Four amplitude rows describe its ideal hull

Together with `L<=delta<=U` and `0<=beta,eta<=1`, the following four rows describe exactly `conv(K)`:

```text
L*eta <= d <= U*beta,
-U*(1-eta) <= d-delta <= -L*(1-beta).             (1)
```

Validity is elementary but phase-sensitive. If `beta=0`, then `d=-p<=0`; if `beta=1`, monotonicity and the unit slope bound give `d<=max(delta,0)<=U`. The lower bound follows symmetrically. Apply these facts to `ReLU(-f)=r-f` and `ReLU(-h)=p-h`, whose bits are `1-beta` and `1-eta`, to obtain the other pair of rows. This includes all independent zero-phase choices.

For completeness, put `e=d-delta`. Four signed-axis endpoints are genuine points of the projected graph, with the indicated free bit:

| Endpoint | d e | Required bit | One concrete realization f h |
| --- | --- | --- | --- |
| A positive | `b,0` | `beta=1`, eta free | `b,0` |
| A negative | `-a,0` | `eta=1`, beta free | `0,a` |
| E positive | `0,a` | `beta=0`, eta free | `-a,0` |
| E negative | `0,-b` | `eta=0`, beta free | `0,-b` |

The origin has both bits free. Every point satisfying (1) is a mixture of two appropriate signed-axis endpoints and the origin:

- If `d,e>=0`, take weights `s=d/b`, `t=e/a`. The beta rows imply `s+t<=1` and `s<=beta<=1-t`; eta is free. The A-positive endpoint contributes beta mass `s`, E-positive contributes zero, and the remaining origin mass achieves the required beta mean. All components have a free eta bit.
- If `d,e<=0`, use `s=-d/a`, `t=-e/b`. The eta rows imply `s+t<=1` and `s<=eta<=1-t`; beta is free. The symmetric construction applies.
- If `d>=0,e<=0`, use `s=d/b`, `t=-e/b`. Now `s+t=delta/b<=1`. The rows require `beta>=s` and `eta<=1-t`. A-positive forces beta only, E-negative forces eta only, and their other bits and the origin remain independently selectable. Thus both target bit means can be attained.
- If `d<=0,e>=0`, use `s=-d/a`, `t=e/a`. Here `s+t=-delta/a<=1`, with `eta>=s` and `beta<=1-t`. The symmetric construction again attains both means.

Fractional free-bit means in this proof are mixtures of legal binary endpoint variants, not a relaxation of the represented HZ bits. This establishes the reverse hull containment. The construction is a finite mathematical proof; an implementation needs only the four linear rows, not a branch over the quadrants or phases.

The current theorem is stated for the ordinary crossing case `L<0<U`. No special numerical-case implementation is proposed. Legal zero activations are nevertheless fully covered by the theorem.

## Relation to phase-free difference bounds

Projecting out beta and eta gives the known phase-free sector hull:

```text
L <= delta <= U,
L*(U-delta)/(U-L) <= d <= U*(delta-L)/(U-L).
```

For example, combining `d<=b*beta` and `e<=a*(1-beta)` gives `(a+b)*d<=b*delta+a*b`. The lower envelope is symmetric. Conversely, any point between those envelopes can choose bits satisfying (1); the ideal-hull theorem also gives this projection directly.

Retaining the bits is therefore not cosmetic. Intersecting phase-free difference bounds with independently relaxed gates can lose correlations that (1) retains. This is an integer-preserving relation: the concrete semantics still uses binary original bits, even though its terminal relaxation uses their convex bounds.

## Strict separation without a zero or boundary example

Use `(x,y) in [-1,1]^2` and nonparallel gates

```text
f=x+y/4,     h=x-y/4.
```

Their individual bounds are `[-5/4,5/4]`; the input difference has bounds `[-1/2,1/2]`. Consider the relaxed tuple

```text
(x,y)=(1/20,0),
r=3/10, p=1/10, beta=1/4, eta=1/2.
```

Both mean preactivations are strictly positive, `f=h=1/20`. The tuple passes the eight ordinary gate rows. It also passes the exact phase-free difference hull: `delta=0,d=1/5`, while its upper envelope is `1/4`. But (1) requires `d<=U*beta=1/8`, which it violates by `3/40`.

The tuple even passes the full retained-input ideal hull of each gate separately. For the f gate, mix an active point `(x,y)=(24/25,24/25)` with weight `1/4` and an inactive point `(-19/75,-8/25)` with weight `3/4`. Their preactivations are respectively `6/5` and `-1/3`; the mixture gives the stated source mean, beta and r. For the h gate, mix `(1/5,0)` active and `(-1/10,0)` inactive with equal weights. This gives the same source mean, eta and p.

All four constituent source points lie strictly inside the box and have strictly signed relevant preactivations. Thus the separation does not depend on a zero-bit ambiguity, box boundary or extreme numerical scale. The two independent gate hulls use incompatible source mixtures, explaining why their intersection permits this tuple. This is a paper separation, not an executed CERT or adversarial witness.

## Safe integration and honest cost

For an actual bounded shared HZ source, keep the entire original source relation, both original gates, bits and decoder. Add (1), with `delta=f-h` and `d=r-p` interpreted as existing affine expressions. Validity makes the complete integer relation unchanged, including all original zero bits; retaining the original rows gives LP containment automatically. No source copies or new continuous variables are needed for this form.

The local ideality theorem must not be extended to this full source gluing. Nor may (1) replace the original eight gate rows: the projected factor deliberately forgot the common baseline. D019's retained-source counterexample still applies to that kind of inference. A graph of pairwise ideal factors also does not automatically have a globally ideal hull.

Each edge adds four rows, not zero rows. If `s_delta` counts the actual nonzero source coefficients of `f-h`, with source, current outputs and bits in distinct coordinates, the added predicate nnz is `12+2*s_delta`, excluding RHS and scalar bounds. If `L<=delta<=U` is not already implied in the retained LP and one wants the complete ideal local factor, add those two rows and another `2*s_delta` nnz. Actual canonical coefficient collisions must be counted rather than using these formulas blindly. Explicit difference columns trade fill for extra variables, equality rows, bounds, metadata and reconstruction links; they are not free.

Exactness does not by itself establish better wall time or a new abstract domain. Ordinary HZ supplied these same valid predicates reproduces the same integer relation and relaxation. The factor supplies a precise semantic building block and a stronger comparator; a broader definition-level contribution still needs a compositional and complete-cost advantage.

## GPU operator form and the remaining composition problem

For a fixed structurally selected graph of neuron pairs, let `B` be its oriented incidence matrix, `f=W z+b0`, `d=B r`, and `delta=B f`. The four edge blocks are diagonal phase scalings plus `B r` and `B(r-f)`. Matrix-vector evaluation can therefore reuse the original affine/Conv operator, followed by edge gathers; the transpose uses the corresponding scatter and original transpose operator. It need not materialize a dense composed `B W` just to evaluate that same linear system.

This is a proposed GPU implementation form, not a measured kernel or a new numerical proof. It still pays for four multipliers per edge, optional difference-bound rows, edge metadata, all original gate work, temporary buffers, both operator directions, directed-rounding or other certified evidence checks, and terminal integration. Vendor LP APIs may require an explicit matrix; matrix-free mathematics does not prove that a selected API accepts callbacks. Transpose multiplication inside the fixed terminal LP is ordinary solver arithmetic, not network backward/dual rescue.

No all-pairs quadratic graph, instance-specific selection, solver-state trigger, phase enumeration or extra optimization loop is proposed. A future graph and certified difference-bound rule must be fixed by mathematical structure and preregistered before experiments. Affine/Add/Concat relations compose exactly in a shared source frame; the unresolved contribution is how these phase-aware factors propagate usefully through the next nonlinear block without retaining an ever-growing copy of known cuts. This theorem alone does not solve that question.

## Rejected residual shortcuts and prior art

Anchored values `r=p+d` with certified `f>=h` satisfy `d=clip(f,0,f-h)`. Retaining the anchor and all original bits gives an exact calculus. Its eight-row LP-strengthening basis is exactly the preceding D018 order theorem in new coordinates; generic dense rows add three predicate nnz versus the original pair, not a compression. Affine/Add/Concat composition is merely the associated invertible coordinate transform unless a further theorem removes cost.

For `f=P-N`, `P,N>=0`, with original gate bounds `ell<0<u`, put `o=min(P,N)` and `r=P-o`. The old four rows become `o<=P`, `o<=N`, `o>=P-u*beta`, `o>=N+ell*(1-beta)`: one variable and four rows remain. Sharing overlap under sums with `c_j>0` is exact precisely when the nonzero differences `P_j-N_j` have the same sign pointwise, because

```text
min(sum c_j P_j, sum c_j N_j) - sum c_j min(P_j,N_j)
 = min(sum c_j (P_j-N_j)^+, sum c_j (P_j-N_j)^-).
```

This returns to known co-sign lumping or the earlier disjoint-amplitude rule. For a generic mixed bank, a new overlap is simply the new nonlinear gate under another name. Neither shortcut is promoted as a new definition.

Difference tracking itself is established prior art. [ReluDiff](https://arxiv.org/pdf/2001.03662), [NeuroDiff section 4.3](https://arxiv.org/pdf/2009.09943) and [DiffPoly section 4.2](https://ggndpsngh.github.io/files/raven.pdf) derive relational activation bounds. D011 had already identified DiffPoly; this is a more specific comparison, not its first discovery. ReluDiff's sampling, backward refinement and input subdivision are not imported. NeuroDiff's convex difference envelopes are useful comparators, not substitutes for the retained nonconvex HZ relation. Generic disjunctive/ideal formulations and Sharp HZ also remain necessary novelty comparators. The four-row proof has been independently checked, but no exhaustive novelty search or claim of superiority over every rule in these systems has been completed.

## Evidence status

Date 2026-09-30, branch `redu-hz`, commit `f1bc0f16612bd3f2112f2970174ec2f7cc3bf5ac`, pre-existing dirty worktree. This file is paper mathematics and primary-source comparison only. Root derived the nonordered factor, signed-axis hull proof and strict interior witness; an independent mathematics reviewer verified them. Separate residual/overlap audits supplied the rejected-shortcut accounting. No numerical test or model run of this factor has occurred. A separate initialization-only GPU diagnostic in this turn does not test or qualify this factor. Formal baseline 1870/2413 and independent external 61/400 remain unchanged.
