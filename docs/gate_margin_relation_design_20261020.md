# M3 static design: do not rename N1 slicing as a new mechanism

2026-10-20. Initial mathematical/source review at `9a92570c9`, strengthened by
the [static integration and strong-baseline audit](gate_margin_relation_static_audit_20261020.md)
at `4edc5f882`. No solver, native control, trained model or GPU experiment for
M3. G1 remains OPEN. The bounded next step is a default-off two-row mechanism,
not another M1b observer repair or a claim of new sigmoid relaxation theory.

## Existing mechanism boundary

`conditioned_difference_support.py::condition_on_affine_path_interval` already
adds two closed margin constraints to shared factors. Its
`build_segmented_weighted_top2_f0` carries the same slice into router, input and
joint expert domains, recomputes ranges, then builds F0 (N1).
`hz_endpoints.py::prepare_request` already makes constant-gate endpoint support
queries on a shared/private joint HZ (H2). Merely wiring N1 to H2 is a combination
baseline, not a new relationship representation.

`weighted_top2.py::build_weighted_top2_f0` allocates an independent lambda factor
and product factor with McCormick constraints. Its gate interval is conditioned
on the router but the lambda factor has no within-slice constraint tying its
value to the affine router margin. This is a concrete representation difference
worth designing against; no literature novelty or real-request gain is claimed.

## Coverage and honest strength

For each tie-legal canonical pair S={a,b}, preserve the full guard domain P_S,
shared router/input factors and independent expert-private factors. Let
m(z)=r_a(z)-r_b(z). Frozen closed intervals I_j cover a justified margin range.
P_{S,j}=P_S intersect {m in I_j}; their union must cover P_S. Unknown feasibility
retains a slice; deletion requires checked infeasibility. Adjacent endpoints
overlap. All active slices, pairs and properties must be checked and aggregated.

A gate interval J_j inside each slice is still a Cartesian outer approximation:
the true gate graph is contained in union_j(P_{S,j} x J_j), contained in P_S x J.
It is NOT the true sigmoid graph. Nested exact optima may improve; finite-budget
candidate bounds and request outcomes need not be monotone.

Keep two factors separate: global versus fixed sliced gate ranges, and independent
expert-property facts versus shared-input support. H2 and N1+F0 are required native
comparators; any endpoint/McCormick representation difference is separately labeled.

## Nonduplicate candidate: within-slice gate–margin rows

For lambda=sigma(m), sigma(0)=1/2 and 0<sigma'(m)<=1/4 imply:

* m>=0: 1/2 <= lambda <= 1/2 + m/4.
* m<=0: 1/2 + m/4 <= lambda <= 1/2.

Intersect these rational affine constraints with the existing justified gate range.
They must use the actual router margin on the SAME factor vector as the experts,
not a newly independent margin variable. With the original McCormick relation,
the true weighted graph is contained in the new relaxation, which is contained
in the old F0 relaxation on the same slice. This implies only a same-domain exact
F0 optimum comparison. It does not imply general dominance over H2 or finite-time
dominance, and is not a new theorem about sigmoid bounds.

Do not substitute the input-dependent upper bound into the objective and still
call it linear endpoint support: multiplying it by a variable expert difference
is generally quadratic. The proposed difference is a joint constraint in HybridZ.
Its implementation/independent lowering checks remain future work, not accomplished.

## Fixed paper separation and necessary failure controls

Take x=m in [0,1], B=9/16+x/4, A=B-1, so the weighted property is B-lambda.
Use the same two slices [0,1/2], [1/2,1], with legal gate ranges [1/2,5/8]
and [1/2,3/4]. In either slice, independent gate H2/F0 has exact minimum -1/16.
The expert difference is constant -1, avoiding a product-relaxation confound.
Those deliberately coarse ranges are not the strongest N1 comparator.
With the *tight* sigmoid ranges, the old minima instead are
9/16-sigma(1/2)<-3/80 and 11/16-sigma(1)<0. The two-slice separation survives.

Adding lambda<=1/2+x/4 gives B-lambda>=1/16. The relaxed point x=0,lambda=1/2
attains 1/16. The former relaxed point x=0,lambda=5/8 is excluded. These are
algebraic witnesses of strict inclusion/separation, NOT a trained-model certificate,
route-changing request or measured result.
Conversely, the existing N1 with four equally spaced slices already proves this
example: every slice lower bound is at least 9/16-sigma(1/4)>0. Thus this example
does not show that old N1 is incapable of proving the property. Preserve a
pre-fixed four-slice native cost reference rather than claiming unconditional
superiority over the old method.

Required controls before a finite implementation experiment: expert equality,
fixed gate, negative-margin orientation, invalid factor mapping, missing slice,
missing tie pair and unsupported gate-source premise. Check rational projection,
row signs, source binding and outward conversion of any floating implementation.
Retain all upstream G3 assumptions; exact affine rows do not repair input inclusion.
Do not trust a floating `margin_bounds` sign or a slice label as the new row's
premise. The linked audit specifies exact box/direct-retained-row premises, the
native lambda center/radius substitution, and the outward row error check.
N1's slice rows are not a complete constraint prefix of the joint expert HZ
when private expert constraints intervene; bind the unsliced frame and explicit
retained rows instead.

Continue only if the representation differs from existing N1/H2 and controlled
same-domain comparisons substantiate the difference. Stop or narrow the claim if
the new rows are redundant or complete-request losses/costs outweigh gain. Do not
claim delta-completeness without an independently established feasible-point,
progress and numerical-error contract. No CEGGR or GPU execution is authorized by
this static design document; they remain separate factors.

## Literature and execution boundary

[GenBaB](https://arxiv.org/html/2405.21063v2), sections 3.1–3.3 and appendix A.1,
already combines general nonlinear branching, interval-dependent linear bounds
and multiplication relaxations. Neither margin slicing nor a sigmoid affine
envelope alone establishes MoE-specific novelty. The candidate here restores a
missing connection in ACT's current F0 representation. Any broader contribution
still requires shared-guard/factor justification, direct ablation and complete
request evidence against a reasonable strong implementation.

The next implementation may reuse existing independent rational construction,
gate-range and row-enclosure checks, but must not silently reset their separate
Work ledgers. This review does not activate a real-request queue, deploy GenBaB,
repeat M1b, enlarge resource caps or close G1/G3.
