# M3 static design: do not rename N1 slicing as a new mechanism

2026-10-20. Read-only mathematical/source review at `9a92570c9`; no solver,
native control, trained model or GPU experiment. G1 remains OPEN.

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

Adding lambda<=1/2+x/4 gives B-lambda>=1/16. The relaxed point x=0,lambda=1/2
attains 1/16. The former relaxed point x=0,lambda=5/8 is excluded. These are
algebraic witnesses of strict inclusion/separation, NOT a trained-model certificate,
route-changing request or measured result.

Required controls before a finite implementation experiment: expert equality,
fixed gate, negative-margin orientation, invalid factor mapping, missing slice,
missing tie pair and unsupported gate-source premise. Check rational projection,
row signs, source binding and outward conversion of any floating implementation.
Retain all upstream G3 assumptions; exact affine rows do not repair input inclusion.

Continue only if the representation differs from existing N1/H2 and controlled
same-domain comparisons substantiate the difference. Stop or narrow the claim if
the new rows are redundant or complete-request losses/costs outweigh gain. Do not
claim delta-completeness without an independently established feasible-point,
progress and numerical-error contract. No CEGGR or GPU execution is authorized by
this static design document; they remain separate factors.
