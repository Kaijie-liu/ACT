# Saved-record localization of the complete ranged-source proof gap

This is a **no-solve diagnostic**, following the closed two-arm range study.
The sole input is its reviewed `range_on` package, input98, pair[1,2], all
nine required classification properties. Every package file is checked against
the externally archived manifest identity before reading source projections
and stored candidate vectors. No model forward, checkpoint loading, new HZ
propagation, native call, new range, time-limit increase or certificate occurs.

## Fixed analysis, not a new safety checker

The last ReLU precedes a final Linear classifier, so its direct margin effect
can be written exactly. From the saved append-only trace, map each local factor
**by identity** into the final joint LP frame. For every saved point, evaluate
the layer6 preactivation a, layer7 relaxed output h, and final expert margins.
All computations use rational interpretations of stored numbers. All last-layer
rows and all nine properties remain in the report; no successful subset is
selected. Missing points remain missing, not zero-gap observations.

For expert i and competitor k, let c_ij=W_i[label,j]-W_i[k,j]. At that one
stored point the signed last-ReLU effect is

`R_ik = weight_i * sum_j c_ij * (h_ij - max(a_ij,0))`.

The sign matters: a large unsigned activation discrepancy need not hurt the
property. Negative contributions are reported as diagnostic harmful effects;
positive ones offset them. Retain the residual between the LP final-output
coordinates and the affine reconstruction from h, rather than assuming that
the approximate point satisfies the defining equalities exactly.

If the LP objective is J=u+w, the saved free gate is lambda, and d is the
expert-margin difference, define P=lambda*d-w and the weighted final-affine
residual A. Check the exact accounting identity

`J + P - A - sum_i R_ik = weighted margin with last ReLU replaced`.

Both J+P and the final replacement are **values at a modified, unverified LP
assignment**, not optimized bounds, feasible repaired points or network
evaluations. Earlier activations/input relations and the correspondence of
lambda to the router remain unchecked in this diagnostic. An apparently
positive replacement therefore proves no safety; a negative one proves no
unsafety or intrinsic gap. The old independent source/bound checks are not
silently converted into a primal-feasibility check.

## Range-only priority versus observed-point priority

For an unstable scalar ReLU with l<0<u, its continuous triangle has maximal
vertical gap `g=-l*u/(u-l)`, attained at a=0. This follows by comparing the
chord `u*(a-l)/(u-l)` with max(a,0) on either side of zero. Stable and zero-width
ranges have g=0. Conditional on this scalar hull and the range, a negative
classifier coefficient can incur at most max(0,-c)*g in the **local unweighted
margin**. We report this local-envelope potential for every property; it is
not an attainable joint-LP loss or a guaranteed gain from tighter ranges.

Separately rank every still-unstable last-layer row by its maximum observed
harmful signed contribution across all nine stored points, with expert/row
tie breaks. This is a post-result development heuristic, not a frozen
production selection policy. Record stable-row discrepancies, negative ReLU
gaps, gate values, preactivation range excess and local triangle-gap excess;
do not assume the approximate point is feasible merely because violations
are small. All earlier ReLU branch counts are retained as context, not
attributed to the last layer.

## Controls and deliverable

Controls cover zero/tie ranges, exact triangle gap and convex-hull examples,
signed property accounting including nonzero affine residual, shared/private
factor mapping with permuted coordinates and rejected aliases, and changed
package/partial-review rejection. They call no solver. The real diagnostic
retains all exact aggregate terms, the full hidden-row inventory, every
property's observed unstable rows, and identity/cost metadata in a new result
directory. Rounded row-level displays are labeled as such and are not proof
inputs. The exact aggregate decomposition must have residual zero.

Run with the existing environment, without site packages:

```sh
python -S -m unittest range_diagnosis.tests
python -S -m range_diagnosis.analyze --output data/moe/results/range_diagnosis_conv98_20260920_v1
```

The next decision must follow these saved-record results. Do not automatically
run the top-ranked rows, expand to a full hidden layer, alter gates or restart
the sealed arithmetic/solver searches. A targeted intervention would be a
separate bounded development protocol, requiring fresh ranges, downstream
matrices and complete new output evidence. The original single-route67.06%
control cannot fulfill the high-accuracy/cross-family route-changing goal.
