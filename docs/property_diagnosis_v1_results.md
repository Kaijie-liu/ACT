# Post-selection diagnosis: stop the input98 follow-up

**Decision: STOP_INPUT98_FOLLOWUP.** Analyze the eight shared properties only;
do not retry property7 or start another range/gate/row/time search. The saved
records identify how the available bounds improved, but still show coupled
product/last-ReLU symptoms at unverified assignments. They do not isolate a
new representation intervention beyond the coupling already used to motivate
this finite selection experiment. No next control protocol is proposed here.

This closes the [bounded diagnosis](property_diagnosis_v1.md), not the overall
MoE project. High-accuracy, cross-family route-changing strict positivity
remains open. Input98 is a single-route,67.06%-clean convolutional case and
cannot independently establish those goals.

## Sources and exact scope

Read the [machine-readable accounting](property_diagnosis_v1_results.json)
and [control/recomputation receipt](property_diagnosis_v1_controls.json).
The input is the archived `property_ranges_v1_review.json`, SHA-256
`0b90c33a8c11d5bbadf2c386812c0189ac78a323a0ea0d1e94dafccfdd112760`,
plus its two hash-bound relocated packages. Both manifests and every listed
file are checked before reading the saved records. All eight properties
1,2,3,4,5,6,8,9 are analyzed in each arm's **own** source/factor frame.
No point or dual is transferred between the different matrices.

Every result below is saved-evidence arithmetic, not a new optimized bound.
There are **zero new solver calls, model forwards, source propagations,
primal-feasibility proofs or LP-optimality proofs**. The ninth obligation
remains in each arm's endpoint ledger: prefix has a checked nonpositive bound;
property7 is missing in the property arm. Its point is not analyzed and its
16s failed call is not repeated. All old results, row counts, query counts,
time budgets, sample identities and acceptance gates remain unchanged.

## Where the observed improvement appears

At a saved assignment write `J=u+w`, `P=lambda*d-w`, final affine discrepancy
`A`, signed last-ReLU contribution `R`, and `T=J+P-A-R`. Let `L=D+C` be the
already checked bound and `e=J-L`. Independently check the exact identity

```
Delta L = Delta T - Delta P + Delta A + Delta R - Delta e.
```

The following rounded terms compare the two independently generated points.
They are **accounting components, not causal effect estimates**.

| Property | Checked bound improvement Delta L | Product-gap decrease -Delta P | Signed ReLU change Delta R | Both-replaced value change Delta T |
|---|---:|---:|---:|---:|
| 1 | 7.987574 | 4.205206 | 3.796312 | -0.013944 |
| 2 | 7.455435 | 5.763001 | 1.674585 | 0.017850 |
| 3 | 9.404863 | 4.924011 | 4.360828 | 0.120024 |
| 4 | 8.634644 | 4.746449 | 3.845711 | 0.042484 |
| 5 | 10.731554 | 5.752696 | 4.849579 | 0.129279 |
| 6 | 11.326839 | 5.443935 | 5.913210 | -0.030305 |
| 8 | 10.326936 | 6.772154 | 3.418999 | 0.135783 |
| 9 | 9.268507 | 4.771409 | 4.474202 | 0.022896 |

`Delta A` and `Delta e` are tiny here (absolute values below5.6e-15 and
2.8e-14 respectively); exact rationals, not rounded table entries, close all
eight identities. The benefit is not just a final affine reconstruction
change. This does **not** prove that either arm's exact optimum improved by
the displayed amount: the point/optimum distinction still applies.

Partitioning all128 last-ReLU rows into the four prefix-selected rows, four
property-selected rows and other120 shows the following descriptive pattern:

* The property-selected group's signed contribution improves by1.7811–6.0915.
* The other120 rows become slightly more harmful, by0.1065–0.2304.
* The prefix-selected group changes by0 to-8.19e-5.

The groups sum exactly to `Delta R`; no unselected row is silently removed.
This locates a useful part of the row-selection effect, but all evaluated
factors and gate values can change between arms. It is not a fixed-point
intervention isolating those four rows.

## What still prevents a conclusion

All last-ReLU rows are mapped and counted, including stable rows. Both arms
have52 unstable last-layer rows and1,618 ReLU binaries overall. The split
changes from36/16 to37/15 between experts1/2: removing the prefix tightening
restores one expert1 unstable row, while the property rule fixes one expert2
row active. The preceding ReLU layers retain1,566 unstable relationships;
their complete pointwise graph consistency is **not** checked in this diagnosis.

The new arm still uses a free gate range `[0,1]` and broad difference ranges.
Its recorded difference-box widths remain about179.06–237.57; the saved gate
values are about0.616–0.655. Those values do not establish equality with the
actual router's softmax. At the new arm's own saved points:

| Property | Original J | Product-only J+P | Last-ReLU/affine-only J-A-R | Both T |
|---|---:|---:|---:|---:|
| 1 | -65.115129 | -10.189211 | -49.657155 | 5.268762 |
| 2 | -58.767015 | -10.532520 | -46.553014 | 1.681482 |
| 3 | -49.944909 | -9.465717 | -36.649587 | 3.829605 |
| 4 | -56.525035 | -7.412581 | -45.453141 | 3.659313 |
| 5 | -53.933907 | -8.312320 | -39.728301 | 5.893286 |
| 6 | -65.010640 | -12.057209 | -50.430071 | 2.523359 |
| 8 | -57.022342 | -6.059279 | -44.016700 | 6.946363 |
| 9 | -61.521060 | -8.122721 | -46.258689 | 7.139650 |

Product discrepancies remain40.4792–54.9259; signed last-ReLU contributions
remain-15.4580 to-11.0719. Product-only replacements are negative8/8;
last-ReLU-only replacements are negative8/8; both replacements are positive8/8.
The prefix points show the same sign pattern. **These are not feasible repairs,
model outputs, global bounds or impossibility witnesses.** In particular,
negative single-component replacements do not prove that tightening that
component and re-optimizing would fail.

Every one of the16 saved points has at least one tiny negative local ReLU gap
(largest magnitude1.83e-15). Stable-ReLU discrepancy and tested local range/
triangle upper excess are zero, but those limited tests are not a full exact
constraint check. The evidence therefore does not supply an exact feasible
point, much less a certified optimum. No model-unsafety claim is warranted.

## Residual-box correction is not a roundoff budget

For each old checked certificate we reread `D`, `C`, residual L1 and nonzero
coordinates, and verify `L=D+C` exactly. In the property arm:

| Property | Dual constant D | Residual-box correction C | Unverified J-L |
|---|---:|---:|---:|
| 1 | -46.970039 | -18.145089 | 3.31e-14 |
| 2 | -43.553217 | -15.213798 | 3.81e-14 |
| 3 | -35.314583 | -14.630326 | 2.96e-14 |
| 4 | -40.436510 | -16.088525 | 3.20e-14 |
| 5 | -39.306040 | -14.627867 | 3.55e-14 |
| 6 | -49.429200 | -15.581439 | 5.10e-14 |
| 8 | -42.337940 | -14.684402 | 2.07e-14 |
| 9 | -45.150122 | -16.370938 | 3.77e-14 |

Across arms, `D` improves5.6268–8.0317 and `C` improves1.0021–3.4196;
these changes sum exactly to each bound improvement. Residual L1 is about
14.6279–18.1451 in the new arm, with17,317–17,971 nonzero coordinates.
Finite-box residual minimization is a legitimate part of the lower bound;
its magnitude cannot be called numerical error or simply deleted. Even the
unjustified deletion would leave all eight recorded `D` values negative.

The tiny `J-L` differences do not create a primal-dual optimality certificate:
the points were never exactly validated, and local discrepancies exist.
The checked correction and final affine discrepancies supply **no direct
motivation for another numerical-repair or precision-search experiment** here.
They also do not exclude every possible numerical or representation effect.

## Stop decision and reproducibility

The new evidence accounts for the observed improvement, while the residual
symptom is still the known coupling between a free weighted product and
relaxed expert relations. Neither the source ranges nor the point arithmetic
isolates a new, finite, sufficiently motivated intervention distinct from
the already completed selection control. Earlier layers, gate dependence,
re-optimization and exact feasibility remain unseparated.

Accordingly, **stop chasing a positive certificate on input98**. Do not
automatically freeze another local control, retry7, enlarge a row quota,
shrink epsilon or raise a cap. This is a research stop, not a proof that the
LP family cannot certify or that the network is unsafe. Any genuinely new
direction needs its own evidence and scope, not an extension of this run.

The stdlib-only analysis took1.461s, outside the sealed experimental timing.
It is analysis cost, not a verifier speed result. The independent summary
rechecks exact accounting/rosters, not the full source or bound proofs again.
Thirty-one no-solver controls/regressions pass; a fresh reread reproduces all
nontiming content.81 frozen execution files and13 captured artifacts match
their registered hashes. No original tables or outcomes are rewritten.

Raw diagnostics remain local at
`data/moe/results/property_diagnosis_conv98_20260920_v1/analysis.json`;
764,091bytes, SHA-256
`dfaac323dd4ec64d0939024f4bb0666afb1bf4268c83a2fa12c27df2c2a6de04`.
Exact aggregates and source identities are committed; raw matrices and point
vectors are not. Reproduce by calling `python -S -m property_diagnosis.analyze`
with a **new** `--output` directory, then `python -S -m property_diagnosis.summarize`
on its analysis JSON. Both operations use saved records only, not solvers.
