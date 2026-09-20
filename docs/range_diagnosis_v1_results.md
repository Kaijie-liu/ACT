# Saved-record diagnosis: final ReLU and weighted-product effects

The [specified no-solve analysis](range_diagnosis_v1.md) completed on the
reviewed range-on source and all nine saved output candidates. It took0.747s,
with **zero new solver calls, source propagations or model forwards**. Full
row-level diagnostics stay in the new local directory
`data/moe/results/range_diagnosis_conv98_20260920_v1`; the
[compact committed result](range_diagnosis_v1_results.json) binds that raw
analysis, source/package/review identities and all nine exact aggregate terms.
No old record or complete-request verdict changed.

## Source locations, not just a total binary count

| Expert | First ReLU unstable | Second ReLU unstable | Last ReLU unstable |
|---|---:|---:|---:|
| 1 | 545 | 490 | 36 |
| 2 | 356 | 175 | 16 |

These sum to the reviewed1,618 binaries. The last layer contributes52of128
rows, not the entire source gap. All four rows queried in the previous fixed
prefix are now inactive; they do not appear in the remaining unstable roster.
The new diagnostic has not proved the earlier1,566 activations harmless.

## Exact accounting at each UNVERIFIED saved point

J is the original relaxed objective. The columns below substitute the actual
scalar product and/or max(a,0) **at the saved LP assignment**, holding the
remaining factors and the free gate fixed. Final-affine residuals are retained.
These are not newly optimized or certified lower bounds.

| Competitor | Original J | Product only replaced | Last ReLU only replaced | Both replaced |
|---|---:|---:|---:|---:|
| 1 | -73.10270 | -13.97158 | -53.84842 | 5.28271 |
| 2 | -66.22245 | -12.22495 | -52.33386 | 1.66363 |
| 3 | -59.34977 | -13.94657 | -41.69362 | 3.70958 |
| 4 | -65.15968 | -11.30078 | -50.24207 | 3.61683 |
| 5 | -64.66546 | -13.29118 | -45.61028 | 5.76401 |
| 6 | -76.33748 | -17.94011 | -55.84370 | 2.55366 |
| 7 | -74.75274 | -11.44161 | -58.01413 | 5.29700 |
| 8 | -67.34928 | -9.61406 | -50.92464 | 6.81058 |
| 9 | -70.78957 | -12.61982 | -51.05299 | 7.11675 |

All nine accounting identities have exact rational residual zero. The product
term contributes45.4032 to63.3111; the signed last-ReLU correction contributes
13.8886 to20.4938 to the displayed margin. The largest absolute final-affine
residual is1.16e-14, which is retained rather than rounded away. Only replacing
both local relationships produces positive values at these particular points.

This gives a **localized interaction hypothesis**, not a unique-root-cause
proof. Reoptimization could select entirely different points. The replacement
need not satisfy the source equalities or earlier graph constraints, and the
free lambda need not equal the actual router softmax. Indeed, all nine saved
vectors have negative local last-ReLU gaps of magnitude1.11e-16 to1.88e-15;
they must not be silently promoted to exact feasible witnesses. Zero measured
preactivation range/triangle-gap excess is not full LP feasibility. There is
no new SAFE, UNSAFE, checked upper bound or proof that a particular strengthened
LP will succeed.

## Which hidden rows matter in these saved points?

Rank every remaining unstable row by its largest harmful signed margin
contribution across all nine points. This is post-result **development
localization**, not a production policy or a selection of successful cases.
The complete52-row ranking is in the committed result.

| Expert / row | Largest observed harmful contribution | Competitor |
|---|---:|---:|
| 1 / 10 | 4.22176 | 7 |
| 1 / 30 | 3.72582 | 8 |
| 1 / 16 | 3.06582 | 5 |
| 1 / 17 | 2.70109 | 9 |
| 1 / 43 | 2.55812 | 6 |
| 2 / 31 | 0.57284 | 3 |
| 2 / 17 | 0.31955 | 9 |

Unlike the old prefix rows, expert1/row10 has range[-14.9892,16.0826] and
row30 has[-14.5047,12.0501]. Their scalar-triangle gaps are7.75834 and6.58198.
A separate range-only score, `max_k max(0,-c_k)*[-l*u/(u-l)]`, ranks expert1
rows10,30,16,54 first, with scores7.40254,6.23849,5.28308,5.15087. The first
three agree with observed-point ranking, while the fourth differs (54versus17).
This range-only score is computable before output solving from current source
bounds and final classifier coefficients; it need not borrow a saved outcome
for free. Neither score is a guaranteed achievable improvement. Full exact
per-property range potentials for all128 rows remain in the raw analysis.

## Verification and research decision

Eight no-solve controls cover triangle/tie arithmetic, signed decomposition,
nonzero affine residuals, permuted/aliased factor mappings, manifest drift,
partial records, complete rosters, false positive claims and exact accounting
that would be hidden by decimal rounding. A fresh saved-record recomputation
matches every non-timing field; summary reconstruction also matches exactly.
This is an accounting/recomputation check, **not an independent reproof of
all network bounds or any stored primal vector**.

```sh
python -S -m unittest range_diagnosis.tests
python -S -m range_diagnosis.summarize data/moe/results/range_diagnosis_conv98_20260920_v1/analysis.json
```

The next justified control would study **property-directed row selection**,
not automatically expand the number of queried rows or optimize parsing again.
The range-only score provides a prospectively computable candidate rule;
the observed-point ranking is corroborating development evidence. Before
execution, fix the same finite query/total budget, select one rule, regenerate
all downstream matrices and require all nine properties. Costs of selection
must be charged. Do not treat these saved replacements as certificates or
assume that only changing ranges will suffice: even replacing every last-ReLU
value at the old points leaves the unrepaired weighted objective negative.

No such new control has been frozen or run here. Changing gate/weighted
representation would be a separate research decision, not reopening the
sealed gate/backend searches or mixing an extra intervention into row selection.
Input98 remains single-route and the model67.06% accurate. High-accuracy,
cross-family route-changing strict output evidence remains open.
