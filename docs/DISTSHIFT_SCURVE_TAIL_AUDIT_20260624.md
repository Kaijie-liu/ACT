# dist_shift S-Curve Tail Audit 2026-06-24

Scope: strict pure HybridZ.  No input split, sampling, LP-witness promotion,
CROWN/backward tightening, Gurobi-counted proof, or per-iid tuned rescue path.

## Current Frozen State

Frozen headline:

`/data1/Kane/ICSE/act_hybridz_clean_20260624_cora25/`

`dist_shift_2023`: `70/72 = 70 CERT + 0 ADV`, `P0=0`.

The two unresolved rows are `iid39` and `iid42`.  In the frozen frontier they
are classified as `s_curve_operator_tail`.

## Existing Best Evidence

Main frozen source:

`audit_results/hz_distshift_k8_stage_full_20260624/dist_shift_2023.jsonl`

Earlier diagnostic logs:

`audit_results/hz_distshift_graphcuts_scout_20260623/iid39_scip.log`

`audit_results/hz_distshift_graphcuts_scout_20260623/iid42_scip.log`

Both rows use the current sound S-curve configuration:

- compressed/pruned sigmoid;
- `K=2` S-curve pieces in the targeted diagnostic;
- domain cuts and graph cuts;
- cutoff-row exact-HZ sparse MILP.

Observed state sizes:

| iid | n_cont | n_bin | n_eq | n_ub | eq_nnz | ub_nnz | result |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 39 | about 5.1k | about 2.5k | about 1.6k | about 27.2k | about 259.6k | about 347.1k | SCIP root timelimit |
| 42 | about 4.7k | about 2.3k | about 1.6k | about 24.9k | about 240.0k | about 321.6k | SCIP root timelimit |

The LP/MILP wall is not representation blow-up.  It is a tightness/root-bound
problem in the S-curve plus downstream ReLU exact-HZ system.

## Midpoint Tangent Experiment

Candidate tried locally: add one extra midpoint tangent cut per S-curve segment.
This was intended as a sound local tightening of the sigmoid graph relaxation.

Audit artifact:

`audit_results/hz_scurve_midpoint_audit_20260624/`

Toy audit:

`audit_results/hz_scurve_midpoint_audit_20260624/scurve_midpoint_audit.json`

The candidate was sound on the toy oracle, but did not improve the canonical
correlated test `sigmoid(x) + sigmoid(-x) = 1`.  For `K=1/2/4`, the LP gap was
unchanged with and without the midpoint cuts.  This is the important negative
signal: more local tangents do not capture the missing cross-neuron
correlation.

Real dist_shift LP-only smoke:

| iid | config | n_ub | ub_nnz | LP margin |
| ---: | --- | ---: | ---: | ---: |
| 39 | K=2 midpoint | 29636 | 411828 | -8439.385688277773 |
| 39 | prior K=2 no midpoint log | about 27.2k | about 347.1k | -8439.716385978834 |
| 42 | K=2 midpoint | 27190 | 381700 | -410.0596308169743 |
| 42 | K=2 no midpoint current control | 24936 | 321626 | -410.05965401854473 |

The improvement is effectively zero while adding about 60k-65k upper-bound
nonzeros per row.  This is not a useful production tradeoff.

## Decision

Do not promote midpoint tangent cuts.

The temporary midpoint implementation was removed after the audit.  Current
source files compile, and no residual midpoint flags or references remain in:

- `scripts/cifar_sparse_exact_probe.py`
- `scripts/hz_sparse_worker.py`

## Pair-Correlation Check

A stronger exact S-curve correlation was also audited:

`audit_results/hz_scurve_pair_correlation_audit_20260624/`

If two sigmoid preactivations satisfy `x_j = -x_i`, then
`sigmoid(x_i) + sigmoid(x_j) = 1` can be added as an exact sparse equality.  On
a toy HZ this closes the current sigmoid relaxation gap completely.  However,
the `dist_shift` sigmoid input layer has no exact complement pairs and no near
pairs even at tolerance `1e-3`; the best pair has relative weight-sum error
about `0.794`.  Therefore this cut is sound but not applicable to the current
`dist_shift` model.

## Next Valid dist_shift Work

The remaining useful path is not "increase K" or add more local tangent rows.
The missing strength is correlation-aware:

1. Add sound aggregate S-curve cuts that couple groups such as
   `sigmoid(x) + sigmoid(-x)`, validated first by a toy exact oracle.
2. Improve structural sparse MILP presolve for the root problem, again with a
   benchmark-wide setting rather than per-iid tuning.  Simple row scaling was
   audited separately and is not a promising axis.
3. Keep the frozen `70/72` result until a full clean dataset rerun proves a
   benchmark-wide improvement with `P0=0`.
