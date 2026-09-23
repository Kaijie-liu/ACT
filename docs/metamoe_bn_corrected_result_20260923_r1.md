# Repaired BN same-object control: PASS, not a robustness result

Implementation `45f9a2094`, frozen manifest `cb91e48cd`, one execution;
[saved-only audit](metamoe_bn_corrected_archive_20260923_r1.json) PASS, 0 issues.
Same original checkpoint, physical MNIST0 float64 tensor, epsilon, label and
19 properties. Only converter source was explicitly rebound. No old manifest,
failed attempt, numerical gate, or old UNKNOWN was overwritten/relabelled.

| Diagnostic | Observed |
| --- | ---: |
| Source vs padded expert | 0 |
| Source vs concrete IR | 5.729e-14 |
| Source vs HZ output | 5.462e-14 |
| Maximum of 26 layer IR/HZ errors | 8.482e-14 |
| Fresh proposal + full stored-matrix check | 0.024294 s |
| Supervised execution including preflight | 7.218234 s |
| Full orchestration including postflight | 7.805485 s |
| New native queries | 0 |

Fresh matrix: 7,950 variables / 4,879 rows, fingerprint
`832189bb57759bb5700b0217d743c5c578d13c53293bca98255b2beda64fd53e`.
Old defective matrix had 6,918 variables / 3,847 rows and is sealed. Correct
BN scaling changes activation bounds and hence factor counts. Do not reuse
old assignments or positive/negative bounds on this matrix.

The independent audit checks stored CSR rows using scalar summation, input
recovery, unchanged source/property arrays, all layer deltas, identity and
outer cost/terminal precedence. It does not rerun the original network or
prove all-domain conversion equivalence. The accepted single point is a
base-feasibility witness under the unchanged float policy, NOT SAFE/UNSAFE.

54 focused tests pass in ACT; 26 conversion/intake/class-separated tests also
pass in the pinned intake environment. Historical defective toy JSON remains
unchanged. New tests exercise branches, repeated BN modules, multiple BN,
negative/unit scale, nonaffine BN and BN1d/2d/3d.

Next: opt-in current-assignment base-query integration, retaining original
expert/request budget, all properties, fallback, native infeasibility gates
and complete-model witness replay. No evidence yet identifies a remaining
relaxation defect. Reassess only from corrected-version records. Any old
source-level claim involving this converter's expanded BN graph needs a
source-identity impact review; do not extend the defect to unrelated MLP
experiments or silently regard affected historical outputs as source proofs.
