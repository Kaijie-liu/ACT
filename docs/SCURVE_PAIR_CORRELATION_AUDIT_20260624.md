# S-Curve Pair Correlation Audit 2026-06-24

Scope: strict pure HybridZ.  This audit tested one sound, structural
correlation cut for sigmoid layers.  It did not use input split, sampling,
LP-witness promotion, CROWN/backward tightening, Gurobi-counted proof, or
per-iid rescue.

## Candidate

If two sigmoid preactivations satisfy the exact affine relation

`x_j = -x_i`,

then the outputs satisfy

`sigmoid(x_i) + sigmoid(x_j) = 1`.

This equality can be added to the post-sigmoid HZ as a sparse equality row.
It is not a ReLU triangle approximation and it does not change the exact-HZ
semantics when the preactivation relation is exact.

## Toy Oracle

Artifact:

`audit_results/hz_scurve_pair_correlation_audit_20260624/`

Toy setup:

`x_0 = r * xi`, `x_1 = -r * xi`, `xi in [-1, 1]`.

The current `K=2` compressed sigmoid encoding with domain and graph cuts leaves
a relaxation gap in `sigmoid(x_0) + sigmoid(x_1)`.  Adding the exact complement
equality closes the gap:

| radius | before gap | after gap |
| ---: | ---: | ---: |
| 1 | 0.00944823596832256 | 0 |
| 2 | 0.05852804552006097 | 0 |
| 4 | 0.23732443013741134 | 2.22e-16 |
| 8 | 0.5480900875148187 | 0 |

This confirms the cut is meaningful when the structure exists.

## dist_shift Scan

The Dense layer immediately before the `dist_shift_2023` sigmoid was scanned
for exact complement rows.

Result for the model used by `iid42`:

- sigmoid layer: `5`
- preceding dense layer: `4`
- dense rows: `784`
- exact complement pairs: `0`
- near complement pairs:
  - tolerance `1e-8`: `0`
  - tolerance `1e-6`: `0`
  - tolerance `1e-4`: `0`
  - tolerance `1e-3`: `0`
- best pair still has relative weight-sum error about `0.794`, so it is not
  remotely close to a safe complement relation.

Because the model weights are shared across `dist_shift` instances, this is a
model-structure result, not an iid42-only observation.

## Decision

Do not add sigmoid complement-pair cuts to the production `dist_shift` path.

The cut is sound and useful for networks that contain exact complement pairs,
but the current `dist_shift` model does not have that structure.  Applying it
approximately would be unsound and violates the pure-HZ rule.

## Implication

For `dist_shift`, the remaining useful S-curve direction must be broader than
exact pair identities:

1. correlation-aware cuts derived from real affine relations in the pre-sigmoid
   HZ, not approximate weight similarity;
2. aggregate cuts validated by an exact toy oracle before production use;
3. structural sparse presolve for the downstream exact MILP root wall.
