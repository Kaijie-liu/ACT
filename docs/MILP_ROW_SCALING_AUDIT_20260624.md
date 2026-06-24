# MILP Row Scaling Audit 2026-06-24

Scope: strict pure HybridZ.  This audit only tested solver conditioning for
the exact sparse-HZ MILP.  It did not use input split, sampling, LP-witness
promotion, CROWN/backward tightening, Gurobi-counted proof, or per-iid rescue.

## Candidate

Prototype: scale every MILP row by a positive constant before passing the sparse
linear system to HiGHS.

This is sound in principle because positive row scaling preserves the same
linear feasible set.  The prototype was added behind a local flag, audited, and
then removed because it did not produce a useful improvement.

## Toy Check

A minimal HZ oracle was checked with and without scaling:

- impossible unsafe set: same certified-empty semantics;
- reachable unsafe set: same target/witness semantics.

The toy case also forced a badly scaled row (`1e-9 * x <= 1e-9`), and the
scaled and unscaled outcomes matched.

## Frontier Diagnostics

Artifact:

`audit_results/hz_milp_row_scale_probe_20260624/`

| Bench / iid | Config | Result | Key solver signal |
| --- | --- | --- | --- |
| `dist_shift_2023` / 42 | no row scale | UNKNOWN | HiGHS root timeout, dual `-2032.18`, margin dual bound `-362.23` |
| `dist_shift_2023` / 42 | row scale | UNKNOWN | HiGHS root timeout, dual `-2061.73`, margin dual bound `-391.78` |
| `tllverifybench_2023` / 7 | no row scale | UNKNOWN | HiGHS timeout, `4460` nodes, same margin dual bound |
| `tllverifybench_2023` / 7 | row scale | UNKNOWN | HiGHS timeout, `4104` nodes, same margin dual bound |

For `dist_shift`, row scaling slightly worsened the dual bound.  For TLL it
reduced node count a little but did not change the bound or verdict.  Neither
case justifies a production option.

## Decision

Do not promote row scaling.

The temporary prototype was removed from:

- `scripts/cifar_sparse_exact_probe.py`
- `scripts/hz_sparse_worker.py`

Both files compile after cleanup, and there are no residual `milp-row-scale`,
`milp_row_scale`, or `row_scale` references in those scripts.

## Implication

The remaining root walls are not primarily caused by trivial row magnitude
conditioning.  Useful next axes should target structure rather than scalar
normalization:

1. block/Schur-style exact sparse presolve for large equality systems;
2. correlation-aware S-curve cuts for `dist_shift`;
3. exact MIP probing/compressed phase elimination that removes binaries or
   proves infeasibility, not just improves numerical scaling.
