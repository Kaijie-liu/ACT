# Sparse Exact-HZ MILP Structure Audit 2026-06-24

Scope: strict pure HybridZ.  This audit only inspects sparse exact-HZ MILP
matrices.  It does not run a solver to produce CERT/ADV and does not use input
split, sampling, LP-witness promotion, CROWN/backward tightening, Gurobi, or
per-iid rescue.

## Question

For `tllverifybench_2023`, is the remaining wall caused by decomposable sparse
blocks that a generic component/block presolve could separate?

## Diagnostic

Script:

`scripts/hz_sparse_mip_structure_audit.py`

Artifact:

`audit_results/hz_sparse_mip_structure_audit_20260624/`

The script builds the same z-space exact MILP matrix:

- continuous HZ variables stay continuous in `[-1, 1]`;
- binary HZ variables use `xi_b = 2z - 1`, `z in {0,1}`;
- equality and upper-bound rows are carried as sparse linear rows;
- no verifier verdict is emitted.

It then reports:

- bipartite row/column connected components;
- objective-connected component size;
- exact singleton/equality substitution effect;
- low-cost Fourier-Motzkin candidate counts for objective-free continuous
  variables.

## TLL Findings

### `iid7`

This is a small unresolved timeout row in the frozen artifact.

After exact equality substitution:

- columns: `2042`
- integer columns: `1020`
- rows: `4080`
- nonzeros: `21552`
- components: `1`
- objective-connected component: all `2042` columns, all `4080` rows, all
  `1020` integer columns
- objective support: only `4` columns, but those four columns connect to the
  entire component
- low-cost FM candidates after substitution: `0`

### `iid21`

This is a larger unresolved TLL row.

After exact equality substitution:

- columns: `18818`
- integer columns: `9408`
- rows: `37632`
- nonzeros: `201576`
- components: `1`
- objective-connected component: all `18818` columns, all `37632` rows, all
  `9408` integer columns
- objective support: only `4` columns, again connected to the entire component
- low-cost FM candidates after substitution: `0`

Before substitution there are many apparent low-degree continuous columns, but
they are exactly the columns removed by the current equality substitution.  The
remaining continuous columns have no cheap exact Fourier-Motzkin projection
under the current matrix.

## Decision

Generic connected-component block presolve is not a promising TLL improvement
axis: the hard rows are already one objective-connected component.

Naive low-degree Fourier-Motzkin is also not a promising immediate production
axis after current exact equality substitution: the audit found no low-cost
candidates on the checked unresolved TLL rows.

## cGAN Extension

The same audit was extended to `cgan_2023 iid8 q0`, a 64x64 representation-drop
frontier row:

`audit_results/hz_sparse_mip_structure_audit_20260624/cgan_2023_iid8_q0.json`

Sparse exact-HZ propagation builds a feasible HZ:

- `n_cont=247419`, `n_bin=123708`, `n_eq=123708`;
- `eq_nnz=25499967`, `ub_nnz=494832`, `value_nnz=512`.

Before exact substitution the MILP matrix has:

- columns: `371127`
- rows: `371124`
- nonzeros: `25994799`
- components: `1`
- objective-connected component: the entire matrix

Row-kind breakdown explains the wall: equality rows dominate the matrix
(`123708` rows, `25.50M` nnz), while simple upper-bound rows are tiny
(`247416` rows, `494832` nnz).

Exact equality substitution is representation-negative on this cGAN case:

- removed columns: `123708`
- rows delta: `+123708`
- nnz delta: `+50257686`
- nnz ratio after/before: `2.93x`

So for cGAN, exact equality substitution should not be enabled merely because
it removes variables. The better research axis is preserving the sparse
equality form while finding formulation-aware cuts/presolve that do not densify
the convolutional dependency rows.

### ReLU Valid Cuts on `iid8`

A second audit checked the same row with exact ReLU valid cuts enabled:

`audit_results/hz_sparse_mip_structure_audit_20260624/cgan_2023_iid8_q0_relucuts.json`

The cuts are mathematically valid, but they are representation-negative here:

- before substitution nnz increases from `25.99M` to `76.50M`;
- upper-bound rows increase from `247416` rows / `494832` nnz to `494832` rows /
  `50.99M` nnz;
- after exact substitution the matrix reaches `126.76M` nnz;
- exact substitution is still negative, with the same `50.26M` nnz increase.

Conclusion: for cGAN 64x64 sparse exact-HZ rows, blanket ReLU cuts should not
be part of the default portfolio.  A future version would need a selective cut
policy that is justified by local row support and measured matrix growth.

## Next Valid TLL Work

Useful work needs to be more formulation-aware:

1. exact ReLU phase/formulation compression that removes binary variables or
   proves infeasibility, not generic component splitting;
2. solver portfolio scheduling on the exact same MILP formulation under the
   official wall;
3. TLL-specific algebraic compression only if it can be expressed as an exact
   forward HZ operator/presolve and passes a toy exact-MILP oracle first.
