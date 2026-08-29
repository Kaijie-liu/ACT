# Retained-path conditioned support lemma

## Statement

Let \(X\) be a hybrid-zonotope domain, let \(m:X\to\mathbb R\) be an
affine path expression, and let \(d:X\to\mathbb R\) be any downstream affine
expression represented in the same factor frame.  Choose finite cuts

\[
  \ell_m=t_0<t_1<\cdots<t_s=u_m
\]

from sound support bounds \(\ell_m\le m(x)\le u_m\).  Define closed retained
path domains

\[
  X_j=X\cap\{x:t_j\le m(x)\le t_{j+1}\}.
\]

Then:

1. \(X=\bigcup_j X_j\), including every point on a cut;
2. for every segment, conditioned support is monotone,
   \(\inf_X d\le\inf_{X_j}d\) and
   \(\sup_{X_j}d\le\sup_X d\);
3. if \([\ell_j,u_j]\) soundly encloses \(d(X_j)\), then

   \[
     d(X)\subseteq
     \bigcup_{j:X_j\ne\varnothing}[\ell_j,u_j]
     \subseteq
     [\min_j\ell_j,\max_j u_j].
   \]

The result is independent of the gate family.  In particular, partitioning an
affine router margin and recomputing \(q^T(E_a-E_b)\) support does not encode
softmax, sigmoid, exponentiation, division, or a gate-function segment.

## Proof

Every \(m(x)\in[\ell_m,u_m]\) lies in at least one consecutive closed
interval; if it equals a cut, it lies in both adjacent intervals.  Therefore
the retained domains cover \(X\), including ties.  Since \(X_j\subseteq X\),
minimizing over \(X_j\) cannot yield a smaller infimum than minimizing over
\(X\), and maximizing over \(X_j\) cannot yield a larger supremum.  Finally,
each concrete \(x\in X\) belongs to some nonempty \(X_j\), whose sound support
interval contains \(d(x)\).  Taking the union, or its interval hull, therefore
remains sound.  □

## Executable policy

`condition_on_affine_path_interval` appends the two weak inequalities directly
to the shared generator frame.  It rejects frame-id matches that have lost or
changed the retained path constraints.  Adjacent segments deliberately overlap
at cuts, so a zero router margin and all other ties are covered by both legal
branches.

`segmented_affine_conditioned_support` intersects each raw segment support with
the unconditional support interval.  This realizes the monotonicity lemma even
when a solver side uses its sound fast fallback.  Infeasible segments are
omitted only after a feasibility proof; unknown segments stay in the union.
Any numerical intersection conflict falls back to the unconditional interval
and is explicitly counted in telemetry.

This mechanism is N1, not F1.  Its telemetry fixes
`segmentation_axis=affine_path_margin`, `gate_function_encoded=false`, and
`sigmoid_segments=0`.  A future gate-function segmentation must remain a
separate, explicitly named ablation.
