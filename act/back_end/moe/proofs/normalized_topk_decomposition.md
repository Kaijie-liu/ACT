# Normalized top-k decomposition lemma

## Statement

Fix a tie-inclusive feasible unordered route set (S), with (|S|=k), and a
guarded input domain (X_S).  Let every selected expert have output
(E_i(x)\in\mathbb{R}^m).  Suppose the gate is normalized and non-negative:

\[
  \lambda_i(x) \ge 0,
  \qquad
  \sum_{i\in S}\lambda_i(x)=1,
\]

and sound bounds (ell_i\le\lambda_i(x)\le u_i) are available for every
(x\in X_S).  Choose any anchor (b\in S).  Then

\[
 F(x)=E_b(x)+\sum_{i\in S\setminus\{b\}}
       \lambda_i(x)\big(E_i(x)-E_b(x)\big).
\]

Consequently, for a linear safety row (q^T F(x)+c\ge0), it is sound to use
exactly (k-1) product variables

\[
 w_i=\lambda_i d_i,
 \qquad
 d_i=q^T(E_i-E_b),
\]

relax each product with its McCormick hull over
([ell_i,u_i]\times[\underline d_i,\overline d_i]), and retain the omitted
anchor bound through

\[
  1-u_b \le \sum_{i\ne b}\lambda_i \le 1-ell_b.
\]

The encoded scalar is

\[
  y=q^TE_b+c+\sum_{i\ne b}w_i.
\]

If the minimum of this relaxation is proved strictly above the registered safe
tolerance, the original row is safe on (X_S).  A non-positive relaxation
candidate is not a concrete counterexample and yields only `UNKNOWN` unless the
full MoE forward validates it.

## Proof

For any (x\in X_S), eliminate the anchor weight using simplex normalization:

\[
\begin{aligned}
F(x)
 &= \sum_{i\in S}\lambda_i E_i \\
 &= \Big(1-\sum_{i\ne b}\lambda_i\Big)E_b
    +\sum_{i\ne b}\lambda_iE_i \\
 &= E_b+\sum_{i\ne b}\lambda_i(E_i-E_b).
\end{aligned}
\]

The sound weight and difference bounds place each concrete triple
((\lambda_i,d_i,\lambda_i d_i)) inside its McCormick hull.  Normalization also
places the free weights inside the two retained anchor inequalities.  Thus every
concrete guarded execution maps to a feasible point of the relaxation with the
same value of (q^TF+c).  The relaxation minimum is therefore no greater than
the concrete minimum.  A certified positive relaxation minimum proves the
concrete minimum positive.  The converse does not follow, which is why a
relaxation violation cannot establish unsafety.  □

## Gate-family scope

| ACT gate | Applies | Reason |
|---|---:|---|
| `hard_top1` | yes | singleton simplex; zero product terms |
| `selected_softmax` | yes | selected weights are positive and normalized |
| `normalized_sigmoid` | yes | selected sigmoid weights are explicitly normalized |
| `switch_prob` | no | selected-expert probability is an unnormalized scale |

`switch_prob` requires a separate scale product even when (k=1).  Treating it
as the singleton simplex weight would replace (p_iE_i) by (E_i) and is not
sound.  The implementation therefore rejects it explicitly.

Tie inclusion is handled outside the lemma: every legal unordered top-k set at
a tie is a separate guarded obligation.  Safety requires all such obligations
to be proved.

## Top-2 compatibility

For (k=2), choose the second canonical expert as the anchor.  The construction
has one free weight and one product, giving

\[
  F=E_b+\lambda_a(E_a-E_b),
\]

which is exactly the existing F0 decomposition.  Supplying the same guarded
top-2 gate range produces identical disagreement bounds and McCormick rows.
